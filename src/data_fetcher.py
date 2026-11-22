"""High-volume incremental OHLCV and Fear & Greed Index ingestion into SQLite.

Phase 1 module for Aegis Alpha:
- Fetches 15m OHLCV candles from Binance via ccxt with rate limiting, retries,
  pagination (limit=1000, since=last_timestamp + 1), and incremental SQLite
  resumption via SELECT MAX(timestamp) FROM ohlcv WHERE symbol = ?.
- Fetches historical Crypto Fear & Greed Index from alternative.me API with
  BeautifulSoup and Playwright fallback scrapers.
- Stores all data in SQLite configured with WAL journaling and composite PKs.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import requests

from src.config import (
    DB_PATH,
    FEAR_GREED_API_URL,
    FEAR_GREED_WEB_URL,
    MAX_RETRIES,
    OHLCV_BATCH_LIMIT,
    REQUEST_TIMEOUT_SEC,
    RETRY_BACKOFF_SEC,
    SQLITE_BUSY_TIMEOUT_MS,
    START_DATE_ISO,
    SYMBOLS,
    TIMEFRAME,
    TIMEFRAME_MS,
    configure_logging,
    ensure_directories,
)

logger = logging.getLogger("aegis_alpha.data_fetcher")

HTTP_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/html;q=0.9, */*;q=0.8",
}


def parse_iso_to_ms(iso_timestamp: str) -> int:
    """Convert an ISO-8601 UTC timestamp string into Unix epoch milliseconds."""
    normalized = iso_timestamp.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    dt = datetime.fromisoformat(normalized)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.astimezone(timezone.utc).timestamp() * 1000)


def epoch_ms_to_iso(timestamp_ms: int) -> str:
    """Convert Unix epoch milliseconds into an ISO-8601 UTC string."""
    dt = datetime.fromtimestamp(timestamp_ms / 1000.0, tz=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def classify_fear_greed(value: int) -> str:
    """Map numeric Fear & Greed score (0-100) to standard regime label."""
    if value <= 24:
        return "Extreme Fear"
    if value <= 44:
        return "Fear"
    if value <= 55:
        return "Neutral"
    if value <= 74:
        return "Greed"
    return "Extreme Greed"


def get_db_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Open a SQLite connection configured with WAL mode and concurrency pragmas."""
    resolved_path = Path(db_path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(
        str(resolved_path),
        timeout=SQLITE_BUSY_TIMEOUT_MS / 1000.0,
    )
    conn.row_factory = sqlite3.Row

    # Enforce WAL mode and safe concurrency settings on every connection
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.execute(f"PRAGMA busy_timeout={SQLITE_BUSY_TIMEOUT_MS};")
    conn.execute("PRAGMA temp_store=MEMORY;")
    return conn


def init_database(conn: sqlite3.Connection) -> None:
    """Create the ohlcv and fear_greed tables and indexes if absent."""
    with conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ohlcv (
                symbol TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                datetime_utc TEXT NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                PRIMARY KEY (symbol, timestamp)
            ) WITHOUT ROWID;
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_ohlcv_timestamp
            ON ohlcv (timestamp);
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS fear_greed (
                timestamp INTEGER PRIMARY KEY,
                date_utc TEXT NOT NULL UNIQUE,
                value INTEGER NOT NULL,
                classification TEXT NOT NULL,
                source TEXT NOT NULL
            );
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_fear_greed_date
            ON fear_greed (date_utc);
            """
        )


def create_binance_exchange() -> Any:
    """Instantiate a rate-limited public ccxt Binance spot client."""
    try:
        import ccxt
    except ImportError as exc:
        raise ImportError(
            "ccxt is required for OHLCV ingestion. Install dependencies via "
            "'pip install -r requirements.txt'."
        ) from exc

    return ccxt.binance(
        {
            "enableRateLimit": True,
            "timeout": REQUEST_TIMEOUT_SEC * 1000,
            "options": {"defaultType": "spot"},
        }
    )


def get_latest_ohlcv_timestamp(conn: sqlite3.Connection, symbol: str) -> int | None:
    """Query the most recent stored candle timestamp (ms) for a symbol."""
    cursor = conn.execute(
        "SELECT MAX(timestamp) FROM ohlcv WHERE symbol = ?;",
        (symbol,),
    )
    row = cursor.fetchone()
    if row is None or row[0] is None:
        return None
    return int(row[0])


def _get_latest_closed_candle_open_ms(
    timeframe_ms: int = TIMEFRAME_MS,
    now_ms: int | None = None,
) -> int:
    """Return the open timestamp (ms) of the latest fully closed candle."""
    current_ms = int(time.time() * 1000) if now_ms is None else now_ms
    current_candle_open = (current_ms // timeframe_ms) * timeframe_ms
    return current_candle_open - timeframe_ms


def _fetch_ohlcv_page_with_retry(
    exchange: Any,
    symbol: str,
    timeframe: str,
    since: int,
    limit: int,
    max_retries: int = MAX_RETRIES,
    base_backoff_sec: float = RETRY_BACKOFF_SEC,
) -> list[list[Any]]:
    """Fetch a single OHLCV page with exponential backoff on transient failures."""
    try:
        import ccxt

        retryable_errors: tuple[type[BaseException], ...] = (
            ccxt.RateLimitExceeded,
            ccxt.DDoSProtection,
            ccxt.NetworkError,
            ccxt.ExchangeNotAvailable,
            ccxt.RequestTimeout,
        )
    except ImportError:
        retryable_errors = (TimeoutError, ConnectionError)

    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            candles = exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=since,
                limit=limit,
            )
            return candles or []
        except retryable_errors as exc:
            last_error = exc
            sleep_sec = base_backoff_sec * (2 ** (attempt - 1))
            logger.warning(
                "Transient ccxt error for %s at since=%d (attempt %d/%d): %s. "
                "Retrying in %.1fs.",
                symbol,
                since,
                attempt,
                max_retries,
                exc,
                sleep_sec,
            )
            time.sleep(sleep_sec)

    raise RuntimeError(
        f"Failed to fetch OHLCV for {symbol} at since={since} after "
        f"{max_retries} attempts."
    ) from last_error


def upsert_ohlcv_batch(
    conn: sqlite3.Connection,
    symbol: str,
    candles: Iterable[Sequence[Any]],
    max_timestamp_ms: int | None = None,
) -> int:
    """Validate and upsert a batch of OHLCV rows into SQLite."""
    rows_to_insert: list[tuple[str, int, str, float, float, float, float, float]] = []
    for candle in candles:
        if len(candle) < 6:
            continue
        ts_raw, open_p, high_p, low_p, close_p, volume = candle[:6]
        if None in (ts_raw, open_p, high_p, low_p, close_p, volume):
            continue
        ts_ms = int(ts_raw)
        if max_timestamp_ms is not None and ts_ms > max_timestamp_ms:
            continue
        rows_to_insert.append(
            (
                symbol,
                ts_ms,
                epoch_ms_to_iso(ts_ms),
                float(open_p),
                float(high_p),
                float(low_p),
                float(close_p),
                float(volume),
            )
        )

    if not rows_to_insert:
        return 0

    with conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO ohlcv (
                symbol, timestamp, datetime_utc, open, high, low, close, volume
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """,
            rows_to_insert,
        )
    return len(rows_to_insert)


def fetch_symbol_ohlcv_incremental(
    conn: sqlite3.Connection,
    exchange: Any,
    symbol: str,
    timeframe: str = TIMEFRAME,
    start_iso: str = START_DATE_ISO,
    end_ms: int | None = None,
    limit: int = OHLCV_BATCH_LIMIT,
    timeframe_ms: int = TIMEFRAME_MS,
) -> int:
    """Incrementally fetch and store OHLCV candles for a single trading pair.

    Queries SELECT MAX(timestamp) FROM ohlcv WHERE symbol = ? before fetching
    so subsequent runs only pull missing candles.
    """
    latest_stored_ts = get_latest_ohlcv_timestamp(conn, symbol)
    if latest_stored_ts is not None:
        since = latest_stored_ts + 1
        logger.info(
            "[%s] Resuming incremental load from %s (since=%d).",
            symbol,
            epoch_ms_to_iso(since),
            since,
        )
    else:
        since = parse_iso_to_ms(start_iso)
        logger.info(
            "[%s] Starting full historical load from %s (since=%d).",
            symbol,
            start_iso,
            since,
        )

    target_end_ms = (
        _get_latest_closed_candle_open_ms(timeframe_ms=timeframe_ms)
        if end_ms is None
        else end_ms
    )

    if since > target_end_ms:
        logger.info(
            "[%s] Already up to date through %s. Skipping.",
            symbol,
            epoch_ms_to_iso(target_end_ms),
        )
        return 0

    total_inserted = 0
    batch_count = 0

    while since <= target_end_ms:
        candles = _fetch_ohlcv_page_with_retry(
            exchange=exchange,
            symbol=symbol,
            timeframe=timeframe,
            since=since,
            limit=limit,
        )
        if not candles:
            logger.info("[%s] Exchange returned empty page at since=%d; stopping.", symbol, since)
            break

        inserted = upsert_ohlcv_batch(
            conn=conn,
            symbol=symbol,
            candles=candles,
            max_timestamp_ms=target_end_ms,
        )
        total_inserted += inserted
        batch_count += 1

        last_ts = int(candles[-1][0])
        if last_ts < since:
            logger.warning(
                "[%s] Non-advancing timestamp encountered (last_ts=%d < since=%d); stopping.",
                symbol,
                last_ts,
                since,
            )
            break

        since = last_ts + 1

        if batch_count % 25 == 0:
            logger.info(
                "[%s] Progress: %d batches | %d new candles | cursor=%s",
                symbol,
                batch_count,
                total_inserted,
                epoch_ms_to_iso(last_ts),
            )

        if last_ts >= target_end_ms:
            break
        if len(candles) < limit and (last_ts + timeframe_ms) > target_end_ms:
            break

    logger.info(
        "[%s] Completed OHLCV sync: %d candles upserted across %d batches.",
        symbol,
        total_inserted,
        batch_count,
    )
    return total_inserted


def fetch_all_ohlcv(
    conn: sqlite3.Connection,
    symbols: Sequence[str] = SYMBOLS,
    timeframe: str = TIMEFRAME,
    start_iso: str = START_DATE_ISO,
    end_ms: int | None = None,
    limit: int = OHLCV_BATCH_LIMIT,
    exchange: Any | None = None,
) -> dict[str, int]:
    """Fetch incremental OHLCV data for all configured trading pairs."""
    client = exchange if exchange is not None else create_binance_exchange()
    results: dict[str, int] = {}

    for symbol in symbols:
        inserted = fetch_symbol_ohlcv_incremental(
            conn=conn,
            exchange=client,
            symbol=symbol,
            timeframe=timeframe,
            start_iso=start_iso,
            end_ms=end_ms,
            limit=limit,
        )
        results[symbol] = inserted

    return results


def _normalize_date_to_utc_midnight_ms(dt: datetime) -> tuple[int, str]:
    """Normalize a datetime object to UTC midnight epoch ms and YYYY-MM-DD."""
    utc_dt = dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    midnight = datetime(utc_dt.year, utc_dt.month, utc_dt.day, tzinfo=timezone.utc)
    return int(midnight.timestamp() * 1000), midnight.strftime("%Y-%m-%d")


def fetch_fear_greed_from_api(
    url: str = FEAR_GREED_API_URL,
    timeout: int = REQUEST_TIMEOUT_SEC,
    max_retries: int = MAX_RETRIES,
) -> list[dict[str, Any]]:
    """Fetch complete historical Crypto Fear & Greed Index from alternative.me JSON API."""
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, headers=HTTP_HEADERS, timeout=timeout)
            response.raise_for_status()
            payload = response.json()
            raw_entries = payload.get("data", [])
            if not raw_entries:
                raise ValueError("Alternative.me API returned an empty 'data' list.")

            records: list[dict[str, Any]] = []
            seen_dates: set[str] = set()
            for item in raw_entries:
                try:
                    value = int(item["value"])
                    ts_sec = int(item["timestamp"])
                except (KeyError, TypeError, ValueError):
                    continue
                if not (0 <= value <= 100):
                    continue

                dt_utc = datetime.fromtimestamp(ts_sec, tz=timezone.utc)
                ts_ms, date_utc = _normalize_date_to_utc_midnight_ms(dt_utc)
                if date_utc in seen_dates:
                    continue
                seen_dates.add(date_utc)

                classification = str(
                    item.get("value_classification") or classify_fear_greed(value)
                ).strip()
                records.append(
                    {
                        "timestamp": ts_ms,
                        "date_utc": date_utc,
                        "value": value,
                        "classification": classification,
                        "source": "alternative_me_api",
                    }
                )

            if not records:
                raise ValueError("No valid Fear & Greed records parsed from API payload.")

            records.sort(key=lambda r: r["timestamp"])
            logger.info("Fetched %d Fear & Greed records via primary API.", len(records))
            return records
        except (requests.RequestException, ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            sleep_sec = RETRY_BACKOFF_SEC * (2 ** (attempt - 1))
            logger.warning(
                "Fear & Greed API attempt %d/%d failed: %s. Retrying in %.1fs.",
                attempt,
                max_retries,
                exc,
                sleep_sec,
            )
            time.sleep(sleep_sec)

    raise RuntimeError("Fear & Greed primary API failed after retries.") from last_error


def parse_fear_greed_html(html: str, source_name: str) -> list[dict[str, Any]]:
    """Parse Fear & Greed values from alternative.me HTML using BeautifulSoup."""
    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:
        raise ImportError(
            "beautifulsoup4 is required for Fear & Greed HTML fallback parsing."
        ) from exc

    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        soup = BeautifulSoup(html, "html.parser")

    records_by_date: dict[str, dict[str, Any]] = {}
    now_utc = datetime.now(timezone.utc)

    # Strategy A: Look for embedded JSON time-series arrays inside <script> blocks
    date_val_pattern = re.compile(
        r'"(\d{4}-\d{2}-\d{2})"\s*[:,\]]\s*(\d{1,3})\b'
    )
    for script in soup.find_all("script"):
        script_text = script.string or script.get_text() or ""
        if not script_text:
            continue
        for match in date_val_pattern.finditer(script_text):
            date_str, val_str = match.group(1), match.group(2)
            val = int(val_str)
            if 0 <= val <= 100:
                try:
                    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                except ValueError:
                    continue
                ts_ms, date_utc = _normalize_date_to_utc_midnight_ms(dt)
                records_by_date[date_utc] = {
                    "timestamp": ts_ms,
                    "date_utc": date_utc,
                    "value": val,
                    "classification": classify_fear_greed(val),
                    "source": source_name,
                }

    # Strategy B: Parse standard alternative.me historical summary blocks
    relative_offsets = {
        "now": 0,
        "today": 0,
        "yesterday": 1,
        "last week": 7,
        "last month": 30,
    }
    for block in soup.select(".fng-value"):
        label_el = block.select_one(".gray")
        circle_el = block.select_one(".fng-circle")
        status_el = block.select_one(".status")
        if not (label_el and circle_el):
            continue
        label_text = label_el.get_text(strip=True).lower()
        val_digits = re.search(r"\d{1,3}", circle_el.get_text(strip=True))
        if not val_digits:
            continue
        val = int(val_digits.group(0))
        if not (0 <= val <= 100):
            continue
        offset_days = relative_offsets.get(label_text)
        if offset_days is None:
            continue
        target_dt = now_utc - timedelta(days=offset_days)
        ts_ms, date_utc = _normalize_date_to_utc_midnight_ms(target_dt)
        classification = (
            status_el.get_text(strip=True) if status_el else classify_fear_greed(val)
        )
        records_by_date.setdefault(
            date_utc,
            {
                "timestamp": ts_ms,
                "date_utc": date_utc,
                "value": val,
                "classification": classification,
                "source": source_name,
            },
        )

    # Strategy C: Fallback single current circle value
    if not records_by_date:
        circle_el = soup.select_one(".fng-circle")
        if circle_el:
            val_digits = re.search(r"\d{1,3}", circle_el.get_text(strip=True))
            if val_digits:
                val = int(val_digits.group(0))
                if 0 <= val <= 100:
                    ts_ms, date_utc = _normalize_date_to_utc_midnight_ms(now_utc)
                    records_by_date[date_utc] = {
                        "timestamp": ts_ms,
                        "date_utc": date_utc,
                        "value": val,
                        "classification": classify_fear_greed(val),
                        "source": source_name,
                    }

    records = sorted(records_by_date.values(), key=lambda r: r["timestamp"])
    return records


def fetch_fear_greed_via_bs4(
    url: str = FEAR_GREED_WEB_URL,
    timeout: int = REQUEST_TIMEOUT_SEC,
) -> list[dict[str, Any]]:
    """Fallback #1: Scrape Fear & Greed Index page with requests + BeautifulSoup."""
    response = requests.get(url, headers=HTTP_HEADERS, timeout=timeout)
    response.raise_for_status()
    records = parse_fear_greed_html(response.text, source_name="bs4_scraper")
    if not records:
        raise ValueError("BeautifulSoup fallback could not extract Fear & Greed records.")
    logger.info("Fetched %d Fear & Greed records via BeautifulSoup fallback.", len(records))
    return records


def fetch_fear_greed_via_playwright(
    url: str = FEAR_GREED_WEB_URL,
    timeout: int = REQUEST_TIMEOUT_SEC,
) -> list[dict[str, Any]]:
    """Fallback #2: Render Fear & Greed page with headless Playwright and parse DOM."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise ImportError(
            "playwright is required for headless browser fallback scraping."
        ) from exc

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            page = browser.new_page(user_agent=HTTP_HEADERS["User-Agent"])
            page.goto(url, wait_until="networkidle", timeout=timeout * 1000)
            html = page.content()
        finally:
            browser.close()

    records = parse_fear_greed_html(html, source_name="playwright_scraper")
    if not records:
        raise ValueError("Playwright fallback could not extract Fear & Greed records.")
    logger.info("Fetched %d Fear & Greed records via Playwright fallback.", len(records))
    return records


def upsert_fear_greed_records(
    conn: sqlite3.Connection,
    records: Sequence[dict[str, Any]],
) -> int:
    """Upsert normalized Fear & Greed Index records into SQLite."""
    if not records:
        return 0

    rows = [
        (
            int(rec["timestamp"]),
            str(rec["date_utc"]),
            int(rec["value"]),
            str(rec["classification"]),
            str(rec["source"]),
        )
        for rec in records
    ]
    with conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO fear_greed (
                timestamp, date_utc, value, classification, source
            ) VALUES (?, ?, ?, ?, ?);
            """,
            rows,
        )
    return len(rows)


def fetch_and_store_fear_greed(
    conn: sqlite3.Connection,
    allow_fallback: bool = True,
) -> int:
    """Fetch Fear & Greed data (API primary, BS4/Playwright fallback) and store in SQLite."""
    records: list[dict[str, Any]] = []
    try:
        records = fetch_fear_greed_from_api()
    except Exception as api_exc:
        if not allow_fallback:
            raise
        logger.warning(
            "Primary Fear & Greed API failed (%s). Trying BeautifulSoup fallback.",
            api_exc,
        )
        try:
            records = fetch_fear_greed_via_bs4()
        except Exception as bs4_exc:
            logger.warning(
                "BeautifulSoup fallback failed (%s). Trying Playwright fallback.",
                bs4_exc,
            )
            records = fetch_fear_greed_via_playwright()

    count = upsert_fear_greed_records(conn, records)
    logger.info("Upserted %d Fear & Greed records into SQLite.", count)
    return count


def get_database_summary(conn: sqlite3.Connection) -> dict[str, Any]:
    """Collect row counts, date bounds, and WAL status from the SQLite database."""
    journal_mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]

    ohlcv_rows = conn.execute(
        """
        SELECT
            symbol,
            COUNT(*) AS row_count,
            MIN(datetime_utc) AS min_utc,
            MAX(datetime_utc) AS max_utc
        FROM ohlcv
        GROUP BY symbol
        ORDER BY symbol;
        """
    ).fetchall()

    ohlcv_by_symbol = [
        {
            "symbol": row["symbol"],
            "row_count": int(row["row_count"]),
            "min_utc": row["min_utc"],
            "max_utc": row["max_utc"],
        }
        for row in ohlcv_rows
    ]
    total_ohlcv = sum(item["row_count"] for item in ohlcv_by_symbol)

    fg_row = conn.execute(
        """
        SELECT
            COUNT(*) AS row_count,
            MIN(date_utc) AS min_date,
            MAX(date_utc) AS max_date
        FROM fear_greed;
        """
    ).fetchone()

    return {
        "journal_mode": str(journal_mode),
        "ohlcv_total_rows": total_ohlcv,
        "ohlcv_by_symbol": ohlcv_by_symbol,
        "fear_greed_total_rows": int(fg_row["row_count"]) if fg_row else 0,
        "fear_greed_min_date": fg_row["min_date"] if fg_row else None,
        "fear_greed_max_date": fg_row["max_date"] if fg_row else None,
    }


def format_database_summary(summary: dict[str, Any]) -> str:
    """Render a clean ASCII summary table of SQLite ingestion state."""
    lines = [
        "==========================================================================",
        "AEGIS ALPHA - SQLITE MARKET DATA SUMMARY",
        "==========================================================================",
        f"SQLite journal_mode : {summary['journal_mode']}",
        f"Total OHLCV rows    : {summary['ohlcv_total_rows']:,}",
        "--------------------------------------------------------------------------",
        f"{'Symbol':<12} | {'Rows':>10} | {'Min UTC':<20} | {'Max UTC':<20}",
        "--------------------------------------------------------------------------",
    ]
    for item in summary["ohlcv_by_symbol"]:
        lines.append(
            f"{item['symbol']:<12} | {item['row_count']:>10,} | "
            f"{str(item['min_utc'] or 'N/A'):<20} | {str(item['max_utc'] or 'N/A'):<20}"
        )
    if not summary["ohlcv_by_symbol"]:
        lines.append("No OHLCV rows stored.")
    lines.extend(
        [
            "--------------------------------------------------------------------------",
            f"Fear & Greed rows   : {summary['fear_greed_total_rows']:,} "
            f"({summary['fear_greed_min_date'] or 'N/A'} to "
            f"{summary['fear_greed_max_date'] or 'N/A'})",
            "==========================================================================",
        ]
    )
    return "\n".join(lines)


def run_ingestion(
    db_path: Path = DB_PATH,
    symbols: Sequence[str] = SYMBOLS,
    timeframe: str = TIMEFRAME,
    start_iso: str = START_DATE_ISO,
    end_ms: int | None = None,
    skip_ohlcv: bool = False,
    skip_fear_greed: bool = False,
) -> dict[str, Any]:
    """Execute end-to-end Phase 1 incremental ingestion into SQLite."""
    ensure_directories()
    conn = get_db_connection(db_path=db_path)
    try:
        init_database(conn)

        ohlcv_counts: dict[str, int] = {}
        if not skip_ohlcv:
            ohlcv_counts = fetch_all_ohlcv(
                conn=conn,
                symbols=symbols,
                timeframe=timeframe,
                start_iso=start_iso,
                end_ms=end_ms,
            )

        fear_greed_count = 0
        if not skip_fear_greed:
            fear_greed_count = fetch_and_store_fear_greed(conn=conn)

        summary = get_database_summary(conn)
        summary["newly_inserted_ohlcv"] = ohlcv_counts
        summary["upserted_fear_greed"] = fear_greed_count
        return summary
    finally:
        conn.close()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aegis Alpha Phase 1 incremental OHLCV and Fear & Greed ingestion."
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DB_PATH,
        help=f"Path to SQLite database file (default: {DB_PATH}).",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=list(SYMBOLS),
        help="Trading pairs to ingest (default: 5 configured USDT pairs).",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=START_DATE_ISO,
        help=f"ISO-8601 UTC start date for initial backfill (default: {START_DATE_ISO}).",
    )
    parser.add_argument(
        "--skip-ohlcv",
        action="store_true",
        help="Skip Binance OHLCV ingestion.",
    )
    parser.add_argument(
        "--skip-fear-greed",
        action="store_true",
        help="Skip Crypto Fear & Greed Index ingestion.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print current SQLite database summary without fetching new data.",
    )
    return parser


def main() -> None:
    """CLI entrypoint for Phase 1 ingestion."""
    configure_logging()
    args = _build_arg_parser().parse_args()

    if args.summary_only:
        ensure_directories()
        conn = get_db_connection(args.db_path)
        try:
            init_database(conn)
            summary = get_database_summary(conn)
            print(format_database_summary(summary))
        finally:
            conn.close()
        return

    summary = run_ingestion(
        db_path=args.db_path,
        symbols=args.symbols,
        start_iso=args.start_date,
        skip_ohlcv=args.skip_ohlcv,
        skip_fear_greed=args.skip_fear_greed,
    )
    print(format_database_summary(summary))


if __name__ == "__main__":
    main()
