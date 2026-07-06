# Databricks notebook source
# MAGIC %md
# MAGIC # Yahoo Finance — price and volume ingestion
# MAGIC
# MAGIC Daily OHLCV + volume for target equities (market-manipulation use case),
# MAGIC written to `yahoo_finance_prices`.
# MAGIC
# MAGIC Uses the Yahoo chart API directly (`query1.finance.yahoo.com/v8/finance/chart`)
# MAGIC rather than the `yfinance` library: no extra dependency, and we control the
# MAGIC User-Agent, which Yahoo requires (the default `python-requests` UA gets a 429).

# COMMAND ----------

# DBTITLE 1,Parameters
dbutils.widgets.text("catalog", "hackathon", "Catalog")
dbutils.widgets.text("schema", "shared_datasets", "Schema")
dbutils.widgets.text("target_table", "yahoo_finance_prices", "Target table")
dbutils.widgets.text("tickers", "GME,AMC,BB,NOK,BBBY,SPY,QQQ", "Tickers (comma-separated)")
dbutils.widgets.text("date_from", "2020-01-01", "From (YYYY-MM-DD)")
dbutils.widgets.text("date_to", "2021-12-31", "To (YYYY-MM-DD)")
dbutils.widgets.dropdown("interval", "1d", ["1d", "1wk", "1mo"], "Interval")

CATALOG = dbutils.widgets.get("catalog").strip()
SCHEMA = dbutils.widgets.get("schema").strip()
TARGET_TABLE = f"{CATALOG}.{SCHEMA}." + dbutils.widgets.get("target_table").strip()
TICKERS = [t.strip().upper() for t in dbutils.widgets.get("tickers").split(",") if t.strip()]
DATE_FROM = dbutils.widgets.get("date_from").strip()
DATE_TO = dbutils.widgets.get("date_to").strip()
INTERVAL = dbutils.widgets.get("interval").strip()

print(f"target_table = {TARGET_TABLE}")
print(f"tickers = {TICKERS}")
print(f"range = {DATE_FROM}..{DATE_TO}  interval = {INTERVAL}")

spark.sql(f"CREATE SCHEMA IF NOT EXISTS `{CATALOG}`.`{SCHEMA}`")

# COMMAND ----------

# DBTITLE 1,Fetch from the Yahoo chart API
import time
from datetime import datetime, timedelta, timezone

import requests
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, LongType,
)

CHART_HOSTS = ["query1.finance.yahoo.com", "query2.finance.yahoo.com"]
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}
MAX_RETRIES = 5
BASE_SLEEP = 1.0


def _epoch(date_str: str) -> int:
    return int(datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())


def fetch_ticker(ticker: str, date_from: str, date_to: str, interval: str) -> list:
    """Return a list of daily OHLCV row dicts for one ticker, or [] on failure."""
    period1 = _epoch(date_from)
    period2 = _epoch(date_to) + 86400  # inclusive of date_to
    params = {"period1": period1, "period2": period2, "interval": interval}
    for attempt in range(MAX_RETRIES):
        host = CHART_HOSTS[attempt % len(CHART_HOSTS)]
        url = f"https://{host}/v8/finance/chart/{ticker}"
        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=30)
        except requests.RequestException as e:
            print(f"  {ticker}: request error {e}")
            time.sleep(BASE_SLEEP * (2 ** attempt))
            continue
        if resp.status_code == 200:
            return _parse_chart(ticker, resp.json())
        if resp.status_code in (429, 500, 502, 503, 504):
            wait = BASE_SLEEP * (2 ** attempt)
            print(f"  {ticker}: {resp.status_code}, backing off {wait:.1f}s")
            time.sleep(wait)
            continue
        print(f"  {ticker}: HTTP {resp.status_code}, giving up")
        return []
    print(f"  {ticker}: exhausted retries")
    return []


def _parse_chart(ticker: str, payload: dict) -> list:
    chart = payload.get("chart", {})
    if chart.get("error"):
        print(f"  {ticker}: API error {chart['error']}")
        return []
    results = chart.get("result") or []
    if not results:
        return []
    result = results[0]
    timestamps = result.get("timestamp") or []
    quote = (result.get("indicators", {}).get("quote") or [{}])[0]
    adj = (result.get("indicators", {}).get("adjclose") or [{}])
    adjclose = adj[0].get("adjclose") if adj and adj[0] else None
    now = datetime.utcnow().isoformat()

    def g(seq, i):
        return seq[i] if seq and i < len(seq) and seq[i] is not None else None

    rows = []
    for i, ts in enumerate(timestamps):
        close = g(quote.get("close"), i)
        if close is None:  # skip empty trading rows
            continue
        rows.append({
            "ticker": ticker,
            "date": datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d"),
            "open": g(quote.get("open"), i),
            "high": g(quote.get("high"), i),
            "low": g(quote.get("low"), i),
            "close": close,
            "adj_close": g(adjclose, i) if adjclose else None,
            "volume": g(quote.get("volume"), i),
            "ingested_at": now,
        })
    print(f"  {ticker}: {len(rows)} rows")
    return rows

# COMMAND ----------

# DBTITLE 1,Ingest all tickers and write Delta
all_rows = []
for t in TICKERS:
    all_rows.extend(fetch_ticker(t, DATE_FROM, DATE_TO, INTERVAL))
    time.sleep(0.5)

if not all_rows:
    raise RuntimeError("No price data fetched for any ticker.")

SCHEMA_STRUCT = StructType([
    StructField("ticker", StringType()),
    StructField("date", StringType()),
    StructField("open", DoubleType()),
    StructField("high", DoubleType()),
    StructField("low", DoubleType()),
    StructField("close", DoubleType()),
    StructField("adj_close", DoubleType()),
    StructField("volume", LongType()),
    StructField("ingested_at", StringType()),
])

sdf = spark.createDataFrame(all_rows, schema=SCHEMA_STRUCT)
(
    sdf.write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .partitionBy("ticker")
    .saveAsTable(TARGET_TABLE)
)
print(f"Wrote {sdf.count()} rows to {TARGET_TABLE}")

# COMMAND ----------

# DBTITLE 1,Verify
from pyspark.sql import functions as F

prices = spark.table(TARGET_TABLE)
display(
    prices.groupBy("ticker").agg(
        F.count("*").alias("rows"),
        F.min("date").alias("first_date"),
        F.max("date").alias("last_date"),
        F.round(F.min("close"), 2).alias("min_close"),
        F.round(F.max("close"), 2).alias("max_close"),
    ).orderBy("ticker")
)
display(prices.orderBy(F.desc("volume")).limit(10))
