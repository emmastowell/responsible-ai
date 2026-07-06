# Databricks notebook source
# MAGIC %md
# MAGIC # Load Reddit (Pushshift / Arctic Shift) corpora into Unity Catalog
# MAGIC
# MAGIC Lands two bronze Delta tables from the **Arctic Shift** API (the maintained
# MAGIC successor to the shut-down Pushshift API):
# MAGIC
# MAGIC | Corpus | Subreddits | Theme |
# MAGIC |---|---|---|
# MAGIC | `unrest` | r/PublicFreakout, r/protest, r/ActivismInAction | Social unrest / historical social media text |
# MAGIC | `wsb`    | r/wallstreetbets                                  | Financial market manipulation |
# MAGIC
# MAGIC The API caps each request at 100 rows, so we paginate backward on `created_utc`
# MAGIC until the per-corpus row cap is reached, with a polite delay between calls.
# MAGIC
# MAGIC **Output:** `{catalog}.{schema}.reddit_unrest_bronze` and `{catalog}.{schema}.reddit_wsb_bronze`

# COMMAND ----------

# MAGIC %md
# MAGIC ## Parameters

# COMMAND ----------

dbutils.widgets.text("catalog", "main", "Catalog")
dbutils.widgets.text("schema", "reddit_pushshift", "Schema")
dbutils.widgets.text("row_cap", "2000", "Rows per corpus")
dbutils.widgets.dropdown("content_type", "posts", ["posts", "comments"], "Content type")

CATALOG = dbutils.widgets.get("catalog").strip()
SCHEMA = dbutils.widgets.get("schema").strip()
ROW_CAP = int(dbutils.widgets.get("row_cap").strip())
CONTENT_TYPE = dbutils.widgets.get("content_type").strip()  # "posts" or "comments"

CORPORA = {
    "unrest": ["PublicFreakout", "protest", "activism"],
    "wsb": ["wallstreetbets"],
}

print(f"catalog={CATALOG} schema={SCHEMA} row_cap={ROW_CAP} content_type={CONTENT_TYPE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ensure catalog / schema

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS `{CATALOG}`.`{SCHEMA}`")
print(f"Using `{CATALOG}`.`{SCHEMA}`")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fetch from Arctic Shift (paginated)

# COMMAND ----------

import time
import requests
from datetime import datetime, timezone

API_BASE = "https://arctic-shift.photon-reddit.com/api"
PAGE_LIMIT = 100          # API hard cap per request
SLEEP_SECONDS = 1.0       # be polite to a free community API (comments endpoint is slower)
REQUEST_TIMEOUT = 45
MAX_RETRIES = 7
HEADERS = {"User-Agent": "databricks-reddit-loader/1.0 (research; contact: field-eng)"}


def _endpoint(content_type: str) -> str:
    return f"{API_BASE}/{'posts' if content_type == 'posts' else 'comments'}/search"


def _get(url: str, params: dict):
    """GET with exponential backoff. Returns the `data` list, or None if the
    request could not be completed after retries (caller keeps partial data)."""
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        except requests.RequestException as e:
            wait = SLEEP_SECONDS * (2 ** attempt)
            print(f"  request error {e}, backing off {wait:.1f}s")
            time.sleep(wait)
            continue
        if resp.status_code == 200:
            return resp.json().get("data", [])
        if resp.status_code in (429, 500, 502, 503, 504):
            wait = SLEEP_SECONDS * (2 ** attempt)
            print(f"  {resp.status_code} on {params}, backing off {wait:.1f}s")
            time.sleep(wait)
            continue
        resp.raise_for_status()
    print(f"  WARNING: exhausted retries for {params}; keeping partial results")
    return None


def fetch_subreddit(subreddit: str, content_type: str, cap: int) -> list:
    """Page backward over created_utc until `cap` unique records collected."""
    url = _endpoint(content_type)
    collected, seen_ids = [], set()
    before = None  # None => newest first
    while len(collected) < cap:
        params = {"subreddit": subreddit, "limit": PAGE_LIMIT, "sort": "desc"}
        if before is not None:
            params["before"] = before
        batch = _get(url, params)
        if batch is None:  # retries exhausted; keep what we have
            break
        if not batch:      # genuinely no more data
            break
        new_in_batch = 0
        for rec in batch:
            rid = rec.get("id")
            if rid in seen_ids:
                continue
            seen_ids.add(rid)
            collected.append(rec)
            new_in_batch += 1
            if len(collected) >= cap:
                break
        # advance the cursor to the oldest created_utc seen this page
        oldest = min(int(r.get("created_utc", 0)) for r in batch)
        before = oldest
        print(f"  r/{subreddit}: {len(collected)}/{cap} (+{new_in_batch})")
        if new_in_batch == 0 or len(batch) < PAGE_LIMIT:
            break
        time.sleep(SLEEP_SECONDS)
    return collected[:cap]


def fetch_corpus(subreddits: list, content_type: str, cap: int) -> list:
    """Split the cap across subreddits; top up if any come up short."""
    records = []
    per_sub = max(1, cap // len(subreddits))
    for sub in subreddits:
        remaining = cap - len(records)
        if remaining <= 0:
            break
        take = min(per_sub, remaining) if sub != subreddits[-1] else remaining
        print(f"Fetching r/{sub} (target {take})")
        records.extend(fetch_subreddit(sub, content_type, take))
    return records

# COMMAND ----------

# MAGIC %md
# MAGIC ## Normalize + write bronze tables

# COMMAND ----------

import json
from pyspark.sql import Row
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, LongType, DoubleType, TimestampType,
)

BRONZE_SCHEMA = StructType([
    StructField("id", StringType()),
    StructField("subreddit", StringType()),
    StructField("corpus", StringType()),
    StructField("author", StringType()),
    StructField("created_utc", LongType()),
    StructField("title", StringType()),
    StructField("body", StringType()),
    StructField("score", LongType()),
    StructField("num_comments", LongType()),
    StructField("permalink", StringType()),
    StructField("raw", StringType()),
])


def _as_long(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def normalize(records: list, corpus: str, content_type: str) -> list:
    rows = []
    for r in records:
        if content_type == "posts":
            title = r.get("title")
            body = r.get("selftext")
        else:
            title = None
            body = r.get("body")
        rows.append(Row(
            id=r.get("id"),
            subreddit=r.get("subreddit"),
            corpus=corpus,
            author=r.get("author"),
            created_utc=_as_long(r.get("created_utc")),
            title=title,
            body=body,
            score=_as_long(r.get("score")),
            num_comments=_as_long(r.get("num_comments")),
            permalink=r.get("permalink"),
            raw=json.dumps(r, ensure_ascii=False),
        ))
    return rows


def write_bronze(corpus: str, subreddits: list) -> str:
    table = f"`{CATALOG}`.`{SCHEMA}`.reddit_{corpus}_{CONTENT_TYPE}_bronze"
    print(f"\n=== Corpus '{corpus}' -> {table} ===")
    records = fetch_corpus(subreddits, CONTENT_TYPE, ROW_CAP)
    rows = normalize(records, corpus, CONTENT_TYPE)
    if not rows:
        print(f"  WARNING: no records fetched for '{corpus}'")
        return table
    df = (
        spark.createDataFrame(rows, schema=BRONZE_SCHEMA)
        .withColumn("created_at", F.to_timestamp(F.col("created_utc")))
        .withColumn("ingested_at", F.current_timestamp())
    )
    df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
        table.replace("`", "")
    )
    print(f"  wrote {df.count()} rows")
    return table


tables = [write_bronze(corpus, subs) for corpus, subs in CORPORA.items()]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify

# COMMAND ----------

for corpus in CORPORA:
    t = f"{CATALOG}.{SCHEMA}.reddit_{corpus}_{CONTENT_TYPE}_bronze"
    print(f"\n### {t}")
    spark.sql(f"SELECT count(*) AS rows, count(DISTINCT subreddit) AS subs FROM {t}").show()
    spark.sql(
        f"SELECT subreddit, id, created_at, score, "
        f"substr(coalesce(title, body), 1, 120) AS preview FROM {t} LIMIT 5"
    ).show(truncate=False)
