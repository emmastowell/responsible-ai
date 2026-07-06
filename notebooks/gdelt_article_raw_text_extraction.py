# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Extract article raw text
from datetime import date, datetime, timedelta, timezone
import importlib
import subprocess
import sys

from pyspark.sql import functions as F, types as T


def ensure_package(package_name: str, import_name: str | None = None):
    module_name = import_name or package_name
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError:
        print(f"Installing {package_name} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package_name])
        importlib.invalidate_caches()
        return importlib.import_module(module_name)


requests = ensure_package("requests")
trafilatura = ensure_package("trafilatura")


# Configurable parameters
source_table = "hackathon.shared_datasets.gdelt_unrest_events"
target_table = "hackathon.shared_datasets.gdelt_article_raw_text"
end_date = date.today()
start_date = end_date - timedelta(days=30)
max_unique_urls = 1000
batch_size = 100

# Optional controls
request_timeout_seconds = 20
store_raw_html = False
user_agent = "Mozilla/5.0 (compatible; Databricks-GDELT-Text-Extractor/1.0)"


if isinstance(start_date, str):
    start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
if isinstance(end_date, str):
    end_date = datetime.strptime(end_date, "%Y-%m-%d").date()

if start_date > end_date:
    raise ValueError("start_date must be on or before end_date")
if max_unique_urls <= 0:
    raise ValueError("max_unique_urls must be positive")

window_start_ts = datetime.combine(start_date, datetime.min.time())
window_end_ts_exclusive = datetime.combine(end_date + timedelta(days=1), datetime.min.time())

print(f"source_table = {source_table}")
print(f"target_table = {target_table}")
print(f"start_date = {start_date}")
print(f"end_date = {end_date}")
print(f"max_unique_urls = {max_unique_urls}")
print(f"batch_size = {batch_size}")

fetch_schema = T.StructType([
    T.StructField("source_url", T.StringType(), True),
    T.StructField("resolved_url", T.StringType(), True),
    T.StructField("fetched_at", T.TimestampType(), True),
    T.StructField("http_status", T.IntegerType(), True),
    T.StructField("content_type", T.StringType(), True),
    T.StructField("raw_text", T.StringType(), True),
    T.StructField("raw_html", T.StringType(), True),
    T.StructField("total_num_mentions", T.LongType(), True),
    T.StructField("max_num_mentions", T.LongType(), True),
    T.StructField("event_row_count", T.LongType(), True),
    T.StructField("latest_event_date", T.DateType(), True),
    T.StructField("max_sql_date", T.TimestampType(), True),
    T.StructField("error", T.StringType(), True),
    T.StructField("source_table", T.StringType(), True),
    T.StructField("request_window_start", T.DateType(), True),
    T.StructField("request_window_end", T.DateType(), True),
])

if not spark.catalog.tableExists(target_table):
    spark.createDataFrame([], fetch_schema).write.format("delta").mode("ignore").saveAsTable(target_table)

existing_url_df = spark.table(target_table).select("source_url").distinct()
existing_url_count = existing_url_df.count()

candidate_url_df = (
    spark.table(source_table)
    .select(
        F.trim(F.col("source_url")).alias("source_url"),
        F.col("num_mentions").cast("long").alias("num_mentions"),
        F.col("event_date"),
        F.col("sql_date")
    )
    .where(F.col("sql_date").isNotNull())
    .where(F.col("sql_date") >= F.lit(window_start_ts))
    .where(F.col("sql_date") < F.lit(window_end_ts_exclusive))
    .where(F.col("source_url").isNotNull())
    .where(F.col("source_url") != "")
    .where(F.lower(F.col("source_url")).rlike(r"^https?://"))
    .groupBy("source_url")
    .agg(
        F.sum(F.coalesce(F.col("num_mentions"), F.lit(0))).alias("total_num_mentions"),
        F.max(F.coalesce(F.col("num_mentions"), F.lit(0))).alias("max_num_mentions"),
        F.count(F.lit(1)).alias("event_row_count"),
        F.max(F.col("event_date")).alias("latest_event_date"),
        F.max(F.col("sql_date")).alias("max_sql_date")
    )
    .orderBy(
        F.desc("total_num_mentions"),
        F.desc("max_num_mentions"),
        F.desc("event_row_count"),
        F.desc("latest_event_date"),
        F.asc("source_url")
    )
)

candidate_url_count = candidate_url_df.count()
print(f"candidate_url_count = {candidate_url_count}")
print(f"existing_target_url_count = {existing_url_count}")
display(candidate_url_df.limit(20))

pending_url_df = (
    candidate_url_df
    .join(existing_url_df, on="source_url", how="left_anti")
    .orderBy(
        F.desc("total_num_mentions"),
        F.desc("max_num_mentions"),
        F.desc("event_row_count"),
        F.desc("latest_event_date"),
        F.asc("source_url")
    )
    .limit(max_unique_urls)
)

pending_url_count = pending_url_df.count()
print(f"pending_url_count = {pending_url_count}")
display(pending_url_df)

session = requests.Session()
session.headers.update({"User-Agent": user_agent})


def fetch_article(row):
    source_url = row["source_url"]
    total_num_mentions = int(row["total_num_mentions"] or 0)
    max_num_mentions = int(row["max_num_mentions"] or 0)
    event_row_count = int(row["event_row_count"] or 0)
    latest_event_date = row["latest_event_date"]
    max_sql_date = row["max_sql_date"]
    fetched_at = datetime.now(timezone.utc).replace(tzinfo=None)
    resolved_url = source_url
    http_status = None
    content_type = None
    raw_text = None
    raw_html = None
    error = None

    try:
        response = session.get(source_url, timeout=request_timeout_seconds, allow_redirects=True)
        resolved_url = response.url
        http_status = int(response.status_code)
        content_type = response.headers.get("Content-Type", "")
        body_text = response.text or ""

        if store_raw_html:
            raw_html = body_text

        if http_status >= 400:
            error = f"HTTP {http_status}"
        elif "html" in content_type.lower() or "xml" in content_type.lower() or content_type == "" or "text/" in content_type.lower():
            raw_text = trafilatura.extract(
                body_text,
                url=resolved_url,
                favor_precision=True,
                include_links=False,
                include_images=False,
                include_tables=False,
                deduplicate=True,
            )
            if not raw_text:
                raw_text = trafilatura.extract(
                    body_text,
                    url=resolved_url,
                    favor_recall=True,
                    include_links=False,
                    include_images=False,
                    include_tables=False,
                    deduplicate=True,
                )
            if not raw_text and "text/" in content_type.lower():
                raw_text = body_text
            if not raw_text:
                error = "No readable text extracted"
        else:
            error = f"Unsupported content type: {content_type}"
    except Exception as exc:
        error = f"{type(exc).__name__}: {str(exc)[:500]}"

    return (
        source_url,
        resolved_url,
        fetched_at,
        http_status,
        content_type,
        raw_text,
        raw_html,
        total_num_mentions,
        max_num_mentions,
        event_row_count,
        latest_event_date,
        max_sql_date,
        error,
        source_table,
        start_date,
        end_date,
    )


pending_rows = [row.asDict() for row in pending_url_df.collect()]
print(f"fetched_url_count = {len(pending_rows)}")

batch_stats = []
first_batch_results_df = None

for batch_start in range(0, len(pending_rows), batch_size):
    batch_number = (batch_start // batch_size) + 1
    batch_rows = pending_rows[batch_start: batch_start + batch_size]
    batch_records = [fetch_article(row) for row in batch_rows]
    batch_results_df = spark.createDataFrame(batch_records, schema=fetch_schema)

    if first_batch_results_df is None:
        first_batch_results_df = batch_results_df

    success_count = batch_results_df.filter(F.col("error").isNull()).count()
    error_count = batch_results_df.filter(F.col("error").isNotNull()).count()
    batch_stats.append((batch_number, len(batch_rows), success_count, error_count))
    print(
        f"completed batch {batch_number} with {len(batch_rows)} URLs "
        f"({success_count} success, {error_count} errors)"
    )

    if batch_records:
        batch_results_df.createOrReplaceTempView("gdelt_article_raw_text_new_results")
        spark.sql(f"""
        MERGE INTO {target_table} AS target
        USING gdelt_article_raw_text_new_results AS source
        ON target.source_url = source.source_url
        WHEN NOT MATCHED THEN INSERT (
          source_url,
          resolved_url,
          fetched_at,
          http_status,
          content_type,
          raw_text,
          raw_html,
          total_num_mentions,
          max_num_mentions,
          event_row_count,
          latest_event_date,
          max_sql_date,
          error,
          source_table,
          request_window_start,
          request_window_end
        ) VALUES (
          source.source_url,
          source.resolved_url,
          source.fetched_at,
          source.http_status,
          source.content_type,
          source.raw_text,
          source.raw_html,
          source.total_num_mentions,
          source.max_num_mentions,
          source.event_row_count,
          source.latest_event_date,
          source.max_sql_date,
          source.error,
          source.source_table,
          source.request_window_start,
          source.request_window_end
        )
        """)

if first_batch_results_df is not None:
    display(
        first_batch_results_df.select(
            "source_url",
            "http_status",
            "content_type",
            "total_num_mentions",
            "event_row_count",
            "error"
        )
    )

batch_stats_df = spark.createDataFrame(
    batch_stats,
    ["batch_number", "batch_size", "success_count", "error_count"]
)
display(batch_stats_df)

final_saved_df = spark.table(target_table)
final_count_df = final_saved_df.selectExpr(
    "count(*) AS saved_url_count",
    "count_if(error IS NULL) AS successful_fetch_count",
    "count_if(error IS NOT NULL) AS errored_fetch_count"
)

display(final_count_df)
display(
    final_saved_df.orderBy(F.desc("fetched_at"))
    .select(
        "source_url",
        "fetched_at",
        "http_status",
        "content_type",
        "total_num_mentions",
        "event_row_count",
        "error"
    )
    .limit(20)
)

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM hackathon.shared_datasets.gdelt_article_raw_text