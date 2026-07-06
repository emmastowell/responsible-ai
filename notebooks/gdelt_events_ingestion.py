# Databricks notebook source
# MAGIC %md
# MAGIC # GDELT unrest events ingestion
# MAGIC
# MAGIC Builds `gdelt_unrest_events` — the source table the
# MAGIC `gdelt_article_raw_text_extraction` notebook reads from.
# MAGIC
# MAGIC Downloads GDELT 2.0 event export files (15-minute intervals) for a date
# MAGIC range, filters to social-unrest CAMEO root codes (14 Protest, 17 Coerce,
# MAGIC 18 Assault, 19 Fight), optionally filters to UK geography, and writes a
# MAGIC Delta table. GDELT is freely available — no authentication required.

# COMMAND ----------

# DBTITLE 1,Parameters
dbutils.widgets.text("catalog", "hackathon", "Catalog")
dbutils.widgets.text("schema", "shared_datasets", "Schema")
dbutils.widgets.text("target_table", "gdelt_unrest_events", "Target table")
dbutils.widgets.text("gdelt_date_from", "20260601", "GDELT From (YYYYMMDD)")
dbutils.widgets.text("gdelt_date_to", "20260607", "GDELT To (YYYYMMDD)")
dbutils.widgets.dropdown("gdelt_filter_uk", "true", ["true", "false"], "Filter to UK only")
dbutils.widgets.text("max_files", "0", "Cap number of export files (0 = no cap)")

CATALOG = dbutils.widgets.get("catalog").strip()
SCHEMA = dbutils.widgets.get("schema").strip()
TARGET_TABLE = f"{CATALOG}.{SCHEMA}." + dbutils.widgets.get("target_table").strip()
GDELT_DATE_FROM = dbutils.widgets.get("gdelt_date_from").strip()
GDELT_DATE_TO = dbutils.widgets.get("gdelt_date_to").strip()
GDELT_FILTER_UK = dbutils.widgets.get("gdelt_filter_uk").strip().lower() == "true"
MAX_FILES = int(dbutils.widgets.get("max_files").strip())

print(f"target_table = {TARGET_TABLE}")
print(f"date range   = {GDELT_DATE_FROM}..{GDELT_DATE_TO}  uk_only={GDELT_FILTER_UK}  max_files={MAX_FILES}")

spark.sql(f"CREATE SCHEMA IF NOT EXISTS `{CATALOG}`.`{SCHEMA}`")

# COMMAND ----------

# DBTITLE 1,Shared utilities
import functools
import io
import logging
import time
import zipfile
from datetime import datetime
from typing import Optional

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")
logger = logging.getLogger("gdelt_ingestion")

GDELT_MASTER_URL = "http://data.gdeltproject.org/gdeltv2/masterfilelist.txt"
# CAMEO event root codes for social unrest: 14=Protest, 17=Coerce, 18=Assault, 19=Fight
GDELT_UNREST_CODES = {"14", "17", "18", "19"}


def retry(max_attempts: int = 5, backoff_base: float = 2.0,
          exceptions: tuple = (requests.RequestException,)):
    """Retry on specified exceptions with exponential back-off."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    if attempt == max_attempts:
                        raise
                    wait = backoff_base ** attempt
                    logger.warning("Attempt %d/%d failed (%s). Retrying in %.1fs.",
                                   attempt, max_attempts, exc, wait)
                    time.sleep(wait)
        return wrapper
    return decorator


@retry()
def rate_limited_get(url: str, min_interval: float = 0.25, timeout: int = 30) -> requests.Response:
    """GET with retry, respecting a minimum interval between calls."""
    time.sleep(min_interval)
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response

# COMMAND ----------

# DBTITLE 1,GDELT export schema (61 columns)
GDELT_COLUMNS = [
    "global_event_id", "day", "month_year", "year", "fraction_date",
    "actor1_code", "actor1_name", "actor1_country_code",
    "actor1_known_group_code", "actor1_ethnic_code",
    "actor1_religion1_code", "actor1_religion2_code",
    "actor1_type1_code", "actor1_type2_code", "actor1_type3_code",
    "actor2_code", "actor2_name", "actor2_country_code",
    "actor2_known_group_code", "actor2_ethnic_code",
    "actor2_religion1_code", "actor2_religion2_code",
    "actor2_type1_code", "actor2_type2_code", "actor2_type3_code",
    "is_root_event", "event_code", "event_base_code", "event_root_code",
    "quad_class", "goldstein_scale", "num_mentions", "num_sources",
    "num_articles", "avg_tone",
    "actor1_geo_type", "actor1_geo_fullname", "actor1_geo_country_code",
    "actor1_geo_adm1_code", "actor1_geo_adm2_code",
    "actor1_geo_lat", "actor1_geo_long", "actor1_geo_feature_id",
    "actor2_geo_type", "actor2_geo_fullname", "actor2_geo_country_code",
    "actor2_geo_adm1_code", "actor2_geo_adm2_code",
    "actor2_geo_lat", "actor2_geo_long", "actor2_geo_feature_id",
    "action_geo_type", "action_geo_fullname", "action_geo_country_code",
    "action_geo_adm1_code", "action_geo_adm2_code",
    "action_geo_lat", "action_geo_long", "action_geo_feature_id",
    "date_added", "source_url",
]

# COMMAND ----------

# DBTITLE 1,Fetch, filter and download GDELT export files
def fetch_gdelt_master_list() -> list[tuple[str, str]]:
    """Return (datetime_str, url) tuples for export files in the master list."""
    response = rate_limited_get(GDELT_MASTER_URL, min_interval=1.0)
    entries = []
    for line in response.text.strip().splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        url = parts[2]
        if ".export.CSV.zip" not in url:
            continue
        dt_str = url.split("/")[-1].split(".")[0]  # 14-char datetime string
        entries.append((dt_str, url))
    return entries


def filter_gdelt_entries(entries, date_from, date_to):
    """Filter master-list entries to the requested date range (YYYYMMDD)."""
    return [(dt, url) for dt, url in entries if date_from <= dt[:8] <= date_to]


def parse_gdelt_zip(content: bytes, filter_uk: bool) -> pd.DataFrame:
    """Parse a GDELT export ZIP: unrest event codes and optional UK geography."""
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        with zf.open(zf.namelist()[0]) as f:
            df = pd.read_csv(f, sep="\t", header=None, names=GDELT_COLUMNS,
                             dtype=str, low_memory=False)
    df = df[df["event_root_code"].isin(GDELT_UNREST_CODES)]
    if filter_uk:
        df = df[
            (df["actor1_geo_country_code"] == "UK")
            | (df["actor2_geo_country_code"] == "UK")
            | (df["action_geo_country_code"] == "UK")
        ]
    return df


def ingest_gdelt(date_from: str, date_to: str, filter_uk: bool, max_files: int) -> None:
    logger.info("Fetching GDELT master file list...")
    entries = filter_gdelt_entries(fetch_gdelt_master_list(), date_from, date_to)
    if max_files > 0:
        entries = entries[:max_files]
    logger.info("GDELT: %d files to process for %s–%s.", len(entries), date_from, date_to)

    all_dfs = []
    for idx, (dt_str, url) in enumerate(entries):
        try:
            df = parse_gdelt_zip(rate_limited_get(url, min_interval=0.5).content, filter_uk)
            df["source_file_datetime"] = dt_str
            all_dfs.append(df)
        except Exception as exc:  # noqa: BLE001 - skip and continue
            logger.warning("GDELT: skipping %s — %s", url, exc)
        if (idx + 1) % 25 == 0:
            logger.info("GDELT: processed %d/%d files.", idx + 1, len(entries))

    if not all_dfs:
        logger.warning("GDELT: no records to write.")
        return

    combined = pd.concat(all_dfs, ignore_index=True)
    combined["ingested_at"] = datetime.utcnow().isoformat()
    sdf = spark.createDataFrame(combined)
    sdf.write.format("delta").mode("overwrite").option(
        "overwriteSchema", "true"
    ).saveAsTable(TARGET_TABLE)
    logger.info("GDELT: wrote %d records to %s.", sdf.count(), TARGET_TABLE)


ingest_gdelt(GDELT_DATE_FROM, GDELT_DATE_TO, GDELT_FILTER_UK, MAX_FILES)

# COMMAND ----------

# DBTITLE 1,Verify
from pyspark.sql import functions as F

events = spark.table(TARGET_TABLE)
print(f"{TARGET_TABLE}: {events.count():,} rows")
display(events.groupBy("event_root_code").count().orderBy(F.desc("count")))
display(events.select(
    "global_event_id", "day", "event_root_code", "action_geo_fullname",
    "num_mentions", "avg_tone", "source_url",
).limit(10))
