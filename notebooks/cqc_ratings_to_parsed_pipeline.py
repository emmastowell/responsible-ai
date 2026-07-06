# Databricks notebook source
# MAGIC %md
# MAGIC # CQC ratings -> sampled PDFs -> ai_parse_document
# MAGIC
# MAGIC One end-to-end pipeline (merges four earlier notebooks):
# MAGIC
# MAGIC 1. **Download ratings** — pull the CQC "Latest ratings" spreadsheet (.ods) and land it raw in Delta.
# MAGIC 2. **Build sample** — filter to Care Homes / "Safe" domain and select a balanced sample across ratings.
# MAGIC 3. **Download PDFs** — resolve each location's inspection-report PDF and save it to a UC Volume.
# MAGIC 4. **Parse** — run `ai_parse_document` over the downloaded PDFs into a Delta table.
# MAGIC
# MAGIC All paths/tables are widget-driven. Set `candidate_limit` > 0 for a fast smoke test.

# COMMAND ----------

# DBTITLE 1,Parameters
dbutils.widgets.text("catalog", "hackathon", "Catalog")
dbutils.widgets.text("schema", "shared_datasets", "Schema")
dbutils.widgets.text(
    "ratings_url",
    "https://www.cqc.org.uk/sites/default/files/2026-06/01_June_2026_Latest_ratings.ods",
    "CQC ratings .ods URL",
)
dbutils.widgets.text("ratings_table", "cqc_latest_ratings", "Raw ratings table")
dbutils.widgets.text("parsed_table", "cqc_parsed_documents", "Parsed output table")
dbutils.widgets.text(
    "volume_dir",
    "/Volumes/hackathon/shared_datasets/cqc_reports/cqc_pdf_sample",
    "Volume dir for PDFs",
)
dbutils.widgets.text("good_sample_size", "53", "Good-rated sample size")
dbutils.widgets.text("candidate_limit", "0", "Cap total candidates (0 = no cap)")

CATALOG = dbutils.widgets.get("catalog").strip()
SCHEMA = dbutils.widgets.get("schema").strip()
RATINGS_URL = dbutils.widgets.get("ratings_url").strip()
RATINGS_TABLE = f"{CATALOG}.{SCHEMA}." + dbutils.widgets.get("ratings_table").strip()
PARSED_TABLE = f"{CATALOG}.{SCHEMA}." + dbutils.widgets.get("parsed_table").strip()
VOLUME_DIR = dbutils.widgets.get("volume_dir").strip().rstrip("/")
GOOD_SAMPLE_SIZE = int(dbutils.widgets.get("good_sample_size").strip())
CANDIDATE_LIMIT = int(dbutils.widgets.get("candidate_limit").strip())

print(f"ratings_table = {RATINGS_TABLE}")
print(f"parsed_table  = {PARSED_TABLE}")
print(f"volume_dir    = {VOLUME_DIR}")
print(f"good_sample_size = {GOOD_SAMPLE_SIZE}  candidate_limit = {CANDIDATE_LIMIT}")

# COMMAND ----------

# MAGIC %md ## Stage 1 — Download CQC ratings spreadsheet to Delta

# COMMAND ----------

# DBTITLE 1,Load ratings workbook (raw columns) to Delta
import importlib.util
import subprocess
import sys

import pandas as pd
import requests

SHEET_NAME = "Locations"

# python-calamine (Rust) parses this 300k-row .ods in ~4s; the pure-Python
# odfpy engine takes 10+ minutes on the same file. Called via its native API
# because the serverless pandas is too old for read_excel(engine="calamine").
if importlib.util.find_spec("python_calamine") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "python-calamine", "--quiet"])

from python_calamine import CalamineWorkbook


def prepare_sheet(df: pd.DataFrame) -> pd.DataFrame:
    """Drop blank rows/cols and promote the real header row if the sheet has a
    banner row above the column names (CQC ships the header on row 2)."""
    prepared = df.dropna(axis=0, how="all").dropna(axis=1, how="all").copy()
    prepared.columns = [str(c).strip() for c in prepared.columns]
    if "Service / Population Group" not in prepared.columns and not prepared.empty:
        first_row = [str(v).strip() if pd.notna(v) else "" for v in prepared.iloc[0].tolist()]
        if "Service / Population Group" in first_row:
            prepared = prepared.iloc[1:].copy()
            prepared.columns = first_row
            prepared = prepared.dropna(axis=0, how="all").dropna(axis=1, how="all")
    prepared.columns = [str(c).strip() for c in prepared.columns]
    return prepared


resp = requests.get(RATINGS_URL, timeout=(10, 120))
resp.raise_for_status()
tmp_ods = "/tmp/cqc_latest_ratings.ods"
with open(tmp_ods, "wb") as fh:
    fh.write(resp.content)

workbook = CalamineWorkbook.from_path(tmp_ods)
sheet_rows = workbook.get_sheet_by_name(SHEET_NAME).to_python(skip_empty_area=True)
raw_df = pd.DataFrame(sheet_rows[1:], columns=sheet_rows[0])
ratings_df = prepare_sheet(raw_df)

# Publication Date must be a real timestamp for the Stage 2 filters.
if "Publication Date" in ratings_df.columns:
    ratings_df["Publication Date"] = pd.to_datetime(
        ratings_df["Publication Date"], errors="coerce", dayfirst=True
    )

# Everything else stays as-is; replace NaN with None so Spark infers cleanly.
ratings_df = ratings_df.where(pd.notna(ratings_df), None)

spark_ratings = spark.createDataFrame(ratings_df)
# CQC keeps spaces / '?' / '/' in column names; Delta needs column mapping for that.
(
    spark_ratings.write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .option("delta.columnMapping.mode", "name")
    .saveAsTable(RATINGS_TABLE)
)

print(f"Columns: {list(ratings_df.columns)}")
print(f"Wrote {spark.table(RATINGS_TABLE).count():,} rows to {RATINGS_TABLE}")
display(spark.table(RATINGS_TABLE).limit(5))

# COMMAND ----------

# MAGIC %md ## Stage 2 — Build the sampled candidate set

# COMMAND ----------

# DBTITLE 1,Filter to Care Homes / Safe and sample across ratings
from pyspark.sql import functions as F

SAMPLED_VIEW = "cqc_pdf_sample_candidates"

selection_sql = f"""
CREATE OR REPLACE TEMP VIEW {SAMPLED_VIEW} AS
WITH filtered AS (
  SELECT
    `Location ID`, `Location Name`, `Provider ID`, `Provider Name`,
    `Location Region`, `Service / Population Group`, Domain,
    `Latest Rating`, `Publication Date`, `Report Type`, URL,
    sha2(concat_ws('||', `Location ID`, cast(`Publication Date` as string), coalesce(URL, '')), 256) AS sample_hash
  FROM {RATINGS_TABLE}
  WHERE `Publication Date` >= TIMESTAMP('2025-01-01')
    AND `Publication Date` <  TIMESTAMP('2027-01-01')
    AND `Service / Population Group` = 'Care Homes'
    AND Domain = 'Safe'
    AND `Care Home?` = 'Y'
),
non_good AS (
  SELECT * FROM filtered
  WHERE `Latest Rating` IN ('Requires improvement', 'Inadequate', 'Outstanding')
),
sampled_good AS (
  SELECT `Location ID`, `Location Name`, `Provider ID`, `Provider Name`,
         `Location Region`, `Service / Population Group`, Domain,
         `Latest Rating`, `Publication Date`, `Report Type`, URL, sample_hash
  FROM (
    SELECT *, row_number() OVER (ORDER BY sample_hash, `Location ID`) AS sample_rank
    FROM filtered WHERE `Latest Rating` = 'Good'
  ) ordered_good
  WHERE sample_rank <= {GOOD_SAMPLE_SIZE}
)
SELECT * FROM non_good
UNION ALL
SELECT * FROM sampled_good
"""

spark.sql(selection_sql)
sampled_df = spark.table(SAMPLED_VIEW)
print(f"Sampled candidates: {sampled_df.count()}")
display(
    sampled_df.groupBy("Latest Rating").count().orderBy(F.desc("count"), "Latest Rating")
)

# COMMAND ----------

# MAGIC %md ## Stage 3 — Resolve and download the inspection-report PDFs

# COMMAND ----------

# DBTITLE 1,Resolve each location's report PDF URL
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urljoin

import requests

RESOLVED_VIEW = "cqc_pdf_sample_resolved"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; CQC-PDF-Sampler/1.0)"}
HREF_PATTERN = re.compile(r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>(.*?)</a>', re.IGNORECASE | re.DOTALL)

candidate_query = spark.table(SAMPLED_VIEW)
if CANDIDATE_LIMIT > 0:
    candidate_query = candidate_query.orderBy("sample_hash").limit(CANDIDATE_LIMIT)
selected_rows = [row.asDict() for row in candidate_query.collect()]


def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip().lower()


def build_filename(row: dict) -> str:
    rating = re.sub(r"[^a-z0-9]+", "-", (row["Latest Rating"] or "unknown").strip().lower()).strip("-")
    location_id = re.sub(r"[^A-Za-z0-9-]+", "-", (row["Location ID"] or "unknown").strip()).strip("-")
    location_name = re.sub(r"[^a-z0-9]+", "-", (row["Location Name"] or "unknown").strip().lower()).strip("-")[:80]
    pub = row["Publication Date"].strftime("%Y-%m-%d") if row["Publication Date"] else "unknown-date"
    return f"{rating}__{pub}__{location_id}__{location_name}.pdf"


def choose_pdf_url(page_url: str, html: str) -> str | None:
    candidates = []
    for href, anchor_html in HREF_PATTERN.findall(html):
        full_url = urljoin(page_url, href.strip())
        anchor_text = clean_text(re.sub(r"<[^>]+>", " ", anchor_html))
        lower_url = full_url.lower()
        score = 0
        if "api.cqc.org.uk/public/v1/reports/" in lower_url:
            score += 10
        if ".pdf" in lower_url:
            score += 8
        if "download full inspection report" in anchor_text:
            score += 8
        if "download" in anchor_text or "download" in lower_url:
            score += 4
        if "inspection" in anchor_text or "inspection" in lower_url:
            score += 3
        if "/sites/default/files/" in lower_url:
            score += 2
        is_pdf_candidate = (
            "api.cqc.org.uk/public/v1/reports/" in lower_url
            or ".pdf" in lower_url
            or "download full inspection report" in anchor_text
        )
        if is_pdf_candidate:
            candidates.append((score, full_url))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return candidates[0][1]


def resolve_pdf(row: dict) -> dict:
    page_url = row["URL"]
    result = {**row, "pdf_url": None, "download_filename": build_filename(row),
              "resolution_status": "unresolved", "resolution_error": None}
    if not page_url:
        result["resolution_error"] = "Missing location URL"
        return result
    try:
        response = requests.get(page_url, headers=HEADERS, timeout=(10, 30))
        response.raise_for_status()
        reports_url = response.url.rstrip("/") + "/reports"
        reports_response = requests.get(reports_url, headers=HEADERS, timeout=(10, 30))
        reports_response.raise_for_status()
        pdf_url = choose_pdf_url(reports_url, reports_response.text)
        if pdf_url:
            result["pdf_url"] = pdf_url
            result["resolution_status"] = "resolved"
        else:
            result["resolution_error"] = "No PDF link found on reports page"
    except Exception as exc:  # noqa: BLE001 - record and continue
        result["resolution_error"] = str(exc)
    return result


resolved_rows = []
with ThreadPoolExecutor(max_workers=16) as executor:
    futures = [executor.submit(resolve_pdf, row) for row in selected_rows]
    for future in as_completed(futures):
        resolved_rows.append(future.result())

resolved_df = spark.createDataFrame(pd.DataFrame(resolved_rows))
resolved_df.createOrReplaceTempView(RESOLVED_VIEW)
print(f"Candidates: {len(selected_rows)}  "
      f"resolved: {resolved_df.filter(F.col('resolution_status') == 'resolved').count()}  "
      f"unresolved: {resolved_df.filter(F.col('resolution_status') != 'resolved').count()}")
display(resolved_df.groupBy("resolution_status").count().orderBy(F.desc("count")))

# COMMAND ----------

# DBTITLE 1,Download resolved PDFs to the Volume
import os

os.makedirs(VOLUME_DIR, exist_ok=True)

download_rows = [
    row.asDict()
    for row in spark.table(RESOLVED_VIEW).filter(F.col("resolution_status") == "resolved").collect()
]


def download_pdf(row: dict) -> dict:
    output_path = os.path.join(VOLUME_DIR, row["download_filename"])
    result = {"Location ID": row["Location ID"], "Location Name": row["Location Name"],
              "Latest Rating": row["Latest Rating"], "pdf_url": row["pdf_url"],
              "output_path": output_path, "download_status": "failed",
              "download_error": None, "bytes_written": 0}
    try:
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            result["download_status"] = "already_exists"
            result["bytes_written"] = os.path.getsize(output_path)
            return result
        response = requests.get(row["pdf_url"], headers=HEADERS, timeout=(10, 60))
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "").lower()
        if "pdf" not in content_type and not row["pdf_url"].lower().endswith(".pdf"):
            raise ValueError(f"Unexpected content type: {content_type}")
        with open(output_path, "wb") as fh:
            fh.write(response.content)
        result["download_status"] = "downloaded"
        result["bytes_written"] = len(response.content)
    except Exception as exc:  # noqa: BLE001 - record and continue
        result["download_error"] = str(exc)
    return result


download_results = []
with ThreadPoolExecutor(max_workers=12) as executor:
    futures = [executor.submit(download_pdf, row) for row in download_rows]
    for future in as_completed(futures):
        download_results.append(future.result())

download_df = spark.createDataFrame(pd.DataFrame(download_results))
print(f"Output dir: {VOLUME_DIR}  submitted: {len(download_rows)}")
display(download_df.groupBy("download_status").count().orderBy(F.desc("count")))

# COMMAND ----------

# MAGIC %md ## Stage 4 — Parse the downloaded PDFs with ai_parse_document

# COMMAND ----------

# DBTITLE 1,Run ai_parse_document over the folder and save Delta table
parsed_pdf_df = spark.sql(f"""
SELECT
  path,
  regexp_extract(path, '[^/]+$', 0) AS file_name,
  lower(split(regexp_extract(path, '[^/]+$', 0), '__')[0]) AS rating_name,
  lower(regexp_extract(path, '[^/]+$', 0)) LIKE '%inadequate%' AS has_inadequate,
  ai_parse_document(content, map('version', '2.0')) AS parsed_json
FROM read_files('{VOLUME_DIR}', format => 'binaryFile')
""")

parsed_pdf_df.write.format("delta").mode("overwrite").option(
    "overwriteSchema", "true"
).saveAsTable(PARSED_TABLE)

print(f"Saved parsed results to {PARSED_TABLE}")

# COMMAND ----------

# DBTITLE 1,Verify parsed table
saved = spark.table(PARSED_TABLE)
display(saved.selectExpr(
    "count(*) as row_count",
    "count_if(try_cast(parsed_json:error_status as string) IS NOT NULL) as parse_errors",
    "count_if(rating_name = 'inadequate') as inadequate",
    "count_if(rating_name = 'good') as good",
    "count_if(rating_name = 'outstanding') as outstanding",
    "count_if(rating_name = 'requires-improvement') as requires_improvement",
))
display(saved.selectExpr(
    "file_name", "rating_name",
    "size(try_cast(parsed_json:document:pages as array<variant>)) as page_count",
    "size(try_cast(parsed_json:document:elements as array<variant>)) as element_count",
).limit(10))
