# Databricks notebook source
# MAGIC %md
# MAGIC # FinancialPhraseBank ingestion
# MAGIC
# MAGIC Loads the FinancialPhraseBank dataset (Malo et al., 2014): ~4,840 unique
# MAGIC financial-news sentences labelled positive / neutral / negative, provided at
# MAGIC four annotator-agreement thresholds (50/66/75/100%). Sentiment baseline for
# MAGIC the fraud / market-manipulation use cases.
# MAGIC
# MAGIC Downloads the raw zip from Hugging Face and parses the `.txt` files directly
# MAGIC (the files are latin-1 `sentence@label`). This avoids the `datasets` library's
# MAGIC `trust_remote_code` path, which newer versions no longer support.

# COMMAND ----------

# DBTITLE 1,Parameters
dbutils.widgets.text("catalog", "hackathon", "Catalog")
dbutils.widgets.text("schema", "shared_datasets", "Schema")
dbutils.widgets.text("target_table", "financial_phrasebank", "Target table")
dbutils.widgets.text(
    "source_url",
    "https://huggingface.co/datasets/takala/financial_phrasebank/resolve/main/data/FinancialPhraseBank-v1.0.zip",
    "Source zip URL",
)

CATALOG = dbutils.widgets.get("catalog").strip()
SCHEMA = dbutils.widgets.get("schema").strip()
TARGET_TABLE = f"{CATALOG}.{SCHEMA}." + dbutils.widgets.get("target_table").strip()
SOURCE_URL = dbutils.widgets.get("source_url").strip()

print(f"target_table = {TARGET_TABLE}")
print(f"source_url   = {SOURCE_URL}")

spark.sql(f"CREATE SCHEMA IF NOT EXISTS `{CATALOG}`.`{SCHEMA}`")

# COMMAND ----------

# DBTITLE 1,Download and parse the sentence files
import io
import re
import zipfile
from datetime import datetime

import requests
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

# Hugging Face label convention for financial_phrasebank
LABEL_ID = {"negative": 0, "neutral": 1, "positive": 2}
SENTENCE_FILE = re.compile(r"Sentences_(\w+)Agree\.txt$")

resp = requests.get(SOURCE_URL, timeout=120)
resp.raise_for_status()

records = []
now = datetime.utcnow().isoformat()
with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
    for name in zf.namelist():
        if "__MACOSX" in name:
            continue
        m = SENTENCE_FILE.search(name)
        if not m:
            continue
        agreement_level = f"sentences_{m.group(1).lower()}agree"
        text = zf.read(name).decode("latin-1")
        n = 0
        for line in text.splitlines():
            line = line.strip()
            if not line or "@" not in line:
                continue
            sentence, label = line.rsplit("@", 1)
            label = label.strip().lower()
            records.append({
                "sentence": sentence.strip(),
                "label": label,
                "label_id": LABEL_ID.get(label),
                "agreement_level": agreement_level,
                "ingested_at": now,
            })
            n += 1
        print(f"  {agreement_level}: {n} sentences")

if not records:
    raise RuntimeError("No sentences parsed from the FinancialPhraseBank zip.")
print(f"Total rows: {len(records)}")

# COMMAND ----------

# DBTITLE 1,Write Delta table
SCHEMA_STRUCT = StructType([
    StructField("sentence", StringType()),
    StructField("label", StringType()),
    StructField("label_id", IntegerType()),
    StructField("agreement_level", StringType()),
    StructField("ingested_at", StringType()),
])

sdf = spark.createDataFrame(records, schema=SCHEMA_STRUCT)
sdf.write.format("delta").mode("overwrite").option(
    "overwriteSchema", "true"
).saveAsTable(TARGET_TABLE)
print(f"Wrote {sdf.count()} rows to {TARGET_TABLE}")

# COMMAND ----------

# DBTITLE 1,Verify
from pyspark.sql import functions as F

fpb = spark.table(TARGET_TABLE)
display(
    fpb.groupBy("agreement_level", "label").count()
    .orderBy("agreement_level", "label")
)
display(fpb.limit(5))
