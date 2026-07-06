# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Configure extraction targets
import os
import zipfile
from pathlib import Path

from pyspark.sql import functions as F

# Configurable parameters
zip_path = "/Volumes/hackathon/shared_datasets/crypto_raw/archive (7).zip"
extraction_root = "/Volumes/hackathon/shared_datasets/crypto_raw/participant_ready/archive_7"
summary_table = "hackathon.shared_datasets.crypto_subgraph_summary"
manifest_table = "hackathon.shared_datasets.crypto_subgraph_manifest"
overwrite_extraction = False

summary_csv_path = f"{extraction_root}/subgraph_summary.csv"
graphml_root = f"{extraction_root}/subgraph/subgraph"
llm4tg_root = f"{extraction_root}/repr_llm4tg/repr_llm4tg"

print(f"zip_path = {zip_path}")
print(f"extraction_root = {extraction_root}")
print(f"summary_table = {summary_table}")
print(f"manifest_table = {manifest_table}")
print(f"overwrite_extraction = {overwrite_extraction}")

# COMMAND ----------

# DBTITLE 1,Inspect zip contents
with zipfile.ZipFile(zip_path, "r") as zf:
    members = [zi for zi in zf.infolist() if not zi.is_dir()]
    member_rows = [
        (zi.filename, zi.file_size, zi.compress_size, Path(zi.filename).suffix.lower())
        for zi in members[:50]
    ]

member_df = spark.createDataFrame(
    member_rows,
    ["member_path", "file_size_bytes", "compressed_size_bytes", "suffix"],
)

print(f"member_count = {len(members)}")
display(member_df)

# COMMAND ----------

# DBTITLE 1,Extract archive into shared volume
os.makedirs(extraction_root, exist_ok=True)

needs_extract = overwrite_extraction or (not os.path.exists(summary_csv_path))

if needs_extract:
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extraction_root)
    print(f"Extracted archive to {extraction_root}")
else:
    print(f"Extraction already present at {extraction_root}")

extracted_entries = sorted(dbutils.fs.ls(extraction_root), key=lambda x: x.name)
for entry in extracted_entries:
    print(f"{entry.name}\t{entry.size}\t{entry.path}")

# COMMAND ----------

# DBTITLE 1,Create summary Delta table
summary_df = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(summary_csv_path)
    .withColumn("summary_csv_path", F.lit(summary_csv_path))
    .withColumn("extraction_root", F.lit(extraction_root))
    .withColumn("ingested_at", F.current_timestamp())
)

(
    summary_df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(summary_table)
)

print(f"Saved summary rows to {summary_table}")
display(spark.table(summary_table).limit(10))

# COMMAND ----------

# DBTITLE 1,Create manifest Delta table
graphml_rows = []
for dirpath, _, filenames in os.walk(graphml_root):
    for filename in filenames:
        if filename.lower().endswith(".graphml"):
            full_path = os.path.join(dirpath, filename)
            graphml_rows.append((
                Path(filename).stem,
                full_path,
                Path(full_path).parent.name,
                os.path.getsize(full_path),
            ))

llm4tg_rows = []
for dirpath, _, filenames in os.walk(llm4tg_root):
    for filename in filenames:
        if filename.lower().endswith(".llm4tg"):
            full_path = os.path.join(dirpath, filename)
            llm4tg_rows.append((
                Path(filename).stem,
                full_path,
                os.path.getsize(full_path),
            ))

graphml_df = spark.createDataFrame(
    graphml_rows,
    ["address", "graphml_path", "path_label_type", "graphml_file_size_bytes"],
)
llm4tg_df = spark.createDataFrame(
    llm4tg_rows,
    ["address", "llm4tg_path", "llm4tg_file_size_bytes"],
)
summary_lookup_df = spark.table(summary_table).select(
    "address",
    F.col("type").alias("summary_type")
)

manifest_df = (
    graphml_df.alias("g")
    .join(llm4tg_df.alias("l"), on="address", how="full_outer")
    .join(summary_lookup_df.alias("s"), on="address", how="left")
    .select(
        F.col("address"),
        F.coalesce(F.col("summary_type"), F.col("path_label_type")).alias("type"),
        F.col("path_label_type"),
        F.col("graphml_path"),
        F.col("llm4tg_path"),
        F.col("graphml_file_size_bytes"),
        F.col("llm4tg_file_size_bytes"),
        F.col("graphml_path").isNotNull().alias("has_graphml"),
        F.col("llm4tg_path").isNotNull().alias("has_llm4tg"),
        F.lit(extraction_root).alias("extraction_root"),
        F.current_timestamp().alias("ingested_at"),
    )
)

(
    manifest_df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(manifest_table)
)

print(f"Saved manifest rows to {manifest_table}")
display(spark.table(manifest_table).orderBy("type", "address").limit(10))

# COMMAND ----------

# DBTITLE 1,Validate extracted assets and tables
validation_df = spark.sql(f"""
SELECT 'summary_table' AS asset_name, COUNT(*) AS row_count FROM {summary_table}
UNION ALL
SELECT 'manifest_table' AS asset_name, COUNT(*) AS row_count FROM {manifest_table}
UNION ALL
SELECT 'manifest_with_graphml' AS asset_name, COUNT(*) AS row_count FROM {manifest_table} WHERE has_graphml
UNION ALL
SELECT 'manifest_with_llm4tg' AS asset_name, COUNT(*) AS row_count FROM {manifest_table} WHERE has_llm4tg
""")

display(validation_df)
display(spark.table(summary_table).groupBy("type").count().orderBy(F.desc("count"), "type"))