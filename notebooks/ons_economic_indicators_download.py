# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,ONS economic indicators overview
# MAGIC %md
# MAGIC This notebook downloads a small, reusable set of public UK Office for National Statistics economic indicator datasets for hackathon use.
# MAGIC
# MAGIC Default datasets included:
# MAGIC * CPIH inflation time series
# MAGIC * UK labour market time series (useful for unemployment and earnings-related indicators)
# MAGIC * GDP monthly estimate
# MAGIC * Retail sales index
# MAGIC
# MAGIC The notebook is self-contained and configurable. It discovers the latest public ONS download URLs from the ONS beta dataset API, downloads the raw files into the current user's workspace home, loads each file into a Spark DataFrame for preview, and builds a manifest with dataset name, source URL, local path, row count, and status.
# MAGIC
# MAGIC If you want a different indicator mix later, update the `datasets` list in the configuration cell and rerun the notebook.

# COMMAND ----------

# DBTITLE 1,Configure datasets and discover latest download URLs
import os
from pathlib import Path

import pandas as pd
import requests
from pyspark.sql import functions as F

base_api = "https://api.beta.ons.gov.uk/v1/datasets"
current_user = spark.sql("SELECT current_user() AS user_name").first()["user_name"]
workspace_download_dir = f"/Workspace/Users/{current_user}/ons_economic_indicators_downloads"
uc_target_schema = "hackathon.shared_datasets"
manifest_table = f"{uc_target_schema}.ons_economic_indicator_download_manifest"
os.makedirs(workspace_download_dir, exist_ok=True)

datasets = [
    {
        "dataset_name": "cpih_inflation",
        "dataset_id": "cpih01",
        "description": "Consumer Prices Index including owner occupiers' housing costs (CPIH)",
        "edition": "time-series",
        "preferred_format": "csv",
    },
    {
        "dataset_name": "uk_labour_market",
        "dataset_id": "labour-market",
        "description": "UK labour market time series, useful for unemployment and earnings-related indicators",
        "edition": "time-series",
        "preferred_format": "csv",
    },
    {
        "dataset_name": "gdp_monthly_estimate",
        "dataset_id": "gdp-to-four-decimal-places",
        "description": "GDP monthly estimate time series",
        "edition": "time-series",
        "preferred_format": "csv",
    },
    {
        "dataset_name": "retail_sales_index",
        "dataset_id": "retail-sales-index",
        "description": "Retail sales index time series",
        "edition": "time-series",
        "preferred_format": "csv",
    },
]


def get_latest_download_metadata(dataset_id: str, edition: str = "time-series", preferred_format: str = "csv") -> dict:
    edition_url = f"{base_api}/{dataset_id}/editions/{edition}"
    edition_response = requests.get(edition_url, timeout=30)
    edition_response.raise_for_status()
    edition_payload = edition_response.json()

    latest_version_url = edition_payload["links"]["latest_version"]["href"]
    version_response = requests.get(latest_version_url, timeout=30)
    version_response.raise_for_status()
    version_payload = version_response.json()

    downloads = version_payload.get("downloads", {})
    selected_format = preferred_format if preferred_format in downloads else next(iter(downloads.keys()))
    download_info = downloads[selected_format]
    download_url = download_info["href"]
    file_extension = Path(download_url).suffix or f".{selected_format}"

    return {
        "edition": edition,
        "version": str(version_payload.get("version")),
        "download_format": selected_format,
        "download_url": download_url,
        "file_extension": file_extension,
    }


dataset_downloads = []
for dataset in datasets:
    latest_info = get_latest_download_metadata(
        dataset_id=dataset["dataset_id"],
        edition=dataset["edition"],
        preferred_format=dataset["preferred_format"],
    )
    dataset_downloads.append({**dataset, **latest_info})

for dataset in dataset_downloads:
    dataset["target_table"] = f"{uc_target_schema}.ons_{dataset['dataset_name']}"

manifest_seed_df = spark.createDataFrame(pd.DataFrame(dataset_downloads))
print(f"workspace_download_dir = {workspace_download_dir}")
print(f"uc_target_schema = {uc_target_schema}")
print(f"manifest_table = {manifest_table}")
display(manifest_seed_df)

# COMMAND ----------

# DBTITLE 1,Download raw files and load previews
import re


def make_unique_spark_safe_columns(columns):
    used_names = {}
    cleaned_columns = []

    for position, column_name in enumerate(columns, start=1):
        cleaned_name = re.sub(r"[^0-9A-Za-z_]+", "_", str(column_name).strip().lower()).strip("_")
        if not cleaned_name:
            cleaned_name = f"column_{position}"
        if cleaned_name[0].isdigit():
            cleaned_name = f"column_{position}_{cleaned_name}"

        duplicate_count = used_names.get(cleaned_name, 0)
        used_names[cleaned_name] = duplicate_count + 1
        final_name = cleaned_name if duplicate_count == 0 else f"{cleaned_name}_{duplicate_count + 1}"
        cleaned_columns.append(final_name)

    return cleaned_columns



download_results = []
preview_dfs = {}

session = requests.Session()
session.headers.update({"User-Agent": "Databricks-ONS-Downloader/1.0"})

for dataset in dataset_downloads:
    output_file_name = f"{dataset['dataset_name']}_v{dataset['version']}{dataset['file_extension']}"
    local_path = os.path.join(workspace_download_dir, output_file_name)
    target_table = dataset["target_table"]
    status = "downloaded"
    row_count = None
    error_message = None

    try:
        response = session.get(dataset["download_url"], timeout=120)
        response.raise_for_status()
        with open(local_path, "wb") as file_handle:
            file_handle.write(response.content)

        if dataset["download_format"] == "csv":
            pandas_df = pd.read_csv(local_path, low_memory=False)
        elif dataset["download_format"] == "json":
            pandas_df = pd.read_json(local_path)
        else:
            raise ValueError(f"Unsupported download format for preview: {dataset['download_format']}")

        pandas_df.columns = make_unique_spark_safe_columns(pandas_df.columns)
        row_count = int(len(pandas_df))
        preview_dfs[dataset["dataset_name"]] = spark.createDataFrame(pandas_df)
        (
            preview_dfs[dataset["dataset_name"]]
            .write
            .format("delta")
            .mode("overwrite")
            .option("overwriteSchema", "true")
            .saveAsTable(target_table)
        )
    except Exception as exc:
        status = "failed"
        error_message = f"{type(exc).__name__}: {str(exc)[:500]}"

    download_results.append(
        {
            "dataset_name": dataset["dataset_name"],
            "dataset_id": dataset["dataset_id"],
            "description": dataset["description"],
            "source_url": dataset["download_url"],
            "local_path": local_path,
            "download_format": dataset["download_format"],
            "version": dataset["version"],
            "target_table": target_table,
            "row_count": row_count,
            "status": status,
            "error_message": error_message,
        }
    )

manifest_df = spark.createDataFrame(pd.DataFrame(download_results))
manifest_df.createOrReplaceTempView("ons_indicator_download_manifest")
(
    manifest_df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(manifest_table)
)
display(manifest_df)

# COMMAND ----------

# DBTITLE 1,Validate downloaded files
downloaded_files = sorted(os.listdir(workspace_download_dir))
file_rows = []
for file_name in downloaded_files:
    full_path = os.path.join(workspace_download_dir, file_name)
    file_rows.append((file_name, full_path, os.path.getsize(full_path)))

files_df = spark.createDataFrame(file_rows, ["file_name", "local_path", "file_size_bytes"])
display(files_df)

for dataset_name in sorted(preview_dfs):
    print(f"Preview for {dataset_name}")
    display(preview_dfs[dataset_name].limit(10))

# COMMAND ----------

# DBTITLE 1,Validate manifest summary
manifest_summary_df = spark.sql("""
SELECT
  status,
  COUNT(*) AS dataset_count,
  SUM(COALESCE(row_count, 0)) AS total_preview_rows
FROM ons_indicator_download_manifest
GROUP BY status
ORDER BY status
""")

display(manifest_summary_df)
display(
    spark.table("ons_indicator_download_manifest")
    if False else manifest_df.select(
        "dataset_name",
        "dataset_id",
        "version",
        "download_format",
        "row_count",
        "status"
    )
)

print(f"Raw files saved under: {workspace_download_dir}")

# COMMAND ----------

# DBTITLE 1,Verify Unity Catalog tables
import re


def make_unique_spark_safe_columns(columns):
    used_names = {}
    cleaned_columns = []

    for position, column_name in enumerate(columns, start=1):
        cleaned_name = re.sub(r"[^0-9A-Za-z_]+", "_", str(column_name).strip().lower()).strip("_")
        if not cleaned_name:
            cleaned_name = f"column_{position}"
        if cleaned_name[0].isdigit():
            cleaned_name = f"column_{position}_{cleaned_name}"

        duplicate_count = used_names.get(cleaned_name, 0)
        used_names[cleaned_name] = duplicate_count + 1
        final_name = cleaned_name if duplicate_count == 0 else f"{cleaned_name}_{duplicate_count + 1}"
        cleaned_columns.append(final_name)

    return cleaned_columns


verification_rows = []
manifest_rows = []

for dataset in dataset_downloads:
    dataset_name = dataset["dataset_name"]
    target_table = dataset["target_table"]
    local_path = os.path.join(workspace_download_dir, f"{dataset_name}_v{dataset['version']}{dataset['file_extension']}")

    if dataset["download_format"] == "csv":
        pandas_df = pd.read_csv(local_path, low_memory=False)
    elif dataset["download_format"] == "json":
        pandas_df = pd.read_json(local_path)
    else:
        raise ValueError(f"Unsupported download format for verification: {dataset['download_format']}")

    pandas_df.columns = make_unique_spark_safe_columns(pandas_df.columns)
    dataset_df = spark.createDataFrame(pandas_df)
    (
        dataset_df.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(target_table)
    )

    row_count = int(dataset_df.count())
    verification_rows.append((dataset_name, target_table, row_count, len(dataset_df.columns)))
    manifest_rows.append({
        "dataset_name": dataset_name,
        "dataset_id": dataset["dataset_id"],
        "description": dataset["description"],
        "source_url": dataset["download_url"],
        "local_path": local_path,
        "download_format": dataset["download_format"],
        "version": dataset["version"],
        "target_table": target_table,
        "row_count": row_count,
        "status": "downloaded",
        "error_message": None,
    })

manifest_df = spark.createDataFrame(pd.DataFrame(manifest_rows))
(
    manifest_df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(manifest_table)
)

verification_rows.append(("download_manifest", manifest_table, spark.table(manifest_table).count(), len(manifest_df.columns)))
verification_df = spark.createDataFrame(
    verification_rows,
    ["dataset_name", "table_name", "row_count", "column_count"],
)

display(verification_df)
display(spark.table("hackathon.shared_datasets.ons_cpih_inflation").limit(5))