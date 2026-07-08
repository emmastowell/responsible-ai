# Databricks notebook source
# MAGIC %md
# MAGIC # Vector Search — Shared Infrastructure Provisioning (job)
# MAGIC
# MAGIC **Public Sector Responsible AI Hackathon | Databricks × RSS AI Task Force × Manuka**
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC This is the **organiser-only** provisioning job for the shared Vector Search assets
# MAGIC that `hackathon_vector_search.py` queries read-only. Run it **once** before the
# MAGIC session (or again if the shared assets are lost). It builds, in order:
# MAGIC
# MAGIC 1. the index-friendly **source table** (a clean projection of the tutorial features table),
# MAGIC 2. the shared, always-on **endpoint**, and
# MAGIC 3. the self-managed Delta Sync **index** (reuses the tutorial's `embedding` column — no re-embedding).
# MAGIC
# MAGIC It is **idempotent**: existing endpoint/index are reused (the index is re-synced), and
# MAGIC the source table is overwritten in place, so it is safe to re-run. We deliberately create
# MAGIC **one** shared endpoint for the whole room rather than one per attendee — an endpoint is
# MAGIC persistent, billed compute, and duplicates would bill for nothing extra. (Principles 2 cost,
# MAGIC 5 lifecycle, 7 shared governed assets, 10 governance.)
# MAGIC
# MAGIC > Teardown (deleting the endpoint/index after the session) lives at the bottom of
# MAGIC > `hackathon_vector_search.py`.

# COMMAND ----------

# MAGIC %pip install -U databricks-vectorsearch

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Configuration
from databricks.vector_search.client import VectorSearchClient
from pyspark.sql import functions as F

UC_CATALOG = "hackathon"
UC_SCHEMA = "default"

# ── SHARED infrastructure names (must match hackathon_vector_search.py) ─────────
VS_ENDPOINT_NAME = "hackathon_vs_shared"
VS_INDEX_NAME    = f"{UC_CATALOG}.{UC_SCHEMA}.tutorial_sci_med_vs_index"
VS_SOURCE_TABLE  = f"{UC_CATALOG}.{UC_SCHEMA}.tutorial_sci_med_vs_source"

# ── Features table the index is built from (the tutorial's FEATURES_TABLE) ──────
FEATURES_TABLE = f"{UC_CATALOG}.{UC_SCHEMA}.tutorial_sci_med_features_tutorial"

# ── Embedding model — MUST match the model that produced the `embedding` column ─
EMBEDDING_ENDPOINT = "databricks-bge-large-en"
EMBEDDING_DIM = 1024  # bge-large-en produces 1024-dim vectors

vsc = VectorSearchClient()  # picks up notebook credentials automatically

print(f"Source table:    {VS_SOURCE_TABLE}")
print(f"Shared endpoint: {VS_ENDPOINT_NAME}")
print(f"Shared index:    {VS_INDEX_NAME}")

# COMMAND ----------

# DBTITLE 1,1. Build the shared, index-friendly source table
# FEATURES_TABLE also holds MLlib VectorUDT columns (features / normalized_features) which
# are NOT valid Delta Sync source columns, so we project to a clean table and cast numeric
# feature columns to index-supported types (Vector Search rejects decimal; distance must be
# double, cluster int).
features_df = spark.table(FEATURES_TABLE)

vs_source_df = features_df.select(
    "article_id", "topic", "subject", "text", "summary", "embedding",
    F.col("cluster").cast("int").alias("cluster"),
    F.col("distance_to_centroid").cast("double").alias("distance_to_centroid"),
    "tension_label", "tension_reasoning",
)

(
    vs_source_df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(VS_SOURCE_TABLE)
)

# Delta Sync requires Change Data Feed on the source table.
spark.sql(f"ALTER TABLE {VS_SOURCE_TABLE} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")

# UC column comments via ALTER TABLE (serverless-safe — no RDD/schema round-trip).
COLUMN_COMMENTS = {
    "article_id":           "Unique identifier; primary key of the Vector Search index",
    "topic":                "Newsgroup topic label; used as a retrieval filter",
    "subject":              "Subject line of the original post (PII masked)",
    "text":                 "Article body: headers stripped, person/email PII masked",
    "summary":              "LLM summary the embedding was generated from",
    "embedding":            "1024-dim databricks-bge-large-en vector; self-managed index column",
    "cluster":              "KMeans cluster assignment from the tutorial (k=3)",
    "distance_to_centroid": "Distance to assigned cluster centroid, from the tutorial",
    "tension_label":        "LLM-classified tone: escalating, de-escalating, or neutral",
    "tension_reasoning":    "One-sentence LLM justification for tension_label",
}
for col, comment in COLUMN_COMMENTS.items():
    safe = comment.replace("'", "''")
    spark.sql(f"ALTER TABLE {VS_SOURCE_TABLE} ALTER COLUMN {col} COMMENT '{safe}'")

spark.sql(f"""
    COMMENT ON TABLE {VS_SOURCE_TABLE} IS
    'Shared Vector Search source: clean projection of {FEATURES_TABLE} with the bge-large-en embedding and key metadata. Change Data Feed enabled for Delta Sync.'
""")

row_count = spark.table(VS_SOURCE_TABLE).count()
print(f"Built source table {VS_SOURCE_TABLE} with {row_count} rows and CDF enabled.")

# COMMAND ----------

# DBTITLE 1,2. Create the shared endpoint (idempotent)
# STANDARD is right here: low latency and dictionary-format filters. A few thousand rows is
# tiny; STORAGE_OPTIMIZED is for 1B+ vectors and would be overkill.
existing = [e["name"] for e in vsc.list_endpoints().get("endpoints", [])]
if VS_ENDPOINT_NAME in existing:
    print(f"Endpoint '{VS_ENDPOINT_NAME}' already exists — reusing it.")
else:
    print(f"Creating STANDARD endpoint '{VS_ENDPOINT_NAME}' (can take several minutes)...")
    vsc.create_endpoint_and_wait(name=VS_ENDPOINT_NAME, endpoint_type="STANDARD")
    print("Endpoint is ONLINE.")

# COMMAND ----------

# DBTITLE 1,3. Create the shared self-managed Delta Sync index (idempotent)
existing_indexes = [i["name"] for i in vsc.list_indexes(VS_ENDPOINT_NAME).get("vector_indexes", [])]
if VS_INDEX_NAME in existing_indexes:
    print(f"Index '{VS_INDEX_NAME}' already exists — reusing it. Triggering a sync...")
    vsc.get_index(VS_ENDPOINT_NAME, VS_INDEX_NAME).sync()
else:
    print(f"Creating self-managed Delta Sync index '{VS_INDEX_NAME}' ...")
    vsc.create_delta_sync_index_and_wait(
        endpoint_name=VS_ENDPOINT_NAME,
        index_name=VS_INDEX_NAME,
        source_table_name=VS_SOURCE_TABLE,
        pipeline_type="TRIGGERED",            # sync on demand with index.sync()
        primary_key="article_id",
        embedding_vector_column="embedding",  # reuse the tutorial's vectors
        embedding_dimension=EMBEDDING_DIM,    # 1024 for bge-large-en
    )
    print("Index is ready.")

# COMMAND ----------

# DBTITLE 1,Verify
index = vsc.get_index(VS_ENDPOINT_NAME, VS_INDEX_NAME)
status = index.describe().get("status", {})
print(f"Index {VS_INDEX_NAME}")
print(f"  ready:          {status.get('ready')}")
print(f"  detailed_state: {status.get('detailed_state')}")
print(f"  indexed rows:   {status.get('indexed_row_count')}")
