# Databricks notebook source
# MAGIC %md
# MAGIC # Vector Search for Semantic Retrieval — Companion Notebook
# MAGIC
# MAGIC **Public Sector Responsible AI Hackathon | Databricks × RSS AI Task Force × Manuka | 9 July 2026**
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## What this notebook is
# MAGIC
# MAGIC This is a **companion** to `hackathon_tutorial.py`. That tutorial embedded post
# MAGIC summaries with `databricks-bge-large-en` and used **KMeans clustering** to find
# MAGIC corpus-wide structure. Here we take the *same embeddings* and put them to a
# MAGIC different use: **semantic retrieval** with **Databricks Vector Search**.
# MAGIC
# MAGIC Clustering and retrieval are two questions asked of the same vectors:
# MAGIC
# MAGIC - **Clustering** (the tutorial) answers *"what groups exist across the whole corpus?"*
# MAGIC - **Vector Search** (this notebook) answers *"what is most similar to **this** item?"*

# COMMAND ----------

# MAGIC %md
# MAGIC ## ⚠️ How this notebook is meant to be used — read first
# MAGIC
# MAGIC A Vector Search **endpoint** is persistent, always-on compute, and an **index** is a
# MAGIC shared governed asset. The cost is modest, but there is little point in every attendee
# MAGIC standing up their own copy just to query the same data — treat this infrastructure as
# MAGIC **cattle, not pets**: provision it once, share it, and tear it down when the work is
# MAGIC done. So this notebook has **two roles**, selected by the `role` widget at the top:
# MAGIC
# MAGIC | Role | Who | What runs | Creates infra? |
# MAGIC |------|-----|-----------|----------------|
# MAGIC | **`admin`** | One organiser, **once**, before the session | The *Admin setup* section: builds the shared source table, endpoint and index | **Yes** |
# MAGIC | **`attendee`** | Everyone, any number of times | Everything else: connects to the **existing shared index** and queries it | **No** |
# MAGIC
# MAGIC **The notebook is idempotent.** Running it as `attendee` is entirely **read-only** —
# MAGIC it creates nothing, deletes nothing, and can be run concurrently by many people
# MAGIC against the one shared index. The `admin` setup and teardown cells are guarded by
# MAGIC `if IS_ADMIN:` so attendees can safely *Run All* without provisioning anything.
# MAGIC The endpoint, index and source-table names below are **shared** (no per-user suffix)
# MAGIC precisely so that everyone points at the same asset. (Principles 2 cost, 5 lifecycle,
# MAGIC 7 shared governed assets, 10 governance.)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Where the AI Principles fit in this notebook
# MAGIC
# MAGIC | Stage | What happens | Principles |
# MAGIC |-------|--------------|------------|
# MAGIC | Reuse existing embeddings | Read the tutorial's features; no re-embedding | P2 (cost), P7 (reuse governed assets) |
# MAGIC | Shared, admin-provisioned index | One endpoint/index for all; attendees read-only | P2, P5, P7, P10 |
# MAGIC | Post-to-post & free-text retrieval | Same-model embedding for index and query | P5, P6 |
# MAGIC | Near-duplicate detection | Flags for a human to confirm, not an automated decision | P4 |
# MAGIC | Few-shot label retrieval | Retrieved labels are a signal, not ground truth | P1, P4 |

# COMMAND ----------

# MAGIC %md
# MAGIC ### Install the Vector Search client
# MAGIC The tutorial used only built-in functions; Vector Search needs its Python client.

# COMMAND ----------

# MAGIC %pip install -U databricks-vectorsearch

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Role: attendee (read-only) or admin (provisions the shared index)
# Attendees leave this as "attendee". The organiser sets it to "admin" once, before the
# session, to run the setup section. Everything gated by IS_ADMIN is skipped otherwise.
dbutils.widgets.dropdown("role", "attendee", ["attendee", "admin"])
IS_ADMIN = dbutils.widgets.get("role") == "admin"
print(f"Running as: {'ADMIN (setup enabled)' if IS_ADMIN else 'attendee (read-only)'}")

# COMMAND ----------

# DBTITLE 1,Configuration
from databricks.vector_search.client import VectorSearchClient
from pyspark.sql import functions as F
from datetime import datetime, UTC

UC_CATALOG = "hackathon"
UC_SCHEMA = "default"

# ── SHARED, admin-provisioned infrastructure ──────────────────────────────────
# These names are deliberately NOT per-user: one endpoint and one index are shared
# by everyone on the day. Attendees connect to them read-only; only the admin setup
# section (below) creates them. Do not add a username suffix here.
VS_ENDPOINT_NAME = "hackathon_vs_shared"
VS_INDEX_NAME    = f"{UC_CATALOG}.{UC_SCHEMA}.tutorial_sci_med_vs_index"
VS_SOURCE_TABLE  = f"{UC_CATALOG}.{UC_SCHEMA}.tutorial_sci_med_vs_source"

# ── ADMIN ONLY: the features table the shared index is built from ──────────────
# Point this at the FEATURES_TABLE your tutorial run produced. Used only by the
# admin setup section; attendees never read it.
ADMIN_FEATURES_TABLE = f"{UC_CATALOG}.{UC_SCHEMA}.tutorial_sci_med_features_tutorial"

# ── Embedding model — MUST match the model that produced the `embedding` column ─
# Otherwise the index and your free-text queries live in different vector spaces (P5).
EMBEDDING_ENDPOINT = "databricks-bge-large-en"
EMBEDDING_DIM = 1024  # bge-large-en produces 1024-dim vectors

vsc = VectorSearchClient()  # picks up notebook credentials automatically

print(f"Shared index:    {VS_INDEX_NAME}")
print(f"Shared endpoint: {VS_ENDPOINT_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## What Vector Search is, and why now
# MAGIC
# MAGIC An embedding turns each summary into a point in a 1024-dimensional space where
# MAGIC semantically similar texts sit close together. In the tutorial we used that geometry
# MAGIC *globally* — KMeans carved the whole cloud of points into three clusters.
# MAGIC
# MAGIC **Vector Search** uses the same geometry *locally*. It builds a managed
# MAGIC approximate-nearest-neighbour (ANN) index so that, given any point, it can return the
# MAGIC closest others in milliseconds — the engine behind semantic search and RAG.
# MAGIC
# MAGIC **Principle 6 (the right tool for the job):** clustering is right for *"describe the
# MAGIC corpus"*; a vector index is right for *"act on one item"*.

# COMMAND ----------

# MAGIC %md
# MAGIC # ══════════════ ADMIN SETUP (run once) ══════════════
# MAGIC
# MAGIC Everything in this section is guarded by `if IS_ADMIN:` and only runs when the
# MAGIC `role` widget is set to `admin`. It is **idempotent** — existing endpoint/index are
# MAGIC reused, and the source table is overwritten in place — so it is safe to re-run.
# MAGIC Attendees skip all of it.
# MAGIC
# MAGIC ### Choosing the index type
# MAGIC
# MAGIC | Option | How it embeds | Trade-off |
# MAGIC |--------|---------------|-----------|
# MAGIC | **(a) Delta Sync, managed embeddings** | Point at the `summary` text column | Simplest to query, but **re-embeds** text you already paid for |
# MAGIC | **(b) Delta Sync, self-managed embeddings** | Point at the existing `embedding` column | **No re-embedding cost**; forces the same-model discipline (P5) |
# MAGIC | **(c) Direct Access** | Manual upsert/delete | For real-time updates; overkill for a static governed batch |
# MAGIC
# MAGIC **We use (b)** — it continues the tutorial's "embeddings are reusable features" thesis
# MAGIC at zero extra embedding spend, and makes the same-model lesson explicit.

# COMMAND ----------

# DBTITLE 1,ADMIN: build the shared, index-friendly source table
if IS_ADMIN:
    # FEATURES_TABLE also holds MLlib VectorUDT columns (features / normalized_features)
    # which are NOT valid Delta Sync source columns, so we project to a clean table.
    # We also cast the numeric feature columns to index-supported types (Vector Search
    # rejects decimal; distance must be double, cluster int).
    features_df = spark.table(ADMIN_FEATURES_TABLE)

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
        'Shared Vector Search source: clean projection of {ADMIN_FEATURES_TABLE} with the bge-large-en embedding and key metadata. Change Data Feed enabled for Delta Sync.'
    """)
    print(f"Built shared source table {VS_SOURCE_TABLE} with CDF enabled.")
else:
    print("Skipped (attendee): shared source table is provisioned by the admin.")

# COMMAND ----------

# DBTITLE 1,ADMIN: create the shared endpoint (idempotent)
if IS_ADMIN:
    # STANDARD is right here: low latency and dictionary-format filters. 200 rows is tiny;
    # STORAGE_OPTIMIZED is for 1B+ vectors and would be overkill.
    existing = [e["name"] for e in vsc.list_endpoints().get("endpoints", [])]
    if VS_ENDPOINT_NAME in existing:
        print(f"Endpoint '{VS_ENDPOINT_NAME}' already exists — reusing it.")
    else:
        print(f"Creating STANDARD endpoint '{VS_ENDPOINT_NAME}' (can take several minutes)...")
        vsc.create_endpoint_and_wait(name=VS_ENDPOINT_NAME, endpoint_type="STANDARD")
        print("Endpoint is ONLINE.")
else:
    print("Skipped (attendee): endpoint is provisioned by the admin.")

# COMMAND ----------

# DBTITLE 1,ADMIN: create the shared self-managed Delta Sync index (idempotent)
if IS_ADMIN:
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
            pipeline_type="TRIGGERED",          # sync on demand with index.sync()
            primary_key="article_id",
            embedding_vector_column="embedding",  # reuse the tutorial's vectors
            embedding_dimension=EMBEDDING_DIM,    # 1024 for bge-large-en
        )
        print("Index is ready.")

    # ── Managed-embeddings alternative (option a), for comparison — do NOT run ──
    # Lets Databricks embed the `summary` text so you could query with query_text.
    # We do not use it because it re-embeds text we already embedded.
    #
    # vsc.create_delta_sync_index_and_wait(
    #     endpoint_name=VS_ENDPOINT_NAME, index_name=VS_INDEX_NAME,
    #     source_table_name=VS_SOURCE_TABLE, pipeline_type="TRIGGERED",
    #     primary_key="article_id", embedding_source_column="summary",
    #     embedding_model_endpoint_name="databricks-bge-large-en")  # pin the SAME model
else:
    print("Skipped (attendee): index is provisioned by the admin.")

# COMMAND ----------

# MAGIC %md
# MAGIC # ══════════════ EVERYONE (read-only from here) ══════════════

# COMMAND ----------

# DBTITLE 1,Connect to the shared index
# Read-only: every attendee runs this and shares the one index. If it is missing, the
# admin has not run the setup section yet.
try:
    index = vsc.get_index(VS_ENDPOINT_NAME, VS_INDEX_NAME)
    status = index.describe().get("status", {})
    print(f"Connected to shared index {VS_INDEX_NAME}")
    print(f"  detailed_state: {status.get('detailed_state')}")
    print(f"  indexed rows:   {status.get('indexed_row_count')}")
except Exception as e:
    raise RuntimeError(
        f"Shared index '{VS_INDEX_NAME}' not found on endpoint '{VS_ENDPOINT_NAME}'. "
        "Ask your workshop admin to run the Admin setup section once "
        "(set the 'role' widget to 'admin')."
    ) from e

RESULT_COLS = ["article_id", "topic", "summary", "cluster", "tension_label"]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Use case 1 — "find the posts most similar to this one"
# MAGIC
# MAGIC The cheapest possible query: take a post's **stored** embedding and ask the index for
# MAGIC its nearest neighbours. No new embedding call is needed. We ask for one extra result
# MAGIC and drop the post itself (a post is always its own nearest neighbour).

# COMMAND ----------

# DBTITLE 1,Post-to-post similarity
# Read a seed post's stored vector from the shared source table (a read-only SELECT —
# attendees need only SELECT on this table, they never provision anything).
seed = (
    spark.table(VS_SOURCE_TABLE)
    .filter(F.col("topic") == "rec.autos")
    .select("article_id", "topic", "summary", "embedding", "cluster")
    .limit(1)
    .collect()[0]
)
print(f"Seed post {seed['article_id']} (topic={seed['topic']}, cluster={seed['cluster']}):")
print(f"  {seed['summary']}\n")

res = index.similarity_search(
    query_vector=[float(x) for x in seed["embedding"]],
    columns=RESULT_COLS,
    num_results=6,  # 5 neighbours + the seed itself
)
seed_id = seed["article_id"]

print(f"{'article_id':<14}{'topic':<20}{'cluster':<9}{'score':<8}summary")
for r in res["result"]["data_array"]:
    aid, topic, summary, cluster, tension, score = r
    if aid == seed_id:
        continue  # drop the seed's self-match
    print(f"{aid:<14}{topic:<20}{str(cluster):<9}{score:<8.3f}{(summary or '')[:70]}")

# COMMAND ----------

# MAGIC %md
# MAGIC Notice the nearest neighbours are almost all from the same newsgroup and the same
# MAGIC KMeans cluster as the seed — retrieval and clustering agree because they read the same
# MAGIC geometry. But retrieval gives a *ranked, per-item* answer with scores, where clustering
# MAGIC gave a single hard label. That finer granularity is the point.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Use case 2 — free-text semantic query
# MAGIC
# MAGIC A query that is **not** already in the table. Because our index is *self-managed*, it
# MAGIC does not embed text for us — so we embed the query string ourselves with the **same**
# MAGIC `databricks-bge-large-en` model the tutorial used for the documents. Embedding with a
# MAGIC different model would place the query in an incompatible space. This is **Principle 5**
# MAGIC made concrete. We also show a **filter** — governed, scoped retrieval.

# COMMAND ----------

# DBTITLE 1,Embed a free-text query with the same model, then search
def embed_query(text: str) -> list:
    """Embed a single query string with the SAME endpoint used for the documents."""
    row = spark.sql(
        "SELECT ai_query(:endpoint, :q) AS v",
        args={"endpoint": EMBEDDING_ENDPOINT, "q": text},
    ).collect()[0]
    return [float(x) for x in row["v"]]

query_text = "spacecraft propulsion and orbital mechanics"
qv = embed_query(query_text)
assert len(qv) == EMBEDDING_DIM

res = index.similarity_search(
    query_vector=qv,
    columns=RESULT_COLS,
    num_results=5,
    filters={"topic": "sci.space"},  # STANDARD endpoint uses dict-format filters
)

print(f"Query: {query_text!r}  (filtered to sci.space)\n")
print(f"{'article_id':<14}{'score':<8}summary")
for r in res["result"]["data_array"]:
    aid, topic, summary, cluster, tension, score = r
    print(f"{aid:<14}{score:<8.3f}{(summary or '')[:80]}")

# COMMAND ----------

# MAGIC %md
# MAGIC **Principle 1 (know the limits):** the scores are *relative* similarities, not
# MAGIC calibrated probabilities, and the results are only as good as the lossy summaries the
# MAGIC embeddings were built from. A high rank means "closest available", not "correct".

# COMMAND ----------

# MAGIC %md
# MAGIC ## Use case 3 — semantic near-duplicate detection
# MAGIC
# MAGIC Official statistics care a lot about **de-duplication**. Exact-match de-dup misses
# MAGIC reworded repeats; semantic similarity catches them. For a sample of posts we look at
# MAGIC each one's single nearest *other* neighbour and flag the pair when similarity is very
# MAGIC high.
# MAGIC
# MAGIC **Principle 4 (human control):** these are **candidates for a human to confirm**, not
# MAGIC an instruction to auto-delete rows.

# COMMAND ----------

# DBTITLE 1,Flag likely near-duplicate pairs for review
NEAR_DUP_THRESHOLD = 0.97  # tune on your data; higher = stricter

# A sample of posts from the shared source table (read-only).
candidates = (
    spark.table(VS_SOURCE_TABLE)
    .select("article_id", "summary", "embedding")
    .limit(60)
    .collect()
)

seen_pairs, flagged = set(), []
for row in candidates:
    res = index.similarity_search(
        query_vector=[float(x) for x in row["embedding"]],
        columns=["article_id", "summary"],
        num_results=2,  # itself + nearest other
    )
    for other_id, other_summary, score in res["result"]["data_array"]:
        if other_id == row["article_id"]:
            continue
        pair = tuple(sorted((row["article_id"], other_id)))
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        if score >= NEAR_DUP_THRESHOLD:
            flagged.append((pair[0], pair[1], round(float(score), 4),
                            (row["summary"] or "")[:60], (other_summary or "")[:60]))

print(f"{len(flagged)} candidate near-duplicate pair(s) at threshold {NEAR_DUP_THRESHOLD}:\n")
for a, b, score, sa, sb in sorted(flagged, key=lambda x: -x[2]):
    print(f"  {score}  {a} ~ {b}\n      A: {sa}\n      B: {sb}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Use case 4 — nearest labelled examples as a feature (few-shot)
# MAGIC
# MAGIC The tutorial produced a `tension_label` (escalating / de-escalating / neutral).
# MAGIC Retrieval lets us use those labels in two ways:
# MAGIC
# MAGIC 1. **As a feature:** for a target post, the label distribution of its nearest
# MAGIC    neighbours is a retrieval-derived feature (a k-NN signal in embedding space).
# MAGIC 2. **As few-shot prompt examples:** the nearest labelled posts are the most relevant
# MAGIC    examples to show an LLM when classifying a new post.
# MAGIC
# MAGIC **Principles 1 and 4:** the retrieved labels are themselves LLM guesses — a signal for
# MAGIC a human or model, never ground truth.

# COMMAND ----------

# DBTITLE 1,Neighbour-label vote as a feature, and few-shot example selection
from collections import Counter

# A target post from the shared source table (read-only).
target = (
    spark.table(VS_SOURCE_TABLE)
    .filter(F.col("tension_label").isNotNull())
    .select("article_id", "summary", "tension_label", "embedding")
    .limit(1)
    .collect()[0]
)
t_id, t_summary, t_label = target["article_id"], target["summary"], target["tension_label"]

res = index.similarity_search(
    query_vector=[float(x) for x in target["embedding"]],
    columns=["article_id", "summary", "tension_label"],
    num_results=6,  # target + 5 neighbours
)
neighbours = [
    (aid, summary, label, score)
    for aid, summary, label, score in res["result"]["data_array"]
    if aid != t_id
]

votes = Counter(label for _, _, label, _ in neighbours if label)
majority = votes.most_common(1)[0][0] if votes else None

print(f"Target {t_id} — its own tension_label: {t_label}")
print(f"  {(t_summary or '')[:90]}\n")
print(f"Neighbour label distribution (the retrieval-derived feature): {dict(votes)}")
print(f"Neighbour majority vote: {majority}\n")

few_shot_block = "\n".join(
    f"- Example ({label}): {(summary or '')[:80]}" for _, summary, label, _ in neighbours if label
)
print("Few-shot examples you could inject into a tension ai_query prompt:")
print(few_shot_block)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Clustering vs vector search — when to use which
# MAGIC
# MAGIC | | KMeans clustering (tutorial) | Vector search retrieval (this notebook) |
# MAGIC |---|---|---|
# MAGIC | Question answered | "What groups exist across the whole corpus?" | "What is most similar to *this* item?" |
# MAGIC | Output | One hard label per row (k=3) + distance-to-centroid | Ranked top-k neighbours per query, with scores |
# MAGIC | When computed | Once, offline, on the full set | On demand, per query |
# MAGIC | Granularity | Coarse (k=3, loose clusters) | Fine; no commitment to a fixed number of groups |
# MAGIC | Best for | Discovering themes/segments, dimensionality reduction | Similar-item lookup, de-duplication, few-shot selection, RAG |
# MAGIC | Cost model | Cheap one-off MLlib fit | Persistent, always-on endpoint |
# MAGIC
# MAGIC **Punchline:** *cluster to understand the corpus as a whole; retrieve to act on a
# MAGIC single item.* They are complementary.

# COMMAND ----------

# MAGIC %md
# MAGIC # ══════════════ ADMIN TEARDOWN ══════════════
# MAGIC
# MAGIC **Do not run this during the workshop** — it deletes the *shared* index and endpoint
# MAGIC that everyone else is using. It is guarded by `if IS_ADMIN:` and is intended only for
# MAGIC the organiser to run **after** the session, so the always-on endpoint stops billing
# MAGIC (Principles 2 and 5).

# COMMAND ----------

# DBTITLE 1,ADMIN: delete the shared index and endpoint (post-workshop only)
TEARDOWN = False  # admin sets this to True, after the session, to actually delete

if IS_ADMIN and TEARDOWN:
    vsc.delete_index(index_name=VS_INDEX_NAME)
    print(f"Deleted index {VS_INDEX_NAME}.")
    vsc.delete_endpoint(name=VS_ENDPOINT_NAME)
    print(f"Deleted endpoint {VS_ENDPOINT_NAME}.")
else:
    print("Teardown skipped. (Admin: set TEARDOWN=True after the session to delete shared infra.)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Now it is your turn!
# MAGIC
# MAGIC You have seen how the embeddings from the tutorial can be reused for semantic
# MAGIC retrieval — similar-item lookup, free-text search, near-duplicate detection, and
# MAGIC few-shot feature engineering — all against a single shared, governed index.
# MAGIC
# MAGIC For your own project, an admin can point a new index at your team's corpus (for
# MAGIC example the parsed CQC inspection reports) and ask: *which reports are most similar to
# MAGIC a known problem case?* Remember to embed with a single, consistent model, share one
# MAGIC index rather than one-per-person, keep a human in the loop on anything you flag, and
# MAGIC tear the endpoint down when the work is done.
