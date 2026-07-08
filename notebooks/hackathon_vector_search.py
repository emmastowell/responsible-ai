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
# MAGIC ## How this notebook is meant to be used — read first
# MAGIC
# MAGIC A Vector Search **endpoint** is persistent, always-on compute, and an **index** is a
# MAGIC shared governed asset. There is little point in every attendee standing up their own
# MAGIC copy just to query the same data, so the endpoint, index and source table have been
# MAGIC **provisioned once and shared** for the whole room. That keeps cost down — one
# MAGIC always-on endpoint instead of dozens — and means everyone queries the same governed
# MAGIC asset.
# MAGIC
# MAGIC Because of that, **this notebook is read-only for you.** You connect to the existing
# MAGIC shared index and query it; you create nothing and delete nothing, and the whole room
# MAGIC can run it at the same time. The provisioning code (source table, endpoint, index) is
# MAGIC included below **for reference but commented out** — an organiser ran it once before
# MAGIC the session. The endpoint, index and source-table names are deliberately **shared**
# MAGIC (no per-user suffix) so everyone points at the same asset. (Principles 2 cost,
# MAGIC 5 lifecycle, 7 shared governed assets, 10 governance.)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Where the AI Principles fit in this notebook
# MAGIC
# MAGIC | Stage | What happens | Principles |
# MAGIC |-------|--------------|------------|
# MAGIC | Reuse existing embeddings | Read the tutorial's features; no re-embedding | P2 (cost), P7 (reuse governed assets) |
# MAGIC | Shared, pre-provisioned index | One endpoint/index for all; attendees read-only | P2, P5, P7, P10 |
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

# DBTITLE 1,Configuration
from databricks.vector_search.client import VectorSearchClient
from pyspark.sql import functions as F

UC_CATALOG = "hackathon"
UC_SCHEMA = "default"

# ── SHARED infrastructure (provisioned once, before the session) ───────────────
# These names are deliberately NOT per-user: one endpoint and one index are shared
# by everyone on the day. You connect to them read-only. The code that created them
# is shown further down but commented out — an organiser ran it once. Do not add a
# username suffix here.
VS_ENDPOINT_NAME = "hackathon_vs_shared"
VS_INDEX_NAME    = f"{UC_CATALOG}.{UC_SCHEMA}.tutorial_sci_med_vs_index"
VS_SOURCE_TABLE  = f"{UC_CATALOG}.{UC_SCHEMA}.tutorial_sci_med_vs_source"

# Features table the shared index was built from — referenced only by the
# (commented-out) provisioning code below; attendees never read it.
FEATURES_TABLE = f"{UC_CATALOG}.{UC_SCHEMA}.tutorial_sci_med_features_tutorial"

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
# MAGIC # ══════════════ HOW THE SHARED INFRASTRUCTURE WAS BUILT (reference only) ══════════════
# MAGIC
# MAGIC The three cells in this section are **commented out** — an organiser ran them once
# MAGIC before the session to build the shared source table, endpoint and index. They are
# MAGIC left here so you can see exactly how the shared assets were made, but running this
# MAGIC notebook creates nothing. **Skip straight to *Connect to the shared index* below.**
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

# DBTITLE 1,Reference: how the shared source table was built (commented out — do not run)
# An organiser ran this once to build an index-friendly source table. FEATURES_TABLE also
# holds MLlib VectorUDT columns (features / normalized_features) which are NOT valid Delta
# Sync source columns, so it projects to a clean table and casts the numeric feature
# columns to index-supported types (Vector Search rejects decimal; distance must be
# double, cluster int).
#
# features_df = spark.table(FEATURES_TABLE)
#
# vs_source_df = features_df.select(
#     "article_id", "topic", "subject", "text", "summary", "embedding",
#     F.col("cluster").cast("int").alias("cluster"),
#     F.col("distance_to_centroid").cast("double").alias("distance_to_centroid"),
#     "tension_label", "tension_reasoning",
# )
#
# (
#     vs_source_df.write
#     .format("delta")
#     .mode("overwrite")
#     .option("overwriteSchema", "true")
#     .saveAsTable(VS_SOURCE_TABLE)
# )
#
# # Delta Sync requires Change Data Feed on the source table.
# spark.sql(f"ALTER TABLE {VS_SOURCE_TABLE} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
#
# # UC column comments via ALTER TABLE (serverless-safe — no RDD/schema round-trip).
# COLUMN_COMMENTS = {
#     "article_id":           "Unique identifier; primary key of the Vector Search index",
#     "topic":                "Newsgroup topic label; used as a retrieval filter",
#     "subject":              "Subject line of the original post (PII masked)",
#     "text":                 "Article body: headers stripped, person/email PII masked",
#     "summary":              "LLM summary the embedding was generated from",
#     "embedding":            "1024-dim databricks-bge-large-en vector; self-managed index column",
#     "cluster":              "KMeans cluster assignment from the tutorial (k=3)",
#     "distance_to_centroid": "Distance to assigned cluster centroid, from the tutorial",
#     "tension_label":        "LLM-classified tone: escalating, de-escalating, or neutral",
#     "tension_reasoning":    "One-sentence LLM justification for tension_label",
# }
# for col, comment in COLUMN_COMMENTS.items():
#     safe = comment.replace("'", "''")
#     spark.sql(f"ALTER TABLE {VS_SOURCE_TABLE} ALTER COLUMN {col} COMMENT '{safe}'")
#
# spark.sql(f"""
#     COMMENT ON TABLE {VS_SOURCE_TABLE} IS
#     'Shared Vector Search source: clean projection of the tutorial features table with the bge-large-en embedding and key metadata. Change Data Feed enabled for Delta Sync.'
# """)

# COMMAND ----------

# DBTITLE 1,Reference: creating the shared endpoint (commented out — do not run)
# We deliberately create ONE shared, always-on endpoint for the whole room rather than one
# per attendee. An endpoint is persistent, billed compute — dozens of duplicates would bill
# for nothing extra, since everyone queries the same data. So an organiser ran this once
# before the session; it is commented out here to keep the notebook read-only.
# (Principles 2 cost, 5 lifecycle.)
#
# STANDARD is the right type here: low latency and dictionary-format filters. 200 rows is
# tiny; STORAGE_OPTIMIZED is for 1B+ vectors and would be overkill.
#
# existing = [e["name"] for e in vsc.list_endpoints().get("endpoints", [])]
# if VS_ENDPOINT_NAME in existing:
#     print(f"Endpoint '{VS_ENDPOINT_NAME}' already exists — reusing it.")
# else:
#     print(f"Creating STANDARD endpoint '{VS_ENDPOINT_NAME}' (can take several minutes)...")
#     vsc.create_endpoint_and_wait(name=VS_ENDPOINT_NAME, endpoint_type="STANDARD")
#     print("Endpoint is ONLINE.")

# COMMAND ----------

# DBTITLE 1,Reference: creating the shared self-managed Delta Sync index (commented out — do not run)
# The index is self-managed: it reuses the tutorial's stored `embedding` column rather than
# re-embedding, so there is no extra embedding cost and the index shares the documents'
# vector space (P5). An organiser ran this once; it is commented out here.
#
# existing_indexes = [i["name"] for i in vsc.list_indexes(VS_ENDPOINT_NAME).get("vector_indexes", [])]
# if VS_INDEX_NAME in existing_indexes:
#     print(f"Index '{VS_INDEX_NAME}' already exists — reusing it. Triggering a sync...")
#     vsc.get_index(VS_ENDPOINT_NAME, VS_INDEX_NAME).sync()
# else:
#     print(f"Creating self-managed Delta Sync index '{VS_INDEX_NAME}' ...")
#     vsc.create_delta_sync_index_and_wait(
#         endpoint_name=VS_ENDPOINT_NAME,
#         index_name=VS_INDEX_NAME,
#         source_table_name=VS_SOURCE_TABLE,
#         pipeline_type="TRIGGERED",            # sync on demand with index.sync()
#         primary_key="article_id",
#         embedding_vector_column="embedding",  # reuse the tutorial's vectors
#         embedding_dimension=EMBEDDING_DIM,    # 1024 for bge-large-en
#     )
#     print("Index is ready.")
#
# # ── Managed-embeddings alternative (option a), for comparison ──
# # Lets Databricks embed the `summary` text so you could query with query_text.
# # We do not use it because it re-embeds text we already embedded.
# #
# # vsc.create_delta_sync_index_and_wait(
# #     endpoint_name=VS_ENDPOINT_NAME, index_name=VS_INDEX_NAME,
# #     source_table_name=VS_SOURCE_TABLE, pipeline_type="TRIGGERED",
# #     primary_key="article_id", embedding_source_column="summary",
# #     embedding_model_endpoint_name="databricks-bge-large-en")  # pin the SAME model

# COMMAND ----------

# MAGIC %md
# MAGIC # ══════════════ EVERYONE (read-only from here) ══════════════

# COMMAND ----------

# DBTITLE 1,Connect to the shared index
# Read-only: every attendee runs this and shares the one index. If it is missing, the
# organiser has not run the (commented-out) provisioning cells above yet.
try:
    index = vsc.get_index(VS_ENDPOINT_NAME, VS_INDEX_NAME)
    status = index.describe().get("status", {})
    print(f"Connected to shared index {VS_INDEX_NAME}")
    print(f"  detailed_state: {status.get('detailed_state')}")
    print(f"  indexed rows:   {status.get('indexed_row_count')}")
except Exception as e:
    raise RuntimeError(
        f"Shared index '{VS_INDEX_NAME}' not found on endpoint '{VS_ENDPOINT_NAME}'. "
        "Ask your workshop organiser to run the provisioning cells above once "
        "(they are commented out by default)."
    ) from e

RESULT_COLS = ["article_id", "topic", "summary", "cluster", "tension_label"]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Use case 1 — "here is a flagged post; find related discussions"
# MAGIC
# MAGIC The tutorial flagged each post with a `tension_label` (escalating / de-escalating /
# MAGIC neutral). A natural analyst question is: *given one post that was flagged as
# MAGIC escalating, what other discussions look like it?* We take that post's **stored**
# MAGIC embedding and ask the index for its nearest neighbours — no new embedding call is
# MAGIC needed. We ask for one extra result and drop the post itself (a post is always its
# MAGIC own nearest neighbour).

# COMMAND ----------

# DBTITLE 1,Post-to-post similarity from a flagged post
# Read a seed post's stored vector from the shared source table (a read-only SELECT —
# attendees need only SELECT on this table, they never provision anything). We start from
# a post the tutorial flagged as escalating.
seed = (
    spark.table(VS_SOURCE_TABLE)
    .filter(F.col("tension_label") == "escalating")
    .select("article_id", "topic", "summary", "embedding", "tension_label")
    .limit(1)
    .collect()[0]
)
print(f"Seed post {seed['article_id']} (topic={seed['topic']}, tension={seed['tension_label']}):")
print(f"  {seed['summary']}\n")

res = index.similarity_search(
    query_vector=[float(x) for x in seed["embedding"]],
    columns=RESULT_COLS,
    num_results=6,  # 5 neighbours + the seed itself
)
seed_id = seed["article_id"]

print(f"{'article_id':<14}{'topic':<18}{'tension':<15}{'score':<8}summary")
for r in res["result"]["data_array"]:
    aid, topic, summary, cluster, tension, score = r
    if aid == seed_id:
        continue  # drop the seed's self-match
    print(f"{aid:<14}{topic:<18}{str(tension):<15}{score:<8.3f}{(summary or '')[:60]}")

# COMMAND ----------

# MAGIC %md
# MAGIC The neighbours are **topically** similar, because the embeddings encode the *theme*
# MAGIC of each post. Now read across to their `tension_label`: where related posts also carry
# MAGIC an escalating tone, you are looking at a cluster of heated discussion on a single theme
# MAGIC — exactly the shape of a discussion-monitoring or early-warning question. Retrieval
# MAGIC gives a *ranked, per-item* answer with scores, where the tutorial's KMeans gave a
# MAGIC single hard label; that finer granularity is the point.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Use case 2 — free-text semantic query, scoped by a filter
# MAGIC
# MAGIC An analyst rarely starts from an existing row — usually they have a *theme in mind* and
# MAGIC want the posts closest to it. That query text is **not** in the table, so we embed it
# MAGIC ourselves. Because our index is *self-managed*, it does not embed text for us — so we
# MAGIC must use the **same** `databricks-bge-large-en` model the documents were embedded with.
# MAGIC A different model would place the query in an incompatible space. This is **Principle 5**
# MAGIC made concrete. We also apply a **filter** — governed, scoped retrieval.

# COMMAND ----------

# DBTITLE 1,Embed a free-text query with the same model, then search
def embed_query(text: str) -> list:
    """Embed a single query string with the SAME endpoint used for the documents."""
    row = spark.sql(
        "SELECT ai_query(:endpoint, :q) AS v",
        args={"endpoint": EMBEDDING_ENDPOINT, "q": text},
    ).collect()[0]
    return [float(x) for x in row["v"]]

query_text = "arguments about the cost and politics of the space programme"
qv = embed_query(query_text)
assert len(qv) == EMBEDDING_DIM

res = index.similarity_search(
    query_vector=qv,
    columns=RESULT_COLS,
    num_results=5,
    filters={"topic": "sci.space"},  # STANDARD endpoint uses dict-format filters
)

print(f"Query: {query_text!r}  (filtered to sci.space)\n")
print(f"{'article_id':<14}{'tension':<15}{'score':<8}summary")
for r in res["result"]["data_array"]:
    aid, topic, summary, cluster, tension, score = r
    print(f"{aid:<14}{str(tension):<15}{score:<8.3f}{(summary or '')[:70]}")

# COMMAND ----------

# MAGIC %md
# MAGIC **Principle 1 (know the limits):** the scores are *relative* similarities, not
# MAGIC calibrated probabilities, and the embeddings were built from lossy one-sentence
# MAGIC summaries — which capture a post's *theme* far better than its *tone*. So retrieval
# MAGIC finds you posts on a subject; read the `tension_label` column (or filter on it, e.g.
# MAGIC `filters={"tension_label": "escalating"}`) to scope to the heated ones. A high rank
# MAGIC means "closest available", not "correct".

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
# MAGIC # ══════════════ TEARDOWN (organiser only, after the session) ══════════════
# MAGIC
# MAGIC The cell below is **commented out** and is intended only for the organiser to run
# MAGIC **after** the session, so the always-on endpoint stops billing (Principles 2 and 5).
# MAGIC It deletes the *shared* index and endpoint that everyone else is using, so it must
# MAGIC never be run during the workshop.

# COMMAND ----------

# DBTITLE 1,Reference: delete the shared index and endpoint (organiser, post-workshop only)
# Uncomment and run ONLY after the session, once nobody else needs the shared index.
#
# vsc.delete_index(index_name=VS_INDEX_NAME)
# print(f"Deleted index {VS_INDEX_NAME}.")
# vsc.delete_endpoint(name=VS_ENDPOINT_NAME)
# print(f"Deleted endpoint {VS_ENDPOINT_NAME}.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Now it is your turn!
# MAGIC
# MAGIC You have seen how the embeddings from the tutorial can be reused for semantic
# MAGIC retrieval — similar-item lookup, free-text search, near-duplicate detection, and
# MAGIC few-shot feature engineering — all against a single shared, governed index.
# MAGIC
# MAGIC For your own project, an organiser can point a new index at your team's corpus (for
# MAGIC example the parsed CQC inspection reports) and ask: *which reports are most similar to
# MAGIC a known problem case?* Remember to embed with a single, consistent model, share one
# MAGIC index rather than one-per-person, keep a human in the loop on anything you flag, and
# MAGIC tear the endpoint down when the work is done.
