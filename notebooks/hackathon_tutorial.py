# Databricks notebook source
# MAGIC %md
# MAGIC # Semantic Features in Statistical Workflows — Tutorial
# MAGIC
# MAGIC **Public Sector Responsible AI Hackathon | Databricks × RSS AI Task Force × Manuka | 9 July 2026**
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## What this notebook is
# MAGIC
# MAGIC This notebook is a **worked example** of the first two stages of a possible pipeline for statistical modelling wth semantic features:  data ingestion and feature engineering. The other stages of the pipeline are more standard statistics/data science project stages where the semantic features are taken as inputs.
# MAGIC We will take the time to work through this notebook on the first day of the hackathon,
# MAGIC and you may subsequently use it as a reference during team working time.
# MAGIC
# MAGIC The dataset used here is the **20 Newsgroups** corpus — a classic collection
# MAGIC of ~18,000 news articles across 20 topic categories, pre-loaded on every
# MAGIC Databricks workspace at `/databricks-datasets/news20.binary/`. It has no
# MAGIC connection to the hackathon use cases, but its structure — unstructured text
# MAGIC that needs to be turned into something a model can act on — is exactly the
# MAGIC same problem you will face on the day.
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Where the AI Principles fit into the Pipeline

# COMMAND ----------

# MAGIC %md
# MAGIC **The UK Government AI Principles**
# MAGIC
# MAGIC We have already had an introduction to these in the presentations, but as a reminder, here are the ten principles for responsible AI use in public sector organisations.  
# MAGIC
# MAGIC - P1:  **You know what AI is and what its limitations are**:  AI is not a single technology, and AI technologies are evolving rapidly.  While it is useful to understand in theory how AI technologies work, the best way to really understand the strengths ad limitations of AI tools is to experiment and evaluate the suitability of tools in low-risk settings that model the sort of work you wish to use it for before embedding them in your work.  This hackathon is just such a setting, so by participating, you are already making progress towards the first principle.  More broadly, we encourage you to keep experimenting as the technology evolves.  Hackathons can be a fun way to do this, so we hope you will enjoy this one as well as learn from it!
# MAGIC
# MAGIC - P2: **You use AI lawfully, ethically and responsibly**:  The basic requirements around lawful, ethical and responsible use of _data_ hold in all statistical workflows, and AI workflows are no different.  The main differences coming from AI tools as opposed to more conventional statistical and data tooling are 
# MAGIC     1. When we use an LLM, we do not control the underpinning model we are using, and the models commonly used are regularly updated by the companies that provide them.  This means that more needs to be done to evaluate the performance of the model against a clear set of suitability criteria and on a representative test set of cases both when initially undertaking work, and also in ongoing monitoring of tools built on top of them.    
# MAGIC     2. LLMs are both financially and environmentally costly to use.  So when designing workflows using calls, work to optimise your workflows to reduce these costs, through model selection, prompt optimisation and efficient architectures.  Databricks has several models available, and you will have the opportunity to compare how these perform to identify the most efficient one to use for your use case.
# MAGIC
# MAGIC - P3: **You know how to use AI securely**:  Because data flows to models and responses back from models, it is important to to work with LLM models and AI tools built on these that have been approved by your organisation for the use case and data you have in mind.  Cybersecurity is a complex field of its own, and it is not possible to undertake adequate security evaluation as an individual statistician.  In this hackathon, as we are not working within the service boundary of your organisation, it is consequently important only to use publicly available data.  
# MAGIC
# MAGIC - P4:  **You have meaningful human control at the right stages** In any project, it is essential to identify the risk level of the use case you are building towards, and understand any legal or departmental governance standards around which decisions or validations must sit with human experts and which may be taken by a system.  In this tutorial, the use case is low-risk, and consequently, our human control is aimed at sense checking and ensuring explainability.
# MAGIC
# MAGIC - P5: **You understand how to manage the full AI life cycle** For any data project, it is necessary to monitor suitability and performance of a tool as the data it is built on may evolve.  Given point 1 of principle 2, when the tool is build using a foundation LLM, it is additionally necessary to monitor the suitability and performance of the underpinning model throughout the project life cycle.
# MAGIC
# MAGIC - P6: **You use the right tool for the job** Different AI and statistical tools are good at different things.  In this hackathon, you will combine an LLM tool to extract semantic features from unstructured text--a job that LLMs are good at.  You will then construct a model using these features and conventional statistical or ml techniques--something LLMs are NOT good at, and where other techniques are both more effective and less costly.
# MAGIC
# MAGIC - P7: **You are open and collaborative** In this hackathon, you will get to know colleagues from several different public sector agencies.  We hope you will take this opportunity to build a community for knowledge sharing with other public sector statisticians.  Additionally, we will show how to ensure that the datasets you create for analysis are discoverable and understandable by other users.
# MAGIC
# MAGIC - P8: **You work with commercial colleagues from the start** Manuka has talked about how to identify suitable use cases for AI models.  A tool may perform extremely well according to a set of evaluation metrics, but still not meet the requirements of the end users.  The best tool is the one that performs best in the setting in which it will be used.  So think carefully about how the tool you will develop in the hackathon will deliver value, and what it needs to provide in order for the intended users to get benefit from it.
# MAGIC
# MAGIC - P9: **You have the skills and expertise needed to implement and use AI solutions**  The skills and experience you develop in this hackathon will complement the statistical skills you already have, and will also give you ideas of new tools to try out and sample data to try them on in the future.
# MAGIC
# MAGIC - P10: **You use these principles alongside your organisation's policies and have the right assurance in place** In this hackathon, we encourage you to use the policies in your organsation to determine the appropriate risk level and required assurance for your project.  If your organisation has not yet provided assurance guidelines for AI projects, we encourage you to consider the framework set out in the EU AI Act.
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC **A seven-stage pipeline for statistical modelling with LLM-derived semantic features**
# MAGIC
# MAGIC | Stage | What happens | Playbook principles |
# MAGIC |-------|-------------|---------------------|
# MAGIC | 1 — Ingest | Load, examine and clean raw data and save as a Delta Table | P2, P4, P7, P10 |
# MAGIC | 2 — Feature Engineering | Generate quantitative and categorical features to feed downstream models| P1, P6, P9 |
# MAGIC | 3 — Human Review | Route feature outputs to a domain expert for sense checking before use | P4, P8 |
# MAGIC | 4 — Model | Train and evaluate model; log with MLflow; explain with SHAP or other tools | P1, P5, P7 |
# MAGIC | 6 — Surface Outputs for Users | Create a dashboard or app to permit business users to engage with the outputs of your model | P6, P8  |
# MAGIC | 7 — Govern | Audit trail, lineage, monitoring | P5, P10 |
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration and Package Loading

# COMMAND ----------

import re
import os
from datetime import datetime, UTC
from typing import Optional
import json

import pandas as pd
import numpy as np

from pyspark.sql import functions as F, types as T
from pyspark.sql.window import Window
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, DoubleType, ArrayType, FloatType, BooleanType
)

from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.feature import Normalizer
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation

import matplotlib.pyplot as plt
import seaborn as sns


# ── Unity Catalog target ──────────────────────────────────────────────────────
UC_CATALOG = "hackathon"
UC_SCHEMA = "default"
USERNAME = "tutorial" # Provide a unique tag to label your data assets

# ── Table names ───────────────────────────────────────────────────────────────
PROCESSED_TABLE = f"{UC_CATALOG}.{UC_SCHEMA}.tutorial_sci_med_posts_{USERNAME}"
SUMMARISED_TABLE = f"{UC_CATALOG}.{UC_SCHEMA}.tutorial_sci_med_summarised_{USERNAME}"
VECTORISED_TABLE = f"{UC_CATALOG}.{UC_SCHEMA}.tutorial_sci_med_vectorised_{USERNAME}"
CLASSIFIED_TABLE = f"{UC_CATALOG}.{UC_SCHEMA}.tutorial_sci_med_classified_{USERNAME}"
FEATURES_TABLE = f"{UC_CATALOG}.{UC_SCHEMA}.tutorial_sci_med_features_{USERNAME}"
AUTO_POST_FEATURES_TABLE = f"{UC_CATALOG}.{UC_SCHEMA}.tutorial_sci_med_auto_post_features_{USERNAME}"
REVIEW_TABLE = f"{UC_CATALOG}.{UC_SCHEMA}.tutorial_sci_med_for_review_{USERNAME}"

# ── Sample size ───────────────────────────────────────────────────────────────
# Keep at 200 for the tutorial to limit AI Function token consumption.
# Increase for your use case work.
SAMPLE_SIZE = 200

# ── MLflow experiment ─────────────────────────────────────────────────────────
EXPERIMENT_NAME = f"/Users/{USERNAME}/hackathon_tutorial"

print(f"UC target: {UC_CATALOG}.{UC_SCHEMA}")
print(f"Sample size: {SAMPLE_SIZE} articles")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Ingest through an ETL pipeline

# COMMAND ----------

# MAGIC %md
# MAGIC ### What we are doing and why
# MAGIC
# MAGIC The first stage of every pipeline is getting your raw data into **Delta Lake**
# MAGIC under Unity Catalog governance, having embedded data protection in the assets and having provided metadata. This matters for four reasons that map
# MAGIC directly to the AI Playbook:
# MAGIC
# MAGIC - **Principle 2:** You need to know the data you are working with and determine if there are any special legal considerations for working with it.  Is the data sensitive?  Is there personal data?  What limitations or biases may the data have that will need to be balanced for in modelling or limit the situations in which the tool can suitably be used.
# MAGIC - **Principle 4:** Starting out with a manual investigation of the data provides human control at the very start of the process, and permits first considerations of tooling costs.
# MAGIC - **Principle 7:** By providing metadata when saving tables, it is possible for other users to discover and understand the datasets you create and how they were generated.
# MAGIC - **Principle 10:** Unity Catalog records who created each table, when, and from what source and tracks the lineage of data assets.  This is the first stage in assurance and audit of AI systems.
# MAGIC
# MAGIC ### About the 20 Newsgroups dataset
# MAGIC
# MAGIC The 20 Newsgroups dataset contains articles from 20 newsgroups with broad categories, indicated by the field "topic".  We will work with the topic sci.space, but you can explore some of the other topics if you like at the end of the tutorial.
# MAGIC Each file is a raw email/newsgroup post, and the raw text has several pieces of information in headers.  So the first thing we need to do is extract relevant header data into separate fields both so that we can examine that data separately and potentially use these fields in our models, but also so that the main messages do not have headers, which may otherwise trigger spurious labels.
# MAGIC
# MAGIC **This is the same operation you will perform in your use case** — the CQC
# MAGIC corpus, for example, arrives as JSON with a `report_text` field. It may make sense to split the text into sections before analysing, etc.  

# COMMAND ----------

# DBTITLE 1,See available newsgroups
spark.read.parquet("/databricks-datasets/news20.binary/data-001/training").groupBy('topic').count().display()

# COMMAND ----------

# DBTITLE 1,Examine raw data
raw_df = spark.read.parquet("/databricks-datasets/news20.binary/data-001/training").filter(F.col('topic').isin(['talk.religion.misc','rec.autos','sci.space'])).drop('label')
display(raw_df.limit(20))

# COMMAND ----------

# DBTITLE 1,helper function to normalise text
KNOWN_HEADERS = [
    "From", "Subject", "Organization", "Lines", "Distribution",
    "Reply-To", "NNTP-Posting-Host", "Nntp-Posting-Host", "Article-I.D.",
    "X-Added", "Original-Sender", "Newsgroups", "Path", "Message-ID",
    "References", "Date", "Sender", "Followup-To", "Keywords",
    "Summary", "Expires", "X-Newsreader", "In-Reply-To", "Approved","News-Software",
    "Supersedes", "X-Last-Updated","X-X-From",
]
# Longest first so e.g. "NNTP-Posting-Host" matches before any shorter prefix could
_sorted_headers = sorted(set(KNOWN_HEADERS), key=len, reverse=True)
HEADER_KEY_RE = re.compile(
    r'(?:^|(?<=\s))(' + '|'.join(re.escape(h) for h in _sorted_headers) + r'):\s'
)

HEADER_SCHEMA = StructType([
    StructField("from_email", StringType(), True),
    StructField("from_name", StringType(), True),
    StructField("subject", StringType(), True),
    StructField("organization", StringType(), True),
    StructField("lines_header", StringType(), True),
    StructField("distribution", StringType(), True),
    StructField("reply_to", StringType(), True),
    StructField("nntp_posting_host", StringType(), True),
    StructField("body", StringType(), True),
])

FROM_RE = re.compile(r'^(\S+)(?:\s*\((.*)\))?\s*$')

def parse_post(content):
    if content is None:
        return (None, None, None, None, None, None, None, None, "")

    # Body boundary: the flattened blank line shows up as 2+ spaces
    parts = re.split(r'\s{2,}', content, maxsplit=1)
    header_block = parts[0]
    body = parts[1].strip() if len(parts) > 1 else ""

    matches = list(HEADER_KEY_RE.finditer(header_block))
    headers = {}
    for i, m in enumerate(matches):
        key = m.group(1).strip().lower()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(header_block)
        headers[key] = header_block[start:end].strip()

    from_raw = headers.get("from")
    from_email, from_name = None, None
    if from_raw:
        m = FROM_RE.match(from_raw)
        from_email = m.group(1) if m else from_raw
        from_name = m.group(2) if m else None

    return (
        from_email, from_name,
        headers.get("subject"),
        headers.get("organization"),
        headers.get("lines"),
        headers.get("distribution"),
        headers.get("reply-to"),
        headers.get("nntp-posting-host"),
        body,
    )

parse_post_udf = F.udf(parse_post, HEADER_SCHEMA)

# COMMAND ----------

# DBTITLE 1,load dataset, normalise, add fields
processed_df = (
    raw_df
    .withColumnRenamed("id", "article_id")
    .withColumn("parsed", parse_post_udf(F.col("text")))
    
    .select(
        "article_id",
        "topic",
        F.col("parsed.subject").alias("subject"),
        F.col("parsed.organization").alias("organization"),
        F.col("parsed.lines_header").alias("lines_header"),
        F.col("parsed.distribution").alias("distribution"),
        F.col("parsed.body").alias("text"),
    )
    .withColumn("char_count", F.length("text"))
    .withColumn("word_count", F.size(F.split(F.trim(F.col("text")), r"\s+")))
    .filter(F.col("word_count")>30)
)

print(f"Rows after cleaning: {processed_df.count()}")
display(processed_df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cap the working set before paying for LLM calls
# MAGIC
# MAGIC Every downstream step in this notebook — `ai_mask`, `ai_summarize`, the embedding
# MAGIC call and the two `ai_query` classifications — sends text to a model and is billed by
# MAGIC the token. That means **cost scales directly with the number of rows.** Before running
# MAGIC any of them, we cap the dataset to `SAMPLE_SIZE` rows (set in the config cell at the
# MAGIC top). Work out your approach and validate quality on this small, cheap sample first;
# MAGIC only scale up once you are happy. This is Principle 2 (use AI responsibly and keep
# MAGIC costs down) and Principle 4 (meaningful human control early) in practice.

# COMMAND ----------

# DBTITLE 1,Cap to SAMPLE_SIZE before any paid LLM calls
# limit() gives an exact, cheap cap; ordering first makes the sample reproducible across runs.
full_row_count = processed_df.count()
processed_df = processed_df.orderBy("article_id").limit(SAMPLE_SIZE)
print(f"Working on {processed_df.count()} of {full_row_count} rows (SAMPLE_SIZE={SAMPLE_SIZE}).")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Getting a feel for tooling costs
# MAGIC
# MAGIC Before spending anything, get a feel for the shape of your data. Next to where it says
# MAGIC **Table** in a display above, click the **+** icon and select **Data Profile**. This
# MAGIC gives summaries of missingness, most common values, and average field lengths. You will
# MAGIC notice, for example, that `distribution` is ~80% missing (so unlikely to be a useful
# MAGIC feature), while `topic`, `subject` and `text` are complete.
# MAGIC
# MAGIC A common rule of thumb for English text is **~4 characters per token**. The next cell
# MAGIC turns that into a rough-order-of-magnitude (ROM) cost estimate. The key points it makes:
# MAGIC
# MAGIC - Cost is driven by **tokens × passes × rate**. This notebook makes *several* LLM passes
# MAGIC   over the data (two `ai_mask` calls, `ai_summarize`, an embedding, and `ai_query` for
# MAGIC   tension and car extraction), so you multiply, not add once.
# MAGIC - **Output tokens** cost money too, and are often priced higher than input tokens.
# MAGIC - **Prompt caching** reduces the input cost of the shared prompt prefix. Databricks AI
# MAGIC   Functions cache the common prefix automatically, and some served models bill cached
# MAGIC   reads at a reduced rate — so the ROM figure is an upper bound on the input side.

# COMMAND ----------

# DBTITLE 1,Rough-order-of-magnitude (ROM) cost estimate
# Rule of thumb: ~4 characters per token for English text.
CHARS_PER_TOKEN = 4

# $ per 1M tokens. THESE ARE PLACEHOLDERS — set them from the current Databricks pricing
# page for your chosen model, or derive them from system.billing.list_prices (next section).
# Foundation Model APIs are billed in DBUs; convert the DBU price to $ via list_prices.
RATE_PER_M_INPUT = 0.50    # $/1M input tokens
RATE_PER_M_OUTPUT = 1.50   # $/1M output tokens

avg_text_chars = processed_df.agg(F.avg("char_count")).collect()[0][0] or 0
avg_subject_chars = processed_df.agg(F.avg(F.length("subject"))).collect()[0][0] or 0
sample_rows = processed_df.count()

text_tokens_in = avg_text_chars / CHARS_PER_TOKEN
subject_tokens_in = avg_subject_chars / CHARS_PER_TOKEN

# Every LLM pass this notebook makes, with rough input/output tokens PER ROW.
# Output sizes are estimates — a summary or JSON blob is far smaller than the input, but not free.
passes = [
    # name,                    input_tokens_per_row,     output_tokens_per_row
    ("ai_mask (text)",         text_tokens_in,           text_tokens_in),   # returns masked text ~ same size
    ("ai_mask (subject)",      subject_tokens_in,        subject_tokens_in),
    ("ai_summarize",           text_tokens_in,           60),               # ~50-word summary
    ("embedding (summary)",    60,                       0),                # embeddings billed on input only
    ("ai_query tension",       text_tokens_in + 120,     40),               # + prompt, small JSON out
    ("ai_query car (subset)",  text_tokens_in + 300,     80),               # rec.autos only, so over-counts here
]

print(f"{'pass':<24}{'in tok/row':>12}{'out tok/row':>13}")
total_in = total_out = 0.0
for name, tin, tout in passes:
    total_in += tin
    total_out += tout
    print(f"{name:<24}{tin:>12,.0f}{tout:>13,.0f}")

def rom_cost(rows):
    c_in = rows * total_in / 1e6 * RATE_PER_M_INPUT
    c_out = rows * total_out / 1e6 * RATE_PER_M_OUTPUT
    return c_in + c_out

print(f"\nPer row: {total_in:,.0f} input + {total_out:,.0f} output tokens across all passes")
print(f"Est. cost on sample     ({sample_rows} rows): ${rom_cost(sample_rows):,.4f}")
print(f"Est. cost on full corpus ({full_row_count} rows): ${rom_cost(full_row_count):,.2f}")
print("\nROM figures only. Prompt caching and cached-token discounts reduce the input side.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Measuring what a run *actually* cost, with system tables
# MAGIC
# MAGIC The estimate above is a planning tool. To see what a run really cost, query the billing
# MAGIC **system tables** after it finishes. `system.billing.usage` records every billable event;
# MAGIC `system.billing.list_prices` gives the price per unit so you can turn DBUs into dollars.
# MAGIC
# MAGIC Foundation Model API / Model Serving and AI Functions usage appear under
# MAGIC `billing_origin_product` values `MODEL_SERVING` and `AI_FUNCTIONS`; the specific function
# MAGIC is in `product_features.ai_functions.ai_function` and the endpoint in
# MAGIC `usage_metadata.endpoint_name`. **Caveat:** these records land with a few hours' latency,
# MAGIC so run the next cell later, not immediately after the cells above. This is the audit and
# MAGIC lifecycle-monitoring evidence called for by Principles 5 and 10.

# COMMAND ----------

# DBTITLE 1,Actual $ cost of recent AI usage (run a few hours after your test run)
spark.sql("""
WITH ai_usage AS (
  SELECT
    u.sku_name,
    u.billing_origin_product,
    u.usage_metadata.endpoint_name                 AS endpoint_name,
    u.product_features.ai_functions.ai_function    AS ai_function,
    u.usage_unit,
    SUM(u.usage_quantity)                          AS usage_quantity
  FROM system.billing.usage u
  WHERE u.billing_origin_product IN ('MODEL_SERVING', 'AI_FUNCTIONS')
    AND u.usage_date >= current_date() - INTERVAL 2 DAYS
    -- Optional: narrow to this notebook only
    -- AND u.usage_metadata.notebook_path = '<your notebook path>'
  GROUP BY ALL
)
SELECT
  a.billing_origin_product,
  a.endpoint_name,
  a.ai_function,
  a.sku_name,
  a.usage_unit,
  a.usage_quantity,
  p.pricing.effective_list.default                                 AS unit_price_usd,
  ROUND(a.usage_quantity * p.pricing.effective_list.default, 4)    AS est_cost_usd
FROM ai_usage a
LEFT JOIN system.billing.list_prices p
  ON a.sku_name = p.sku_name
  AND p.price_end_time IS NULL          -- current price
ORDER BY est_cost_usd DESC
""").display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### GDPR Considerations

# COMMAND ----------

# MAGIC %md
# MAGIC Notice that this dataset contains personal data, in particular PII data, in the form of names, email addresses and institutions, as well as text written by identifiable natural persons in the 1990s.  Because the data is in newsgroups related to religion, it is also possible that it will contain sensitive data related to religion or beliefs of the poster or another identifiable individual.  This means that the data is protected by GDPR.  Therefore:
# MAGIC
# MAGIC - **We should apply appropriate anonymisation/pseudonimisation prior to processing according to the standards set by our organisation**:  in the cleaning step, we therefore need code that will allow us to strip names and email addresses from the data.
# MAGIC - **We need a legal basis for processing the data**.  Because this is a publicly available dataset that has been widely used for research, and because we are using this for training and the outputs of our work will not feed into any decision making process, we can take Legitimate Interests as our legal basis for processing.
# MAGIC
# MAGIC Removal of email addresses and phone numbers is fairly straightforward with regex, but it is much trickier to identify names to redact.  This is a nice use for ai tooling, such as the ai_mask function available in Databricks.  Similar functions can be created outside of Databricks by having an LLM identify all names in a text field, then using a replace to redact.  Note this isn't perfect, so where risk level on the project is high, additional methods should be used.
# MAGIC

# COMMAND ----------

# DBTITLE 1,Remove personal identifiers using ai_mask and regex
BARE_HOST_EMAIL_PATTERN = r'\b[\w.+-]+@[\w-]+\b'
EMAIL_PATTERN = r'[\w.+-]+@[\w-]+\.[\w.-]+'
UUCP_PATTERN = r'\bUUCP:\s*\S+'
PHONE_PATTERN = r'\+?\d[\d\-\s]{6,}\d'


def redact_structured(df, text_col="text"):
    return (
        df
        .withColumn(text_col, F.regexp_replace(F.col(text_col), EMAIL_PATTERN, "[EMAIL]"))
        .withColumn(text_col, F.regexp_replace(F.col(text_col), BARE_HOST_EMAIL_PATTERN, "[EMAIL]"))
        .withColumn(text_col, F.regexp_replace(F.col(text_col), UUCP_PATTERN, "[UUCP_ID]"))
        .withColumn(text_col, F.regexp_replace(F.col(text_col), PHONE_PATTERN, "[PHONE]"))
    )

redacted_df = (
    processed_df
    .withColumn("text", F.expr("ai_mask(text,array('person', 'email'))"))
    .withColumn("subject",F.expr("ai_mask(subject,array('person', 'email'))"))
)
redacted_df = redact_structured(redacted_df,text_col="text")
redacted_df = redact_structured(redacted_df,text_col="subject")


display(redacted_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Masking strategies: who sees the raw data, and when?
# MAGIC
# MAGIC We redacted PII **in memory, before the first table write** — so raw personal data never
# MAGIC lands in a governed table. That is one valid strategy, but not the only one. Which you
# MAGIC choose depends on your risk level and on who genuinely needs access to the raw data.
# MAGIC
# MAGIC | Strategy | How it works | Trade-off | Who can see raw |
# MAGIC |---|---|---|---|
# MAGIC | **A. Redact at source** (this notebook) | Mask before writing any table; raw never persists | Simplest and safest, but you cannot re-derive if masking was too aggressive | Only the ingest job / its owner |
# MAGIC | **B. Tiered medallion** | Bronze = raw (locked down), Silver = masked, Gold = aggregated | Raw stays available for re-processing, but Bronze must be locked down hard | A small DE / DPO group, on Bronze only |
# MAGIC | **C. Dynamic masking** | One table; Unity Catalog **column masks** + **row filters** reveal raw only to a privileged group | Flexible, single source of truth, but needs careful group management | Members of a privileged UC group |
# MAGIC
# MAGIC **Persona guidance.** Raw PII should reach only a Data Protection / data-engineering group.
# MAGIC Model builders and analysts should work from the masked Silver layer; business users should
# MAGIC only ever see aggregated Gold outputs or a dashboard. Note that `ai_mask` **itself** sends
# MAGIC raw text to the model endpoint, so the masking step must run under the privileged persona —
# MAGIC masking is a way to avoid *persisting or exposing* raw data, not a way to avoid *processing*
# MAGIC it. (Principles 2, 3 and 4.)

# COMMAND ----------

# DBTITLE 1,Strategy C illustration: a Unity Catalog column mask (read-only, not executed)
# This shows how you would expose one table to two personas. It needs a real UC group,
# so it is left here as a string to read rather than run.
COLUMN_MASK_EXAMPLE = """
-- 1. A mask function: the privileged group sees the value, everyone else a placeholder.
CREATE OR REPLACE FUNCTION hackathon.default.mask_pii(val STRING)
RETURN CASE
  WHEN is_account_group_member('data_protection_officers') THEN val
  ELSE '[REDACTED]'
END;

-- 2. Apply it to a column holding raw text on the (locked-down) raw table.
ALTER TABLE hackathon.default.raw_posts
  ALTER COLUMN text SET MASK hackathon.default.mask_pii;

-- 3. Optionally, a ROW FILTER to hide whole rows from non-privileged users:
--    CREATE FUNCTION ... RETURN is_account_group_member(...);
--    ALTER TABLE ... SET ROW FILTER ...;
"""
print(COLUMN_MASK_EXAMPLE)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Add metadata

# COMMAND ----------

# DBTITLE 1,Define table schema with field level metadata
def field(name, dtype, comment):
    """Shorthand for a StructField carrying a UC column comment."""
    return StructField(name, dtype, True, {"comment": comment})

raw_schema = StructType([
    field("article_id",        StringType(),  "Unique identifier from the source dataset"),
    field("topic",          StringType(),  "Newsgroup topic label (filtered to talk.religion.misc, rec.autos, sci.space)"),
    field("subject",           StringType(),  "Subject line of the original post with emails and names redacted"),
    field("organization",      StringType(),  "Organization header, self-reported by the poster"),
    field("lines_header",      StringType(),  "Line count as given by the header"),
    field("distribution",      StringType(),  "Distribution header, where present"),
    field("text",              StringType(),  "Article body with headers stripped and emails and names redacted"),
    field("char_count",        IntegerType(), "Character count of the stripped body text"),
    field("word_count",        IntegerType(), "Word count of the stripped body text"),
])

# Apply the schema to your existing DataFrame (column order/names must match)
processed_df_with_comments = spark.createDataFrame(redacted_df.rdd, schema=raw_schema)

(
    processed_df_with_comments.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(PROCESSED_TABLE)
)

# COMMAND ----------

# DBTITLE 1,Define table metadata including data provenance
spark.sql(f"""
    COMMENT ON TABLE {PROCESSED_TABLE} IS
    'Posts from three newsgroups (talk.religion.misc, rec.autos, sci.space) from the news20 public training dataset of newsgroup posts, available at https://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/news20.html.  Names and email addresses have been redacted using the ai_mask function and regex.'
""")

# COMMAND ----------

# DBTITLE 1,Define table properties
# ── Provenance metadata ───────────────────────────────────────────────────────
PROCESSED_BY = spark.sql("SELECT current_user()").collect()[0][0]

try:
    NOTEBOOK_PATH = (
        dbutils.notebook.entry_point.getDbutils().notebook()
        .getContext().notebookPath().get()
    )
except Exception as exc:
    NOTEBOOK_PATH = None
    print("Could not determine notebook path: %s", exc)

spark.sql(f"""
    ALTER TABLE {PROCESSED_TABLE} SET TBLPROPERTIES (
        'notebook_path' = '{NOTEBOOK_PATH}',
        'processed_by' = '{PROCESSED_BY}',
        'last_ingested_at' = '{datetime.now(UTC).isoformat()}'
    )
""")

# COMMAND ----------

# DBTITLE 1,read back in from table
text_dataset = spark.table(PROCESSED_TABLE)

# COMMAND ----------

# MAGIC %md
# MAGIC Go to the Catalog in the menu column to the left of the screen and find the dataset you just saved.  You can look through the various tabs to find a range of information about the dataset, its lineage, when it was created, by whom, etc. as well as the metadata you have provided above.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Engineering Semantic Features

# COMMAND ----------

# MAGIC %md
# MAGIC ### What we are doing and why
# MAGIC
# MAGIC Raw text is not useful for a statistical model on its own. We need to
# MAGIC **extract structured semantic features** from it — properties that can be
# MAGIC represented as columns in a table and used downstream.  This is one of the most powerful capabilities that LLMs provide, and opens up many new approaches for working statistically with unstructured text data.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Principles we embed at this stage
# MAGIC
# MAGIC **P1:  You know what AI is and what its limitations are**
# MAGIC Before running these functions, it is worth being explicit about what they
# MAGIC cannot do reliably:
# MAGIC
# MAGIC - `ai_summarize` can drop important information. A summary is lossy by
# MAGIC   design. Do not use summaries as the sole input to downstream decisions.
# MAGIC
# MAGIC - clustering on text embeddings may not create categories that are meaningful in your use-case.
# MAGIC This is why subject matter expert review is critical at this stage.
# MAGIC   
# MAGIC - `ai_query` is not adapted automatically to the context of your data and use-case, so it is useful to craft your prompt to include contextual clues and test its performance on a representative sample of data.
# MAGIC
# MAGIC These limitations are not reasons to avoid LLM extraction — they are
# MAGIC reasons to test, validate, and build human review into your pipeline.
# MAGIC
# MAGIC **P6: You use the right tool for the job**
# MAGIC Databricks provides several SQL AI Functions for working with free text. You can read about them here:
# MAGIC https://docs.databricks.com/aws/en/large-language-models/ai-functions
# MAGIC
# MAGIC The dataset we are using provides text already in a field in a table.  But some of the datasets that have been provided for the hackathon were ingested originally as pdf files into a volume. You can see them in the catalog to the left (click on the icon of a triangle over a circle and a square under the folder icon in the Workspace panel).  Go to My organisation -> dbacademy -> hackathon -> shared_datasets -> Volumes -> cqc_safe_inadequate_reports to see these, one per folder with the cqc location id as a folder name.  When starting with raw documents, the ai_parse_document function can be used to initially extract text and formatting information from the files, which can then be passed to other ai functions.
# MAGIC
# MAGIC In addition to the built-in databricks functions, it is also possible to call models directly.  You can experiment with this in the Playground, towards the bottom of the far left menu panel.  In this tutorial, we will use a text embedding model to convert text summaries into vectors we can use to examine semantic similarity.
# MAGIC
# MAGIC We will use the following workflow to classify the subjects of the posts:
# MAGIC 1. Summarise the post using ai_summarize function
# MAGIC 2. Embed the summaries using a text embedding
# MAGIC 3. Look for semantic clusters
# MAGIC 4. Expert feedback
# MAGIC
# MAGIC Then we will extract some additional structured fields using ai_query on the original text field, which model two different types of tasks that can be undertaken with this function:
# MAGIC 1. Across the full dataset, we will identify the tone of the post: escalating, de-escalating, or neutral, together with a brief justification of the category.
# MAGIC 2. For posts from the rec.autos newsgroup, we will extract two lists:  one of car models and makes mentioned, and one of automobile parts or features mentioned.
# MAGIC
# MAGIC
# MAGIC **P9:  You have the skills and expertise needed to implement and use AI solutions**  Make sure to explore these tools and test their capabilities before implementing them into your project pipeline.  You should be able to justify the choices you made through evidence you collected during this exploration stage.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Features from semantic clusters

# COMMAND ----------

# MAGIC %md
# MAGIC #### Step 1: One-sentence summary with `ai_summarize`
# MAGIC
# MAGIC If you embed the full text and try to create clusters, the length and diversity of the text will have a tendency to create unclear clusters.  So as a pre-processing step, we use ai_summarize to extract the general topic of a post in a shorter phrase.

# COMMAND ----------

# DBTITLE 1,Summarise full text into shorter text to use for clustering
summarised_df = (
    text_dataset
    .filter(F.col('text').isNotNull())
    .withColumn("summary", F.expr("ai_summarize(text,50)"))
)
display(summarised_df)

# COMMAND ----------

# DBTITLE 1,Create checkpoint to force materialisation
summarised_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(SUMMARISED_TABLE)
summarised_df = spark.table(SUMMARISED_TABLE)  # read back — this "cuts" the lazy lineage here

# COMMAND ----------

# MAGIC %md
# MAGIC #### Step 2: Text embedding
# MAGIC
# MAGIC Next we embed the text using a vector embedding model

# COMMAND ----------

# DBTITLE 1,Text embedding
EMBEDDING_ENDPOINT = "databricks-bge-large-en"  

embedded_df = (
    summarised_df  
    .withColumn(
        "embedding",
        F.expr(f"ai_query('{EMBEDDING_ENDPOINT}', summary)").cast(T.ArrayType(T.FloatType()))
    )
)

display(embedded_df.select("article_id", "summary", "embedding").limit(5))

# COMMAND ----------

# DBTITLE 1,convert array to vector for KMeans and normalise
to_vector_udf = F.udf(lambda arr: Vectors.dense(arr), VectorUDT())
vectorized_df = embedded_df.filter(F.col('text').isNotNull()).withColumn("features", to_vector_udf(F.col("embedding")))
normalizer = Normalizer(inputCol="features", outputCol="normalized_features", p=2.0)
vectorized_df = normalizer.transform(vectorized_df)

# COMMAND ----------

# DBTITLE 1,vectorised checkpoint
vectorized_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(VECTORISED_TABLE)
vectorized_df = spark.table(VECTORISED_TABLE)  # read back — this "cuts" the lazy lineage here

# COMMAND ----------

# MAGIC %md
# MAGIC #### Step 3: Cluster Analysis

# COMMAND ----------

# MAGIC %md
# MAGIC We will use spherical k means for this clustering, which means it is clustering on cosine similarity, the standard similarity used for vector embedded text.  This is why we needed to normalise the vectors we obtained from the vector embedding.  This a bit of a quick first pass at assigning clusters that doesn't require installing any additional libraries.  It is enough to indicate if there is any discernable clustering going on, but in full workflows it will be necessary to evaluate different clustering methods to see what produces sensible clusters that make sense to your domain experts -- part of the human-in-the-loop called for by Principle 4 as well as choosing the correct tools, as in Principle 6.  
# MAGIC

# COMMAND ----------

# DBTITLE 1,Determine the appropriate number of clusters
for k in range(2, 9):
    km = KMeans(featuresCol="features", predictionCol="cluster", k=k, seed=42)
    m = km.fit(vectorized_df)
    preds = m.transform(vectorized_df)
    evaluator = ClusteringEvaluator(featuresCol="features", predictionCol="cluster")
    score = evaluator.evaluate(preds)
    print(f"k={k}: silhouette={score:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC As we chose 3 different newsgroups, it is not surprising to find that three clusters is the best fit.  Note also that these are not very tight clusters -- the highest silouette is only 0.1244.  So we can see that the discussions in newsgroups can tend to wander.

# COMMAND ----------

# DBTITLE 1,compare clusters to topics
K =3  
kmeans = KMeans(featuresCol="features", predictionCol="cluster", k=K, seed=42)
model = kmeans.fit(vectorized_df)
clustered_df = model.transform(vectorized_df)

display(clustered_df.crosstab('topic','cluster'))

# COMMAND ----------

# MAGIC %md
# MAGIC We can see from this that topic 0 is roughly religion, topic 1 is roughly space and topic 2 is roughly autos. 
# MAGIC
# MAGIC It may be interesting to examine the posts that do not correspond to the expected cluster, as these are "outlying" posts in that newsgroup.

# COMMAND ----------

# DBTITLE 1,auto posts that end up in religion cluster
display(clustered_df
        .filter(F.col('topic')=='rec.autos')
        .filter(F.col('cluster')==0)
        )

# COMMAND ----------

# MAGIC %md
# MAGIC Next we can look at how far a given post summary is from the center of its assigned cluster and use that to examine the posts whose topics are most central to these clusters.

# COMMAND ----------

centers = model.clusterCenters()  # list of numpy arrays, one per cluster
centers_broadcast = spark.sparkContext.broadcast(centers)

def distance_to_centroid(features, cluster):
    center = centers_broadcast.value[cluster]
    return float(np.linalg.norm(np.array(features) - center))

distance_udf = F.udf(distance_to_centroid, T.DoubleType())

clustered_with_distance_df = clustered_df.withColumn(
    "distance_to_centroid",
    distance_udf(F.col("normalized_features"), F.col("cluster"))  # use whichever feature column you actually clustered on
)

# COMMAND ----------

# DBTITLE 1,Most central to rec.autos
display(clustered_with_distance_df
        .filter(F.col('topic')=='rec.autos')
        .orderBy('distance_to_centroid')
        .limit(10)
        )

# COMMAND ----------

# MAGIC %md
# MAGIC So far we have obtained a cluster label as a feature and a distance from the assigned cluster centroid feature that indicates how close the summary is to the central theme of the cluster.
# MAGIC
# MAGIC It would also be possible to define a field giving the distance to the centroid of each cluster for each row, as features that could be used downstream as a set if coordinates that position the post with respect to each of the three central topics.  This is a form of dimensionality reduction where we have assigned different semantic dimensions.

# COMMAND ----------

# DBTITLE 1,Checkpoint table
clustered_with_distance_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(CLASSIFIED_TABLE)
classified_df = spark.table(CLASSIFIED_TABLE)  # read back — this "cuts" the lazy lineage here

# COMMAND ----------

# MAGIC %md
# MAGIC ### Structured field extraction with `ai_query`
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC #### Step 1:  Tension rating

# COMMAND ----------

TENSION_PROMPT = (
    "Classify the tone of this Usenet reply as one of: "
    "'escalating' (hostile, inflammatory, personal attacks, deliberately provocative), "
    "'de-escalating' (conciliatory, seeks common ground, defuses conflict), or "
    "'neutral' (informational, no clear emotional charge either way). "
    "Base your judgment on the reply itself, not on any quoted text from a previous "
    "message (lines starting with '>'). "
    "Return ONLY JSON: {\"tension_label\": \"...\", \"reasoning\": \"one sentence\"}. "
    "Text: "
)

rated_df = classified_df.withColumn(
    "tension_raw",
    F.call_function(
        "ai_query",
        F.lit("databricks-meta-llama-3-3-70b-instruct"),
        F.concat(F.lit(TENSION_PROMPT), F.col("text"))
    )
)

# COMMAND ----------

TENSION_SCHEMA = StructType([
    StructField("tension_label", StringType(), True),
    StructField("reasoning", StringType(), True),
])

unpacked_df = (
    rated_df
    .withColumn("tension_parsed", F.from_json(F.col("tension_raw"), TENSION_SCHEMA))
    .withColumn("tension_label", F.col("tension_parsed.tension_label"))
    .withColumn("tension_reasoning", F.col("tension_parsed.reasoning"))
    .drop("tension_parsed")
)

display(unpacked_df.select("article_id", "tension_label", "tension_reasoning", "tension_raw"))

# COMMAND ----------

# DBTITLE 1,Which is the most tense newsgroup?
display(unpacked_df.crosstab('topic','tension_label'))

# COMMAND ----------

# MAGIC %md
# MAGIC Now we want to save the table with features with appropriate metadata.

# COMMAND ----------

def field(name, dtype, comment):
    """Shorthand for a StructField carrying a UC column comment."""
    return StructField(name, dtype, True, {"comment": comment})

unpacked_schema = StructType([
    field("article_id",          StringType(),               "Unique identifier from the source dataset"),
    field("topic",                StringType(),               "Newsgroup topic label (filtered to talk.religion.misc, rec.autos, sci.space)"),
    field("subject",              StringType(),               "Subject line of the original post"),
    field("organization",         StringType(),               "Organization header, self-reported by the poster"),
    field("lines_header",         StringType(),               "Original Lines: header value (not verified against actual line count)"),
    field("distribution",         StringType(),               "Distribution header, where present"),
    field("text",                 StringType(),               "Article body: headers and footer stripped, person/email PII masked"),
    field("char_count",           IntegerType(),              "Character count of the cleaned body text"),
    field("word_count",           IntegerType(),              "Word count of the cleaned body text"),
    field("summary",              StringType(),               "LLM-generated summary of the article body (ai_summarize / ai_query, ~30 words)"),
    field("embedding",            ArrayType(FloatType()),     "Raw embedding vector from the databricks-bge-large-en endpoint, generated from the summary"),
    field("features",             VectorUDT(),                "Embedding cast to MLlib Vector type, input to KMeans"),
    field("normalized_features",  VectorUDT(),                "L2-normalized embedding vector; KMeans was fit on this column, not raw features"),
    field("cluster",              IntegerType(),              "KMeans cluster assignment (k=3, fit on normalized_features)"),
    field("distance_to_centroid", DoubleType(),                "Euclidean distance from this point to its assigned cluster's centroid, in normalized_features space"),
    field("tension_label",        StringType(),               "LLM-classified tone of the post: escalating, de-escalating, or neutral"),
    field("tension_reasoning",    StringType(),               "One-sentence LLM justification for the tension_label"),
])

# Apply schema to the DataFrame with tension_raw dropped
unpacked_df_with_comments = spark.createDataFrame(
    unpacked_df.drop("tension_raw").rdd,
    schema=unpacked_schema
)

(
    unpacked_df_with_comments.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(FEATURES_TABLE)  
)

# COMMAND ----------

spark.sql(f"""
    COMMENT ON TABLE {FEATURES_TABLE} IS
    'Final tutorial dataset: three newsgroups (talk.religion.misc, rec.autos, sci.space), cleaned, redacted, summarized, embedded, clustered (k=3), and tone-classified.'
""")

# COMMAND ----------

spark.sql(f"""
    ALTER TABLE {FEATURES_TABLE} SET TBLPROPERTIES (
        'notebook_path' = '{NOTEBOOK_PATH}',
        'processed_by' = '{PROCESSED_BY}',
        'last_ingested_at' = '{datetime.now(UTC).isoformat()}'
    )
""")

# COMMAND ----------

feature_table = spark.table(FEATURES_TABLE)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Step 2:  Extract specific data in structured form

# COMMAND ----------

# MAGIC %md
# MAGIC The `ai_query` function can also be used to extract specific pieces of information from text.  
# MAGIC
# MAGIC As an example, we will extract a list from posts from the rec.autos newsgroup of all car makes mentioned in the post.  This can be converted to a set of features for inference.

# COMMAND ----------

# DBTITLE 1,Filter to rec.autos newsgroup
rec_autos_df = feature_table.filter(F.col('topic')=='rec.autos')

# COMMAND ----------

# MAGIC %md
# MAGIC #### Old-school NLP vs `ai_query`: use both
# MAGIC
# MAGIC Entity extraction is a great illustration of **Principle 6 (the right tool for the job)**.
# MAGIC You do not have to choose between regex / dictionary methods and LLMs — the strongest
# MAGIC pipelines combine them:
# MAGIC
# MAGIC - **Car *makes* are a closed, known set.** A dictionary / gazetteer lookup (the next cell)
# MAGIC   is free, deterministic, reproducible and fully auditable — properties that matter a lot
# MAGIC   for official statistics, where you must be able to explain and repeat a result exactly.
# MAGIC - **Car *models*, slang and typos are open-ended and context-dependent**
# MAGIC   (`Bimmer`→BMW, `Vette`→Corvette, `Ponitac`→Pontiac). This fuzzy, judgement-heavy
# MAGIC   resolution is exactly what an LLM is good at and what regex is bad at.
# MAGIC
# MAGIC So a good design is **hybrid**: cheap deterministic methods for the easy, enumerable cases,
# MAGIC and `ai_query` only for the hard residual — or LLM for recall, then a gazetteer to validate
# MAGIC for precision. (We already did this in the redaction step: regex for emails/phones +
# MAGIC `ai_mask` for names.) Remember too that `ai_query` is **stochastic** and model versions
# MAGIC change over time, so its outputs need the ongoing monitoring called for by Principles 2 and
# MAGIC 5 — a gazetteer does not.
# MAGIC
# MAGIC Below we first run a pure-gazetteer pass (cheap, deterministic), then the `ai_query` pass,
# MAGIC which picks up the fuzzy cases the gazetteer misses.

# COMMAND ----------

# DBTITLE 1,Deterministic baseline: a gazetteer of known makes (no LLM, no cost)
KNOWN_MAKES = [
    "Ford", "Toyota", "Honda", "Chevrolet", "Nissan", "Volkswagen", "BMW",
    "Mercedes-Benz", "Mazda", "Subaru", "Mitsubishi", "Pontiac", "Oldsmobile",
    "Chrysler", "Dodge", "Audi", "Volvo", "Saab", "Porsche", "Jaguar",
]

# Case-insensitive, word-boundary match for each make; keep the ones that hit.
make_hits = F.array_distinct(F.array_remove(F.array(*[
    F.when(F.col("text").rlike(r"(?i)\b" + re.escape(m) + r"\b"), F.lit(m)).otherwise(F.lit(None))
    for m in KNOWN_MAKES
]), None))

gazetteer_df = rec_autos_df.withColumn("makes_gazetteer", make_hits)
display(gazetteer_df.select("article_id", "makes_gazetteer"))

# COMMAND ----------

# MAGIC %md
# MAGIC The gazetteer is instant and costs nothing, but it only finds makes we listed and cannot
# MAGIC resolve slang, typos, or model-only mentions (`Civic` with no `Honda`). That residual is
# MAGIC where the `ai_query` pass below earns its cost.

# COMMAND ----------

CAR_EXTRACTION_PROMPT = (
    "Read this Usenet post about cars, including any quoted text from earlier messages. "
    "Extract a list of: makes_and_models: any car manufacturer or model name mentioned, anywhere in the text "
    "(e.g. Honda, Civic, Ford Probe GT, BMW, 626, Mercedes). "
    "If a category has no mentions, return an empty list for it, not null. "
    "For each input string, return a list with elements in a normalised form: "
    "- Resolve slang and abbreviations to the real name (e.g. 'Bimmer' -> 'BMW', 'Benz'/'Merc' -> 'Mercedes-Benz', 'Vette' -> 'Corvette'). "
    "- Fix obvious typos (e.g. 'Ponitac' -> 'Pontiac', 'Mistubishi' -> 'Mitsubishi'). "
    "- Strip simple plurals (e.g. 'Mustangs' -> 'Mustang'). "
    "- If the input is 'Make Model' (e.g. 'Honda Civic'), keep both, standardized: 'Honda Civic'. "
    "- If the input is a model alone (e.g. 'Civic'), return it as 'Make Model' if the make is unambiguous (e.g. 'Civic' -> 'Honda Civic'). "
    "- If the input is a make alone (e.g. 'Ford'), return just the make, standardized. "
    "- If the input is not a real car make/model (e.g. a person's name, a company unrelated to cars), return null for that entry. "
    "Return ONLY JSON in this exact shape: "
    "{\"makes_and_models\": [], \"features_and_parts\": []}. "
    "Post text: "
)

extracted_df = rec_autos_df.withColumn(
    "car_data_raw",
    F.call_function(
        "ai_query",
        F.lit("databricks-meta-llama-3-3-70b-instruct"),
        F.concat(F.lit(CAR_EXTRACTION_PROMPT), F.col("text"))
    )
)

CAR_SCHEMA = StructType([
    StructField("makes_and_models", T.ArrayType(StringType()), True),
    StructField("features_and_parts", T.ArrayType(StringType()), True),
])

unpacked_car_df = (
    extracted_df
    .withColumn("car_parsed", F.from_json(F.col("car_data_raw"), CAR_SCHEMA))
    .withColumn("makes_and_models", F.col("car_parsed.makes_and_models"))
    .withColumn("features_and_parts", F.col("car_parsed.features_and_parts"))
    .drop("car_parsed")
)

display(unpacked_car_df.select("article_id", "makes_and_models", "features_and_parts"))

# COMMAND ----------

# MAGIC %md
# MAGIC Although we asked the llm to normalise the list elements, it is generally necessary to review and further normalise manually.  This is a place that domain experts should also review to ensure that the categories make sense for the use-case.
# MAGIC
# MAGIC Just a caveat here that I am not a car expert, so this is a very rough job of cleaning!

# COMMAND ----------

# DBTITLE 1,Examine extracted fields
display(
    unpacked_car_df
    .select('article_id', F.explode("makes_and_models").alias("make_or_model"))
    .withColumn('make', 
                F.when(F.lower(F.col("make_or_model")).contains('alfa romeo'),'Alfa Romeo')
                .when(F.lower(F.col("make_or_model")).contains('t-bird'),'Ford')
                .when(F.lower(F.col("make_or_model")).contains('thunderbird'),'Ford')
                .when(F.lower(F.col("make_or_model")).contains('vw'),'Volkswagen')
                .when(F.lower(F.col("make_or_model")).contains('chevy'),'Chevrolet')
                .when(F.lower(F.col("make_or_model")).contains('sho'),'Ford')
                .when(F.lower(F.col("make_or_model")).contains('240sx'),'Nissan')
                .when(F.lower(F.col("make_or_model")).contains('olds'),'Oldsmobile')
                .when(F.lower(F.col("make_or_model"))=='lh','Chrysler')
                .otherwise(F.split(F.col("make_or_model"),' ')[0]))
    .groupBy("make")
    .count()
    .orderBy(F.desc("count"))
)

# COMMAND ----------

# DBTITLE 1,Create features for inference
car_mentions = (unpacked_car_df
    .select('article_id', F.explode("makes_and_models").alias("make_or_model"))
    .withColumn('make', 
                F.when(F.lower(F.col("make_or_model")).contains('alfa romeo'),'Alfa Romeo')
                .when(F.lower(F.col("make_or_model")).contains('t-bird'),'Ford')
                .when(F.lower(F.col("make_or_model")).contains('thunderbird'),'Ford')
                .when(F.lower(F.col("make_or_model")).contains('vw'),'Volkswagen')
                .when(F.lower(F.col("make_or_model")).contains('chevy'),'Chevrolet')
                .when(F.lower(F.col("make_or_model")).contains('sho'),'Ford')
                .when(F.lower(F.col("make_or_model")).contains('240sx'),'Nissan')
                .when(F.lower(F.col("make_or_model")).contains('olds'),'Oldsmobile')
                .when(F.lower(F.col("make_or_model"))=='lh','Chrysler')
                .otherwise(F.split(F.col("make_or_model"),' ')[0]))
    .groupBy('article_id')
    .agg(
        F.collect_set('make').alias('make_list'),
        F.count_distinct('make').alias('make_count')
    )
    .withColumn('mentions_ford',F.array_contains(F.col('make_list'),F.lit('Ford')))
    .withColumn('mentions_toyota',F.array_contains(F.col('make_list'),F.lit('Toyota')))
    .withColumn('mentions_honda',F.array_contains(F.col('make_list'),F.lit('Honda')))
    .withColumn('mentions_chevrolet',F.array_contains(F.col('make_list'),F.lit('Chevrolet')))
    .withColumn('mentions_nissan',F.array_contains(F.col('make_list'),F.lit('Nissan')))
    .withColumn('mentions_vw',F.array_contains(F.col('make_list'),F.lit('Volkswagen')))

    )

                

# COMMAND ----------

auto_post_features = (rec_autos_df
                      .join(car_mentions, on='article_id', how='left')
                      .na.fill(False,subset=['mentions_ford','mentions_toyota','mentions_honda','mentions_chevrolet','mentions_nissan','mentions_vw'])
                      .na.fill(0,subset =['make_count'])

)
display(auto_post_features)

# COMMAND ----------

# DBTITLE 1,checkpoint
AUTO_POST_FEATURES_RAW_TABLE = f"{UC_CATALOG}.{UC_SCHEMA}.tutorial_auto_post_features_raw_{USERNAME}"
auto_post_features.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(AUTO_POST_FEATURES_RAW_TABLE)
auto_post_features = spark.table(AUTO_POST_FEATURES_RAW_TABLE)

# COMMAND ----------

# DBTITLE 1,What makes are mentioned together
boolean_cols = ['mentions_ford','mentions_toyota','mentions_honda','mentions_chevrolet','mentions_nissan','mentions_vw']

assembler = VectorAssembler(inputCols=boolean_cols, outputCol="features_vec")
vector_df = assembler.transform(auto_post_features).select("features_vec")

corr_matrix = Correlation.corr(vector_df, "features_vec", method="pearson").head()[0]
corr_array = corr_matrix.toArray()

corr_df = pd.DataFrame(corr_array, index=boolean_cols, columns=boolean_cols)

plt.figure(figsize=(max(8, len(boolean_cols) * 0.5), max(6, len(boolean_cols) * 0.4)))
sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1, square=True)
plt.title("Correlation between boolean features")
plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,Write table with metadata
auto_post_features_schema = StructType([
    field("article_id",          StringType(),               "Unique identifier from the source dataset"),
    field("topic",                StringType(),               "Newsgroup topic label (filtered to rec.autos for this table)"),
    field("subject",              StringType(),               "Subject line of the original post"),
    field("organization",         StringType(),               "Organization header, self-reported by the poster"),
    field("lines_header",         StringType(),               "Original Lines: header value (not verified against actual line count)"),
    field("distribution",         StringType(),               "Distribution header, where present"),
    field("text",                 StringType(),               "Article body: headers and footer stripped, person/email PII masked"),
    field("char_count",           IntegerType(),              "Character count of the cleaned body text"),
    field("word_count",           IntegerType(),              "Word count of the cleaned body text"),
    field("summary",              StringType(),               "LLM-generated summary of the article body (ai_summarize / ai_query, ~30 words)"),
    field("embedding",            ArrayType(FloatType()),     "Raw embedding vector from the databricks-bge-large-en endpoint, generated from the summary"),
    field("features",             VectorUDT(),                "Embedding cast to MLlib Vector type, input to KMeans"),
    field("normalized_features",  VectorUDT(),                "L2-normalized embedding vector; KMeans was fit on this column, not raw features"),
    field("cluster",              IntegerType(),              "KMeans cluster assignment (k=3, fit on normalized_features across all three newsgroups)"),
    field("distance_to_centroid", DoubleType(),                "Euclidean distance from this point to its assigned cluster's centroid, in normalized_features space"),
    field("tension_label",        StringType(),               "LLM-classified tone of the post: escalating, de-escalating, or neutral"),
    field("tension_reasoning",    StringType(),               "One-sentence LLM justification for the tension_label"),
    field("make_list",            ArrayType(StringType()),   "LLM-extracted, LLM-normalized list of distinct car makes/models mentioned in the post (raw extraction canonicalized via a separate ai_query mapping pass)"),
    field("make_count",           IntegerType(),              "Count of distinct entries in make_list"),
    field("mentions_ford",        BooleanType(),              "True if make_list contains a Ford make or model"),
    field("mentions_toyota",      BooleanType(),              "True if make_list contains a Toyota make or model"),
    field("mentions_honda",       BooleanType(),              "True if make_list contains a Honda make or model"),
    field("mentions_chevrolet",   BooleanType(),              "True if make_list contains a Chevrolet make or model"),
    field("mentions_nissan",      BooleanType(),              "True if make_list contains a Nissan make or model"),
    field("mentions_vw",          BooleanType(),              "True if make_list contains a Volkswagen make or model"),
])

auto_post_features_df_with_comments = spark.createDataFrame(
    auto_post_features.rdd,  # your final DataFrame before this write
    schema=auto_post_features_schema
)

(
    auto_post_features_df_with_comments.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(AUTO_POST_FEATURES_TABLE)
)

spark.sql(f"""
    COMMENT ON TABLE {AUTO_POST_FEATURES_TABLE} IS
    'rec.autos subset: cleaned, redacted, summarized, embedded, clustered, tone-classified, plus LLM-extracted and normalized car make/model mentions with per-make boolean flags for the six most common manufacturers found in this dataset.'
""")

spark.sql(f"""
    ALTER TABLE {AUTO_POST_FEATURES_TABLE} SET TBLPROPERTIES (
        'notebook_path' = '{NOTEBOOK_PATH}',
        'processed_by' = '{PROCESSED_BY}',
        'last_ingested_at' = '{datetime.now(UTC).isoformat()}'
    )
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Now it is your turn!
# MAGIC
# MAGIC We have gone over just a few ways to create semantic features from free text data, and thought about how to embed the government AI principles in the workflows to define them.  I am sure you can find others.  We look forward to seeing what you do!