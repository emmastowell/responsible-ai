# responsible-ai
Materials for the databricks responsible AI hackathon

## Notebooks

All notebooks run on Databricks serverless and write to Unity Catalog (`hackathon.shared_datasets` by default). Open in the workspace and set the widgets, or run via `databricks jobs submit`.

### `notebooks/hackathon_tutorial.py`
Day-one worked example: **"Semantic Features in Statistical Workflows"**. Walks through the first two pipeline stages (ETL ingestion and semantic feature engineering) on the 20 Newsgroups corpus, using AI functions (`ai_mask`, `ai_summarize`) and framed around the ten UK Government AI Principles. A reference for team working time; no connection to the specific use cases, but the same unstructured-text problem.

### `notebooks/load_reddit_pushshift_corpora.py`
Lands two Reddit corpora as bronze Delta tables from the [Arctic Shift](https://arctic-shift.photon-reddit.com) API (the maintained successor to the shut-down Pushshift API). Paginates backward over `created_utc` up to a configurable row cap.

| Corpus | Subreddits | Theme |
|--------|-----------|-------|
| `unrest` | r/PublicFreakout, r/protest, r/activism | Social unrest / historical social media text |
| `wsb` | r/wallstreetbets | Financial market manipulation |

Output tables: `reddit_{corpus}_{posts,comments}_bronze`. Run once with `content_type=posts` and once with `content_type=comments` to populate all four.

Widgets: `catalog`, `schema`, `row_cap` (default 2000), `content_type` (`posts` or `comments`).

### `notebooks/cqc_ratings_to_parsed_pipeline.py`
End-to-end CQC (UK Care Quality Commission) pipeline in four stages:

1. **Download ratings** — pull the CQC "Latest ratings" spreadsheet (.ods, ~310k rows) and land it raw in Delta. Parsed with `python-calamine` (~4s; the `odfpy` engine takes 10+ minutes on this file).
2. **Build sample** — filter to Care Homes / "Safe" domain, keep all Requires-improvement / Inadequate / Outstanding, plus a hash-sampled slice of Good.
3. **Download PDFs** — resolve each location's inspection-report PDF from the CQC site and save it to a UC Volume.
4. **Parse** — run `ai_parse_document` over the downloaded PDFs into a Delta table.

Output tables: `cqc_latest_ratings` (raw), `cqc_parsed_documents` (parsed). PDFs land in the `cqc_reports` Volume.

Widgets: `catalog`, `schema`, `ratings_url`, `ratings_table`, `parsed_table`, `volume_dir`, `good_sample_size`, `candidate_limit` (set > 0 for a fast smoke test).

### `notebooks/gdelt_events_ingestion.py`
Builds the `gdelt_unrest_events` source table. Downloads GDELT 2.0 event export files (15-minute intervals) for a date range, filters to social-unrest CAMEO root codes (14 Protest, 17 Coerce, 18 Assault, 19 Fight), and optionally filters to UK geography. No authentication required. Run this **before** the extraction notebook below.

Widgets: `catalog`, `schema`, `target_table`, `gdelt_date_from`, `gdelt_date_to`, `gdelt_filter_uk`, `max_files` (cap for a fast test).

### `notebooks/gdelt_article_raw_text_extraction.py`
Reads unrest-related news events from `gdelt_unrest_events`, fetches each unique article URL, and extracts clean article body text with `trafilatura`. Writes `gdelt_article_raw_text`.

Configurable in-notebook: date window (default last 30 days), `max_unique_urls`, `batch_size`, and whether to store raw HTML.

### `notebooks/yahoo_finance_prices_ingestion.py`
Daily OHLCV + volume for target equities (market-manipulation use case, pairs with the WSB Reddit data). Defaults to the meme-stock set GME, AMC, BB, NOK, BBBY plus SPY/QQQ baselines, 2020-2021. Uses the Yahoo chart API directly (`query1.finance.yahoo.com/v8/finance/chart`) with a browser User-Agent — the default `python-requests` UA gets a 429. Writes `yahoo_finance_prices` (long format, partitioned by ticker).

Widgets: `catalog`, `schema`, `target_table`, `tickers`, `date_from`, `date_to`, `interval` (`1d`/`1wk`/`1mo`).

### `notebooks/financial_phrasebank_ingestion.py`
Loads the FinancialPhraseBank dataset (Malo et al., 2014): ~4,840 financial-news sentences labelled positive / neutral / negative, at four annotator-agreement thresholds. Sentiment baseline for the fraud / market-manipulation use cases. Downloads the raw zip from Hugging Face and parses the latin-1 `sentence@label` files directly (avoids the `datasets` `trust_remote_code` path). Writes `financial_phrasebank` with an `agreement_level` column (~14.8k rows across the four levels).

Widgets: `catalog`, `schema`, `target_table`, `source_url`.

### `notebooks/crypto_archive_extraction.py`
Extracts a crypto transaction-subgraph archive (a zip staged in the `crypto_raw` Volume) into a shared Volume for participants, then builds two Delta tables: `crypto_subgraph_summary` and `crypto_subgraph_manifest` (GraphML subgraphs plus LLM4TG text representations). Configurable in-notebook: `zip_path`, `extraction_root`, table names, `overwrite_extraction`.

### `notebooks/synthetic_crypto_fraud_narratives.py`
Generates a synthetic crypto transaction narratives dataset for pump-and-dump fraud-detection exercises. Writes `synthetic_crypto_fraud_narratives` with price/volume/wallet/social spike features (`price_change_pct`, `volume_spike_x`, `unique_buy_wallets`, `top10_holder_pct`, `social_mention_spike_x`), a free-text `narrative`, and a `label`.

## Serverless notes

- Parse large `.ods` files with `python-calamine`, not `odfpy`. The serverless pandas is too old for `read_excel(engine="calamine")`, so call the native `CalamineWorkbook` API.
- Do not set `spark.sql.execution.arrow.pyspark.enabled` on serverless (managed by Spark Connect).
- Delta needs `delta.columnMapping.mode = name` for column names with spaces or special characters (e.g. the raw CQC columns).
