# responsible-ai
Materials for the databricks responsible AI hackathon

## Notebooks

All notebooks run on Databricks serverless and write to Unity Catalog (`hackathon.shared_datasets` by default). Open in the workspace and set the widgets, or run via `databricks jobs submit`.

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

### `notebooks/gdelt_article_raw_text_extraction.py`
Reads unrest-related news events from `gdelt_unrest_events`, fetches each unique article URL, and extracts clean article body text with `trafilatura`. Writes `gdelt_article_raw_text`.

Configurable in-notebook: date window (default last 30 days), `max_unique_urls`, `batch_size`, and whether to store raw HTML.

## Serverless notes

- Parse large `.ods` files with `python-calamine`, not `odfpy`. The serverless pandas is too old for `read_excel(engine="calamine")`, so call the native `CalamineWorkbook` API.
- Do not set `spark.sql.execution.arrow.pyspark.enabled` on serverless (managed by Spark Connect).
- Delta needs `delta.columnMapping.mode = name` for column names with spaces or special characters (e.g. the raw CQC columns).
