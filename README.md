# Public Sector Responsible AI Hackathon

Materials for a public sector Responsible AI hackathon run by **Databricks**, the **Royal Statistical Society (RSS) AI Task Force**, and **Manuka.AI**.

## Background

This repository is the basis for a hands-on hackathon exploring the responsible use of AI in public sector statistical work. The premise is simple: the best way to understand where AI tools help, and where they fall short, is to experiment with them in low-risk settings that mirror real work before embedding them in live services.

The exercises are framed around the ten UK Government AI Principles (knowing what AI is and its limits, using it lawfully and securely, keeping meaningful human control, and so on). Every dataset here is public, so participants can experiment freely without touching sensitive or in-service data.

The work is organised as a tutorial that teaches the core technique, followed by a set of real-world datasets that teams use for their own projects.

## Start here: the tutorial

**[`notebooks/hackathon_tutorial.py`](notebooks/hackathon_tutorial.py)** is the main notebook and the starting point for day one.

*Semantic Features in Statistical Workflows* is a worked example of the first two stages of a modelling pipeline: data ingestion and semantic feature engineering. It uses the 20 Newsgroups corpus (unstructured text, the same shape of problem the use cases present) and Databricks AI functions such as `ai_mask` and `ai_summarize`, with the UK Government AI Principles called out at each stage. Work through it first, then use it as a reference during team time.

## Datasets for real-world practice

Each use case below has one or more notebooks that ingest public data into Unity Catalog (`hackathon.shared_datasets`), ready for teams to build on. Open a notebook in the workspace and set its widgets, or run it via `databricks jobs submit`.

### Social unrest early warning

| Notebook | What it builds |
|----------|----------------|
| [`gdelt_events_ingestion.py`](notebooks/gdelt_events_ingestion.py) | GDELT 2.0 news events filtered to unrest CAMEO codes (protest, coerce, assault, fight), optionally UK-only. Table: `gdelt_unrest_events`. |
| [`gdelt_article_raw_text_extraction.py`](notebooks/gdelt_article_raw_text_extraction.py) | Fetches each event's article and extracts clean body text with `trafilatura`. Table: `gdelt_article_raw_text`. Run after the ingestion notebook. |
| [`load_reddit_pushshift_corpora.py`](notebooks/load_reddit_pushshift_corpora.py) | Reddit unrest discussion (r/PublicFreakout, r/protest, r/activism) via the Arctic Shift API. Tables: `reddit_unrest_posts_bronze`, `reddit_unrest_comments_bronze`. |
| [`ons_economic_indicators_download.py`](notebooks/ons_economic_indicators_download.py) | UK ONS indicators (CPIH inflation, labour market, monthly GDP, retail sales) via the ONS beta API, as contextual features. Tables: `ons_*`. |

### Market manipulation detection

| Notebook | What it builds |
|----------|----------------|
| [`load_reddit_pushshift_corpora.py`](notebooks/load_reddit_pushshift_corpora.py) | r/WallStreetBets posts and comments via Arctic Shift. Tables: `reddit_wsb_posts_bronze`, `reddit_wsb_comments_bronze`. |
| [`yahoo_finance_prices_ingestion.py`](notebooks/yahoo_finance_prices_ingestion.py) | Daily OHLCV and volume for target equities (meme-stock set plus SPY/QQQ baselines, 2020-2021) via the Yahoo chart API. Table: `yahoo_finance_prices`. |
| [`financial_phrasebank_ingestion.py`](notebooks/financial_phrasebank_ingestion.py) | FinancialPhraseBank sentiment sentences (Malo et al., 2014) at four agreement levels, as a sentiment baseline. Table: `financial_phrasebank`. |

### CQC inspection report analysis

| Notebook | What it builds |
|----------|----------------|
| [`cqc_ratings_to_parsed_pipeline.py`](notebooks/cqc_ratings_to_parsed_pipeline.py) | End-to-end pipeline: download the CQC "Latest ratings" spreadsheet, sample Care Homes on the "Safe" domain, download their inspection PDFs, and structure them with `ai_parse_document`. Tables: `cqc_latest_ratings`, `cqc_parsed_documents`; PDFs land in the `cqc_reports` Volume. |

### Financial and crypto fraud detection

| Notebook | What it builds |
|----------|----------------|
| [`crypto_archive_extraction.py`](notebooks/crypto_archive_extraction.py) | Unpacks a crypto transaction-subgraph archive into a shared Volume and builds summary and manifest tables (GraphML subgraphs plus LLM4TG text representations). Tables: `crypto_subgraph_summary`, `crypto_subgraph_manifest`. |
| [`synthetic_crypto_fraud_narratives.py`](notebooks/synthetic_crypto_fraud_narratives.py) | Generates a synthetic pump-and-dump dataset with price, volume, wallet and social-signal features plus a narrative and label. Table: `synthetic_crypto_fraud_narratives`. |

### External dataset: IEEE-CIS Fraud Detection (Kaggle)

Used for the financial fraud-detection use case. There is no notebook for it here because it requires Kaggle credentials and accepting the competition rules.

- Download: https://www.kaggle.com/competitions/ieee-fraud-detection/data
- Accept the rules first: https://www.kaggle.com/competitions/ieee-fraud-detection/rules
- API access needs a Kaggle token (username and key) from https://www.kaggle.com/settings > API

The files (`train_transaction`, `test_transaction`, `train_identity`, `test_identity`) are already loaded to `hackathon.shared_datasets` as the matching `train_*` / `test_*` tables.
