# responsible-ai
Materials for the databricks responsible AI hackathon

## Datasets

`notebooks/load_reddit_pushshift_corpora.py` lands two Reddit corpora as bronze Delta tables in Unity Catalog. Source is the [Arctic Shift](https://arctic-shift.photon-reddit.com) API, the maintained successor to the shut-down Pushshift API. It paginates backward over `created_utc` up to a configurable row cap.

| Corpus | Subreddits | Theme |
|--------|-----------|-------|
| `unrest` | r/PublicFreakout, r/protest, r/activism | Social unrest / historical social media text |
| `wsb` | r/wallstreetbets | Financial market manipulation |

**Output tables** (in `{catalog}.{schema}`): `reddit_{corpus}_{posts,comments}_bronze`, e.g. `reddit_unrest_posts_bronze`, `reddit_wsb_comments_bronze`.

**Run it:** open the notebook in Databricks and set the widgets.

| Widget | Default | Purpose |
|--------|---------|---------|
| `catalog` | `main` | UC catalog to write to |
| `schema` | `reddit_pushshift` | created if missing |
| `row_cap` | `2000` | max rows per corpus |
| `content_type` | `posts` | `posts` (title + selftext) or `comments` (reply threads) |

Run once with `content_type=posts` and once with `content_type=comments` to populate all four tables.
