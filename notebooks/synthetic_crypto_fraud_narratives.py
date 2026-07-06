# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Synthetic dataset overview
# MAGIC %md
# MAGIC This notebook builds a synthetic crypto transaction narratives dataset for hackathon participants. It is designed for pump-and-dump style fraud detection exercises where simple spike features are not enough on their own.
# MAGIC
# MAGIC The generated table keeps the same core columns as the earlier small example:
# MAGIC * `tx_id`
# MAGIC * `timestamp`
# MAGIC * `token`
# MAGIC * `price_change_pct`
# MAGIC * `volume_spike_x`
# MAGIC * `unique_buy_wallets`
# MAGIC * `top10_holder_pct`
# MAGIC * `social_mention_spike_x`
# MAGIC * `narrative`
# MAGIC * `label`
# MAGIC
# MAGIC The notebook creates a larger synthetic dataset, displays sample rows, and saves the output as a shared Delta table for participants to query.

# COMMAND ----------

# DBTITLE 1,Generate and save synthetic dataset
import random
from datetime import datetime, timedelta

from pyspark.sql import functions as F

seed = 42
row_count = 1000
fraud_share = 0.6
target_table = "hackathon.shared_datasets.synthetic_crypto_fraud_narratives"

rng = random.Random(seed)
base_timestamp = datetime(2026, 3, 1, 0, 0)

fraud_event_templates = [
    "Coordinated buy wave from newly funded wallets pushed {token} up {price_change_pct}% before concentrated holders exited within {exit_window} minutes.",
    "Anonymous channel activity signalled entry and {token} spiked rapidly; {supply_shift}% of supply moved toward exchange deposit addresses shortly after the rally.",
    "{token} volume surged without credible protocol news and the largest wallet cluster liquidated {liquidated_pct}% of holdings near the intraday peak.",
    "A cluster of recently created wallets accumulated {accumulated_pct}% of {token} supply, then sold in a coordinated {exit_window}-minute window.",
    "Influencer promotion coincided with a sharp {token} rally while promoter-linked wallets had accumulated inventory {preload_hours} hours earlier.",
    "A low-liquidity {token} pool repriced after a large buy sequence, followed by rapid distribution across {fresh_wallets} newly active wallets.",
    "Liquidity tied to original deployer wallets was withdrawn shortly after a scripted social campaign accelerated retail buying in {token}."
]

legit_event_templates = [
    "{token} rallied after a confirmed product launch and the holder base broadened with no single wallet exceeding {max_wallet_pct}% of supply.",
    "Price rose steadily following a verified exchange listing for {token}; sell pressure stayed muted after the initial move.",
    "A token incentive campaign brought {unique_buy_wallets} participating wallets into {token} and distribution remained broad after the increase.",
    "{token} gained after a routine protocol upgrade, with volume normalising within {normalise_hours} hours and no concentration spike.",
    "Publicly disclosed accumulation ahead of a scheduled token event supported a gradual move in {token} while holder concentration stayed diffuse.",
    "Developer adoption news drove higher mentions of {token}, but inflows were spread across existing wallets rather than a fresh wallet cluster.",
    "Treasury and ecosystem announcements increased interest in {token}; order flow stayed diversified and there was no fast post-spike unwind."
]

fraud_tokens = ["ZYNC", "KRUX", "GLIMR", "OBLIX", "FENRA", "VORTX", "PLYTH", "DRAKE", "MYSTR", "QNTX"]
legit_tokens = ["AXEL", "DRYFT", "NOVAQ", "HELXA", "QUOR", "LUMENX", "SOLARA", "TREON", "ORBIX", "VYNE"]


def make_tx_id(index: int) -> str:
    return f"TX{index:05d}"


def make_timestamp(index: int) -> datetime:
    return base_timestamp + timedelta(
        hours=6 * index + rng.randint(0, 5),
        minutes=rng.randint(0, 59),
    )


def fraud_row(index: int) -> dict:
    token = rng.choice(fraud_tokens)
    price_change_pct = rng.randint(165, 620)
    volume_spike_x = rng.randint(18, 65)
    unique_buy_wallets = rng.randint(380, 2600)
    top10_holder_pct = rng.randint(52, 80)
    social_mention_spike_x = rng.randint(12, 36)
    narrative = rng.choice(fraud_event_templates).format(
        token=token,
        price_change_pct=price_change_pct,
        exit_window=rng.randint(15, 90),
        supply_shift=rng.randint(35, 70),
        liquidated_pct=rng.randint(25, 55),
        accumulated_pct=rng.randint(20, 38),
        preload_hours=rng.randint(18, 72),
        fresh_wallets=rng.randint(20, 140),
    )
    return {
        "tx_id": make_tx_id(index),
        "timestamp": make_timestamp(index),
        "token": token,
        "price_change_pct": price_change_pct,
        "volume_spike_x": volume_spike_x,
        "unique_buy_wallets": unique_buy_wallets,
        "top10_holder_pct": top10_holder_pct,
        "social_mention_spike_x": social_mention_spike_x,
        "narrative": narrative,
        "label": "fraud",
    }


def legitimate_row(index: int) -> dict:
    token = rng.choice(legit_tokens)
    price_change_pct = rng.randint(45, 190)
    volume_spike_x = rng.randint(4, 20)
    unique_buy_wallets = rng.randint(1200, 6500)
    top10_holder_pct = rng.randint(12, 32)
    social_mention_spike_x = rng.randint(5, 18)
    narrative = rng.choice(legit_event_templates).format(
        token=token,
        unique_buy_wallets=unique_buy_wallets,
        max_wallet_pct=rng.randint(2, 5),
        normalise_hours=rng.randint(24, 72),
    )
    return {
        "tx_id": make_tx_id(index),
        "timestamp": make_timestamp(index),
        "token": token,
        "price_change_pct": price_change_pct,
        "volume_spike_x": volume_spike_x,
        "unique_buy_wallets": unique_buy_wallets,
        "top10_holder_pct": top10_holder_pct,
        "social_mention_spike_x": social_mention_spike_x,
        "narrative": narrative,
        "label": "legitimate",
    }

rows = []
fraud_rows = int(row_count * fraud_share)
for index in range(1, row_count + 1):
    if index <= fraud_rows:
        rows.append(fraud_row(index))
    else:
        rows.append(legitimate_row(index))

rng.shuffle(rows)
synthetic_df = spark.createDataFrame(rows).orderBy("timestamp", "tx_id")
synthetic_df.createOrReplaceTempView("synthetic_crypto_fraud_narratives")

(
    synthetic_df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(target_table)
)

print(f"Saved {synthetic_df.count()} rows to {target_table}")
display(synthetic_df.limit(20))

# COMMAND ----------

# DBTITLE 1,Validate class balance and feature patterns
saved_df = spark.table(target_table)

class_balance_df = saved_df.groupBy("label").count().orderBy("label")
feature_summary_df = saved_df.groupBy("label").agg(
    F.round(F.avg("price_change_pct"), 1).alias("avg_price_change_pct"),
    F.round(F.avg("volume_spike_x"), 1).alias("avg_volume_spike_x"),
    F.round(F.avg("unique_buy_wallets"), 1).alias("avg_unique_buy_wallets"),
    F.round(F.avg("top10_holder_pct"), 1).alias("avg_top10_holder_pct"),
    F.round(F.avg("social_mention_spike_x"), 1).alias("avg_social_mention_spike_x"),
)

high_spike_lookalikes_df = (
    saved_df
    .filter((F.col("label") == "legitimate") & (F.col("volume_spike_x") >= 12))
    .select(
        "tx_id",
        "token",
        "price_change_pct",
        "volume_spike_x",
        "top10_holder_pct",
        "social_mention_spike_x",
        "label",
    )
    .orderBy(F.desc("volume_spike_x"), F.desc("price_change_pct"))
)

display(class_balance_df)
display(feature_summary_df)
display(high_spike_lookalikes_df.limit(10))

# COMMAND ----------

# DBTITLE 1,Design logic notes
# MAGIC %md
# MAGIC Design logic for the synthetic dataset:
# MAGIC
# MAGIC * Fraud rows are biased toward large price and volume spikes, elevated holder concentration, and narratives that imply coordinated distribution or liquidity withdrawal.
# MAGIC * Legitimate rows intentionally include some strong price, volume, and social spikes so a naive "big move = fraud" rule performs poorly.
# MAGIC * The most useful separating signals are intended to be concentration and distribution-style features such as `top10_holder_pct`, combined with narrative clues about coordinated exits versus broad-based adoption.
# MAGIC * This makes the dataset useful for workshops on feature engineering, prompt-based triage, weak supervision, and simple classification baselines.