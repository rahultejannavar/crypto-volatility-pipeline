"""
featurizer.py — Feature Engineering Pipeline
Reads raw ticks (from Kafka or list), computes windowed features,
and outputs to Kafka topic ticks.features and/or Parquet file.
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime, timezone
from collections import deque

import yaml
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from confluent_kafka import Consumer, Producer

# ── Load config ──
load_dotenv()

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

KAFKA_BROKER = os.getenv("KAFKA_BROKER", config["kafka"]["broker"])
RAW_TOPIC = config["kafka"]["topics"]["raw"]
FEATURES_TOPIC = config["kafka"]["topics"]["features"]
HORIZON = config["features"]["horizon_seconds"]


def parse_tick(raw_msg):
    """Extract flat fields from a nested Coinbase ticker message.
    
    Returns a dict with: timestamp, price, best_bid, best_ask,
    bid_qty, ask_qty, volume_24h. Returns None if parsing fails.
    """
    try:
        ticker = raw_msg["events"][0]["tickers"][0]
        return {
            "timestamp": raw_msg["timestamp"],
            "received_at": raw_msg.get("received_at", ""),
            "price": float(ticker["price"]),
            "best_bid": float(ticker["best_bid"]),
            "best_ask": float(ticker["best_ask"]),
            "bid_qty": float(ticker["best_bid_quantity"]),
            "ask_qty": float(ticker["best_ask_quantity"]),
            "volume_24h": float(ticker["volume_24_h"]),
        }
    except (KeyError, IndexError, ValueError) as e:
        print(f"[PARSE ERROR] {e}")
        return None
    
def compute_features(tick_buffer):
    """Compute features from a buffer of parsed ticks.
    
    Args:
        tick_buffer: list of parsed tick dicts (from parse_tick),
                     ordered oldest to newest
    
    Returns:
        dict of computed features, or None if buffer is too small
    """
    if len(tick_buffer) < 10:
        return None  # need minimum ticks for meaningful features

    # Current tick (most recent)
    current = tick_buffer[-1]

    # ── Midprice ──
    # Average of best bid and best ask — more stable than last trade price
    midprices = [(t["best_bid"] + t["best_ask"]) / 2 for t in tick_buffer]
    current_midprice = midprices[-1]

    # ── Midprice Returns ──
    # Percentage change between consecutive midprices
    returns = []
    for i in range(1, len(midprices)):
        if midprices[i - 1] != 0:
            ret = (midprices[i] - midprices[i - 1]) / midprices[i - 1]
            returns.append(ret)

    if len(returns) < 5:
        return None

    returns_arr = np.array(returns)

    # ── Features ──
    features = {
        "timestamp": current["timestamp"],
        "received_at": current["received_at"],
        "midprice": current_midprice,

        # Bid-ask spread: wider spread = more uncertainty
        "spread": current["best_ask"] - current["best_bid"],
        "spread_pct": (current["best_ask"] - current["best_bid"]) / current_midprice,

        # Order book imbalance: positive = more buy pressure
        "book_imbalance": (current["bid_qty"] - current["ask_qty"]) /
                          (current["bid_qty"] + current["ask_qty"] + 1e-10),

        # Rolling return statistics over the window
        "return_mean": float(np.mean(returns_arr)),
        "return_std": float(np.std(returns_arr)),
        "return_skew": float(pd.Series(returns_arr).skew()) if len(returns_arr) >= 3 else 0.0,

        # Trade intensity: how many ticks in this window
        "tick_count": len(tick_buffer),

        # Price range in window
        "price_range": max(midprices) - min(midprices),
        "price_range_pct": (max(midprices) - min(midprices)) / current_midprice,

        # Volume
        "volume_24h": current["volume_24h"],
    }

    return features
def add_labels(features_list):
    """Add volatility spike labels to a list of feature dicts.
    
    For each row, look ahead HORIZON seconds and compute the std of
    midprice returns in that future window. If it exceeds threshold τ,
    label = 1 (spike), else label = 0.
    
    This can ONLY be done offline (after data collection), not in real-time,
    because it requires future data.
    """
    df = pd.DataFrame(features_list)
    df["timestamp_dt"] = pd.to_datetime(df["timestamp"])

    # Forward-looking volatility: for each row, compute return_std
    # of the rows within the next HORIZON seconds
    future_vol = []

    for i in range(len(df)):
        current_time = df["timestamp_dt"].iloc[i]
        cutoff_time = current_time + pd.Timedelta(seconds=HORIZON)

        # Get future rows within the horizon
        future_mask = (df["timestamp_dt"] > current_time) & (df["timestamp_dt"] <= cutoff_time)
        future_rows = df.loc[future_mask]

        if len(future_rows) >= 5:
            # Compute midprice returns in the future window
            future_midprices = future_rows["midprice"].values
            future_returns = np.diff(future_midprices) / future_midprices[:-1]
            future_vol.append(float(np.std(future_returns)))
        else:
            future_vol.append(np.nan)  # not enough future data

    df["future_vol"] = future_vol
    df.drop(columns=["timestamp_dt"], inplace=True)

    return df


def save_to_parquet(df, output_path="data/processed/features.parquet"):
    """Save features DataFrame to Parquet file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"[SAVE] {len(df)} rows saved to {output_path}")

def run_live_consumer(window_seconds=60):
    """Consume from Kafka ticks.raw, compute features, publish to ticks.features."""

    consumer = Consumer({
        "bootstrap.servers": KAFKA_BROKER,
        "group.id": "featurizer",
        "auto.offset.reset": "earliest"
    })
    consumer.subscribe([RAW_TOPIC])

    producer = Producer({"bootstrap.servers": KAFKA_BROKER})

    tick_buffer = deque()
    features_list = []
    count = 0

    print(f"[FEATURIZER] Listening on '{RAW_TOPIC}'...")
    print(f"[FEATURIZER] Window: {window_seconds}s | Horizon: {HORIZON}s")

    try:
        while True:
            msg = consumer.poll(timeout=2.0)

            if msg is None:
                continue
            if msg.error():
                print(f"[KAFKA ERROR] {msg.error()}")
                continue

            # Parse the raw tick
            raw = json.loads(msg.value().decode("utf-8"))
            tick = parse_tick(raw)
            if tick is None:
                continue

            # Add to buffer
            tick_buffer.append(tick)

            # Remove ticks older than the window
            current_time = pd.to_datetime(tick["timestamp"])
            while tick_buffer:
                oldest_time = pd.to_datetime(tick_buffer[0]["timestamp"])
                if (current_time - oldest_time).total_seconds() > window_seconds:
                    tick_buffer.popleft()
                else:
                    break

            # Compute features
            features = compute_features(list(tick_buffer))
            if features is None:
                continue

            features_list.append(features)
            count += 1

            # Publish to Kafka
            producer.produce(
                FEATURES_TOPIC,
                value=json.dumps(features).encode("utf-8")
            )
            producer.poll(0)

            if count % 50 == 0:
                print(f"[FEATURIZER] {count} feature rows computed")

    except KeyboardInterrupt:
        print(f"\n[FEATURIZER] Stopped. {count} feature rows computed.")
    finally:
        consumer.close()
        producer.flush()

        # Save collected features to Parquet
        if features_list:
            df = add_labels(features_list)
            save_to_parquet(df)


def run_from_file(input_files, output_path="data/processed/features.parquet", window_seconds=60):
    """Process saved NDJSON files through the same feature logic.
    
    This is the replay function — used by scripts/replay.py
    """
    all_ticks = []

    # Load all ticks from all files
    for filepath in input_files:
        print(f"[REPLAY] Loading {filepath}")
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)
                tick = parse_tick(raw)
                if tick:
                    all_ticks.append(tick)

    print(f"[REPLAY] Loaded {len(all_ticks)} ticks from {len(input_files)} file(s)")

    # Sort by timestamp to ensure correct order
    all_ticks.sort(key=lambda t: t["timestamp"])

    # Slide through ticks computing features
    tick_buffer = deque()
    features_list = []

    for tick in all_ticks:
        tick_buffer.append(tick)

        # Remove ticks older than the window
        current_time = pd.to_datetime(tick["timestamp"])
        while tick_buffer:
            oldest_time = pd.to_datetime(tick_buffer[0]["timestamp"])
            if (current_time - oldest_time).total_seconds() > window_seconds:
                tick_buffer.popleft()
            else:
                break

        # Compute features
        features = compute_features(list(tick_buffer))
        if features:
            features_list.append(features)

    print(f"[REPLAY] {len(features_list)} feature rows computed")

    # Add labels and save
    if features_list:
        df = add_labels(features_list)
        save_to_parquet(df, output_path)
        return df

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature Engineering Pipeline")
    parser.add_argument("--mode", choices=["live", "replay"], default="live",
                        help="Run mode: 'live' (Kafka consumer) or 'replay' (from files)")
    parser.add_argument("--input", nargs="+", default=None,
                        help="Input NDJSON files (replay mode only)")
    parser.add_argument("--output", default="data/processed/features.parquet",
                        help="Output Parquet path")
    parser.add_argument("--window", type=int, default=60,
                        help="Sliding window size in seconds (default: 60)")
    args = parser.parse_args()

    if args.mode == "live":
        run_live_consumer(window_seconds=args.window)
    elif args.mode == "replay":
        if not args.input:
            print("[ERROR] Replay mode requires --input files")
            sys.exit(1)
        run_from_file(args.input, args.output, args.window)