"""
kafka_consume_check.py — Kafka Consumer Sanity Check
Reads messages from a Kafka topic and prints them to verify data is flowing.
"""

import os
import sys
import json
import argparse

import yaml
from dotenv import load_dotenv
from confluent_kafka import Consumer

# ── Load config ──
load_dotenv()

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

KAFKA_BROKER = os.getenv("KAFKA_BROKER", config["kafka"]["broker"])


def check_topic(topic, min_messages):
    """Read messages from a Kafka topic and print them."""

    consumer = Consumer({
        "bootstrap.servers": KAFKA_BROKER,
        "group.id": "sanity-check",        # consumer group name
        "auto.offset.reset": "earliest"     # start from the very first message
    })

    consumer.subscribe([topic])
    print(f"[CHECK] Reading from topic '{topic}' (looking for {min_messages} messages)...")

    count = 0
    empty_polls = 0

    while count < min_messages:
        msg = consumer.poll(timeout=2.0)  # wait up to 2 seconds for a message

        if msg is None:
            empty_polls += 1
            if empty_polls >= 5:
                print(f"[CHECK] No more messages after {count} received. Stopping.")
                break
            continue

        if msg.error():
            print(f"[KAFKA ERROR] {msg.error()}")
            continue

        # Reset empty poll counter when we get data
        empty_polls = 0
        count += 1

        data = json.loads(msg.value().decode("utf-8"))

        # Print first 3 and every 50th message
        if count <= 3 or count % 50 == 0:
            ticker = data["events"][0]["tickers"][0]
            print(f"  [{count}] {ticker['product_id']} "
                  f"price={ticker['price']} "
                  f"bid={ticker['best_bid']} "
                  f"ask={ticker['best_ask']} "
                  f"time={data['timestamp']}")

    consumer.close()
    print(f"\n[RESULT] Received {count} messages from '{topic}'")

    if count >= min_messages:
        print("[RESULT] ✓ Kafka is working correctly!")
    else:
        print(f"[RESULT] ✗ Expected {min_messages}, only got {count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kafka Consumer Sanity Check")
    parser.add_argument("--topic", type=str, default="ticks.raw",
                        help="Kafka topic to read from (default: ticks.raw)")
    parser.add_argument("--min", type=int, default=100,
                        help="Minimum messages to read (default: 100)")
    args = parser.parse_args()

    check_topic(args.topic, args.min)