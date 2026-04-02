"""
ws_ingest.py — Coinbase WebSocket Ingestor
Connects to Coinbase Advanced Trade WebSocket API,
receives live ticker data, and publishes to Kafka topic ticks.raw.
Also saves raw data to data/raw/ as NDJSON for replay.
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime, timezone

import yaml
import websocket
from dotenv import load_dotenv
from confluent_kafka import Producer

# ── Load environment variables from .env ──
load_dotenv()

# ── Load config from config.yaml ──
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

KAFKA_BROKER = os.getenv("KAFKA_BROKER", config["kafka"]["broker"])
RAW_TOPIC = config["kafka"]["topics"]["raw"]
WS_URL = "wss://advanced-trade-ws.coinbase.com"

# ── Set up Kafka producer ──
producer = Producer({"bootstrap.servers": KAFKA_BROKER})

def delivery_report(err, msg):
    """Callback: called once per message to confirm delivery to Kafka."""
    if err:
        print(f"[KAFKA ERROR] Failed to deliver: {err}")


def setup_raw_file(pair):
    """Create an NDJSON file in data/raw/ named with the pair and timestamp."""
    os.makedirs("data/raw", exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename = f"data/raw/{pair}_{timestamp}.ndjson"
    print(f"[FILE] Saving raw ticks to {filename}")
    return open(filename, "a")

# ── Global state ──
raw_file = None       # will hold the open NDJSON file
tick_count = 0
        # counts how many ticks we've received


def on_open(ws):
    """Called when WebSocket connection is established."""
    pair = ws.pair  # we'll attach this to the ws object later
    print(f"[WS] Connected to Coinbase WebSocket")

    # Subscribe to ticker channel
    ticker_sub = {
        "type": "subscribe",
        "product_ids": [pair],
        "channel": "ticker"
    }
    ws.send(json.dumps(ticker_sub))
    print(f"[WS] Subscribed to ticker for {pair}")

    # Subscribe to heartbeats channel (keeps connection alive)
    heartbeat_sub = {
        "type": "subscribe",
        "channel": "heartbeats"
    }
    ws.send(json.dumps(heartbeat_sub))
    print(f"[WS] Subscribed to heartbeats")


def on_message(ws, message):
    """Called every time a message arrives from Coinbase."""
    global tick_count

    data = json.loads(message)
    channel = data.get("channel", "")

    # Skip non-ticker messages (subscriptions confirmations, heartbeats)
    if channel != "ticker":
        return

    # Add a local timestamp for when we received it
    data["received_at"] = datetime.now(timezone.utc).isoformat()

    # Publish to Kafka
    producer.produce(
        RAW_TOPIC,
        value=json.dumps(data).encode("utf-8"),
        callback=delivery_report
    )
    producer.poll(0)  # trigger delivery callbacks without blocking

    # Save to NDJSON file
    if raw_file:
        raw_file.write(json.dumps(data) + "\n")
        raw_file.flush()  # write immediately, don't buffer
    

    tick_count += 1
    if tick_count % 10 == 0:
        print(f"[TICK] {tick_count} ticks received")
    
    # Stop if time is up
    if hasattr(ws, 'end_time') and time.time() >= ws.end_time:
        print(f"[DONE] Time limit reached.")
        ws.close()
        return


def on_error(ws, error):
    """Called when a WebSocket error occurs."""
    print(f"[WS ERROR] {error}")


def on_close(ws, close_status, close_msg):
    """Called when the WebSocket connection closes."""
    print(f"[WS] Connection closed (status={close_status}, msg={close_msg})")

def run_ingestor(pair, minutes):
    """Main loop: connect to WebSocket, reconnect on failure, stop after duration."""
    global raw_file, tick_count

    raw_file = setup_raw_file(pair)
    start_time = time.time()
    end_time = start_time + (minutes * 60)

    print(f"[START] Ingesting {pair} for {minutes} minutes")
    print(f"[START] Kafka topic: {RAW_TOPIC}")
    print(f"[START] Kafka broker: {KAFKA_BROKER}")

    while time.time() < end_time:
        remaining = int(end_time - time.time())
        print(f"[WS] Connecting... ({remaining}s remaining)")

        # Create WebSocket connection
        ws = websocket.WebSocketApp(
            WS_URL,
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )
        ws.pair = pair  # attach pair so on_open can access it
        ws.end_time = end_time # attach so on_message can check it


        # Run until connection drops or time runs out
        # ping_interval sends a ping every 30s to detect dead connections
        ws.run_forever(ping_interval=30, ping_timeout=10)

        # If we get here, the connection closed
        if time.time() < end_time:
            print(f"[WS] Reconnecting in 5 seconds...")
            time.sleep(5)

    # ── Clean up ──
    print(f"\n[DONE] Ingestion complete.")
    print(f"[DONE] Total ticks received: {tick_count}")
    producer.flush()  # wait for all Kafka messages to be delivered
    raw_file.close()
    print(f"[DONE] Kafka producer flushed. File closed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Coinbase WebSocket Ingestor")
    parser.add_argument("--pair", type=str, default="BTC-USD",
                        help="Trading pair to ingest (default: BTC-USD)")
    parser.add_argument("--minutes", type=int, default=15,
                        help="How many minutes to ingest (default: 15)")
    args = parser.parse_args()

    try:
        run_ingestor(args.pair, args.minutes)
    except KeyboardInterrupt:
        print(f"\n[STOP] Interrupted by user. {tick_count} ticks received.")
        producer.flush()
        if raw_file:
            raw_file.close()