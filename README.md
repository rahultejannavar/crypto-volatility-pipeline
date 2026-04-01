# Crypto Volatility Detection Pipeline

Real-time pipeline for detecting short-term cryptocurrency volatility spikes using streaming data from Coinbase.

## Overview
This project connects to Coinbase's Advanced Trade WebSocket API, streams live market data through Kafka, engineers features, trains models to predict 60-second volatility spikes, and monitors data quality with Evidently.

## Quick Start
```
docker compose -f docker/compose.yaml up -d
python scripts/ws_ingest.py --pair BTC-USD --minutes 15
```

## Tech Stack
- **Streaming:** Apache Kafka (KRaft mode)
- **Tracking:** MLflow
- **Monitoring:** Evidently
- **Containerization:** Docker Compose

## Project Structure
(To be filled in)

## Author
Rahul Tejannavar — Carnegie Mellon University, Heinz College
94-879 Fundamentals of Operationalizing AI, Spring 2026
