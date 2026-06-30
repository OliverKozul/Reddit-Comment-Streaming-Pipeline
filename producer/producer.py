import json
import os
import time
import logging

from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

BOOTSTRAP_SERVERS = os.environ["KAFKA_BOOTSTRAP_SERVERS"]
TOPIC = os.environ["KAFKA_TOPIC"]
INTERVAL = float(os.environ.get("PRODUCE_INTERVAL_SECONDS", 0.01))
PARQUET_PATH = os.environ.get("PARQUET_PATH", "/app/data/askreddit-stream.parquet")
TEXT_COLUMN = os.environ.get("TEXT_COLUMN", "body")
SUBREDDIT_COLUMN = os.environ.get("SUBREDDIT_COLUMN", "subreddit")


def connect_to_kafka(servers, retries=30, delay=5.0):
    for attempt in range(1, retries + 1):
        try:
            producer = KafkaProducer(
                bootstrap_servers=servers,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            )
            log.info("Connected to Kafka at %s", servers)
            return producer
        except NoBrokersAvailable:
            log.warning("Kafka not ready (attempt %d/%d), retrying in %.0fs", attempt, retries, delay)
            time.sleep(delay)
    raise RuntimeError(f"Could not connect to Kafka after {retries} attempts")


def stream_comments(producer, topic, interval):
    log.info("Streaming comments from parquet %s", PARQUET_PATH)
    dataset = load_dataset("parquet", data_files={"train": PARQUET_PATH},
                           split="train", streaming=True)
    sent = 0
    for i, row in enumerate(dataset):
        text = row.get(TEXT_COLUMN, "")
        if not text or text in ("[deleted]", "[removed]"):
            continue
        record = {
            "id": row.get("id", str(i)),
            "timestamp": str(time.time()),
            "text": text[:2000],
            "label": "unknown",
            "subreddit": row.get(SUBREDDIT_COLUMN, "unknown"),
        }
        producer.send(topic, value=record)
        sent += 1
        if interval:
            time.sleep(interval)
    producer.flush()
    log.info("Done: sent %d comments in a single pass", sent)


def main():
    producer = connect_to_kafka(BOOTSTRAP_SERVERS)
    stream_comments(producer, TOPIC, INTERVAL)


if __name__ == "__main__":
    main()
