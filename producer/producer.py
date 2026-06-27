import json
import os
import time
import logging

from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable

from hf_producer import stream_comments

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

BOOTSTRAP_SERVERS = os.environ["KAFKA_BOOTSTRAP_SERVERS"]
TOPIC = os.environ["KAFKA_TOPIC"]
INTERVAL = float(os.environ.get("PRODUCE_INTERVAL_SECONDS", 0.01))


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


def main():
    producer = connect_to_kafka(BOOTSTRAP_SERVERS)
    stream_comments(producer, TOPIC, INTERVAL)


if __name__ == "__main__":
    main()
