import os
import logging

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, current_timestamp
from pyspark.sql.types import StructType, StructField, StringType

from scorer import add_predictions

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

KAFKA_SERVERS = os.environ["KAFKA_BOOTSTRAP_SERVERS"]
KAFKA_TOPIC   = os.environ["KAFKA_TOPIC"]
PG_HOST       = os.environ["POSTGRES_HOST"]
PG_PORT       = os.environ["POSTGRES_PORT"]
PG_DB         = os.environ["POSTGRES_DB"]
PG_USER       = os.environ["POSTGRES_USER"]
PG_PASS       = os.environ["POSTGRES_PASSWORD"]

JDBC_URL = f"jdbc:postgresql://{PG_HOST}:{PG_PORT}/{PG_DB}"
JDBC_PROPS = {
    "user":     PG_USER,
    "password": PG_PASS,
    "driver":   "org.postgresql.Driver",
}

MESSAGE_SCHEMA = StructType([
    StructField("id",        StringType(), True),
    StructField("timestamp", StringType(), True),
    StructField("text",      StringType(), True),
    StructField("label",     StringType(), True),
    StructField("subreddit", StringType(), True),
])


def write_to_postgres(batch_df, batch_id: int):
    if batch_df.isEmpty():
        return

    parsed = (
        batch_df
        .select(from_json(col("value").cast("string"), MESSAGE_SCHEMA).alias("msg"))
        .select(
            col("msg.id").alias("event_id"),
            col("msg.text").alias("text"),
            col("msg.subreddit").alias("subreddit"),
        )
        .filter(col("text").isNotNull())
    )

    parsed.cache()
    try:
        if parsed.isEmpty():
            return

        (
            parsed
            .withColumn("received_at", current_timestamp())
            .select("event_id", "received_at", "text")
            .write.jdbc(JDBC_URL, "events", mode="append", properties=JDBC_PROPS)
        )

        (
            add_predictions(parsed)
            .withColumn("processed_at", current_timestamp())
            .select("event_id", "processed_at", "text", "sentiment", "score", "subreddit")
            .write.jdbc(JDBC_URL, "ml_results", mode="append", properties=JDBC_PROPS)
        )

        log.info("Batch %d: wrote %d rows", batch_id, parsed.count())
    finally:
        parsed.unpersist()


def main():
    spark = (
        SparkSession.builder
        .appName("SparkStreamingPipeline")
        .config("spark.jars.packages",
                "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0,"
                "org.postgresql:postgresql:42.7.3")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    log.info("Connecting to Kafka %s, topic=%s", KAFKA_SERVERS, KAFKA_TOPIC)

    stream = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_SERVERS)
        .option("subscribe", KAFKA_TOPIC)
        .option("startingOffsets", "earliest")
        .option("failOnDataLoss", "false")
        .option("maxOffsetsPerTrigger", 2000)
        .load()
    )

    query = (
        stream.writeStream
        .foreachBatch(write_to_postgres)
        .option("checkpointLocation", "/tmp/checkpoint")
        .trigger(processingTime="5 seconds")
        .start()
    )

    log.info("Streaming query started. Waiting for termination…")
    query.awaitTermination()


if __name__ == "__main__":
    main()
