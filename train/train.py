import os
import sys

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

import pandas as pd
from pyspark.sql import SparkSession

from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer, NGram, HashingTF, IDF, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

TRAIN_DATA        = os.environ.get("TRAIN_DATA", "data/askreddit-train.parquet")
MODEL_OUTPUT_PATH = os.environ.get("MODEL_OUTPUT_PATH", "models/comment_scorer")
TRAIN_SAMPLE_SIZE = int(os.environ.get("TRAIN_SAMPLE_SIZE", "50000"))
LOW_PCT           = float(os.environ.get("LOW_PCT", "0.35"))
HIGH_PCT          = float(os.environ.get("HIGH_PCT", "0.80"))
SEED              = 42
TEXT_COLUMN       = "body"
SCORE_COLUMN      = "score"


def load_comments(spark, path):
    df = spark.read.parquet(path).select(TEXT_COLUMN, SCORE_COLUMN).toPandas()
    df = df.rename(columns={TEXT_COLUMN: "text", SCORE_COLUMN: "score"})
    df["text"] = df["text"].astype(str)
    df = df[~df["text"].isin(["[deleted]", "[removed]", "nan", ""])]
    df = df.dropna(subset=["score"])
    df["score"] = df["score"].astype(int)
    print(f"Loaded {len(df)} comments from {path}")
    return df


def thresholds(df):
    low_thr = df["score"].quantile(LOW_PCT, interpolation="lower")
    high_thr = df["score"].quantile(HIGH_PCT, interpolation="lower")
    if low_thr >= high_thr:
        high_thr = low_thr + 1
    return low_thr, high_thr


def label_extremes(df, low_thr, high_thr, max_per_class):
    low = df[df["score"] <= low_thr]
    high = df[df["score"] >= high_thr]
    n = min(len(low), len(high), max_per_class)
    low = low.sample(n, random_state=SEED)[["text"]].assign(label=0.0)
    high = high.sample(n, random_state=SEED)[["text"]].assign(label=1.0)
    print(f"  {n} low (score <= {low_thr}) / {n} high (score >= {high_thr})")
    return pd.concat([low, high]).sample(frac=1, random_state=SEED).reset_index(drop=True)


def build_pipeline():
    return Pipeline(stages=[
        RegexTokenizer(inputCol="text", outputCol="chars", pattern=r"[a-z\s!?.]", gaps=False, toLowercase=True),
        NGram(n=3, inputCol="chars", outputCol="cg3"),
        NGram(n=4, inputCol="chars", outputCol="cg4"),
        NGram(n=5, inputCol="chars", outputCol="cg5"),
        HashingTF(inputCol="cg3", outputCol="tf3", numFeatures=20000),
        HashingTF(inputCol="cg4", outputCol="tf4", numFeatures=20000),
        HashingTF(inputCol="cg5", outputCol="tf5", numFeatures=20000),
        IDF(inputCol="tf3", outputCol="idf3"),
        IDF(inputCol="tf4", outputCol="idf4"),
        IDF(inputCol="tf5", outputCol="idf5"),
        VectorAssembler(inputCols=["idf3", "idf4", "idf5"], outputCol="features"),
        LogisticRegression(maxIter=200, regParam=0.01, labelCol="label", featuresCol="features"),
    ])


def main():
    spark = SparkSession.builder.appName("train").master("local[*]").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    df_pd = load_comments(spark, TRAIN_DATA)
    low_thr, high_thr = thresholds(df_pd)
    labeled = label_extremes(df_pd, low_thr, high_thr, TRAIN_SAMPLE_SIZE)

    df = spark.createDataFrame(labeled)
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=SEED)

    model = build_pipeline().fit(train_df)

    predictions = model.transform(test_df)
    evaluator = MulticlassClassificationEvaluator(labelCol="label", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print(f"Accuracy: {accuracy:.3f}")

    os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
    model.write().overwrite().save(MODEL_OUTPUT_PATH)
    print(f"Model saved to {MODEL_OUTPUT_PATH}")
    spark.stop()


if __name__ == "__main__":
    main()
