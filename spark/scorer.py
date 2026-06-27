import os
import pickle
import logging
from functools import lru_cache

import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import StructType, StructField, StringType, DoubleType

log = logging.getLogger(__name__)

MODEL_PATH = os.environ.get("MODEL_PATH", "/app/models/comment_scorer.pkl")

RESULT_SCHEMA = StructType([
    StructField("sentiment", StringType(), nullable=True),
    StructField("score",     DoubleType(), nullable=True),
])


@lru_cache(maxsize=1)
def _load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@lru_cache(maxsize=None)
def _warn_once(message):
    log.warning(message)


@pandas_udf(RESULT_SCHEMA)
def _predict(text: pd.Series) -> pd.DataFrame:
    texts = text.fillna("").tolist()
    try:
        model = _load_model()
        labels = model.predict(texts)
        proba = model.predict_proba(texts)
        sentiment = ["high" if label == 1 else "low" for label in labels]
        score = [float(proba[i][int(labels[i])]) for i in range(len(labels))]
    except Exception as exc:
        _warn_once(f"Scoring failed, defaulting to low: {exc}")
        sentiment = ["low"] * len(texts)
        score = [0.0] * len(texts)
    return pd.DataFrame({"sentiment": sentiment, "score": score})


def add_predictions(df: DataFrame) -> DataFrame:
    scored = df.withColumn("_p", _predict(df["text"]))
    return (
        scored
        .withColumn("sentiment", scored["_p"]["sentiment"])
        .withColumn("score",     scored["_p"]["score"])
        .drop("_p")
    )
