import os
import logging
from functools import lru_cache

from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, when

log = logging.getLogger(__name__)

MODEL_PATH = os.environ.get("MODEL_PATH", "/app/models/comment_scorer")


@lru_cache(maxsize=1)
def _load_model():
    log.info("Loading SparkML model from %s", MODEL_PATH)
    return PipelineModel.load(MODEL_PATH)


def add_predictions(df: DataFrame) -> DataFrame:
    scored = _load_model().transform(df)
    prob = vector_to_array(col("probability"))
    return (
        scored
        .withColumn("sentiment", when(col("prediction") == 1.0, "high").otherwise("low"))
        .withColumn("score", when(col("prediction") == 1.0, prob[1]).otherwise(prob[0]))
        .drop("chars", "cg3", "cg4", "cg5",
              "tf3", "tf4", "tf5", "idf3", "idf4", "idf5",
              "rawPrediction", "probability", "prediction", "features")
    )
