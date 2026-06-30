# Reddit Comment Streaming Pipeline

A containerized real-time pipeline that streams Reddit comments through Kafka into a Spark Structured Streaming job, classifies each comment as a high or low performer with a SparkML model, stores the results in PostgreSQL, and visualizes them in a live Grafana dashboard.

## Setup

**1. Configure environment**

```bash
cp .env.example .env
```

**2. Train the model**

The training data is included at `data/askreddit-comments.parquet` (~25k r/AskReddit comments with real upvote scores, filtered from the Kaggle [1M Reddit comments](https://www.kaggle.com/datasets/smagnan/1-million-reddit-comments-from-40-subreddits) dataset). Train the model once (held-out 80/20 accuracy is printed, model written to `models/comment_scorer/`):

```bash
docker-compose run --rm train
```

The producer replays the same parquet into Kafka, so no dataset download is needed.

**3. Start the stack**

```bash
docker-compose up --build
```

## Services

| URL | Service |
|-----|---------|
| http://localhost:3000 | Grafana (admin / admin) |
| http://localhost:9000 | Kafdrop (Kafka UI) |
| http://localhost:8080 | Spark master UI |

## Resetting

Schema changes require a fresh database volume:

```bash
docker-compose down -v
```
