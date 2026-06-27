# Reddit Comment Streaming Pipeline

A containerized real-time pipeline that streams Reddit comments through Kafka into a Spark Structured Streaming job, classifies each comment as a high or low performer, stores the results in PostgreSQL, and visualizes them in a live Grafana dashboard.

## Setup

**1. Train the model**

A pre-trained model is included at `models/comment_scorer.pkl`, so this step is optional. To retrain it, download the data from [Kaggle](https://www.kaggle.com/datasets/smagnan/1-million-reddit-comments-from-40-subreddits), save it to `train/reddit_comments.csv`, and run:

```bash
pip install -r train/requirements.txt
python train/train.py
```

This overwrites `models/comment_scorer.pkl`.

**2. Configure environment**

```bash
cp .env.example .env
```

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
