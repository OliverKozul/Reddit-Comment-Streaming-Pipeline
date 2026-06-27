import os
import pickle

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

TRAIN_CSV          = os.environ.get("TRAIN_CSV", "train/reddit_comments.csv")
TRAIN_SAMPLE_SIZE = int(os.environ.get("TRAIN_SAMPLE_SIZE", "10000"))
LOW_PCT           = float(os.environ.get("LOW_PCT", "0.35"))
HIGH_PCT          = float(os.environ.get("HIGH_PCT", "0.80"))
MODEL_OUTPUT_PATH = os.environ.get("MODEL_OUTPUT_PATH", "models/comment_scorer.pkl")
SEED              = 42


def load_comments():
    if not os.path.exists(TRAIN_CSV):
        raise RuntimeError(
            f"Training CSV not found at '{TRAIN_CSV}'.\n"
            "Download it from "
            "https://www.kaggle.com/datasets/smagnan/1-million-reddit-comments-from-40-subreddits\n"
            f"and save it to '{TRAIN_CSV}' (or set TRAIN_CSV)."
        )
    df = pd.read_csv(TRAIN_CSV, usecols=["body", "score"])
    df = df.rename(columns={"body": "text"})
    df["text"] = df["text"].astype(str)
    df = df[~df["text"].isin(["[deleted]", "[removed]", "nan", ""])]
    df = df.dropna(subset=["score"])
    df["score"] = df["score"].astype(int)
    print(f"Loaded {len(df)} comments from {TRAIN_CSV}")
    return df


def label_extremes(df):
    low_thr = df["score"].quantile(LOW_PCT, interpolation="lower")
    high_thr = df["score"].quantile(HIGH_PCT, interpolation="lower")
    if low_thr >= high_thr:
        high_thr = low_thr + 1

    low = df[df["score"] <= low_thr]
    high = df[df["score"] >= high_thr]
    n = min(len(low), len(high), TRAIN_SAMPLE_SIZE)
    low = low.sample(n, random_state=SEED)
    high = high.sample(n, random_state=SEED)
    print(f"Labels: {n} low (score <= {low_thr}) / {n} high (score >= {high_thr})")

    texts = pd.concat([low["text"], high["text"]]).tolist()
    labels = [0] * n + [1] * n
    return texts, labels


def build_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            max_features=20000,
            min_df=5,
            sublinear_tf=True,
        )),
        ("clf", LogisticRegression(max_iter=2000, C=1.0, n_jobs=-1)),
    ])


def main():
    df = load_comments()
    texts, labels = label_extremes(df)
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=SEED, stratify=labels
    )
    model = build_pipeline()
    model.fit(X_train, y_train)
    print(classification_report(y_test, model.predict(X_test), target_names=["low", "high"]))

    os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
    with open(MODEL_OUTPUT_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {MODEL_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
