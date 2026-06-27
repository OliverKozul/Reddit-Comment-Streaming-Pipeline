import os
import time
import logging

from datasets import load_dataset

log = logging.getLogger(__name__)

DATASET = os.environ.get("HF_DATASET", "webis/tldr-17")
SPLIT = os.environ.get("HF_SPLIT", "train")
TEXT_COLUMN = os.environ.get("HF_TEXT_COLUMN", "content")
SUBREDDIT_COLUMN = os.environ.get("HF_SUBREDDIT_COLUMN", "subreddit")
SUBREDDITS = {s.strip().lower() for s in os.environ.get("HF_SUBREDDITS", "AskReddit").split(",")}


def stream_comments(producer, topic, interval):
    log.info("Streaming r/%s from %s", ", r/".join(sorted(SUBREDDITS)), DATASET)
    while True:
        dataset = load_dataset(DATASET, split=SPLIT, streaming=True, trust_remote_code=True)
        for i, row in enumerate(dataset):
            text = row.get(TEXT_COLUMN, "")
            if not text or text in ("[deleted]", "[removed]"):
                continue
            subreddit = row.get(SUBREDDIT_COLUMN, "unknown")
            if subreddit.lower() not in SUBREDDITS:
                continue
            record = {
                "id": row.get("id", str(i)),
                "timestamp": str(time.time()),
                "text": text[:2000],
                "label": "unknown",
                "subreddit": subreddit,
            }
            producer.send(topic, value=record)
            if i % 100 == 0:
                log.info("Sent comment (number %d) %s from r/%s", i, record["id"], subreddit)
            time.sleep(interval)
        log.info("Reached the end of the dataset, starting over")
