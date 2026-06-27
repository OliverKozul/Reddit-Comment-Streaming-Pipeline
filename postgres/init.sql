CREATE TABLE IF NOT EXISTS events (
    id          BIGSERIAL PRIMARY KEY,
    event_id    TEXT        NOT NULL,
    received_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    text        TEXT        NOT NULL
);

CREATE TABLE IF NOT EXISTS ml_results (
    id           BIGSERIAL PRIMARY KEY,
    event_id     TEXT        NOT NULL,
    processed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    text         TEXT        NOT NULL,
    sentiment    TEXT,
    score        DOUBLE PRECISION,
    subreddit    TEXT
);

CREATE INDEX IF NOT EXISTS ml_results_processed_at_idx ON ml_results (processed_at);
CREATE INDEX IF NOT EXISTS ml_results_sentiment_idx    ON ml_results (sentiment);
CREATE INDEX IF NOT EXISTS ml_results_subreddit_idx    ON ml_results (subreddit);
