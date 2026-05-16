# Naive Reporter

A PDF ingestion pipeline. Users upload PDFs and chat with the extracted content via Telegram or similar clients.

## Setup

```bash
python -m pip install -e ".[dev]"
```

## Run

```bash
python -m naive_reporter
```

Or, once installed:

```bash
naive-reporter
```

## Docling Serve

Start the document conversion service:

```bash
docker compose up -d
```

API is available at `http://localhost:5001`.
