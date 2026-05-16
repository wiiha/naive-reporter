# Docling Serve — Research Notes

Date: 2026-05-15

---

## What It Is

Docling Serve is a FastAPI-based HTTP service that wraps [Docling](https://github.com/docling-project/docling) for converting documents (PDFs, images, Office files, etc.) into structured text formats like Markdown, JSON, HTML, or plain text.

This service will be used in this project for text extraction from PDFs.

---

## Docker Images

| Image | Purpose | Size |
|---|---|---|
| `quay.io/docling-project/docling-serve` | Default, CPU + GPU (PyPI torch) | ~4.4 GB (arm64), ~8.7 GB (amd64) |
| `quay.io/docling-project/docling-serve-cpu` | CPU-only (smaller) | ~4.4 GB |
| `quay.io/docling-project/docling-serve-cu128` | CUDA 12.8 | ~11.4 GB |
| `quay.io/docling-project/docling-serve-cu130` | CUDA 13.0 | TBD |

**Important:** CUDA images do **not** have a `latest` tag. Always use an explicit version tag. For example:

```bash
docker pull quay.io/docling-project/docling-serve-cpu:1.12.0
```

`latest` and `main` are available for base and CPU images only.

---

## Running the Container (Simplest)

```bash
# CPU-only
docker run -p 5001:5001 quay.io/docling-project/docling-serve-cpu:1.12.0

# With UI enabled
docker run -p 5001:5001 -e DOCLING_SERVE_ENABLE_UI=1 quay.io/docling-project/docling-serve-cpu:1.12.0
```

Default port: **5001**  
API docs: `http://localhost:5001/docs`  
UI playground: `http://localhost:5001/ui`

---

## API Endpoints (v1)

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/v1/convert/source` | Convert from URLs or base64-encoded files (JSON payload) |
| POST | `/v1/convert/file` | Convert by uploading files directly (`multipart/form-data`) |
| POST | `/v1/convert/source/async` | Async URL conversion |
| POST | `/v1/convert/file/async` | Async file upload conversion |
| GET | `/v1/status/poll/{task_id}` | Poll async task status |
| GET | `/v1/result/{task_id}` | Get async task result |

---

## Request / Response Formats

### Source endpoint (`/v1/convert/source`)

Request (JSON):

```json
{
  "sources": [
    {"kind": "http", "url": "https://arxiv.org/pdf/2501.17887"}
  ],
  "options": {
    "to_formats": ["md"]
  }
}
```

Response (JSON, single document):

```json
{
  "document": {
    "md_content": "...",
    "text_content": "...",
    "json_content": "...",
    "html_content": "...",
    "doctags_content": "..."
  },
  "status": "success",
  "processing_time": 1.23,
  "timings": {...},
  "errors": []
}
```

### File endpoint (`/v1/convert/file`)

Send `multipart/form-data` with:
- `files` — the file(s) to convert
- `options` — optional conversion settings as form fields

### Async endpoints

Return a task detail object:

```json
{
  "task_id": "abc123",
  "task_status": "pending",
  "task_position": 0,
  "task_meta": {...}
}
```

Poll with `GET /v1/status/poll/{task_id}`, fetch result with `GET /v1/result/{task_id}`.

---

## Simple curl Example

```bash
curl -X POST 'http://localhost:5001/v1/convert/source' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "sources": [{"kind": "http", "url": "https://arxiv.org/pdf/2501.17887"}]
  }'
```

---

## Key Environment Variables

| Variable | Purpose | Default |
|---|---|---|
| `UVICORN_HOST` | Bind address | `0.0.0.0` |
| `UVICORN_PORT` | Port | `5001` |
| `UVICORN_WORKERS` | Webserver workers | `1` |
| `DOCLING_SERVE_ENG_KIND` | Compute engine: `local`, `rq`, `kfp` | `local` |
| `DOCLING_SERVE_ENG_LOC_NUM_WORKERS` | Local conversion workers | `2` |
| `DOCLING_SERVE_LOAD_MODELS_AT_BOOT` | Preload models at startup | `true` |
| `DOCLING_SERVE_ARTIFACTS_PATH` | Model weights directory | `/opt/app-root/src/.cache/docling/models` |
| `DOCLING_DEVICE` | Device: `cpu`, `cuda`, `mps` | auto |
| `DOCLING_SERVE_ENABLE_UI` | Enable web UI | `0` |
| `DOCLING_NUM_THREADS` | Number of threads | `4` |
| `DOCLING_PERF_PAGE_BATCH_SIZE` | Page batch size | `4` |
| `DOCLING_SERVE_OPTIONS_CACHE_SIZE` | DocumentConverter cache size | `2` |
| `DOCLING_SERVE_CONFIG_FILE` | Path to YAML/JSON config file | — |

Configuration priority: Environment variables > Config file > Defaults.

---

## Supported Formats

**Input (`from_formats`):** `docx`, `pptx`, `html`, `image`, `pdf`, `asciidoc`, `md`, `csv`, `xlsx`, `xml_uspto`, `xml_jats`, `xml_xbrl`, `mets_gbs`, `json_docling`, `audio`, `vtt`, `latex`

**Output (`to_formats`):** `md` (default), `json`, `yaml`, `html`, `html_split_page`, `text`, `doctags`, `vtt`

---

## Common Conversion Options

Both endpoints accept an `options` object with fields like:

| Option | Description | Default |
|---|---|---|
| `do_ocr` | Enable OCR | `true` |
| `force_ocr` | Force OCR even if text is present | `false` |
| `ocr_lang` | OCR language(s) | — |
| `pdf_backend` | PDF parsing backend | `docling_parse` |
| `table_mode` | Table extraction: `fast` or `accurate` | — |
| `do_table_structure` | Extract table structure | — |
| `include_images` | Include extracted images | — |
| `images_scale` | Image scaling factor | — |
| `page_range` | Page range to process | — |
| `document_timeout` | Timeout per document | — |
| `abort_on_error` | Abort on error | — |

---

## References

- Main repo: https://github.com/docling-project/docling-serve
- Docs: https://github.com/docling-project/docling-serve/blob/main/docs/README.md
- Configuration: https://github.com/docling-project/docling-serve/blob/main/docs/configuration.md
- Usage: https://github.com/docling-project/docling-serve/blob/main/docs/usage.md
- Deployment: https://github.com/docling-project/docling-serve/blob/main/docs/deployment.md
