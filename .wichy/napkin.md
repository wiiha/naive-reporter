# Napkin Runbook

> **HARD LIMIT: This file must never exceed 8200 characters.**
> If approaching the limit, recurate aggressively: merge duplicates, remove stale notes, re-prioritize by importance. No exceptions.

## Curation Rules

- Re-prioritize on every read.
- Keep recurring, high-value notes only.
- Max 10 items per category.
- Each item includes date + "Do instead".

## Execution & Validation (Highest Priority)

1. **[2026-05-15] No Python file may exceed 8200 characters**
   Do instead: if a `.py` file approaches or exceeds 8200 chars, treat it as a signal to refactor — split into smaller modules or extract functions/classes. Enforce this at build time; never let files grow past the limit.

## Shell & Command Reliability

1. **[2026-05-15] On startup, check for venv at `/home/wichy/venv`**
   Do instead: if the venv does not exist, create it (`python -m venv /home/wichy/venv`), activate it, and install the project from `/workspace` in editable mode using that new venv (`pip install -e ".[dev]"`). This fixes permission problems with the system venv under `/opt/wichy`.

## Domain Behavior Guardrails

1. **[2026-05-15] Docling-serve may return `text_content: null`**
   Do instead: always fall back to `md_content` when `text_content` is absent or empty. The Markdown content has all the text.

2. **[2026-05-15] Use `host.docker.internal` to reach Docker host from containers**
   Do instead: for docling-serve or Ollama running on the host system, use `http://host.docker.internal:<port>` as the URL.

3. **[2026-05-15] Content-hash deduplication, not filename-based**
   Do instead: compute SHA-256 of PDF contents, store in `data/seen_hashes/`. Skip if hash exists. Only use filename suffix (`_1`, `_2`) when hash is new but filename collides in `processed/`.

## User Directives

1. **[2026-05-15] Prefer simple, direct solutions**
   Do instead: solve problems with the most straightforward, boring, readable code possible. When in doubt, pick the approach a junior dev would understand immediately.

2. **[2026-05-15] Reject "smart" solutions**
   Do instead: explicitly avoid clever one-liners, unnecessary abstractions, over-generalized frameworks, or premature optimization. If a solution feels clever, it's wrong.

3. **[2026-05-15] Keep it simple, keep it straight**
   Do instead: state assumptions plainly, write linear code, minimize indirection, and prefer duplication over the wrong abstraction.
