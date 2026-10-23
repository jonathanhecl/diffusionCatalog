# DiffusionCatalog

**DiffusionCatalog** is a local-first web application written in Go for cataloging, searching, and organizing AI-generated diffusion images and videos directly from a source folder.

It treats your media directory as **read-only**, extracting embedded generation prompts and technical parameters (Automatic1111, Forge, ComfyUI, Fooocus, SwarmUI, InvokeAI) in pure Go and indexing them into a local SQLite + FTS5 database.

## Key Features

- **Non-Destructive Folder Indexing**: Watches and indexes your existing media folder without moving, renaming, or altering your files.
- **Pure-Go Prompt & Engine Extraction**: Reads PNG `tEXt` / `iTXt` chunks, JPEG/WebP EXIF `UserComment`, and `.txt`/`.json` sidecars to extract positive/negative prompts, checkpoint models, LoRAs, sampler, scheduler, steps, CFG scale, and seed.
- **Agnostic Tags & Facets**: Organize assets with arbitrary user-defined tags and facets (`style`, `lighting`, `composition`, `subject`, `lora`, or any custom facet).
- **Agnostic Groups**: Build virtual collections and moodboards independent of disk hierarchy (while also auto-grouping by top-level subfolder).
- **Character & Concept Folders**: Store reusable character profiles containing canonical positive/negative generation prompts, trigger words, recommended checkpoints, and curated reference galleries.
- **Pluggable Face & Subject Explorer**: Hexagonal `VisionEngine` port in Go with an optional Python FastAPI sidecar (`scripts/vision_worker/app.py`) for face/subject detection, cropping, and clustering.
- **Default Local Admin**: Automatically seeds a default `admin` account (`admin` / `admin123`) on first boot.

## Quick Start

### 1. Build and Run

```bash
go build -o catalogd ./cmd/catalogd
./catalogd --media-dir=./media --db=./catalog.db --addr=:8080
```

Then open **http://localhost:8080** in your browser.

> If `./media` is empty on first launch and `--seed-demo=true` (default), `catalogd` generates sample diffusion PNG files with embedded A1111, ComfyUI, Fooocus, SwarmUI, and InvokeAI metadata chunks, along with sample Character Folders, Agnostic Groups, and cropped focus regions so you can explore every view immediately.

### 2. Command-Line Flags & Environment Variables

| Flag | Env Variable | Default | Description |
|------|--------------|---------|-------------|
| `--addr` | `CATALOG_ADDR` | `:8080` | HTTP listen address |
| `--media-dir` | `CATALOG_MEDIA_DIR` | `./media` | Read-only folder containing diffusion images/videos |
| `--cache-dir` | `CATALOG_CACHE_DIR` | `./.cache` | Local directory for cropped regions & thumbnails |
| `--db` | `CATALOG_DB_PATH` | `./catalog.db` | Local SQLite database path |
| `--admin-user` | `CATALOG_ADMIN_USER` | `admin` | Default administrator username |
| `--admin-pass` | `CATALOG_ADMIN_PASS` | `admin123` | Default administrator password |
| `--vision-url` | `CATALOG_VISION_URL` | `""` | Optional Python Vision Sidecar URL (`http://127.0.0.1:8091`) |
| `--auto-scan` | `CATALOG_AUTO_SCAN` | `true` | Automatically scan `--media-dir` on startup |
| `--seed-demo` | `CATALOG_SEED_DEMO` | `true` | Seed sample PNGs with diffusion metadata if `--media-dir` is empty |

### 3. Optional Python Vision Worker (Face Detection & Cropping)

```bash
cd scripts/vision_worker
pip install fastapi uvicorn pillow opencv-python-headless
uvicorn app:app --host 127.0.0.1 --port 8091
```

Then start `catalogd` with `--vision-url=http://127.0.0.1:8091`.

## Documentation

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the Hexagonal Architecture overview, database schema, and metadata extraction pipeline.
