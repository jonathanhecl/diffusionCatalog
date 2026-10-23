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

### Option A: Run Directly Without Compiling (`run.sh`)

```bash
./run.sh
```

`run.sh` checks whether Go and optional dependencies are present (asking if you want to install anything missing) and executes `go run ./cmd/catalogd` directly without leaving a binary behind.

### Option B: Build the Portable Self-Installing Binary (`build.sh`)

```bash
./build.sh
```

This compiles a 100% static, single-file binary (`CGO_ENABLED=0`) into both `./catalogd` and `./build/catalogd` (along with `./build/start-portable.sh`).

The compiled `catalogd` binary is **self-contained and self-installing**:
- Embeds the HTML/CSS dark-navy UI and the Python Vision Worker (`app.py` + `requirements.txt`) inside the binary itself.
- On first interactive launch (or anytime you pass `--setup`), it checks for optional system/runtime components (`ffmpeg` for video container metadata, `python3` + `.venv` for local Face Detection & Cropping) and prompts `[y/N]` to install and start them automatically.

```bash
./catalogd --media-dir=./media --db=./catalog.db --addr=:1080
# Re-run interactive dependency installer at any time:
./catalogd --setup
```

Then open **http://localhost:1080** in your browser.

> Port 1080 is an unprivileged port (≥ 1024), so it runs directly on Linux without sudo or special permissions. You can change the port at any time in `config.json` (e.g. `"port": 8080`) or by passing `--port=8080`.

### 2. Configuration (`config.json`), Flags & Environment Variables

DiffusionCatalog automatically reads `config.json` from the current directory, which can be overridden via command-line flags or environment variables:

```json
{
  "port": 1080,
  "addr": ":1080",
  "media_dir": "./media",
  "cache_dir": "./.cache",
  "db_path": "./catalog.db",
  "admin_user": "admin",
  "admin_pass": "admin123",
  "vision_url": "",
  "auto_scan": true,
  "auto_scan_interval": 10,
  "seed_demo": false
}
```

| Flag | Env Variable | Default | Description |
|------|--------------|---------|-------------|
| `--port` | `PORT` / `CATALOG_PORT` | `1080` | HTTP listen port |
| `--addr` | `CATALOG_ADDR` | `:1080` | HTTP listen address |
| `--config` | `CATALOG_CONFIG` | `config.json` | Path to JSON configuration file |
| `--media-dir` | `CATALOG_MEDIA_DIR` | `./media` | Read-only folder containing diffusion images/videos |
| `--cache-dir` | `CATALOG_CACHE_DIR` | `./.cache` | Local directory for cropped regions & thumbnails |
| `--db` | `CATALOG_DB_PATH` | `./catalog.db` | Local SQLite database path |
| `--admin-user` | `CATALOG_ADMIN_USER` | `admin` | Default administrator username |
| `--admin-pass` | `CATALOG_ADMIN_PASS` | `admin123` | Default administrator password |
| `--vision-url` | `CATALOG_VISION_URL` | `""` | Optional Python Vision Sidecar URL (`http://127.0.0.1:8091`) |
| `--auto-scan` | `CATALOG_AUTO_SCAN` | `true` | Automatically scan media directories on startup and keep synced in background |
| `--auto-scan-interval` | `CATALOG_AUTO_SCAN_INTERVAL` | `10s` | Interval for automatic background media folder sync (e.g. `10s`, `30s`; `0s` disables) |
| `--seed-demo` | `CATALOG_SEED_DEMO` | `false` | Generate sample diffusion images in media-dir if folder does not exist or is empty |

### 3. Optional Python Vision Worker (Face Detection & Cropping)

```bash
cd scripts/vision_worker
pip install fastapi uvicorn pillow opencv-python-headless
uvicorn app:app --host 127.0.0.1 --port 8091
```

Then start `catalogd` with `--vision-url=http://127.0.0.1:8091`.

## Documentation

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the Hexagonal Architecture overview, database schema, and metadata extraction pipeline.
