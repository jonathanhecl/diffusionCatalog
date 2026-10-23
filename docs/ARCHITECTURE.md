# Architecture & Design Reference

## 1. Hexagonal / Clean Architecture

`DiffusionCatalog` follows a strict separation between core domain logic and external adapters:

- **`internal/domain`**: Core domain entities (`MediaAsset`, `GenerationParams`, `Tag`, `Group`, `Character`, `DetectedFace`, `User`) and port interfaces (`CatalogRepository`, `MetadataExtractor`, `VisionEngine`).
- **`internal/usecase`**: Application services (`CatalogService`, `ScannerService`, `AuthService`) orchestrating non-destructive folder scanning, character sheets, agnostic tagging, and authentication.
- **`internal/adapter/metadata`**: Pure-Go parser for PNG `tEXt`/`iTXt` chunks (`parameters`, `prompt`, `workflow`, `sui_image_params`, `invokeai_metadata`), EXIF byte streams, and companion sidecar files.
- **`internal/adapter/storage/sqlite`**: Pure-Go SQLite adapter (`modernc.org/sqlite`) with WAL mode and FTS5 full-text indexing.
- **`internal/adapter/vision`**: Pluggable `VisionEngine` implementations (`NoopEngine` by default, `HTTPWorkerEngine` for connecting a Python OpenCV/InsightFace sidecar).
- **`internal/adapter/web`**: Embedded `html/template` + CSS HTTP server delivering the high-density dark-navy catalog interface and JSON API (`/api/v1/catalog`).

## 2. Supported Diffusion Metadata Formats

| Generator / UI | Source | Extracted Fields |
|----------------|--------|------------------|
| **Automatic1111 / Forge** | PNG `tEXt`/`iTXt` (`parameters`), EXIF `UserComment` | Positive/Negative prompt, Steps, Sampler, Scheduler, CFG, Seed, Model, Model Hash, `<lora:...>` tags |
| **ComfyUI** | PNG `tEXt` (`prompt` & `workflow` JSON graphs) | `CLIPTextEncode` prompts, `KSampler` steps/CFG/seed/sampler/scheduler, `CheckpointLoader` / `UNETLoader`, `LoraLoader` |
| **Fooocus** | PNG `parameters` (A1111 or JSON) | Prompts, Model, Sampler, Seed, Steps |
| **SwarmUI** | PNG `sui_image_params` JSON | Prompts, Model, Seed, Steps, CFG, Sampler |
| **InvokeAI** | PNG `invokeai_metadata` / `sd-metadata` JSON | Prompts, Model, Seed, Steps, CFG |
| **Sidecars (Video/Image)** | Companion `.txt`, `.caption`, `.json` | Prompts and JSON generation parameters |
