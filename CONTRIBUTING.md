# Contributing to DiffusionCatalog

Thank you for your interest in contributing to **DiffusionCatalog**!

We welcome pull requests, bug reports, feature suggestions, and documentation improvements.

## 🛠️ Development Setup

DiffusionCatalog is written in Go (1.22+) and relies on pure-Go dependencies (including `modernc.org/sqlite`), meaning **no C compiler (CGO) or external databases are required**.

### Prerequisites

- Go 1.22 or higher installed
- Git

### Running from Source

```bash
# Clone the repository
git clone https://github.com/jonathanhecl/diffusionCatalog.git
cd diffusionCatalog

# Run directly
go run ./cmd/catalogd
# or using the runner script
./run.sh
```

### Running Tests

Before submitting changes, make sure all unit and integration tests pass:

```bash
make test
# or
go test -v -count=1 ./...
```

Make sure your code is formatted and passes `go vet`:

```bash
make vet
go fmt ./...
```

## 🏗️ Architecture Overview

The codebase is organized following **Hexagonal / Clean Architecture**:

- `internal/domain`: Core domain models and port interfaces. Independent of external frameworks.
- `internal/usecase`: Application services (`CatalogService`, `ScannerService`, `AuthService`).
- `internal/adapter/metadata`: Pure-Go parsers for PNG chunks, EXIF, and sidecars (A1111, ComfyUI, Fooocus, InvokeAI, SwarmUI).
- `internal/adapter/storage/sqlite`: Pure-Go SQLite adapter with WAL mode and FTS5 full-text indexing.
- `internal/adapter/ollama`: Dual Ollama / OpenAI-compatible REST client for embeddings and prompt assistance.
- `internal/adapter/vision`: Pluggable face and subject detection engines (embedded YuNet + SFace sidecar).
- `internal/adapter/web`: Embedded web templates, styles, and HTTP handler.

Refer to [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed design specifications.

## 📝 Commit Conventions

We follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat: add new feature`
- `fix: resolve a bug`
- `refactor: optimize internal implementation without changing behavior`
- `test: add or update tests`
- `docs: update documentation or guides`
- `chore: maintenance, dependency updates`

Please keep commit messages concise, descriptive, and avoid AI attribution trailers.
