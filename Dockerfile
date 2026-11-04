# Multi-stage Dockerfile for DiffusionCatalog
# ------------------------------------------------------------------------------
# 1. Builder Stage
# ------------------------------------------------------------------------------
FROM golang:1.24-alpine AS builder

WORKDIR /build

# Pre-fetch dependencies
COPY go.mod go.sum ./
RUN go mod download

# Copy source tree and compile static binary
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -trimpath -ldflags="-s -w" -o catalogd ./cmd/catalogd

# ------------------------------------------------------------------------------
# 2. Runtime Stage
# ------------------------------------------------------------------------------
FROM alpine:3.20

# Install runtime dependencies for video processing and SSL certificates
RUN apk add --no-cache ca-certificates tzdata ffmpeg

WORKDIR /app

# Copy compiled binary from builder stage
COPY --from=builder /build/catalogd /app/catalogd

# Create media mount and persistent data directories
RUN mkdir -p /media /data/.cache

EXPOSE 1080

VOLUME ["/media", "/data"]

ENV CATALOG_MEDIA_DIR=/media \
    CATALOG_DB_PATH=/data/catalog.db \
    CATALOG_CACHE_DIR=/data/.cache \
    CATALOG_ADDR=:1080

ENTRYPOINT ["/app/catalogd"]
