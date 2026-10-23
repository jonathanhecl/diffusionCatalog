package vision

import (
	"context"

	"github.com/jonathanhecl/diffusionCatalog/internal/domain"
)

// NoopEngine implements domain.VisionEngine when no external Python/ONNX worker is configured.
type NoopEngine struct{}

// NewNoopEngine returns a no-op vision adapter.
func NewNoopEngine() *NoopEngine {
	return &NoopEngine{}
}

// Name returns the engine identifier.
func (n *NoopEngine) Name() string {
	return "noop-vision-adapter"
}

// DetectAndCropFaces returns no faces when external vision detection is disabled.
func (n *NoopEngine) DetectAndCropFaces(ctx context.Context, assetID int64, absImagePath, cacheDir string) ([]domain.DetectedFace, error) {
	return nil, nil
}
