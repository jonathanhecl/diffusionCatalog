package vision

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/jonathanhecl/diffusionCatalog/internal/domain"
)

// HTTPWorkerEngine implements domain.VisionEngine by delegating face detection, cropping,
// and 512-d embedding extraction to an external HTTP service (e.g. Python InsightFace/MediaPipe sidecar).
type HTTPWorkerEngine struct {
	endpoint string
	client   *http.Client
}

// NewHTTPWorkerEngine creates an HTTP-backed VisionEngine adapter.
func NewHTTPWorkerEngine(endpoint string) *HTTPWorkerEngine {
	return &HTTPWorkerEngine{
		endpoint: endpoint,
		client: &http.Client{
			Timeout: 30 * time.Second,
		},
	}
}

// Name returns the engine identifier.
func (h *HTTPWorkerEngine) Name() string {
	return "http-python-vision-worker"
}

type detectRequest struct {
	AssetID   int64  `json:"asset_id"`
	ImagePath string `json:"image_path"`
	CacheDir  string `json:"cache_dir"`
}

type detectResponse struct {
	Faces []domain.DetectedFace `json:"faces"`
}

// DetectAndCropFaces sends the image path and cache directory to the configured vision sidecar.
func (h *HTTPWorkerEngine) DetectAndCropFaces(ctx context.Context, assetID int64, absImagePath, cacheDir string) ([]domain.DetectedFace, error) {
	if h.endpoint == "" {
		return nil, nil
	}
	payload, err := json.Marshal(detectRequest{
		AssetID:   assetID,
		ImagePath: absImagePath,
		CacheDir:  cacheDir,
	})
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, h.endpoint+"/detect", bytes.NewReader(payload))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := h.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("vision worker returned status %d", resp.StatusCode)
	}

	var out detectResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, err
	}
	for i := range out.Faces {
		out.Faces[i].AssetID = assetID
	}
	return out.Faces, nil
}
