package usecase

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/jonathanhecl/diffusionCatalog/internal/domain"
)

// ScanReport summarizes the outcome of a non-destructive directory scan across all registered media roots.
type ScanReport struct {
	ScannedRoots int       `json:"scanned_roots"`
	SkippedRoots int       `json:"skipped_roots"`
	ScannedFiles int       `json:"scanned_files"`
	ImagesFound  int       `json:"images_found"`
	VideosFound  int       `json:"videos_found"`
	FacesFound   int       `json:"faces_found"`
	StartedAt    time.Time `json:"started_at"`
	CompletedAt  time.Time `json:"completed_at"`
}

// ScannerService walks the read-only media folder, extracts diffusion metadata, and indexes assets into SQLite.
type ScannerService struct {
	mediaDir  string
	cacheDir  string
	repo      domain.CatalogRepository
	extractor domain.MetadataExtractor
	vision    domain.VisionEngine

	mu         sync.Mutex
	lastReport ScanReport
}

// NewScannerService creates a new folder scanner service.
func NewScannerService(
	mediaDir, cacheDir string,
	repo domain.CatalogRepository,
	extractor domain.MetadataExtractor,
	vision domain.VisionEngine,
) *ScannerService {
	return &ScannerService{
		mediaDir:  mediaDir,
		cacheDir:  cacheDir,
		repo:      repo,
		extractor: extractor,
		vision:    vision,
	}
}

// LastReport returns the most recent scan report.
func (s *ScannerService) LastReport() ScanReport {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.lastReport
}

// Scan walks the configured media directory and indexes all supported image and video files.
func (s *ScannerService) Scan(ctx context.Context) (ScanReport, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	report := ScanReport{StartedAt: time.Now()}
	_ = os.MkdirAll(s.mediaDir, 0755)
	_ = os.MkdirAll(s.cacheDir, 0755)

	err := filepath.WalkDir(s.mediaDir, func(absPath string, d fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return nil
		}
		if d.IsDir() {
			if strings.HasPrefix(d.Name(), ".") {
				return filepath.SkipDir
			}
			return nil
		}

		mediaType, ok := classifyMediaExtension(filepath.Ext(d.Name()))
		if !ok {
			return nil
		}

		relPath, err := filepath.Rel(s.mediaDir, absPath)
		if err != nil {
			return nil
		}
		relPath = filepath.ToSlash(relPath)

		info, err := d.Info()
		if err != nil {
			return nil
		}

		hash, err := hashFilePrefix(absPath, info.Size())
		if err != nil {
			return nil
		}

		meta, width, height, _ := s.extractor.Extract(ctx, absPath, mediaType)

		asset := &domain.MediaAsset{
			RelPath:   relPath,
			FileName:  d.Name(),
			FileHash:  hash,
			MediaType: mediaType,
			Width:     width,
			Height:    height,
			FileSize:  info.Size(),
			Metadata:  meta,
		}

		assetID, err := s.repo.UpsertAsset(ctx, asset)
		if err != nil {
			return nil
		}

		report.ScannedFiles++
		if mediaType == domain.MediaTypeImage {
			report.ImagesFound++
		} else {
			report.VideosFound++
		}

		// Auto-attach subfolder name as an agnostic Group if nested inside mediaDir
		dirRel := filepath.Dir(relPath)
		if dirRel != "." && dirRel != "/" {
			parts := strings.Split(filepath.ToSlash(dirRel), "/")
			groupName := parts[0]
			if gid, err := s.repo.CreateGroup(ctx, &domain.Group{
				Name:        groupName,
				Description: "Auto-indexed from directory: " + groupName,
				Category:    "Folder",
			}); err == nil {
				_ = s.repo.AttachAssetToGroup(ctx, gid, assetID)
			}
		}

		// Auto-attach LoRAs as agnostic tags under facet "lora"
		for _, lora := range meta.LoRAs {
			if t, err := s.repo.UpsertTag(ctx, lora, "lora", "#7c3aed"); err == nil {
				_ = s.repo.AttachTagToAsset(ctx, assetID, t.ID)
			}
		}

		// Run pluggable vision engine for images if active
		if mediaType == domain.MediaTypeImage && s.vision != nil {
			faces, err := s.vision.DetectAndCropFaces(ctx, assetID, absPath, s.cacheDir)
			if err == nil && len(faces) > 0 {
				for i := range faces {
					_, _ = s.repo.SaveDetectedFace(ctx, &faces[i])
					report.FacesFound++
				}
			}
		}

		return nil
	})

	report.CompletedAt = time.Now()
	s.lastReport = report
	return report, err
}

func classifyMediaExtension(ext string) (domain.MediaType, bool) {
	switch strings.ToLower(ext) {
	case ".png", ".jpg", ".jpeg", ".webp":
		return domain.MediaTypeImage, true
	case ".mp4", ".webm", ".mov", ".gif":
		return domain.MediaTypeVideo, true
	default:
		return "", false
	}
}

func hashFilePrefix(absPath string, size int64) (string, error) {
	f, err := os.Open(absPath)
	if err != nil {
		return "", err
	}
	defer f.Close()

	h := sha256.New()
	_, _ = io.CopyN(h, f, 64*1024)
	return hex.EncodeToString(h.Sum(nil))[:16], nil
}
