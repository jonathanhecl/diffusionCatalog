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

// Scan walks all registered and accessible media root folders, syncs their .diffusion-catalog.json manifests,
// and indexes all supported image and video files up to accessible subdirectories.
func (s *ScannerService) Scan(ctx context.Context) (ScanReport, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	report := ScanReport{StartedAt: time.Now()}
	_ = os.MkdirAll(s.mediaDir, 0755)
	_ = os.MkdirAll(s.cacheDir, 0755)

	// Ensure the primary configured mediaDir is registered in media_roots
	roots, _ := s.repo.ListMediaRoots(ctx)
	if len(roots) == 0 && strings.TrimSpace(s.mediaDir) != "" {
		_, _ = s.repo.UpsertMediaRoot(ctx, &domain.MediaRoot{
			Path:    s.mediaDir,
			Label:   filepath.Base(s.mediaDir),
			Enabled: true,
		})
		roots, _ = s.repo.ListMediaRoots(ctx)
	}

	defaultCats, _ := s.repo.ListDefaultCategories(ctx)

	for _, root := range roots {
		if !root.Enabled {
			continue
		}
		info, statErr := os.Stat(root.Path)
		if statErr != nil || !info.IsDir() {
			report.SkippedRoots++
			continue
		}

		report.ScannedRoots++
		manifest, _ := EnsureFolderManifest(root.Path, root.Label, defaultCats)
		manifestDirty := false

		// Register manifest categories as tags under their respective facets so sidebar filters reflect folder categories
		if manifest != nil {
			for _, cat := range manifest.Categories {
				facet := strings.TrimSpace(cat.Facet)
				if facet == "" {
					facet = "category"
				}
				_, _ = s.repo.UpsertTag(ctx, cat.Name, facet, cat.Color)
			}
		}

		_ = filepath.WalkDir(root.Path, func(absPath string, d fs.DirEntry, walkErr error) error {
			if walkErr != nil {
				// Gracefully skip directories or files where the OS user lacks read permission
				if d != nil && d.IsDir() {
					return filepath.SkipDir
				}
				return nil
			}
			if d.IsDir() {
				if absPath != root.Path && strings.HasPrefix(d.Name(), ".") {
					return filepath.SkipDir
				}
				return nil
			}

			mediaType, ok := classifyMediaExtension(filepath.Ext(d.Name()))
			if !ok {
				return nil
			}

			relToRoot, err := filepath.Rel(root.Path, absPath)
			if err != nil {
				return nil
			}
			relToRoot = filepath.ToSlash(relToRoot)

			// Preserve backward-compatible rel_path for root #1 (primary mediaDir); namespace secondary roots with @root_<id>/
			storedRelPath := relToRoot
			if root.ID > 1 && root.Path != s.mediaDir {
				storedRelPath = "@root_" + itoa64(root.ID) + "/" + relToRoot
			}

			fInfo, err := d.Info()
			if err != nil {
				return nil
			}

			hash, err := hashFilePrefix(absPath, fInfo.Size())
			if err != nil {
				return nil
			}

			meta, width, height, _ := s.extractor.Extract(ctx, absPath, mediaType)

			var mEntry domain.ManifestAssetEntry
			var hasMEntry bool
			if manifest != nil {
				mEntry, hasMEntry = manifest.Assets[relToRoot]
			}

			// Default category from top-level subfolder if not explicitly classified in manifest yet
			dirRel := filepath.Dir(relToRoot)
			subFolder := ""
			if dirRel != "." && dirRel != "/" {
				parts := strings.Split(filepath.ToSlash(dirRel), "/")
				subFolder = parts[0]
			}
			category := mEntry.Category
			subcategory := mEntry.Subcategory
			if category == "" && subFolder != "" {
				category = strings.ReplaceAll(subFolder, "_", " ")
			}

			asset := &domain.MediaAsset{
				RootID:      root.ID,
				RootPath:    root.Path,
				RootLabel:   root.Label,
				RelPath:     storedRelPath,
				FileName:    d.Name(),
				FileHash:    hash,
				MediaType:   mediaType,
				Width:       width,
				Height:      height,
				FileSize:    fInfo.Size(),
				Rating:      mEntry.Rating,
				Category:    category,
				Subcategory: subcategory,
				Metadata:    meta,
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

			// Auto-attach subfolder name as an agnostic Group if nested inside root.Path
			if subFolder != "" {
				if gid, err := s.repo.CreateGroup(ctx, &domain.Group{
					Name:        subFolder,
					Description: "Auto-indexed from directory: " + subFolder + " (" + root.Label + ")",
					Category:    "Folder",
				}); err == nil {
					_ = s.repo.AttachAssetToGroup(ctx, gid, assetID)
				}
			}

			// Sync groups, tags, and characters from the folder's .diffusion-catalog.json manifest
			if hasMEntry {
				for _, gName := range mEntry.Groups {
					if gid, err := s.repo.CreateGroup(ctx, &domain.Group{
						Name:     gName,
						Category: "Manifest",
					}); err == nil {
						_ = s.repo.AttachAssetToGroup(ctx, gid, assetID)
					}
				}
				for _, mt := range mEntry.Tags {
					if t, err := s.repo.UpsertTag(ctx, mt.Name, mt.Facet, mt.Color); err == nil {
						_ = s.repo.AttachTagToAsset(ctx, assetID, t.ID)
					}
				}
				for _, cName := range mEntry.Characters {
					if cid, err := s.repo.CreateCharacter(ctx, &domain.Character{Name: cName}); err == nil {
						_ = s.repo.AttachAssetToCharacter(ctx, cid, assetID)
					}
				}
			} else if manifest != nil {
				// Record newly discovered file in folder manifest
				manifest.Assets[relToRoot] = domain.ManifestAssetEntry{
					Category:    category,
					Subcategory: subcategory,
				}
				manifestDirty = true
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

		if manifest != nil && manifestDirty {
			_ = SaveFolderManifest(root.Path, manifest)
		}
	}

	report.CompletedAt = time.Now()
	s.lastReport = report
	return report, nil
}

func itoa64(v int64) string {
	if v == 0 {
		return "0"
	}
	var buf [20]byte
	i := len(buf)
	for v > 0 {
		i--
		buf[i] = byte('0' + v%10)
		v /= 10
	}
	return string(buf[i:])
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
