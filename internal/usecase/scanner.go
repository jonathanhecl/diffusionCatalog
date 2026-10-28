package usecase

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"io/fs"
	"log"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/jonathanhecl/diffusionCatalog/internal/domain"
)

// ScanReport summarizes the outcome of a non-destructive directory scan across all registered media roots.
type ScanReport struct {
	ScannedRoots   int       `json:"scanned_roots"`
	SkippedRoots   int       `json:"skipped_roots"`
	ScannedFiles   int       `json:"scanned_files"`
	UnchangedFiles int       `json:"unchanged_files"`
	DeletedFiles   int       `json:"deleted_files"`
	ImagesFound    int       `json:"images_found"`
	VideosFound    int       `json:"videos_found"`
	FacesFound     int       `json:"faces_found"`
	StartedAt      time.Time `json:"started_at"`
	CompletedAt    time.Time `json:"completed_at"`
}

// ScanStatus summarizes the real-time progress of an in-progress or completed media scan.
type ScanStatus struct {
	IsScanning     bool       `json:"is_scanning"`
	IsManual       bool       `json:"is_manual"`
	Phase          string     `json:"phase"` // "starting", "discovering", "indexing", "completed", "error"
	ProgressPct    int        `json:"progress_pct"`
	TotalFiles     int        `json:"total_files"`
	ScannedFiles   int        `json:"scanned_files"`
	UnchangedFiles int        `json:"unchanged_files"`
	DeletedFiles   int        `json:"deleted_files"`
	CurrentFile    string     `json:"current_file"`
	CurrentRoot    string     `json:"current_root"`
	StartedAt      time.Time  `json:"started_at"`
	CompletedAt    time.Time  `json:"completed_at"`
	Speed          float64    `json:"speed"`       // files per second
	EtaSeconds     int64      `json:"eta_seconds"` // remaining seconds
	EtaFormatted   string     `json:"eta_formatted"`
	LastReport     ScanReport `json:"last_report"`
	LastError      string     `json:"last_error,omitempty"`
}

func formatETA(seconds int64) string {
	if seconds <= 0 {
		return "< 5s"
	}
	if seconds < 60 {
		return fmt.Sprintf("%ds", seconds)
	}
	m := seconds / 60
	s := seconds % 60
	if m < 60 {
		if s == 0 {
			return fmt.Sprintf("%dm", m)
		}
		return fmt.Sprintf("%dm %02ds", m, s)
	}
	h := m / 60
	m = m % 60
	return fmt.Sprintf("%dh %02dm", h, m)
}

// ScannerService walks the read-only media folder, extracts diffusion metadata, and indexes assets into SQLite.
type ScannerService struct {
	mediaDir  string
	cacheDir  string
	repo      domain.CatalogRepository
	extractor domain.MetadataExtractor
	vision    domain.VisionEngine
	ollama    domain.OllamaProvider

	mu     sync.Mutex
	status ScanStatus
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

// SetOllamaProvider attaches an Ollama provider for optional multimodal vector embeddings.
func (s *ScannerService) SetOllamaProvider(p domain.OllamaProvider) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.ollama = p
}

// Status returns the current live scan status and progress percentage.
func (s *ScannerService) Status() ScanStatus {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.status
}

// IsScanning reports whether a media scan is currently in progress.
func (s *ScannerService) IsScanning() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.status.IsScanning
}

// LastReport returns the most recent scan report.
func (s *ScannerService) LastReport() ScanReport {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.status.LastReport
}

func (s *ScannerService) recordProgress(fileName, rootLabel string, isUnchanged bool, indexStartTime time.Time) (scanned, total, pct int) {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.status.ScannedFiles++
	if isUnchanged {
		s.status.UnchangedFiles++
	}
	s.status.CurrentFile = fileName
	s.status.CurrentRoot = rootLabel

	if s.status.TotalFiles > 0 {
		s.status.ProgressPct = (s.status.ScannedFiles * 100) / s.status.TotalFiles
		if s.status.ProgressPct > 100 {
			s.status.ProgressPct = 100
		}
	}

	elapsed := time.Since(indexStartTime).Seconds()
	if elapsed >= 0.2 && s.status.ScannedFiles > 0 {
		speed := float64(s.status.ScannedFiles) / elapsed
		s.status.Speed = speed
		remaining := s.status.TotalFiles - s.status.ScannedFiles
		if remaining > 0 && speed > 0 {
			s.status.EtaSeconds = int64(float64(remaining) / speed)
			s.status.EtaFormatted = formatETA(s.status.EtaSeconds)
		} else {
			s.status.EtaSeconds = 0
			s.status.EtaFormatted = "< 5s"
		}
	}

	return s.status.ScannedFiles, s.status.TotalFiles, s.status.ProgressPct
}

// StartScanAsync starts the directory scan in a background goroutine if one is not already running.
func (s *ScannerService) StartScanAsync() bool {
	s.mu.Lock()
	if s.status.IsScanning {
		s.mu.Unlock()
		return false
	}
	s.status.IsScanning = true
	s.status.IsManual = true
	s.status.Phase = "starting"
	s.status.ProgressPct = 0
	s.status.ScannedFiles = 0
	s.status.UnchangedFiles = 0
	s.status.DeletedFiles = 0
	s.status.TotalFiles = 0
	s.status.Speed = 0
	s.status.EtaSeconds = -1
	s.status.EtaFormatted = ""
	s.status.CurrentFile = ""
	s.status.CurrentRoot = ""
	s.status.StartedAt = time.Now()
	s.status.LastError = ""
	s.mu.Unlock()

	log.Printf("[SCAN] Manual folder scan started")
	go func() {
		_, err := s.Scan(context.Background())
		if err != nil {
			log.Printf("[SCAN] Manual folder scan error: %v", err)
			s.mu.Lock()
			s.status.IsScanning = false
			s.status.Phase = "error"
			s.status.LastError = err.Error()
			s.mu.Unlock()
		}
	}()
	return true
}

// Scan walks all registered and accessible media root folders and indexes all supported image and video files up to accessible subdirectories.
func (s *ScannerService) Scan(ctx context.Context) (ScanReport, error) {
	s.mu.Lock()
	s.status.IsScanning = true
	s.status.Phase = "discovering"
	s.status.StartedAt = time.Now()
	s.status.ProgressPct = 0
	s.status.ScannedFiles = 0
	s.status.UnchangedFiles = 0
	s.status.DeletedFiles = 0
	s.status.TotalFiles = 0
	s.status.Speed = 0
	s.status.EtaSeconds = -1
	s.status.EtaFormatted = ""
	s.status.CurrentFile = ""
	s.status.CurrentRoot = ""
	s.status.LastError = ""
	isManual := s.status.IsManual
	s.mu.Unlock()

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

	// Check multimodal embedding settings
	appSettings, _ := s.repo.GetSettings(ctx)
	s.mu.Lock()
	hasOllama := s.ollama != nil
	s.mu.Unlock()
	embeddingActive := appSettings.EmbeddingEnabled && strings.TrimSpace(appSettings.EmbeddingModel) != "" && hasOllama
	activeModel := strings.TrimSpace(appSettings.EmbeddingModel)
	activeProvider := appSettings.EmbeddingProvider
	if activeProvider == "" {
		activeProvider = domain.ProviderOllama
	}
	embeddingBase := strings.TrimSpace(appSettings.OllamaBaseURL)
	embeddingKey := ""
	if activeProvider == domain.ProviderOpenAI || activeProvider == domain.ProviderOMLX {
		embeddingBase = strings.TrimSpace(appSettings.OpenAIBaseURL)
		if embeddingBase == "" {
			embeddingBase = strings.TrimSpace(appSettings.OMLXBaseURL)
		}
		if embeddingBase == "" {
			embeddingBase = "http://127.0.0.1:8000"
		}
		embeddingKey = strings.TrimSpace(appSettings.OpenAIAPIKey)
		if embeddingKey == "" {
			embeddingKey = strings.TrimSpace(appSettings.OMLXAPIKey)
		}
	} else {
		if embeddingBase == "" {
			embeddingBase = "http://127.0.0.1:11434"
		}
	}
	if embeddingActive {
		_ = s.repo.InvalidateEmbeddingsExcept(ctx, activeModel)
	}

	// Phase 1: Rapid discovery count of candidate files across enabled roots to calculate %
	totalCandidateFiles := 0
	for _, root := range roots {
		if !root.Enabled {
			continue
		}
		info, statErr := os.Stat(root.Path)
		if statErr != nil || !info.IsDir() {
			continue
		}
		s.mu.Lock()
		s.status.CurrentRoot = root.Label
		s.mu.Unlock()

		_ = filepath.WalkDir(root.Path, func(absPath string, d fs.DirEntry, walkErr error) error {
			if walkErr != nil {
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
			if _, ok := classifyMediaExtension(filepath.Ext(d.Name())); ok {
				totalCandidateFiles++
				if totalCandidateFiles%20 == 0 {
					s.mu.Lock()
					s.status.TotalFiles = totalCandidateFiles
					s.mu.Unlock()
				}
			}
			return nil
		})
	}

	indexStartTime := time.Now()
	s.mu.Lock()
	s.status.TotalFiles = totalCandidateFiles
	s.status.Phase = "indexing"
	s.mu.Unlock()

	if isManual {
		log.Printf("[SCAN] Discovered %d candidate media file(s) across %d registered root(s)", totalCandidateFiles, len(roots))
	}

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

		// Register default categories as tags under their respective facets so sidebar filters reflect categories
		for _, cat := range defaultCats {
			facet := strings.TrimSpace(cat.Facet)
			if facet == "" {
				facet = "category"
			}
			_, _ = s.repo.UpsertTag(ctx, cat.Name, facet, cat.Color)
		}

		// Load existing asset fingerprints from database for this root
		dbFingerprints, _ := s.repo.GetExistingAssetFingerprints(ctx, root.ID)
		if dbFingerprints == nil {
			dbFingerprints = make(map[string]domain.MediaAsset)
		}

		seenRelToRoot := make(map[string]bool)
		seenStoredRelPaths := make(map[string]bool)

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

			seenRelToRoot[relToRoot] = true
			seenStoredRelPaths[storedRelPath] = true

			fInfo, err := d.Info()
			if err != nil {
				return nil
			}

			fileSize := fInfo.Size()
			fileMTime := fInfo.ModTime().UTC()
			mtimeUnix := fileMTime.Unix()

			// FAST PATH: Differential check against SQLite database
			// If file already exists in SQLite with identical size and modification time,
			// it is already indexed. We skip disk extraction, hashing, and database upserts completely.
			existing, inDB := dbFingerprints[storedRelPath]
			if inDB {
				if existing.FileSize == fileSize && (existing.FileMTime.Unix() == mtimeUnix || existing.FileMTime.Equal(fileMTime)) {
					report.UnchangedFiles++
					s.recordProgress(d.Name(), root.Label, true, indexStartTime)
					if embeddingActive {
						s.processAssetEmbedding(ctx, existing.ID, absPath, mediaType, activeProvider, activeModel, embeddingBase, embeddingKey)
					}
					return nil
				}
			}

			// File is new or changed: extract hash and metadata
			hash, hErr := hashFilePrefix(absPath, fileSize)
			if hErr != nil {
				return nil
			}
			meta, width, height, _ := s.extractor.Extract(ctx, absPath, mediaType)

			// Default category from top-level subfolder if not already classified in database
			dirRel := filepath.Dir(relToRoot)
			subFolder := ""
			if dirRel != "." && dirRel != "/" {
				parts := strings.Split(filepath.ToSlash(dirRel), "/")
				subFolder = parts[0]
			}
			category := ""
			subcategory := ""
			rating := 0
			favorite := false
			if inDB {
				category = existing.Category
				subcategory = existing.Subcategory
				rating = existing.Rating
				favorite = existing.Favorite
			} else if subFolder != "" {
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
				FileSize:    fileSize,
				FileMTime:   fileMTime,
				Rating:      rating,
				Favorite:    favorite,
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

			scannedCount, totalCount, pct := s.recordProgress(d.Name(), root.Label, false, indexStartTime)

			log.Printf("[SCAN] [%d/%d] (%d%%) Indexed %s (%s)", scannedCount, totalCount, pct, d.Name(), root.Label)

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

			// Auto-attach LoRAs as agnostic tags under facet "lora"
			for _, lora := range meta.LoRAs {
				if t, err := s.repo.UpsertTag(ctx, lora, "lora", "#7c3aed"); err == nil {
					_ = s.repo.AttachTagToAsset(ctx, assetID, t.ID)
				}
			}

			// Run pluggable vision engine for images if active
			if mediaType == domain.MediaTypeImage && s.vision != nil {
				faces, err := s.vision.DetectAndCropFaces(ctx, assetID, absPath, s.cacheDir)
				if err == nil {
					for i := range faces {
						_, _ = s.repo.SaveDetectedFace(ctx, &faces[i])
						report.FacesFound++
					}
					_ = s.repo.MarkAssetFacesScanned(ctx, assetID, time.Now())
				}
			}

			// Run multimodal embedding if enabled
			if embeddingActive {
				s.processAssetEmbedding(ctx, assetID, absPath, mediaType, activeProvider, activeModel, embeddingBase, embeddingKey)
			}

			return nil
		})

		// Detect and remove assets that were deleted from disk
		var toDeleteIDs []int64
		for storedPath, existing := range dbFingerprints {
			if !seenStoredRelPaths[storedPath] {
				// Safety check: verify file is genuinely missing on disk before deleting
				relClean := normalizeManifestRelPath(root.Path, existing.RelPath)
				fullCheckPath := filepath.Join(root.Path, filepath.FromSlash(relClean))
				if _, statErr := os.Stat(fullCheckPath); os.IsNotExist(statErr) {
					toDeleteIDs = append(toDeleteIDs, existing.ID)
				}
			}
		}

		if len(toDeleteIDs) > 0 {
			if err := s.repo.DeleteAssetsByIDs(ctx, toDeleteIDs); err != nil {
				log.Printf("[SCAN] Error removing %d deleted assets from database: %v", len(toDeleteIDs), err)
			} else {
				report.DeletedFiles += len(toDeleteIDs)
				s.mu.Lock()
				s.status.DeletedFiles += len(toDeleteIDs)
				s.mu.Unlock()
				log.Printf("[SCAN] Cleaned up %d deleted media file(s) from catalog in root %s", len(toDeleteIDs), root.Label)
			}
		}
	}

	report.CompletedAt = time.Now()
	duration := report.CompletedAt.Sub(report.StartedAt).Round(time.Millisecond)

	s.mu.Lock()
	s.status.LastReport = report
	s.status.IsScanning = false
	s.status.IsManual = false
	s.status.Phase = "completed"
	s.status.CompletedAt = report.CompletedAt
	s.status.ProgressPct = 100
	s.status.EtaSeconds = 0
	s.status.EtaFormatted = "0s"
	s.status.CurrentFile = ""
	s.mu.Unlock()

	if isManual || report.ScannedFiles > 0 || report.DeletedFiles > 0 {
		log.Printf("[SCAN] Completed scan in %v: %d indexed (%d images, %d videos), %d removed, %d unchanged across %d roots (%d skipped)",
			duration, report.ImagesFound+report.VideosFound, report.ImagesFound, report.VideosFound, report.DeletedFiles, report.UnchangedFiles, report.ScannedRoots, report.SkippedRoots)
	}
	return report, nil
}

func (s *ScannerService) processAssetEmbedding(ctx context.Context, assetID int64, absPath string, mediaType domain.MediaType, provider domain.AIProviderType, activeModel, baseURL, apiKey string) {
	if mediaType != domain.MediaTypeImage || activeModel == "" {
		return
	}
	s.mu.Lock()
	ollama := s.ollama
	s.mu.Unlock()
	if ollama == nil {
		return
	}

	has, err := s.repo.HasAssetEmbedding(ctx, assetID, activeModel)
	if err == nil && has {
		return
	}

	data, err := os.ReadFile(absPath)
	if err != nil || len(data) == 0 {
		return
	}

	vec, err := ollama.EmbedImage(ctx, provider, baseURL, apiKey, activeModel, data)
	if err != nil {
		log.Printf("[SCAN] Note: could not generate embedding for asset %d (%s) with model %s: %v", assetID, filepath.Base(absPath), activeModel, err)
		return
	}
	if len(vec) > 0 {
		_ = s.repo.SaveAssetEmbedding(ctx, assetID, activeModel, vec)
		log.Printf("[SCAN] Generated embedding for asset %d (%s) with model %s", assetID, filepath.Base(absPath), activeModel)
	}
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

func resolveAssetAbsolutePath(asset domain.MediaAsset, roots []domain.MediaRoot) string {
	cleanRel := normalizeManifestRelPath("", asset.RelPath)
	if asset.RootPath != "" {
		p := filepath.Join(asset.RootPath, filepath.FromSlash(cleanRel))
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}
	for _, r := range roots {
		if r.ID == asset.RootID {
			p := filepath.Join(r.Path, filepath.FromSlash(cleanRel))
			if _, err := os.Stat(p); err == nil {
				return p
			}
		}
	}
	for _, r := range roots {
		p := filepath.Join(r.Path, filepath.FromSlash(cleanRel))
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}
	return ""
}

// StartReextractMissingMetadataAsync re-scans media files that have no extracted prompts or metadata.
func (s *ScannerService) StartReextractMissingMetadataAsync() bool {
	s.mu.Lock()
	if s.status.IsScanning {
		s.mu.Unlock()
		return false
	}
	s.status.IsScanning = true
	s.status.IsManual = true
	s.status.Phase = "reextracting"
	s.status.ProgressPct = 0
	s.status.ScannedFiles = 0
	s.status.UnchangedFiles = 0
	s.status.DeletedFiles = 0
	s.status.TotalFiles = 0
	s.status.Speed = 0
	s.status.EtaSeconds = -1
	s.status.EtaFormatted = ""
	s.status.CurrentFile = ""
	s.status.CurrentRoot = ""
	s.status.StartedAt = time.Now()
	s.status.LastError = ""
	s.mu.Unlock()

	log.Printf("[METADATA_RESCAN] Re-extracting missing metadata started")
	go func() {
		ctx := context.Background()
		roots, _ := s.repo.ListMediaRoots(ctx)
		missing, err := s.repo.ListAssetsMissingMetadata(ctx)
		if err != nil {
			s.mu.Lock()
			s.status.IsScanning = false
			s.status.Phase = "error"
			s.status.LastError = err.Error()
			s.mu.Unlock()
			return
		}

		s.mu.Lock()
		s.status.TotalFiles = len(missing)
		s.mu.Unlock()

		startTime := time.Now()
		updatedCount := 0

		for _, asset := range missing {
			absPath := resolveAssetAbsolutePath(asset, roots)
			if absPath == "" {
				s.recordProgress(asset.FileName, asset.RootLabel, true, startTime)
				continue
			}

			meta, _, _, err := s.extractor.Extract(ctx, absPath, asset.MediaType)
			now := time.Now()
			if err == nil {
				_ = s.repo.UpdateAssetMetadata(ctx, asset.ID, meta, now)
				for _, lora := range meta.LoRAs {
					if t, tErr := s.repo.UpsertTag(ctx, lora, "lora", "#7c3aed"); tErr == nil {
						_ = s.repo.AttachTagToAsset(ctx, asset.ID, t.ID)
					}
				}
				updatedCount++
			} else {
				_ = s.repo.UpdateAssetMetadata(ctx, asset.ID, asset.Metadata, now)
			}
			s.recordProgress(asset.FileName, asset.RootLabel, false, startTime)
		}

		s.mu.Lock()
		s.status.IsScanning = false
		s.status.Phase = "completed"
		s.status.CompletedAt = time.Now()
		s.status.ProgressPct = 100
		s.status.EtaSeconds = 0
		s.status.EtaFormatted = "0s"
		s.status.CurrentFile = ""
		s.mu.Unlock()
		log.Printf("[METADATA_RESCAN] Completed re-extracting metadata for %d assets (%d updated)", len(missing), updatedCount)
	}()
	return true
}

// StartFaceScanAsync scans image assets to detect faces and compute 128-d embeddings, then clusters them.
func (s *ScannerService) StartFaceScanAsync() bool {
	s.mu.Lock()
	if s.status.IsScanning {
		s.mu.Unlock()
		return false
	}
	s.status.IsScanning = true
	s.status.IsManual = true
	s.status.Phase = "scanning_faces"
	s.status.ProgressPct = 0
	s.status.ScannedFiles = 0
	s.status.UnchangedFiles = 0
	s.status.DeletedFiles = 0
	s.status.TotalFiles = 0
	s.status.Speed = 0
	s.status.EtaSeconds = -1
	s.status.EtaFormatted = ""
	s.status.CurrentFile = ""
	s.status.CurrentRoot = ""
	s.status.StartedAt = time.Now()
	s.status.LastError = ""
	s.mu.Unlock()

	log.Printf("[FACE_SCAN] Face extraction scan started")
	go func() {
		ctx := context.Background()
		roots, _ := s.repo.ListMediaRoots(ctx)
		images, err := s.repo.ListImageAssetsMissingFaces(ctx)
		if err != nil {
			s.mu.Lock()
			s.status.IsScanning = false
			s.status.Phase = "error"
			s.status.LastError = err.Error()
			s.mu.Unlock()
			return
		}

		s.mu.Lock()
		s.status.TotalFiles = len(images)
		s.mu.Unlock()

		startTime := time.Now()
		facesFound := 0

		for _, img := range images {
			absPath := resolveAssetAbsolutePath(img, roots)
			if absPath == "" || s.vision == nil {
				s.recordProgress(img.FileName, img.RootLabel, true, startTime)
				continue
			}

			faces, err := s.vision.DetectAndCropFaces(ctx, img.ID, absPath, s.cacheDir)
			if err == nil {
				if len(faces) > 0 {
					_ = s.repo.DeleteFacesForAsset(ctx, img.ID)
					for i := range faces {
						_, _ = s.repo.SaveDetectedFace(ctx, &faces[i])
						facesFound++
					}
				}
				_ = s.repo.MarkAssetFacesScanned(ctx, img.ID, time.Now())
			}
			s.recordProgress(img.FileName, img.RootLabel, false, startTime)
		}

		// After extracting all faces, cluster them by similarity >= 50%
		_, _ = s.ClusterDetectedFaces(ctx, 0.50)

		s.mu.Lock()
		s.status.IsScanning = false
		s.status.Phase = "completed"
		s.status.CompletedAt = time.Now()
		s.status.ProgressPct = 100
		s.status.EtaSeconds = 0
		s.status.EtaFormatted = "0s"
		s.status.CurrentFile = ""
		s.mu.Unlock()
		log.Printf("[FACE_SCAN] Completed face extraction: %d faces found and clustered across %d images", facesFound, len(images))
	}()
	return true
}

// ClusterDetectedFaces groups faces with cosine similarity >= similarityThreshold into clusters.
// Uses exemplar-based leader clustering (best face first) to prevent transitive chaining.
func (s *ScannerService) ClusterDetectedFaces(ctx context.Context, similarityThreshold float64) (int, error) {
	if similarityThreshold <= 0 {
		similarityThreshold = 0.45
	}
	faces, err := s.repo.ListFacesWithEmbeddings(ctx)
	if err != nil {
		return 0, err
	}
	if len(faces) == 0 {
		return 0, nil
	}

	// Sort faces by quality (area * confidence descending) so highest quality crops become cluster exemplars
	type faceWithIdx struct {
		face domain.DetectedFace
		idx  int
	}
	sorted := make([]faceWithIdx, len(faces))
	for i := range faces {
		sorted[i] = faceWithIdx{face: faces[i], idx: i}
	}
	sort.Slice(sorted, func(i, j int) bool {
		q1 := float64(sorted[i].face.Box.Width*sorted[i].face.Box.Height) * sorted[i].face.Confidence
		q2 := float64(sorted[j].face.Box.Width*sorted[j].face.Box.Height) * sorted[j].face.Confidence
		return q1 > q2
	})

	type cluster struct {
		exemplarEmbedding []float32
		faceIndices       []int
	}

	var clusters []cluster
	for _, item := range sorted {
		bestSim := -1.0
		bestClusterIdx := -1
		for cIdx, c := range clusters {
			sim := cosineSimilarity(item.face.Embedding, c.exemplarEmbedding)
			if sim > bestSim {
				bestSim = sim
				bestClusterIdx = cIdx
			}
		}

		if bestSim >= similarityThreshold && bestClusterIdx >= 0 {
			clusters[bestClusterIdx].faceIndices = append(clusters[bestClusterIdx].faceIndices, item.idx)
		} else {
			clusters = append(clusters, cluster{
				exemplarEmbedding: item.face.Embedding,
				faceIndices:       []int{item.idx},
			})
		}
	}

	// Sort clusters by size (largest cluster gets Person #1)
	sort.Slice(clusters, func(i, j int) bool {
		return len(clusters[i].faceIndices) > len(clusters[j].faceIndices)
	})

	clusterCount := 0
	for idx, c := range clusters {
		clusterCount++
		// Determine label: if any face in the cluster already has a user-defined name, keep it
		chosenLabel := ""
		var linkedCharID *int64
		for _, faceIdx := range c.faceIndices {
			lbl := strings.TrimSpace(faces[faceIdx].ClusterLabel)
			if lbl != "" && lbl != "Unassigned" && lbl != "Detected Subject" && !strings.HasPrefix(lbl, "Person #") && !strings.HasPrefix(lbl, "Cluster #") {
				chosenLabel = lbl
				if faces[faceIdx].CharacterID != nil {
					linkedCharID = faces[faceIdx].CharacterID
				}
				break
			}
		}
		if chosenLabel == "" {
			chosenLabel = fmt.Sprintf("Person #%d", idx+1)
		}

		for _, faceIdx := range c.faceIndices {
			f := faces[faceIdx]
			_ = s.repo.UpdateFaceCluster(ctx, f.ID, chosenLabel, linkedCharID)
		}
	}

	return clusterCount, nil
}

func cosineSimilarity(a, b []float32) float64 {
	if len(a) == 0 || len(a) != len(b) {
		return 0
	}
	var dot, normA, normB float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		normA += float64(a[i]) * float64(a[i])
		normB += float64(b[i]) * float64(b[i])
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}

// ScanAssetFaces detects and crops faces for a single asset, saving them and updating clusters.
func (s *ScannerService) ScanAssetFaces(ctx context.Context, assetID int64) (int, error) {
	if s.vision == nil {
		return 0, fmt.Errorf("vision engine not configured")
	}
	asset, err := s.repo.GetAssetByID(ctx, assetID)
	if err != nil || asset == nil {
		return 0, fmt.Errorf("asset not found: %w", err)
	}
	roots, _ := s.repo.ListMediaRoots(ctx)
	absPath := resolveAssetAbsolutePath(*asset, roots)
	if absPath == "" {
		return 0, fmt.Errorf("unable to resolve absolute path for asset %d (%s)", assetID, asset.RelPath)
	}

	faces, err := s.vision.DetectAndCropFaces(ctx, asset.ID, absPath, s.cacheDir)
	if err != nil {
		return 0, err
	}
	_ = s.repo.DeleteFacesForAsset(ctx, asset.ID)
	for i := range faces {
		_, _ = s.repo.SaveDetectedFace(ctx, &faces[i])
	}
	_ = s.repo.MarkAssetFacesScanned(ctx, asset.ID, time.Now())
	_, _ = s.ClusterDetectedFaces(ctx, 0.50)
	return len(faces), nil
}
