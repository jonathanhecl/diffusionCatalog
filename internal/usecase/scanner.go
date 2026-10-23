package usecase

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"io"
	"io/fs"
	"log"
	"os"
	"path/filepath"
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
	ProgressPct    int        `json:"progress_pct"`
	TotalFiles     int        `json:"total_files"`
	ScannedFiles   int        `json:"scanned_files"`
	UnchangedFiles int        `json:"unchanged_files"`
	DeletedFiles   int        `json:"deleted_files"`
	CurrentFile    string     `json:"current_file"`
	CurrentRoot    string     `json:"current_root"`
	StartedAt      time.Time  `json:"started_at"`
	CompletedAt    time.Time  `json:"completed_at"`
	LastReport     ScanReport `json:"last_report"`
	LastError      string     `json:"last_error,omitempty"`
}

// ScannerService walks the read-only media folder, extracts diffusion metadata, and indexes assets into SQLite.
type ScannerService struct {
	mediaDir  string
	cacheDir  string
	repo      domain.CatalogRepository
	extractor domain.MetadataExtractor
	vision    domain.VisionEngine

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

// StartScanAsync starts the directory scan in a background goroutine if one is not already running.
func (s *ScannerService) StartScanAsync() bool {
	s.mu.Lock()
	if s.status.IsScanning {
		s.mu.Unlock()
		return false
	}
	s.status.IsScanning = true
	s.status.IsManual = true
	s.status.ProgressPct = 0
	s.status.ScannedFiles = 0
	s.status.DeletedFiles = 0
	s.status.TotalFiles = 0
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
		}
	}()
	return true
}

// Scan walks all registered and accessible media root folders, syncs their .diffusion-catalog.json manifests,
// and indexes all supported image and video files up to accessible subdirectories.
func (s *ScannerService) Scan(ctx context.Context) (ScanReport, error) {
	s.mu.Lock()
	s.status.IsScanning = true
	s.status.StartedAt = time.Now()
	s.status.ProgressPct = 0
	s.status.ScannedFiles = 0
	s.status.DeletedFiles = 0
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
			}
			return nil
		})
	}

	s.mu.Lock()
	s.status.TotalFiles = totalCandidateFiles
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

			var mEntry domain.ManifestAssetEntry
			var hasMEntry bool
			if manifest != nil {
				mEntry, hasMEntry = manifest.Assets[relToRoot]
			}

			// FAST PATH: Differential check against SQLite database
			// If file already exists in SQLite with identical size and modification time,
			// it is already indexed. We skip disk extraction, hashing, and database upserts completely.
			if existing, inDB := dbFingerprints[storedRelPath]; inDB {
				if existing.FileSize == fileSize && (existing.FileMTime.Unix() == mtimeUnix || existing.FileMTime.Equal(fileMTime)) {
					report.UnchangedFiles++
					s.mu.Lock()
					s.status.ScannedFiles++
					s.status.UnchangedFiles++
					s.status.CurrentFile = d.Name()
					s.status.CurrentRoot = root.Label
					if s.status.TotalFiles > 0 {
						s.status.ProgressPct = (s.status.ScannedFiles * 100) / s.status.TotalFiles
						if s.status.ProgressPct > 100 {
							s.status.ProgressPct = 100
						}
					}
					s.mu.Unlock()
					return nil
				}
			}

			// Preprocessed cache check:
			// If .diffusion-catalog.json already has preprocessed dimensions, hash, and metadata matching current file mtime/size,
			// use the cached metadata without having to re-parse the media file.
			var meta domain.GenerationParams
			var width, height int
			var hash string
			usedPreprocessedCache := false

			if hasMEntry && mEntry.FileSize == fileSize && mEntry.FileMTime == mtimeUnix && mEntry.FileHash != "" {
				width = mEntry.Width
				height = mEntry.Height
				hash = mEntry.FileHash
				if mEntry.Metadata != nil {
					meta = *mEntry.Metadata
				}
				usedPreprocessedCache = true
			} else {
				// File is new or changed: extract hash and metadata
				h, hErr := hashFilePrefix(absPath, fileSize)
				if hErr != nil {
					return nil
				}
				hash = h
				meta, width, height, _ = s.extractor.Extract(ctx, absPath, mediaType)
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
				FileSize:    fileSize,
				FileMTime:   fileMTime,
				Rating:      mEntry.Rating,
				Favorite:    mEntry.Favorite,
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

			s.mu.Lock()
			s.status.ScannedFiles++
			s.status.CurrentFile = d.Name()
			s.status.CurrentRoot = root.Label
			if s.status.TotalFiles > 0 {
				s.status.ProgressPct = (s.status.ScannedFiles * 100) / s.status.TotalFiles
				if s.status.ProgressPct > 100 {
					s.status.ProgressPct = 100
				}
			}
			scannedCount := s.status.ScannedFiles
			totalCount := s.status.TotalFiles
			pct := s.status.ProgressPct
			s.mu.Unlock()

			if usedPreprocessedCache {
				log.Printf("[SCAN] [%d/%d] (%d%%) Synced from manifest: %s (%s)", scannedCount, totalCount, pct, d.Name(), root.Label)
			} else {
				log.Printf("[SCAN] [%d/%d] (%d%%) Indexed %s (%s)", scannedCount, totalCount, pct, d.Name(), root.Label)
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
			}

			// Update / enrich manifest with preprocessed technical metadata so subsequent scans/re-indexing are instant
			if manifest != nil {
				metaCopy := meta
				mEntry.Category = category
				mEntry.Subcategory = subcategory
				mEntry.FileSize = fileSize
				mEntry.FileMTime = mtimeUnix
				mEntry.FileHash = hash
				mEntry.Width = width
				mEntry.Height = height
				mEntry.Metadata = &metaCopy
				manifest.Assets[relToRoot] = mEntry
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

		// Also clean up missing entries in .diffusion-catalog.json manifest
		if manifest != nil {
			for manifestRelPath := range manifest.Assets {
				if !seenRelToRoot[manifestRelPath] {
					fullCheckPath := filepath.Join(root.Path, filepath.FromSlash(manifestRelPath))
					if _, statErr := os.Stat(fullCheckPath); os.IsNotExist(statErr) {
						delete(manifest.Assets, manifestRelPath)
						manifestDirty = true
					}
				}
			}
		}

		if manifest != nil && manifestDirty {
			_ = SaveFolderManifest(root.Path, manifest)
		}
	}

	report.CompletedAt = time.Now()
	duration := report.CompletedAt.Sub(report.StartedAt).Round(time.Millisecond)

	s.mu.Lock()
	s.status.LastReport = report
	s.status.IsScanning = false
	s.status.IsManual = false
	s.status.CompletedAt = report.CompletedAt
	s.status.ProgressPct = 100
	s.mu.Unlock()

	if isManual || report.ScannedFiles > 0 || report.DeletedFiles > 0 {
		log.Printf("[SCAN] Completed scan in %v: %d indexed (%d images, %d videos), %d removed, %d unchanged across %d roots (%d skipped)",
			duration, report.ImagesFound+report.VideosFound, report.ImagesFound, report.VideosFound, report.DeletedFiles, report.UnchangedFiles, report.ScannedRoots, report.SkippedRoots)
	}
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
