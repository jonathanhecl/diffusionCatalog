package usecase

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"io"
	"log"
	"math"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/jonathanhecl/diffusionCatalog/internal/adapter/metadata"
	"github.com/jonathanhecl/diffusionCatalog/internal/domain"
)

// CatalogService coordinates catalog querying, agnostic tagging, grouping, character sheets,
// multi-root media folders, per-folder .diffusion-catalog.json manifests, runtime DB switching,
// and Ollama / oMLX prompt generation (Vision, Image2Prompt, Text2Prompt).
type CatalogService struct {
	mediaDir string
	repo     domain.CatalogRepository
	ollama   domain.OllamaProvider

	aiStatusMu    sync.RWMutex
	aiRefreshMu   sync.Mutex
	aiStatusCache AIBackendStatus
	aiStatusTime  time.Time
}

// NewCatalogService creates a new CatalogService.
func NewCatalogService(mediaDir string, repo domain.CatalogRepository, ollama domain.OllamaProvider) *CatalogService {
	s := &CatalogService{
		mediaDir: mediaDir,
		repo:     repo,
		ollama:   ollama,
	}
	if ollama != nil && repo != nil {
		go func() {
			bgCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
			defer cancel()
			_ = s.InspectAIBackendsFresh(bgCtx)
		}()
	}
	return s
}

// CurrentDBPath returns the active global SQLite database path.
func (s *CatalogService) CurrentDBPath() string {
	return s.repo.CurrentDBPath()
}

// SwitchDatabase switches the active global SQLite database file at runtime.
func (s *CatalogService) SwitchDatabase(ctx context.Context, newDBPath string) error {
	if err := s.repo.SwitchDB(ctx, newDBPath); err != nil {
		return err
	}
	roots, _ := s.repo.ListMediaRoots(ctx)
	if len(roots) == 0 && strings.TrimSpace(s.mediaDir) != "" {
		_, _ = s.repo.UpsertMediaRoot(ctx, &domain.MediaRoot{
			Path:    s.mediaDir,
			Label:   filepath.Base(s.mediaDir),
			Enabled: true,
		})
	}
	return nil
}

// ListMediaRoots returns all registered media root folders enriched with their default categories.
func (s *CatalogService) ListMediaRoots(ctx context.Context) ([]domain.MediaRoot, error) {
	roots, err := s.repo.ListMediaRoots(ctx)
	if err != nil {
		return nil, err
	}
	defaults, _ := s.repo.ListDefaultCategories(ctx)
	for i := range roots {
		roots[i].Categories = defaults
	}
	return roots, nil
}

// AddMediaRoot registers a new media root folder (local directory or external drive).
func (s *CatalogService) AddMediaRoot(ctx context.Context, path, label string) (int64, error) {
	root := &domain.MediaRoot{
		Path:    strings.TrimSpace(path),
		Label:   strings.TrimSpace(label),
		Enabled: true,
	}
	return s.repo.UpsertMediaRoot(ctx, root)
}

// DeleteMediaRoot removes a media root folder from the catalog index without deleting any files on disk.
func (s *CatalogService) DeleteMediaRoot(ctx context.Context, id int64) error {
	return s.repo.DeleteMediaRoot(ctx, id)
}

// ListDefaultCategories returns the global default agnostic categories and subcategories.
func (s *CatalogService) ListDefaultCategories(ctx context.Context) ([]domain.DefaultCategory, error) {
	return s.repo.ListDefaultCategories(ctx)
}

// UpsertDefaultCategory creates or updates a global default category and its subcategories.
func (s *CatalogService) UpsertDefaultCategory(ctx context.Context, name, facet, color, subcategoriesCSV string) (int64, error) {
	var subs []string
	for _, part := range strings.Split(subcategoriesCSV, ",") {
		p := strings.TrimSpace(part)
		if p != "" {
			subs = append(subs, p)
		}
	}
	cat := &domain.DefaultCategory{
		Name:          strings.TrimSpace(name),
		Facet:         strings.TrimSpace(facet),
		Color:         strings.TrimSpace(color),
		Subcategories: subs,
	}
	id, err := s.repo.UpsertDefaultCategory(ctx, cat)
	if err != nil {
		return 0, err
	}
	return id, nil
}

// DeleteDefaultCategory removes a global default category by ID.
func (s *CatalogService) DeleteDefaultCategory(ctx context.Context, id int64) error {
	return s.repo.DeleteDefaultCategory(ctx, id)
}

// ResolveAssetAbsPath resolves the absolute filesystem path of an asset across any registered MediaRoot.
func (s *CatalogService) ResolveAssetAbsPath(ctx context.Context, asset *domain.MediaAsset) string {
	if asset == nil {
		return ""
	}
	rootPath := strings.TrimSpace(asset.RootPath)
	if rootPath == "" && asset.RootID > 0 {
		if mr, err := s.repo.GetMediaRootByID(ctx, asset.RootID); err == nil && mr != nil {
			rootPath = mr.Path
		}
	}
	if rootPath == "" {
		rootPath = s.mediaDir
	}
	cleanRel := normalizeManifestRelPath(rootPath, asset.RelPath)
	return filepath.Join(rootPath, filepath.FromSlash(cleanRel))
}

// Query returns a paginated, facet-filtered catalog page.
func (s *CatalogService) Query(ctx context.Context, filter domain.CatalogFilter) (*domain.CatalogPage, error) {
	if strings.TrimSpace(filter.Query) != "" && s.ollama != nil {
		if settings, err := s.repo.GetSettings(ctx); err == nil && settings.EmbeddingEnabled && strings.TrimSpace(settings.EmbeddingModel) != "" {
			prov := settings.EmbeddingProvider
			if prov == "" {
				prov = domain.ProviderOllama
			}
			baseURL := strings.TrimSpace(settings.OllamaBaseURL)
			apiKey := ""
			if prov == domain.ProviderOpenAI || prov == domain.ProviderOMLX {
				baseURL = strings.TrimSpace(settings.OpenAIBaseURL)
				if baseURL == "" {
					baseURL = strings.TrimSpace(settings.OMLXBaseURL)
				}
				if baseURL == "" {
					baseURL = "http://127.0.0.1:8000"
				}
				apiKey = strings.TrimSpace(settings.OpenAIAPIKey)
				if apiKey == "" {
					apiKey = strings.TrimSpace(settings.OMLXAPIKey)
				}
			} else {
				if baseURL == "" {
					baseURL = "http://127.0.0.1:11434"
				}
			}
			queryVec, err := s.ollama.EmbedText(ctx, prov, baseURL, apiKey, settings.EmbeddingModel, filter.Query)
			if err != nil {
				log.Printf("[CATALOG] Embedding query error for %q: %v", filter.Query, err)
			} else if len(queryVec) > 0 {
				if similarIDs, err := s.repo.FindSimilarAssetIDs(ctx, queryVec, settings.EmbeddingModel, 0.20, 50); err == nil && len(similarIDs) > 0 {
					filter.EmbeddingAssetIDs = similarIDs
					log.Printf("[CATALOG] Semantic search for %q with %s matched %d asset(s)", filter.Query, settings.EmbeddingModel, len(similarIDs))
				} else if err != nil {
					log.Printf("[CATALOG] FindSimilarAssetIDs error for %q: %v", filter.Query, err)
				}
			}
		}
	}

	page, err := s.repo.QueryCatalog(ctx, filter)
	if err != nil {
		return nil, err
	}
	defaults, _ := s.repo.ListDefaultCategories(ctx)
	for i := range page.MediaRoots {
		page.MediaRoots[i].Categories = defaults
	}
	return page, nil
}

// GetAsset fetches an asset by ID and increments its view count.
func (s *CatalogService) GetAsset(ctx context.Context, id int64) (*domain.MediaAsset, error) {
	_ = s.repo.IncrementAssetView(ctx, id)
	return s.repo.GetAssetByID(ctx, id)
}

// DeleteAsset deletes a media file from disk (including sidecars), removes it from folder manifest,
// and deletes the asset record from the repository.
func (s *CatalogService) DeleteAsset(ctx context.Context, id int64) error {
	asset, err := s.repo.GetAssetByID(ctx, id)
	if err != nil {
		return err
	}
	if asset == nil {
		return fmt.Errorf("asset %d not found", id)
	}

	rootPath := s.mediaDir
	if asset.RootPath != "" {
		rootPath = asset.RootPath
	} else if asset.RootID > 0 {
		if roots, err := s.repo.ListMediaRoots(ctx); err == nil {
			for _, r := range roots {
				if r.ID == asset.RootID && r.Path != "" {
					rootPath = r.Path
					break
				}
			}
		}
	}

	cleanRel := normalizeManifestRelPath(rootPath, asset.RelPath)
	fullPath := filepath.Join(rootPath, filepath.FromSlash(cleanRel))

	// 1. Remove from disk if file exists
	if _, err := os.Stat(fullPath); err == nil {
		_ = os.Remove(fullPath)
	}

	// 2. Remove companion sidecars (.txt, .caption, .json)
	base := strings.TrimSuffix(fullPath, filepath.Ext(fullPath))
	for _, ext := range []string{".txt", ".caption", ".json"} {
		sidecar := base + ext
		if sidecar != fullPath {
			_ = os.Remove(sidecar)
		}
	}

	// 3. Remove from folder manifest if it was tracked
	_ = RemoveAssetFromFolderManifest(rootPath, asset.RelPath)

	// 4. Delete from database (cascades to tags, groups, characters, faces, fts)
	return s.repo.DeleteAsset(ctx, id)
}

// GetAdjacentAssetIDs returns the previous and next asset IDs relative to the specified asset ID,
// respecting the specified filter criteria and sort order.
func (s *CatalogService) GetAdjacentAssetIDs(ctx context.Context, id int64, filter domain.CatalogFilter) (int64, int64, error) {
	return s.repo.GetAdjacentAssetIDs(ctx, id, filter)
}

// RelatedAssetsBundle holds related assets sharing the same Category, Character, or Group as a selected asset.
type RelatedAssetsBundle struct {
	SameCategory  []domain.MediaAsset
	SameCharacter []domain.MediaAsset
	SameGroup     []domain.MediaAsset
	SameModel     []domain.MediaAsset
}

// GetRelatedAssets finds other cataloged assets that share the same Category, Character, Group, or Checkpoint Model.
func (s *CatalogService) GetRelatedAssets(ctx context.Context, asset *domain.MediaAsset) RelatedAssetsBundle {
	var b RelatedAssetsBundle
	if asset == nil {
		return b
	}

	excludeSelf := func(items []domain.MediaAsset, max int) []domain.MediaAsset {
		var out []domain.MediaAsset
		for _, a := range items {
			if a.ID != asset.ID {
				out = append(out, a)
				if len(out) >= max {
					break
				}
			}
		}
		return out
	}

	if asset.Category != "" {
		if p, err := s.repo.QueryCatalog(ctx, domain.CatalogFilter{Category: asset.Category, PerPage: 12}); err == nil && p != nil {
			b.SameCategory = excludeSelf(p.Assets, 8)
		}
	}
	if len(asset.Characters) > 0 {
		if p, err := s.repo.QueryCatalog(ctx, domain.CatalogFilter{CharacterID: asset.Characters[0].ID, PerPage: 12}); err == nil && p != nil {
			b.SameCharacter = excludeSelf(p.Assets, 8)
		}
	}
	if len(asset.Groups) > 0 {
		if p, err := s.repo.QueryCatalog(ctx, domain.CatalogFilter{GroupID: asset.Groups[0].ID, PerPage: 12}); err == nil && p != nil {
			b.SameGroup = excludeSelf(p.Assets, 8)
		}
	}
	if asset.Metadata.ModelName != "" {
		if p, err := s.repo.QueryCatalog(ctx, domain.CatalogFilter{ModelName: asset.Metadata.ModelName, PerPage: 12}); err == nil && p != nil {
			b.SameModel = excludeSelf(p.Assets, 8)
		}
	}
	return b
}

// SetAssetCategory updates an asset's category and subcategory in SQLite.
func (s *CatalogService) SetAssetCategory(ctx context.Context, assetID int64, category, subcategory string) error {
	if err := s.repo.UpdateAssetCategory(ctx, assetID, category, subcategory); err != nil {
		return err
	}
	if strings.TrimSpace(category) != "" {
		if t, err := s.repo.UpsertTag(ctx, strings.TrimSpace(category), "category", "#2563eb"); err == nil {
			_ = s.repo.AttachTagToAsset(ctx, assetID, t.ID)
		}
	}
	if strings.TrimSpace(subcategory) != "" {
		if t, err := s.repo.UpsertTag(ctx, strings.TrimSpace(subcategory), "subcategory", "#059669"); err == nil {
			_ = s.repo.AttachTagToAsset(ctx, assetID, t.ID)
		}
	}
	return nil
}

// RateAsset updates an asset's 0-10 score and favorite flag in SQLite.
func (s *CatalogService) RateAsset(ctx context.Context, assetID int64, rating int, favorite bool) error {
	if rating < 0 {
		rating = 0
	}
	if rating > 10 {
		rating = 10
	}
	return s.repo.UpdateAssetRating(ctx, assetID, rating, favorite)
}

// AddTagToAsset creates or updates an agnostic tag and attaches it to an asset in SQLite.
func (s *CatalogService) AddTagToAsset(ctx context.Context, assetID int64, name, facet string) error {
	name = strings.TrimSpace(strings.TrimPrefix(name, "#"))
	if name == "" {
		return nil
	}
	if facet == "" {
		facet = "general"
	}
	t, err := s.repo.UpsertTag(ctx, name, strings.ToLower(strings.TrimSpace(facet)), "#2563eb")
	if err != nil {
		return err
	}
	return s.repo.AttachTagToAsset(ctx, assetID, t.ID)
}

// RemoveTagFromAsset detaches an agnostic tag from an asset.
func (s *CatalogService) RemoveTagFromAsset(ctx context.Context, assetID, tagID int64) error {
	return s.repo.DetachTagFromAsset(ctx, assetID, tagID)
}

// CreateGroup creates a new agnostic group and optionally links an initial asset.
func (s *CatalogService) CreateGroup(ctx context.Context, name, category, description string, initialAssetID int64) (int64, error) {
	g := &domain.Group{
		Name:        strings.TrimSpace(name),
		Category:    strings.TrimSpace(category),
		Description: strings.TrimSpace(description),
	}
	if initialAssetID > 0 {
		g.CoverAssetID = &initialAssetID
	}
	gid, err := s.repo.CreateGroup(ctx, g)
	if err != nil {
		return 0, err
	}
	if initialAssetID > 0 {
		_ = s.repo.AttachAssetToGroup(ctx, gid, initialAssetID)
	}
	return gid, nil
}

// AssignAssetToGroup attaches an existing asset to a group.
func (s *CatalogService) AssignAssetToGroup(ctx context.Context, groupID, assetID int64) error {
	return s.repo.AttachAssetToGroup(ctx, groupID, assetID)
}

// RemoveAssetFromGroup detaches an asset from a group.
func (s *CatalogService) RemoveAssetFromGroup(ctx context.Context, groupID, assetID int64) error {
	return s.repo.DetachAssetFromGroup(ctx, groupID, assetID)
}

// UpdateGroup updates a collection's title and description.
func (s *CatalogService) UpdateGroup(ctx context.Context, id int64, name, description string) error {
	return s.repo.UpdateGroup(ctx, id, name, description)
}

// DeleteGroup deletes a collection by ID.
func (s *CatalogService) DeleteGroup(ctx context.Context, id int64) error {
	return s.repo.DeleteGroup(ctx, id)
}

// ListGroups returns all agnostic groups.
func (s *CatalogService) ListGroups(ctx context.Context) ([]domain.Group, error) {
	return s.repo.ListGroups(ctx)
}

// CreateCharacter creates a character/concept folder with its canonical prompts and optional example asset.
func (s *CatalogService) CreateCharacter(ctx context.Context, c *domain.Character, exampleAssetID int64) (int64, error) {
	if exampleAssetID > 0 && c.AvatarAssetID == nil {
		c.AvatarAssetID = &exampleAssetID
	}
	cid, err := s.repo.CreateCharacter(ctx, c)
	if err != nil {
		return 0, err
	}
	if exampleAssetID > 0 {
		_ = s.repo.AttachAssetToCharacter(ctx, cid, exampleAssetID)
	}
	return cid, nil
}

// AssignAssetToCharacter links an asset as a reference example for a character.
func (s *CatalogService) AssignAssetToCharacter(ctx context.Context, characterID, assetID int64) error {
	return s.repo.AttachAssetToCharacter(ctx, characterID, assetID)
}

// RemoveAssetFromCharacter detaches an example asset from a character folder.
func (s *CatalogService) RemoveAssetFromCharacter(ctx context.Context, characterID, assetID int64) error {
	return s.repo.DetachAssetFromCharacter(ctx, characterID, assetID)
}

// GetCharacter retrieves a character profile and its example gallery.
func (s *CatalogService) GetCharacter(ctx context.Context, id int64) (*domain.Character, error) {
	return s.repo.GetCharacterByID(ctx, id)
}

// ListCharacters returns all character profiles.
func (s *CatalogService) ListCharacters(ctx context.Context) ([]domain.Character, error) {
	return s.repo.ListCharacters(ctx)
}

// GetSettings retrieves application and Ollama model routing settings.
func (s *CatalogService) GetSettings(ctx context.Context) (domain.AppSettings, error) {
	return s.repo.GetSettings(ctx)
}

// SaveSettings updates application and Ollama model routing settings.
func (s *CatalogService) SaveSettings(ctx context.Context, cfg domain.AppSettings) error {
	return s.repo.SaveSettings(ctx, cfg)
}

// SetDefaultTargetGenerator updates the active default target generator slug.
func (s *CatalogService) SetDefaultTargetGenerator(ctx context.Context, slug string) error {
	return s.repo.SetDefaultTargetGenerator(ctx, slug)
}

// ListPromptProfiles returns all generator system prompt profiles (flux, krea2, moss, minimaxh3, etc.).
func (s *CatalogService) ListPromptProfiles(ctx context.Context) ([]domain.PromptProfile, error) {
	return s.repo.ListPromptProfiles(ctx)
}

// UpsertPromptProfile saves or updates a generator system prompt profile.
func (s *CatalogService) UpsertPromptProfile(ctx context.Context, p *domain.PromptProfile) (int64, error) {
	return s.repo.UpsertPromptProfile(ctx, p)
}

// DeletePromptProfile deletes a generator system prompt profile by ID.
func (s *CatalogService) DeletePromptProfile(ctx context.Context, id int64) error {
	return s.repo.DeletePromptProfile(ctx, id)
}

// ListModelLoRAs lists model-scoped LoRA documentation entries.
func (s *CatalogService) ListModelLoRAs(ctx context.Context, profileSlug string) ([]domain.ModelLoRA, error) {
	return s.repo.ListModelLoRAs(ctx, profileSlug)
}

// UpsertModelLoRA creates or updates a model-scoped LoRA documentation entry.
func (s *CatalogService) UpsertModelLoRA(ctx context.Context, l *domain.ModelLoRA) (int64, error) {
	return s.repo.UpsertModelLoRA(ctx, l)
}

// DeleteModelLoRA deletes a model-scoped LoRA documentation entry by ID.
func (s *CatalogService) DeleteModelLoRA(ctx context.Context, id int64) error {
	return s.repo.DeleteModelLoRA(ctx, id)
}

func (s *CatalogService) buildLoRAContextBlock(ctx context.Context, targetSlug string, loraIDs []int64) string {
	if len(loraIDs) == 0 {
		return ""
	}
	loras, err := s.repo.GetModelLoRAsByIDs(ctx, loraIDs)
	if err != nil || len(loras) == 0 {
		return ""
	}

	var b strings.Builder
	b.WriteString(fmt.Sprintf("Selected LoRAs for Target Model '%s' (MUST be applied in the generated prompt following their rules and examples):\n", targetSlug))
	for i, l := range loras {
		triggerInfo := l.TriggerWords
		if strings.TrimSpace(triggerInfo) == "" {
			triggerInfo = "None (triggerless LoRA — apply via style/description and syntax tag)"
		}
		b.WriteString(fmt.Sprintf(
			"%d) LoRA Name: %s (Target Engine: %s)\n   - Syntax / Weight Tag: %s (Recommended Weight: %s)\n   - Trigger Words: %s\n   - Purpose & Behavior: %s\n   - Reference Usage Examples:\n     %s\n",
			i+1, l.Name, l.ProfileSlug, l.SyntaxTag, l.RecommendedWeight, triggerInfo, l.Description, l.UsageExamples,
		))
	}
	return b.String()
}

// AIBackendStatus summarizes the live connectivity and discovered models of both Ollama and oMLX.
type AIBackendStatus struct {
	Checked       bool                     `json:"checked"`
	OllamaOnline  bool                     `json:"ollama_online"`
	OllamaVersion string                   `json:"ollama_version"`
	OllamaModels  []domain.OllamaModelInfo `json:"ollama_models"`
	OMLXOnline    bool                     `json:"omlx_online"`
	OMLXModels    []domain.OllamaModelInfo `json:"omlx_models"`
	AllModels     []domain.OllamaModelInfo `json:"all_models"`
}

// InvalidateAICache resets the cache timestamp so the next call or background refresh fetches fresh data.
func (s *CatalogService) InvalidateAICache() {
	s.aiStatusMu.Lock()
	s.aiStatusTime = time.Time{}
	s.aiStatusMu.Unlock()
}

// InspectAIBackends returns cached AI backend status if fresh (< 1 min).
// If the cache is stale but present, it returns the stale cache immediately and refreshes in the background (stale-while-revalidate).
// If the cache is cold, it performs a live probe.
func (s *CatalogService) InspectAIBackends(ctx context.Context) AIBackendStatus {
	s.aiStatusMu.RLock()
	cached := s.aiStatusCache
	cachedAt := s.aiStatusTime
	s.aiStatusMu.RUnlock()

	// 1. Fresh cache hit (< 1 minute)
	if !cachedAt.IsZero() && time.Since(cachedAt) < 1*time.Minute {
		return cached
	}

	// 2. Stale cache: return immediately and revalidate in background
	if !cachedAt.IsZero() {
		go func() {
			bgCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
			defer cancel()
			_ = s.InspectAIBackendsFresh(bgCtx)
		}()
		return cached
	}

	// 3. Cold cache: run fresh inspection
	return s.InspectAIBackendsFresh(ctx)
}

// InspectAIBackendsFresh probes live servers and updates the cache.
func (s *CatalogService) InspectAIBackendsFresh(ctx context.Context) AIBackendStatus {
	s.aiRefreshMu.Lock()
	defer s.aiRefreshMu.Unlock()

	// Double-check if another goroutine just refreshed the cache while waiting for the lock
	s.aiStatusMu.RLock()
	if !s.aiStatusTime.IsZero() && time.Since(s.aiStatusTime) < 5*time.Second {
		cached := s.aiStatusCache
		s.aiStatusMu.RUnlock()
		return cached
	}
	s.aiStatusMu.RUnlock()

	// Use an independent probe context with timeout so client cancellations (e.g. browser F5 reload)
	// never abort backend network checks mid-flight or cache false offline states.
	probeCtx, cancel := context.WithTimeout(context.Background(), 8*time.Second)
	defer cancel()

	st := s.probeAIBackends(probeCtx)
	st.Checked = true

	s.aiStatusMu.Lock()
	s.aiStatusCache = st
	s.aiStatusTime = time.Now()
	s.aiStatusMu.Unlock()

	return st
}

// probeAIBackends checks both the configured Ollama server and oMLX server concurrently and merges their model lists.
func (s *CatalogService) probeAIBackends(ctx context.Context) AIBackendStatus {
	var st AIBackendStatus
	if s.ollama == nil || s.repo == nil {
		return st
	}
	cfg, err := s.repo.GetSettings(ctx)
	if err != nil {
		return st
	}

	var (
		wg            sync.WaitGroup
		ollamaOnline  bool
		ollamaVersion string
		ollamaModels  []domain.OllamaModelInfo

		omlxOnline bool
		omlxModels []domain.OllamaModelInfo
	)

	if cfg.OllamaEnabled && strings.TrimSpace(cfg.OllamaBaseURL) != "" {
		wg.Add(1)
		go func() {
			defer wg.Done()
			ver, online := s.ollama.CheckStatus(ctx, cfg.OllamaBaseURL)
			if online {
				models, _ := s.ollama.ListModels(ctx, cfg.OllamaBaseURL)
				ollamaOnline = true
				ollamaVersion = ver
				ollamaModels = models
			}
		}()
	}

	openaiURL := cfg.OpenAIBaseURL
	if openaiURL == "" {
		openaiURL = cfg.OMLXBaseURL
	}
	apiKey := cfg.OpenAIAPIKey
	if apiKey == "" {
		apiKey = cfg.OMLXAPIKey
	}
	if cfg.OpenAIEnabled && strings.TrimSpace(openaiURL) != "" {
		wg.Add(1)
		go func() {
			defer wg.Done()
			models, err := s.ollama.ListOMLXModels(ctx, openaiURL, apiKey)
			if err == nil {
				omlxOnline = true
				omlxModels = models
			}
		}()
	}

	wg.Wait()

	st.OllamaOnline = ollamaOnline
	st.OllamaVersion = ollamaVersion
	st.OllamaModels = ollamaModels
	st.OMLXOnline = omlxOnline
	st.OMLXModels = omlxModels

	totalLen := len(ollamaModels) + len(omlxModels)
	if totalLen > 0 {
		st.AllModels = make([]domain.OllamaModelInfo, 0, totalLen)
		st.AllModels = append(st.AllModels, ollamaModels...)
		st.AllModels = append(st.AllModels, omlxModels...)
	}

	return st
}

// ListGeneratedPrompts returns stored prompts generated from assets or characters.
func (s *CatalogService) ListGeneratedPrompts(ctx context.Context, assetID, characterID int64, limit int) ([]domain.GeneratedPrompt, error) {
	return s.repo.ListGeneratedPrompts(ctx, assetID, characterID, limit)
}

// ClearGeneratedPrompts deletes recorded prompts from storage.
func (s *CatalogService) ClearGeneratedPrompts(ctx context.Context, assetID, characterID int64) error {
	if s.repo == nil {
		return nil
	}
	return s.repo.ClearGeneratedPrompts(ctx, assetID, characterID)
}

// ResolveExecutionRoute resolves the provider, model, baseURL, and thinking level for a prompt generation task,
// applying any per-request quick overrides on top of the persistent Settings defaults.
func (s *CatalogService) ResolveExecutionRoute(cfg domain.AppSettings, defaultProvider domain.AIProviderType, defaultModel, defaultThink string, override domain.PromptExecutionOverride) (domain.AIProviderType, string, string, string) {
	return resolveExecutionRoute(cfg, defaultProvider, defaultModel, defaultThink, override)
}

func resolveExecutionRoute(cfg domain.AppSettings, defaultProvider domain.AIProviderType, defaultModel, defaultThink string, override domain.PromptExecutionOverride) (domain.AIProviderType, string, string, string) {
	prov := defaultProvider
	model := defaultModel
	think := defaultThink

	if override.Provider != "" {
		prov = override.Provider
		if pMod, pThink := cfg.GetProviderDefaults(prov); pMod != "" {
			model = pMod
			think = pThink
		}
	}

	if override.Model != "" {
		model = override.Model
	}

	prov = domain.AIProviderType(strings.Trim(strings.TrimSpace(string(prov)), "\"' \t\r\n"))
	model = strings.Trim(strings.TrimSpace(model), "\"' \t\r\n")
	think = strings.Trim(strings.TrimSpace(think), "\"' \t\r\n")

	// Auto-detect provider if model name indicates MLX/OpenAI but provider was left as ollama
	modelUpper := strings.ToUpper(model)
	if prov == domain.ProviderOllama && (strings.Contains(modelUpper, "-MLX") || strings.Contains(modelUpper, "MLX-") || strings.HasPrefix(modelUpper, "MLX/") || strings.Contains(modelUpper, ".MLX")) {
		prov = domain.ProviderOpenAI
	}

	if prov == "" {
		prov = domain.ProviderOllama
	}

	if override.ThinkLevel != "" && override.ThinkLevel != "settings" && override.ThinkLevel != "default" {
		think = override.ThinkLevel
	}
	if think == "" {
		think = "off"
	}

	baseURL := cfg.OllamaBaseURL
	if prov == domain.ProviderOpenAI || prov == domain.ProviderOMLX {
		baseURL = cfg.OpenAIBaseURL
		if baseURL == "" {
			baseURL = cfg.OMLXBaseURL
		}
	}
	return prov, model, baseURL, think
}

// RunImage2Prompt runs the configured Ollama or oMLX Image2Prompt (or Remix Text2Prompt) on an asset
// using the selected generator profile's system prompt (e.g., flux, krea2, moss, minimaxh3), selected model LoRAs,
// optional character/scene direction, and optional quick model/thinking overrides.
func (s *CatalogService) RunImage2Prompt(ctx context.Context, assetID int64, profileSlug, mode, userHint string, characterID int64, loraIDs []int64, override domain.PromptExecutionOverride, saveToAsset bool, frameImagePath string) (string, error) {
	if s.ollama == nil {
		return "", fmt.Errorf("AI provider is not initialized")
	}

	var asset *domain.MediaAsset
	if assetID > 0 {
		var err error
		asset, err = s.repo.GetAssetByID(ctx, assetID)
		if err != nil {
			return "", err
		}
	}

	// If no asset and no attached image or video frame, delegate to Text2Prompt
	if asset == nil && frameImagePath == "" {
		return s.RunText2Prompt(ctx, userHint, profileSlug, loraIDs, override)
	}

	cfg, err := s.repo.GetSettings(ctx)
	if err != nil {
		return "", err
	}
	if profileSlug == "" {
		profileSlug = cfg.DefaultTargetGenerator
	}

	var sysPrompt string
	if prof, err := s.repo.GetPromptProfileBySlug(ctx, profileSlug); err == nil && prof != nil {
		sysPrompt = prof.SystemPrompt
	}

	// Build rich context combining existing asset metadata, optional character sheet, selected LoRA docs, and user scene direction
	var contextParts []string
	if asset != nil && asset.Metadata.PositivePrompt != "" {
		contextParts = append(contextParts, "Reference Asset Original Prompt: "+asset.Metadata.PositivePrompt)
	}
	var charPtr *int64
	if characterID > 0 {
		if ch, err := s.repo.GetCharacterByID(ctx, characterID); err == nil && ch != nil {
			cid := ch.ID
			charPtr = &cid
			contextParts = append(contextParts, fmt.Sprintf(
				"Character Identity to Preserve: %s | Mandatory Trigger Words: %s | Base Character Prompt: %s",
				ch.Name, ch.TriggerWords, ch.BasePositivePrompt,
			))
		}
	}
	if loraDoc := s.buildLoRAContextBlock(ctx, profileSlug, loraIDs); loraDoc != "" {
		contextParts = append(contextParts, loraDoc)
	}
	if strings.TrimSpace(userHint) != "" {
		contextParts = append(contextParts, "New Scene / Movie / Style Direction: "+strings.TrimSpace(userHint))
	}
	combinedHint := strings.Join(contextParts, "\n\n")

	var generated string
	var usedProvider domain.AIProviderType
	var usedModel string
	var usedThink string

	mode = strings.ToLower(strings.TrimSpace(mode))
	if mode == "" {
		if asset != nil || frameImagePath != "" {
			mode = "vision"
		} else {
			mode = "remix"
		}
	}

	apiKey := cfg.OpenAIAPIKey
	if apiKey == "" {
		apiKey = cfg.OMLXAPIKey
	}

	if mode == "remix" {
		// Remix via Text2Prompt model using the extracted prompt + character + direction
		defProv, defMod, defThink := cfg.GetRemixDefaults()
		var baseURL string
		usedProvider, usedModel, baseURL, usedThink = resolveExecutionRoute(
			cfg, defProv, defMod, defThink, override,
		)
		promptInstruction := "Transform the following reference asset and scene direction into a final generation prompt for target engine '" + profileSlug + "'. Preserve any mandatory character trigger words if present:\n\n" + combinedHint
		if strings.TrimSpace(combinedHint) == "" {
			promptInstruction = "Generate a rich, detailed, high-quality generation prompt for target engine '" + profileSlug + "'."
		}
		generated, err = s.ollama.TextToPrompt(ctx, usedProvider, baseURL, apiKey, usedModel, sysPrompt, promptInstruction, usedThink)
		if err != nil {
			return "", err
		}
	} else {
		// Vision analysis via Image2Prompt model
		defProv, defMod, defThink := cfg.GetRemixDefaults()
		var baseURL string
		usedProvider, usedModel, baseURL, usedThink = resolveExecutionRoute(cfg, defProv, defMod, defThink, override)

		var absPath string
		if frameImagePath != "" {
			absPath = frameImagePath
			log.Printf("[AI] Studio using frame or attached image: %s (model: %s)", absPath, usedModel)
		} else if asset != nil {
			absPath = s.ResolveAssetAbsPath(ctx, asset)
		} else {
			return "", fmt.Errorf("no media asset or image provided for vision analysis")
		}

		generated, err = s.ollama.ImageToPrompt(ctx, usedProvider, baseURL, apiKey, usedModel, sysPrompt, absPath, combinedHint, usedThink)
		if err != nil {
			return "", err
		}
	}

	if generated != "" {
		var aid *int64
		if asset != nil {
			aid = &asset.ID
		}
		_, _ = s.repo.SaveGeneratedPrompt(ctx, &domain.GeneratedPrompt{
			AssetID:           aid,
			CharacterID:       charPtr,
			TargetProfileSlug: profileSlug,
			SceneDirection:    strings.TrimSpace(userHint),
			OutputPrompt:      generated,
			Provider:          usedProvider,
			ModelName:         fmt.Sprintf("%s (think:%s)", usedModel, usedThink),
		})
	}

	if saveToAsset && asset != nil && generated != "" {
		_ = s.repo.UpdateAssetPrompt(ctx, asset.ID, generated, profileSlug)
		asset.Metadata.PositivePrompt = generated
		asset.Metadata.Engine = domain.EngineType(profileSlug)
		absPath := s.ResolveAssetAbsPath(ctx, asset)
		_ = metadata.WritePromptSidecarTXT(absPath, asset.Metadata)
	}
	return generated, nil
}

// RunCharacterPrompt synthesizes a new prompt for a specific Character in a given scene, movie style, or action
// tailored to a target generator profile (e.g., flux, krea2, qwen-image, moss, minimaxh3 video), optional model-specific LoRAs, and optional quick model/thinking overrides.
func (s *CatalogService) RunCharacterPrompt(ctx context.Context, characterID int64, profileSlug, sceneDirection string, refAssetID int64, loraIDs []int64, override domain.PromptExecutionOverride) (string, error) {
	if s.ollama == nil {
		return "", fmt.Errorf("AI provider is not initialized")
	}
	ch, err := s.repo.GetCharacterByID(ctx, characterID)
	if err != nil {
		return "", fmt.Errorf("character not found: %w", err)
	}
	cfg, err := s.repo.GetSettings(ctx)
	if err != nil {
		return "", err
	}
	if profileSlug == "" {
		profileSlug = cfg.DefaultTargetGenerator
	}

	var sysPrompt string
	if prof, err := s.repo.GetPromptProfileBySlug(ctx, profileSlug); err == nil && prof != nil {
		sysPrompt = prof.SystemPrompt
	}

	loraBlock := s.buildLoRAContextBlock(ctx, profileSlug, loraIDs)

	charSpec := fmt.Sprintf(
		"Character Sheet:\n- Name: %s (%s)\n- Visual Description: %s\n- Mandatory Trigger Words: %s\n- Canonical Base Prompt: %s\n- Character Default LoRAs: %s\n\n%s\nRequested Scene / Movie / Shot Direction:\n%s\n\nWrite a complete, cohesive generation prompt for '%s' that places this exact character into the requested scene/movie context while preserving their trigger words, visual identity, and any selected target-model LoRA syntax/triggers. Output only the final prompt.",
		ch.Name,
		ch.Archetype,
		ch.Description,
		ch.TriggerWords,
		ch.BasePositivePrompt,
		strings.Join(ch.LoRAs, ", "),
		loraBlock,
		strings.TrimSpace(sceneDirection),
		profileSlug,
	)

	var generated string
	var usedProvider domain.AIProviderType
	var usedModel string
	var usedThink string
	var assetPtr *int64

	apiKey := cfg.OpenAIAPIKey
	if apiKey == "" {
		apiKey = cfg.OMLXAPIKey
	}

	if refAssetID > 0 {
		if refAsset, err := s.repo.GetAssetByID(ctx, refAssetID); err == nil && refAsset != nil {
			aid := refAsset.ID
			assetPtr = &aid
			defProv, defMod, defThink := cfg.GetRemixDefaults()
			var baseURL string
			usedProvider, usedModel, baseURL, usedThink = resolveExecutionRoute(cfg, defProv, defMod, defThink, override)
			absPath := s.ResolveAssetAbsPath(ctx, refAsset)
			generated, err = s.ollama.ImageToPrompt(ctx, usedProvider, baseURL, apiKey, usedModel, sysPrompt, absPath, charSpec, usedThink)
			if err != nil {
				return "", err
			}
		}
	}

	if generated == "" {
		defProv, defMod, defThink := cfg.GetStudioDefaults()
		var baseURL string
		usedProvider, usedModel, baseURL, usedThink = resolveExecutionRoute(
			cfg, defProv, defMod, defThink, override,
		)
		generated, err = s.ollama.TextToPrompt(ctx, usedProvider, baseURL, apiKey, usedModel, sysPrompt, charSpec, usedThink)
		if err != nil {
			return "", err
		}
	}

	if generated != "" {
		cid := ch.ID
		_, _ = s.repo.SaveGeneratedPrompt(ctx, &domain.GeneratedPrompt{
			AssetID:           assetPtr,
			CharacterID:       &cid,
			TargetProfileSlug: profileSlug,
			SceneDirection:    strings.TrimSpace(sceneDirection),
			OutputPrompt:      generated,
			Provider:          usedProvider,
			ModelName:         fmt.Sprintf("%s (think:%s)", usedModel, usedThink),
		})
	}

	return generated, nil
}

// RunText2Prompt expands or converts a text idea/prompt into a target generator prompt using the configured
// Ollama or oMLX Text2Prompt model, the selected generator profile's system prompt, any selected model-specific LoRAs, and optional quick model/thinking overrides.
func (s *CatalogService) RunText2Prompt(ctx context.Context, inputConcept, profileSlug string, loraIDs []int64, override domain.PromptExecutionOverride) (string, error) {
	if s.ollama == nil {
		return "", fmt.Errorf("AI provider is not initialized")
	}
	cfg, err := s.repo.GetSettings(ctx)
	if err != nil {
		return "", err
	}
	if profileSlug == "" {
		profileSlug = cfg.DefaultTargetGenerator
	}

	var sysPrompt string
	if prof, err := s.repo.GetPromptProfileBySlug(ctx, profileSlug); err == nil && prof != nil {
		sysPrompt = prof.SystemPrompt
	}

	fullUserPrompt := inputConcept
	if loraDoc := s.buildLoRAContextBlock(ctx, profileSlug, loraIDs); loraDoc != "" {
		fullUserPrompt = fmt.Sprintf("Concept / Scene Request:\n%s\n\n%s\nIncorporate the selected LoRA trigger words, stylistic guidance, and syntax tag into the final prompt.", inputConcept, loraDoc)
	}

	apiKey := cfg.OpenAIAPIKey
	if apiKey == "" {
		apiKey = cfg.OMLXAPIKey
	}

	defProv, defMod, defThink := cfg.GetStudioDefaults()
	provider, model, baseURL, think := resolveExecutionRoute(
		cfg, defProv, defMod, defThink, override,
	)

	generated, err := s.ollama.TextToPrompt(ctx, provider, baseURL, apiKey, model, sysPrompt, fullUserPrompt, think)
	if err != nil {
		return "", err
	}
	if generated != "" {
		_, _ = s.repo.SaveGeneratedPrompt(ctx, &domain.GeneratedPrompt{
			TargetProfileSlug: profileSlug,
			SceneDirection:    strings.TrimSpace(inputConcept),
			OutputPrompt:      generated,
			Provider:          provider,
			ModelName:         fmt.Sprintf("%s (think:%s)", model, think),
		})
	}
	return generated, nil
}

// Studio Composition and Pose persistence methods

func (s *CatalogService) SaveStudioComposition(ctx context.Context, comp *domain.StudioComposition) (int64, error) {
	return s.repo.SaveStudioComposition(ctx, comp)
}

func (s *CatalogService) GetStudioComposition(ctx context.Context, id int64) (*domain.StudioComposition, error) {
	return s.repo.GetStudioCompositionByID(ctx, id)
}

func (s *CatalogService) ListStudioCompositions(ctx context.Context) ([]domain.StudioComposition, error) {
	return s.repo.ListStudioCompositions(ctx)
}

func (s *CatalogService) DeleteStudioComposition(ctx context.Context, id int64) error {
	return s.repo.DeleteStudioComposition(ctx, id)
}

func (s *CatalogService) UpsertStudioPose(ctx context.Context, pose *domain.StudioPose) (int64, error) {
	return s.repo.UpsertStudioPose(ctx, pose)
}

func (s *CatalogService) ListStudioPoses(ctx context.Context) ([]domain.StudioPose, error) {
	return s.repo.ListStudioPoses(ctx)
}

func (s *CatalogService) DeleteStudioPose(ctx context.Context, id int64) error {
	return s.repo.DeleteStudioPose(ctx, id)
}

// Scene Director DTOs for spatial prompt compilation
type SceneElementDTO struct {
	ID          string                 `json:"id"`
	Type        string                 `json:"type"` // "character", "silhouette", "prop", "light", "camera", "note"
	Subtype     string                 `json:"subtype,omitempty"`
	Name        string                 `json:"name,omitempty"`
	Label       string                 `json:"label,omitempty"`
	PropType    string                 `json:"propType,omitempty"`
	CharacterID int64                  `json:"characterId,omitempty"`
	X           float64                `json:"x"`
	Y           float64                `json:"y"`
	Width       float64                `json:"width,omitempty"`
	Height      float64                `json:"height,omitempty"`
	Scale       float64                `json:"scale,omitempty"`
	Rotation    float64                `json:"rotation,omitempty"`
	FlipH       bool                   `json:"flipH,omitempty"`
	Facing      string                 `json:"facing,omitempty"`
	PoseName    string                 `json:"poseName,omitempty"`
	Pose        map[string]interface{} `json:"pose,omitempty"`
	Anatomy     string                 `json:"anatomy,omitempty"`
	Expression  string                 `json:"expression,omitempty"`
	Features    string                 `json:"features,omitempty"`
	Hair        string                 `json:"hair,omitempty"`
	Clothing    string                 `json:"clothing,omitempty"`
	Details       string                 `json:"details,omitempty"`
	Description   string                 `json:"description,omitempty"`
	MarkerRole        string             `json:"markerRole,omitempty"`        // "environment" or "character_feature"
	TargetCharacterID string             `json:"targetCharacterId,omitempty"` // ID of target character if linked
	TargetBodyPart    string             `json:"targetBodyPart,omitempty"`    // e.g. "Left Arm", "Right Knee", "Head / Face"
	LookingAtID   string                 `json:"lookingAtId,omitempty"`
	LookingAtName string                 `json:"lookingAtName,omitempty"`
	LightType     string                 `json:"lightType,omitempty"`
	Color       string                 `json:"color,omitempty"`
	Angle       string                 `json:"angle,omitempty"`
	FOV         string                 `json:"fov,omitempty"`
	Text        string                 `json:"text,omitempty"`
	Layer       int                    `json:"layer,omitempty"`
}

type SceneDirectorDTO struct {
	AspectRatio     string            `json:"aspectRatio"`
	CanvasWidth     float64           `json:"canvasWidth"`
	CanvasHeight    float64           `json:"canvasHeight"`
	Perspective     string            `json:"perspective,omitempty"` // "front" or "side"
	EnvironmentType string            `json:"environmentType,omitempty"` // "any", "indoor", "outdoor"
	Environment     string            `json:"environment,omitempty"`
	TimeOfDay       string            `json:"timeOfDay,omitempty"`
	Elements        []SceneElementDTO `json:"elements"`
	DirectorNote    string            `json:"directorNote,omitempty"`
}

// CompileSceneDirectorManifest converts a structured 2D scene JSON into a spatial composition manifest.
func (s *CatalogService) CompileSceneDirectorManifest(ctx context.Context, sceneJSON string) (string, error) {
	if strings.TrimSpace(sceneJSON) == "" || sceneJSON == "{}" {
		return "Standard composition, eye-level framing, central subject focus.", nil
	}

	var scene SceneDirectorDTO
	if err := json.Unmarshal([]byte(sceneJSON), &scene); err != nil {
		return "Scene elements arranged in 2D space: " + sceneJSON, nil
	}

	width := scene.CanvasWidth
	if width <= 0 {
		width = 1000
	}
	height := scene.CanvasHeight
	if height <= 0 {
		height = 562
	}

	aspectRatio := scene.AspectRatio
	if aspectRatio == "" {
		aspectRatio = "16:9 widescreen"
	}

	cameraSummary := "Cinematic standard framing, eye-level angle."
	var subjectLines []string
	var propLines []string
	var lightLines []string
	var noteLines []string

	if strings.TrimSpace(scene.DirectorNote) != "" {
		noteLines = append(noteLines, scene.DirectorNote)
	}

	// 1. Index characters to resolve character feature markers (e.g. robotic arm, injured knee)
	type charRef struct {
		id   string
		name string
		x    float64
		y    float64
	}
	var chars []charRef
	for _, el := range scene.Elements {
		if el.Type == "character" || el.Type == "silhouette" {
			name := el.Name
			if name == "" {
				name = el.Label
			}
			if name == "" {
				name = "Person / Character"
			}
			chars = append(chars, charRef{id: el.ID, name: name, x: el.X, y: el.Y})
		}
	}

	var charFeatures = make(map[string][]string)
	var featureLines []string
	for _, el := range scene.Elements {
		isFeatureMarker := el.MarkerRole == "character_feature" || el.Subtype == "marker" || (el.MarkerRole != "environment" && strings.TrimSpace(el.TargetBodyPart) != "")
		if !isFeatureMarker {
			continue
		}

		targetCharID := el.TargetCharacterID
		targetCharName := ""
		if targetCharID != "" {
			for _, c := range chars {
				if c.id == targetCharID {
					targetCharName = c.name
					break
				}
			}
		}
		if targetCharID == "" && len(chars) > 0 {
			bestDist := 1e12
			for _, c := range chars {
				dx := c.x - el.X
				dy := c.y - el.Y
				dist := dx*dx + dy*dy
				if dist < bestDist {
					bestDist = dist
					targetCharID = c.id
					targetCharName = c.name
				}
			}
		}
		if targetCharName == "" {
			targetCharName = "Subject"
		}

		bodyPart := strings.TrimSpace(el.TargetBodyPart)
		if bodyPart == "" {
			bodyPart = "Targeted Body Area"
		}

		desc := strings.TrimSpace(el.Description)
		if desc == "" {
			desc = strings.TrimSpace(el.Details)
		}
		if desc == "" {
			desc = strings.TrimSpace(el.Label)
		}
		if desc == "" {
			desc = "special feature detail"
		}

		featStr := fmt.Sprintf("[%s]: %s", bodyPart, desc)
		if targetCharID != "" {
			charFeatures[targetCharID] = append(charFeatures[targetCharID], featStr)
		}
		featureLines = append(featureLines, fmt.Sprintf("- [%s] -> %s (indicated by targeted body marker)", targetCharName, featStr))
	}

	for _, el := range scene.Elements {
		xNorm := el.X / width
		hPos := "Center"
		if xNorm < 0.35 {
			hPos = "Left"
		} else if xNorm > 0.65 {
			hPos = "Right"
		}

		yNorm := el.Y / height
		vPos := "Midground"
		if yNorm > 0.65 || el.Scale > 1.25 {
			vPos = "Foreground"
		} else if yNorm < 0.35 || (el.Scale > 0 && el.Scale < 0.75) {
			vPos = "Background"
		}
		loc := fmt.Sprintf("%s-%s", vPos, hPos)

		switch el.Type {
		case "camera":
			ang := el.Angle
			if ang == "" {
				ang = "eye-level shot"
			}
			fov := el.FOV
			if fov == "" {
				fov = "standard cinematic field of view"
			}
			cameraSummary = fmt.Sprintf("%s angle, %s.", ang, fov)

		case "character", "silhouette":
			facingDesc := "facing camera forward"
			switch el.Facing {
			case "back":
				facingDesc = "facing away / back to viewer"
			case "left":
				facingDesc = "profile facing left"
			case "right":
				facingDesc = "profile facing right"
			case "three_quarter_left":
				facingDesc = "three-quarters view looking left"
			case "three_quarter_right":
				facingDesc = "three-quarters view looking right"
			}

			mirrorDesc := ""
			if el.FlipH {
				mirrorDesc = " (mirrored pose)"
			}

			poseDesc := ""
			if el.PoseName != "" {
				poseDesc = fmt.Sprintf(", in %s posture", el.PoseName)
			}

			charName := el.Name
			if charName == "" {
				charName = el.Label
			}
			if charName == "" {
				charName = "Person / Character"
			}

			if el.CharacterID > 0 {
				if char, err := s.repo.GetCharacterByID(ctx, el.CharacterID); err == nil && char != nil {
					charName = fmt.Sprintf("%s (Trigger: %s)", char.Name, char.TriggerWords)
				}
			}

			scaleDesc := ""
			if el.Scale > 1.3 {
				scaleDesc = ", prominent close scale"
			} else if el.Scale < 0.7 && el.Scale > 0 {
				scaleDesc = ", smaller distant scale"
			}

			var charTraits []string
			if strings.TrimSpace(el.Anatomy) != "" {
				charTraits = append(charTraits, "anatomy: "+strings.TrimSpace(el.Anatomy))
			}
			if strings.TrimSpace(el.Features) != "" {
				charTraits = append(charTraits, "features: "+strings.TrimSpace(el.Features))
			}
			if strings.TrimSpace(el.Expression) != "" {
				charTraits = append(charTraits, "expression: "+strings.TrimSpace(el.Expression))
			}
			if strings.TrimSpace(el.Hair) != "" {
				charTraits = append(charTraits, "hair: "+strings.TrimSpace(el.Hair))
			}
			if strings.TrimSpace(el.Clothing) != "" {
				charTraits = append(charTraits, "clothing: "+strings.TrimSpace(el.Clothing))
			}
			if strings.TrimSpace(el.Details) != "" {
				charTraits = append(charTraits, "details: "+strings.TrimSpace(el.Details))
			}

			gazeTarget := strings.TrimSpace(el.LookingAtName)
			if gazeTarget == "" && el.LookingAtID != "" {
				for _, other := range scene.Elements {
					if other.ID == el.LookingAtID {
						gazeTarget = other.Name
						if gazeTarget == "" {
							gazeTarget = other.Label
						}
						if gazeTarget == "" {
							gazeTarget = other.Subtype
						}
						if gazeTarget == "" {
							gazeTarget = other.PropType
						}
						break
					}
				}
			}
			if gazeTarget != "" {
				charTraits = append(charTraits, "gaze: looking directly at ["+gazeTarget+"]")
			}
			if feats, ok := charFeatures[el.ID]; ok && len(feats) > 0 {
				for _, f := range feats {
					charTraits = append(charTraits, "targeted body feature "+f)
				}
			}
			traitsDesc := ""
			if len(charTraits) > 0 {
				traitsDesc = " [" + strings.Join(charTraits, ", ") + "]"
			}

			subjectLines = append(subjectLines, fmt.Sprintf("- [%s]%s positioned in %s: %s%s%s%s.",
				charName, traitsDesc, loc, facingDesc, poseDesc, mirrorDesc, scaleDesc))

		case "prop":
			isFeatureMarker := el.MarkerRole == "character_feature" || el.Subtype == "marker" || (el.MarkerRole != "environment" && strings.TrimSpace(el.TargetBodyPart) != "")
			if isFeatureMarker {
				// Handled as character feature marker, skip environment prop
				continue
			}
			propName := el.Label
			if propName == "" {
				propName = el.PropType
			}
			if propName == "" {
				propName = "Prop object"
			}
			desc := strings.TrimSpace(el.Description)
			if desc == "" {
				desc = strings.TrimSpace(el.Details)
			}
			if desc != "" {
				propName += fmt.Sprintf(" (%s)", desc)
			}
			propLines = append(propLines, fmt.Sprintf("- [%s] located in %s.", propName, loc))

		case "light":
			lType := el.LightType
			if lType == "" {
				lType = "Directional light"
			}
			colorDesc := ""
			if el.Color != "" {
				colorDesc = fmt.Sprintf(" (tint %s)", el.Color)
			}
			lightLines = append(lightLines, fmt.Sprintf("- %s positioned at %s shining into the scene at %0.0f°%s.",
				lType, loc, el.Rotation, colorDesc))

		case "note":
			if strings.TrimSpace(el.Text) != "" {
				noteLines = append(noteLines, fmt.Sprintf("- Directive note at %s: \"%s\"", loc, el.Text))
			}
		}
	}

	var sb strings.Builder
	sb.WriteString("[SPATIAL SCENE COMPOSITION MANIFEST]\n")
	sb.WriteString(fmt.Sprintf("Aspect Ratio: %s\n", aspectRatio))
	if scene.Perspective == "side" {
		sb.WriteString("Perspective: Lateral Side-View / Profile Scene Staging\n")
	}
	sb.WriteString(fmt.Sprintf("Framing & Camera Perspective: %s\n", cameraSummary))

	if strings.TrimSpace(scene.Environment) != "" || strings.TrimSpace(scene.TimeOfDay) != "" {
		envDesc := strings.TrimSpace(scene.Environment)
		if scene.EnvironmentType != "" && scene.EnvironmentType != "any" {
			envDesc = fmt.Sprintf("[%s] %s", strings.ToUpper(scene.EnvironmentType), envDesc)
		}
		if envDesc != "" {
			sb.WriteString(fmt.Sprintf("Setting & Environment: %s\n", envDesc))
		}
		if strings.TrimSpace(scene.TimeOfDay) != "" {
			sb.WriteString(fmt.Sprintf("Time of Day & Atmosphere: %s\n", strings.TrimSpace(scene.TimeOfDay)))
		}
	}

	if len(subjectLines) > 0 {
		sb.WriteString("Subjects & Characters:\n")
		sb.WriteString(strings.Join(subjectLines, "\n") + "\n")
	} else {
		sb.WriteString("Subjects & Characters:\n- Standard subject composition.\n")
	}

	if len(featureLines) > 0 {
		sb.WriteString("Character Targeted Features & Body Modifications:\n")
		sb.WriteString(strings.Join(featureLines, "\n") + "\n")
	}

	if len(propLines) > 0 {
		sb.WriteString("Environment, Props & Staging Elements:\n")
		sb.WriteString(strings.Join(propLines, "\n") + "\n")
	}

	if len(lightLines) > 0 {
		sb.WriteString("Lighting & Illumination Vectors:\n")
		sb.WriteString(strings.Join(lightLines, "\n") + "\n")
	} else {
		sb.WriteString("Lighting: Balanced studio/natural cinematic illumination.\n")
	}

	if len(noteLines) > 0 {
		sb.WriteString("Director Scene Directives & Annotations:\n")
		sb.WriteString(strings.Join(noteLines, "\n") + "\n")
	}

	return sb.String(), nil
}

// SynthesizeScenePrompt compiles the scene JSON into a spatial manifest and calls ImageToPrompt if a canvas screenshot (file path or in-memory data URL) is provided,
// falling back to TextToPrompt.
func (s *CatalogService) SynthesizeScenePrompt(ctx context.Context, sceneJSON, targetProfileSlug, userHint string, loraIDs []int64, override domain.PromptExecutionOverride, canvasImage string) (manifest string, prompt string, err error) {
	compiledManifest, err := s.CompileSceneDirectorManifest(ctx, sceneJSON)
	if err != nil {
		return "", "", err
	}

	inputConcept := compiledManifest
	if strings.TrimSpace(userHint) != "" {
		inputConcept = fmt.Sprintf("%s\n\nStyle & Atmosphere Directives:\n%s", compiledManifest, strings.TrimSpace(userHint))
	}

	cfg, err := s.repo.GetSettings(ctx)
	if err != nil {
		return compiledManifest, "", err
	}
	if targetProfileSlug == "" {
		targetProfileSlug = cfg.DefaultTargetGenerator
	}

	var sysPrompt string
	if prof, err := s.repo.GetPromptProfileBySlug(ctx, targetProfileSlug); err == nil && prof != nil {
		sysPrompt = prof.SystemPrompt
	}

	fullUserPrompt := inputConcept
	if loraDoc := s.buildLoRAContextBlock(ctx, targetProfileSlug, loraIDs); loraDoc != "" {
		fullUserPrompt = fmt.Sprintf("%s\n\n%s\nIncorporate the selected LoRA trigger words, stylistic guidance, and syntax tag into the final prompt.", inputConcept, loraDoc)
	}

	// If a canvas screenshot is provided, attempt multimodal vision synthesis so the LLM visually sees the layout
	if strings.TrimSpace(canvasImage) != "" && s.ollama != nil {
		defProv, defMod, defThink := cfg.GetStudioDefaults()
		provider, model, baseURL, think := resolveExecutionRoute(cfg, defProv, defMod, defThink, override)
		visualHint := fmt.Sprintf("This image is a visual canvas capture of a 2D scene staging layout with positioned subjects, poses, lighting, camera framing and environment elements.\n\nSpatial Layout Manifest:\n%s\n\nAnalyze the visual staging in the canvas image together with the spatial manifest to generate an evocative, highly-detailed diffusion prompt matching the exact positions, depth, poses, lighting, and elements depicted in the visual composition.", fullUserPrompt)

		apiKey := cfg.OpenAIAPIKey
		if apiKey == "" {
			apiKey = cfg.OMLXAPIKey
		}

		generated, vErr := s.ollama.ImageToPrompt(ctx, provider, baseURL, apiKey, model, sysPrompt, canvasImage, visualHint, think)
		if vErr == nil && strings.TrimSpace(generated) != "" {
			_, _ = s.repo.SaveGeneratedPrompt(ctx, &domain.GeneratedPrompt{
				TargetProfileSlug: targetProfileSlug,
				SceneDirection:    strings.TrimSpace(inputConcept),
				OutputPrompt:      generated,
				Provider:          provider,
				ModelName:         fmt.Sprintf("%s (canvas vision, think:%s)", model, think),
			})
			return compiledManifest, generated, nil
		}
		log.Printf("[SCENE] Vision prompt synthesis with canvas screenshot failed (%v), falling back to TextToPrompt", vErr)
	}

	generatedPrompt, err := s.RunText2Prompt(ctx, fullUserPrompt, targetProfileSlug, nil, override)
	if err != nil {
		return compiledManifest, "", err
	}

	return compiledManifest, generatedPrompt, nil
}

// ImageToSceneSystemPrompt provides instructions for multimodal models to deconstruct an image into SceneDirector JSON.
const ImageToSceneSystemPrompt = `You are an expert visual scene director and spatial composition analyst.
Your task is to analyze the provided image/photo and deconstruct it into a 2D scene staging composition matching the SceneDirector JSON schema.

Analyze:
1. Aspect ratio: "16:9", "4:3", "1:1", "9:16", or "21:9" based on image orientation.
2. Canvas dimensions: canvasWidth=1024, canvasHeight=768 (or 1024x576 for 16:9, 768x1024 for 9:16, 900x900 for 1:1).
3. Environment:
   - environmentType: "indoor" or "outdoor"
   - environment: vivid description of the background and surroundings (e.g., "cyberpunk alley with neon signage", "sunlit modern living room")
   - timeOfDay: "day", "night", "golden hour", "dusk", "foggy morning", etc.
   - perspective: "front" (default) or "side"
4. Characters / People:
   For every visible person or major humanoid figure:
   - type: "character"
   - subtype: "male", "female", or "child"
   - name: brief descriptive label (e.g. "Protagonist", "Woman in trenchcoat")
   - x, y: center coordinates on canvas (0 to canvasWidth, 0 to canvasHeight)
   - width: default 150 (scaled with character)
   - height: default 270 (scaled with character)
   - scale: 0.6 (distant background) to 1.5 (close foreground). Normal is 1.0.
   - facing: one of "front", "back", "left", "right", "three_quarter_left", "three_quarter_right"
   - poseName: CHOOSE THE CLOSEST MATCH from this list of standard poses:
     "Standing Idle", "Walking / Striding", "Running / Sprint", "Sitting Relaxed",
     "Crouching Action", "Arms Crossed", "Hands on Hips", "Reaching Dynamic",
     "Lying Down", "Kneeling (Both Knees)", "Kneeling (One Knee)", "Crawling / All Fours",
     "Hands Up (Surrender / Cheer)", "Pointing Forward", "Hands on Face (Shock)",
     "Boxing Guard (Defense)", "Sitting Cross-Legged (Lotus)", "High Kick (Martial Arts)",
     "Leaning Against Wall", "Victory (Chest Out)", "Defeated (Collapsed)", "Model Cross-Legged (Runway)",
     "Standing Idle (Side)", "Walking (Side)", "Running (Side)", "Sitting (Side)", "Crouching (Side)"
   - clothing: concise description of attire and colors
   - expression: facial expression or mood
   - hair: hair style and color
   - anatomy: body build / posture notes
5. Props / Objects:
   For major furniture, vehicles, nature, architectural elements:
   - type: "prop"
   - subtype: choose from "sofa", "chair", "table", "bed", "car", "tree", "streetlight", "window", "door", "marker", "circle", "generic"
   - label: concise name (e.g. "Leather Couch", "Sports Car", "Oak Tree")
   - description: brief details
   - x, y: center coordinates
   - width, height: approximate proportional bounding size
   - scale: 1.0 default
6. Lighting:
   Identify 1 or 2 key light sources:
   - type: "light"
   - lightType: "Key Light", "Rim Light", "Ambient Light", or "Neon Glow"
   - x, y: position where the light source is located
   - rotation: angle in degrees shining toward the subject (0-360)
   - color: hex color code (e.g., "#fbbf24" for warm sunlight, "#38bdf8" for cool daylight, "#ec4899" for neon pink)
7. Director Note:
   - directorNote: concise cinematography guidance (camera framing, focal depth, lens, atmosphere)

Output format:
Return ONLY a valid, parseable JSON object matching this schema. Do not wrap in markdown or include conversational text.
{
  "aspectRatio": "16:9",
  "canvasWidth": 1024,
  "canvasHeight": 576,
  "perspective": "front",
  "environmentType": "indoor",
  "environment": "...",
  "timeOfDay": "golden hour",
  "directorNote": "...",
  "elements": [
    ...
  ]
}`

// cleanJSONResponse strips markdown fences and extracts the outermost JSON block.
func cleanJSONResponse(raw string) string {
	raw = strings.TrimSpace(raw)
	if strings.HasPrefix(raw, "```") {
		lines := strings.Split(raw, "\n")
		if len(lines) >= 2 && strings.HasPrefix(lines[0], "```") {
			lines = lines[1:]
		}
		if len(lines) >= 1 && strings.HasPrefix(lines[len(lines)-1], "```") {
			lines = lines[:len(lines)-1]
		}
		raw = strings.TrimSpace(strings.Join(lines, "\n"))
	}
	start := strings.Index(raw, "{")
	end := strings.LastIndex(raw, "}")
	if start != -1 && end != -1 && end > start {
		return raw[start : end+1]
	}
	return raw
}

// getCanvasDimensionsForRatio returns standard width and height for a target aspect ratio matching Scene Director canvas.
func getCanvasDimensionsForRatio(ratio string) (float64, float64) {
	switch ratio {
	case "1:1":
		return 900, 900
	case "9:16":
		return 720, 1280
	case "3:4":
		return 768, 1024
	case "4:3":
		return 1024, 768
	case "21:9":
		return 1440, 617
	case "16:9":
		return 1280, 720
	default:
		return 1024, 768
	}
}

// detectImageDimensionsAndRatio decodes an image config to find its natural dimensions and closest matching standard ratio.
func detectImageDimensionsAndRatio(imageSource string) (string, float64, float64) {
	type ratioCandidate struct {
		name   string
		ratio  float64
		width  float64
		height float64
	}
	candidates := []ratioCandidate{
		{name: "9:16", ratio: 9.0 / 16.0, width: 720, height: 1280},
		{name: "3:4", ratio: 3.0 / 4.0, width: 768, height: 1024},
		{name: "1:1", ratio: 1.0, width: 900, height: 900},
		{name: "4:3", ratio: 4.0 / 3.0, width: 1024, height: 768},
		{name: "16:9", ratio: 16.0 / 9.0, width: 1280, height: 720},
		{name: "21:9", ratio: 21.0 / 9.0, width: 1440, height: 617},
	}

	var imgReader io.Reader
	trimmed := strings.TrimSpace(imageSource)
	if strings.HasPrefix(trimmed, "data:image/") {
		idx := strings.Index(trimmed, ";base64,")
		if idx != -1 {
			b, err := base64.StdEncoding.DecodeString(trimmed[idx+8:])
			if err == nil {
				imgReader = bytes.NewReader(b)
			}
		}
	} else if strings.HasPrefix(trimmed, "base64:") {
		b, err := base64.StdEncoding.DecodeString(strings.TrimPrefix(trimmed, "base64:"))
		if err == nil {
			imgReader = bytes.NewReader(b)
		}
	} else if _, err := os.Stat(trimmed); err == nil {
		if f, err := os.Open(trimmed); err == nil {
			defer f.Close()
			imgReader = f
		}
	}

	if imgReader != nil {
		if cfg, _, err := image.DecodeConfig(imgReader); err == nil && cfg.Width > 0 && cfg.Height > 0 {
			actual := float64(cfg.Width) / float64(cfg.Height)
			best := candidates[4] // default 16:9
			minDiff := math.Abs(math.Log(actual) - math.Log(best.ratio))
			for _, c := range candidates {
				diff := math.Abs(math.Log(actual) - math.Log(c.ratio))
				if diff < minDiff {
					minDiff = diff
					best = c
				}
			}
			return best.name, best.width, best.height
		}
	}

	return "16:9", 1280, 720
}

// ImageToScene analyzes an input image using a multimodal Vision LLM and converts it into a SceneDirectorDTO composition.
func (s *CatalogService) ImageToScene(ctx context.Context, imageSource, userHint, targetAspectRatio string, canvasWidth, canvasHeight float64, override domain.PromptExecutionOverride) (string, *SceneDirectorDTO, error) {
	if s.ollama == nil {
		return "", nil, fmt.Errorf("AI vision provider is not configured")
	}
	if strings.TrimSpace(imageSource) == "" {
		return "", nil, fmt.Errorf("image source cannot be empty")
	}

	cfg, err := s.repo.GetSettings(ctx)
	if err != nil {
		return "", nil, err
	}

	defProv, defMod, defThink := cfg.GetRemixDefaults()
	provider, model, baseURL, think := resolveExecutionRoute(cfg, defProv, defMod, defThink, override)

	apiKey := cfg.OpenAIAPIKey
	if apiKey == "" {
		apiKey = cfg.OMLXAPIKey
	}

	if strings.TrimSpace(targetAspectRatio) == "" || canvasWidth <= 0 || canvasHeight <= 0 {
		detRatio, detW, detH := detectImageDimensionsAndRatio(imageSource)
		if strings.TrimSpace(targetAspectRatio) == "" {
			targetAspectRatio = detRatio
		}
		if canvasWidth <= 0 || canvasHeight <= 0 {
			canvasWidth = detW
			canvasHeight = detH
		}
	}

	userPrompt := fmt.Sprintf("Analyze this image and reconstruct it into the SceneDirector 2D staging JSON format according to your system prompt instructions.\n\nTARGET STAGING CANVAS:\n- Aspect Ratio: %s\n- Canvas Dimensions: %0.0fx%0.0f (width x height)\n- Coordinate System: all element coordinates (x, y) MUST be within x in [0, %0.0f] and y in [0, %0.0f].\n\nEnsure 'aspectRatio' in your JSON is '%s', 'canvasWidth' is %0.0f, and 'canvasHeight' is %0.0f. Return ONLY the JSON object.",
		targetAspectRatio, canvasWidth, canvasHeight, canvasWidth, canvasHeight, targetAspectRatio, canvasWidth, canvasHeight)
	if strings.TrimSpace(userHint) != "" {
		userPrompt += "\nAdditional user directives / hints:\n" + strings.TrimSpace(userHint)
	}

	rawJSON, err := s.ollama.AnalyzeImage(ctx, provider, baseURL, apiKey, model, ImageToSceneSystemPrompt, imageSource, userPrompt, think)
	if err != nil {
		return "", nil, fmt.Errorf("vision analysis failed: %w", err)
	}

	cleaned := cleanJSONResponse(rawJSON)
	var scene SceneDirectorDTO
	if err := json.Unmarshal([]byte(cleaned), &scene); err != nil {
		return cleaned, nil, fmt.Errorf("failed to parse generated scene JSON: %w", err)
	}

	// Validate & sanitize defaults
	scene.AspectRatio = targetAspectRatio
	scene.CanvasWidth = canvasWidth
	scene.CanvasHeight = canvasHeight
	if scene.Perspective == "" {
		scene.Perspective = "front"
	}
	for i := range scene.Elements {
		if scene.Elements[i].ID == "" {
			scene.Elements[i].ID = fmt.Sprintf("el_%d_%d", time.Now().UnixNano(), i)
		}
		if scene.Elements[i].Type == "character" {
			if scene.Elements[i].Width <= 0 {
				scene.Elements[i].Width = 150
			}
			if scene.Elements[i].Height <= 0 {
				scene.Elements[i].Height = 270
			}
			if scene.Elements[i].Scale <= 0 {
				scene.Elements[i].Scale = 1.0
			}
			if scene.Elements[i].Facing == "" {
				scene.Elements[i].Facing = "front"
			}
			if scene.Elements[i].PoseName == "" {
				scene.Elements[i].PoseName = "Standing Idle"
			}
		}
	}

	reencoded, err := json.Marshal(scene)
	if err != nil {
		return cleaned, &scene, nil
	}

	return string(reencoded), &scene, nil
}
