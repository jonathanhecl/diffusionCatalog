package usecase

import (
	"context"
	"fmt"
	"path/filepath"
	"strings"

	"github.com/jonathanhecl/diffusionCatalog/internal/domain"
)

// CatalogService coordinates catalog querying, agnostic tagging, grouping, character sheets,
// application settings, and Ollama prompt generation (Vision, Image2Prompt, Text2Prompt).
type CatalogService struct {
	mediaDir string
	repo     domain.CatalogRepository
	ollama   domain.OllamaProvider
}

// NewCatalogService creates a new CatalogService.
func NewCatalogService(mediaDir string, repo domain.CatalogRepository, ollama domain.OllamaProvider) *CatalogService {
	return &CatalogService{
		mediaDir: mediaDir,
		repo:     repo,
		ollama:   ollama,
	}
}

// Query returns a paginated, facet-filtered catalog page.
func (s *CatalogService) Query(ctx context.Context, filter domain.CatalogFilter) (*domain.CatalogPage, error) {
	return s.repo.QueryCatalog(ctx, filter)
}

// GetAsset fetches an asset by ID and increments its view count.
func (s *CatalogService) GetAsset(ctx context.Context, id int64) (*domain.MediaAsset, error) {
	_ = s.repo.IncrementAssetView(ctx, id)
	return s.repo.GetAssetByID(ctx, id)
}

// AddTagToAsset creates or updates an agnostic tag and attaches it to an asset.
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

// InspectOllama checks if the configured Ollama instance is reachable and lists installed models with capabilities.
func (s *CatalogService) InspectOllama(ctx context.Context) (string, bool, []domain.OllamaModelInfo) {
	if s.ollama == nil {
		return "", false, nil
	}
	cfg, _ := s.repo.GetSettings(ctx)
	ver, online := s.ollama.CheckStatus(ctx, cfg.OllamaBaseURL)
	if !online {
		return "", false, nil
	}
	models, _ := s.ollama.ListModels(ctx, cfg.OllamaBaseURL)
	return ver, true, models
}

// RunImage2Prompt runs the configured Ollama Image2Prompt (or Vision) model on an asset using the selected
// generator profile's system prompt (e.g., flux, krea2, moss, minimaxh3).
func (s *CatalogService) RunImage2Prompt(ctx context.Context, assetID int64, profileSlug, userHint string, saveToAsset bool) (string, error) {
	if s.ollama == nil {
		return "", fmt.Errorf("ollama provider is not initialized")
	}
	asset, err := s.repo.GetAssetByID(ctx, assetID)
	if err != nil {
		return "", err
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

	model := cfg.Image2PromptModel
	if model == "" {
		model = cfg.VisionModel
	}

	absPath := filepath.Join(s.mediaDir, filepath.FromSlash(asset.RelPath))
	generated, err := s.ollama.ImageToPrompt(ctx, cfg.OllamaBaseURL, model, sysPrompt, absPath, userHint)
	if err != nil {
		return "", err
	}

	if saveToAsset && generated != "" {
		_ = s.repo.UpdateAssetPrompt(ctx, asset.ID, generated, profileSlug)
	}
	return generated, nil
}

// RunText2Prompt expands or converts a text idea/prompt into a target generator prompt using the configured
// Ollama Text2Prompt model and the selected generator profile's system prompt.
func (s *CatalogService) RunText2Prompt(ctx context.Context, inputConcept, profileSlug string) (string, error) {
	if s.ollama == nil {
		return "", fmt.Errorf("ollama provider is not initialized")
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

	return s.ollama.TextToPrompt(ctx, cfg.OllamaBaseURL, cfg.Text2PromptModel, sysPrompt, inputConcept)
}
