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
	OllamaOnline  bool
	OllamaVersion string
	OllamaModels  []domain.OllamaModelInfo
	OMLXOnline    bool
	OMLXModels    []domain.OllamaModelInfo
	AllModels     []domain.OllamaModelInfo
}

// InspectAIBackends checks both the configured Ollama server and oMLX server and merges their model lists.
func (s *CatalogService) InspectAIBackends(ctx context.Context) AIBackendStatus {
	var st AIBackendStatus
	if s.ollama == nil {
		return st
	}
	cfg, _ := s.repo.GetSettings(ctx)

	if strings.TrimSpace(cfg.OllamaBaseURL) != "" {
		ver, online := s.ollama.CheckStatus(ctx, cfg.OllamaBaseURL)
		st.OllamaOnline = online
		st.OllamaVersion = ver
		if online {
			st.OllamaModels, _ = s.ollama.ListModels(ctx, cfg.OllamaBaseURL)
			st.AllModels = append(st.AllModels, st.OllamaModels...)
		}
	}

	if strings.TrimSpace(cfg.OMLXBaseURL) != "" {
		models, err := s.ollama.ListOMLXModels(ctx, cfg.OMLXBaseURL, cfg.OMLXAPIKey)
		if err == nil {
			st.OMLXOnline = true
			st.OMLXModels = models
			st.AllModels = append(st.AllModels, models...)
		}
	}

	return st
}

// ListGeneratedPrompts returns stored prompts generated from assets or characters.
func (s *CatalogService) ListGeneratedPrompts(ctx context.Context, assetID, characterID int64, limit int) ([]domain.GeneratedPrompt, error) {
	return s.repo.ListGeneratedPrompts(ctx, assetID, characterID, limit)
}

// resolveExecutionRoute resolves the provider, model, baseURL, and thinking level for a prompt generation task,
// applying any per-request quick overrides on top of the persistent Settings defaults.
func resolveExecutionRoute(cfg domain.AppSettings, defaultProvider domain.AIProviderType, defaultModel, defaultThink string, override domain.PromptExecutionOverride) (domain.AIProviderType, string, string, string) {
	prov := defaultProvider
	model := defaultModel
	think := defaultThink

	if override.Provider != "" && override.Model != "" {
		prov = override.Provider
		model = override.Model
	} else if override.Model != "" {
		model = override.Model
	}
	if prov == "" {
		prov = domain.ProviderOllama
	}

	if override.ThinkLevel != "" && override.ThinkLevel != "settings" {
		think = override.ThinkLevel
	}
	if think == "" {
		think = "off"
	}

	baseURL := cfg.OllamaBaseURL
	if prov == domain.ProviderOMLX {
		baseURL = cfg.OMLXBaseURL
	}
	return prov, model, baseURL, think
}

// RunImage2Prompt runs the configured Ollama or oMLX Image2Prompt (or Remix Text2Prompt) on an asset
// using the selected generator profile's system prompt (e.g., flux, krea2, moss, minimaxh3), selected model LoRAs,
// optional character/scene direction, and optional quick model/thinking overrides.
func (s *CatalogService) RunImage2Prompt(ctx context.Context, assetID int64, profileSlug, mode, userHint string, characterID int64, loraIDs []int64, override domain.PromptExecutionOverride, saveToAsset bool) (string, error) {
	if s.ollama == nil {
		return "", fmt.Errorf("AI provider is not initialized")
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

	// Build rich context combining existing asset metadata, optional character sheet, selected LoRA docs, and user scene direction
	var contextParts []string
	if asset.Metadata.PositivePrompt != "" {
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

	if mode == "remix" && combinedHint != "" {
		// Remix via Text2Prompt model using the extracted prompt + character + direction
		var baseURL string
		usedProvider, usedModel, baseURL, usedThink = resolveExecutionRoute(
			cfg, cfg.Text2PromptProvider, cfg.Text2PromptModel, cfg.Text2PromptThink, override,
		)
		promptInstruction := "Transform the following reference asset and scene direction into a final generation prompt for target engine '" + profileSlug + "'. Preserve any mandatory character trigger words if present:\n\n" + combinedHint
		generated, err = s.ollama.TextToPrompt(ctx, usedProvider, baseURL, cfg.OMLXAPIKey, usedModel, sysPrompt, promptInstruction, usedThink)
		if err != nil {
			return "", err
		}
	} else {
		// Vision analysis via Image2Prompt model
		defProv := cfg.Image2PromptProvider
		defMod := cfg.Image2PromptModel
		defThink := cfg.Image2PromptThink
		if defMod == "" {
			defProv = cfg.VisionProvider
			defMod = cfg.VisionModel
			defThink = cfg.VisionThink
		}
		var baseURL string
		usedProvider, usedModel, baseURL, usedThink = resolveExecutionRoute(cfg, defProv, defMod, defThink, override)
		absPath := filepath.Join(s.mediaDir, filepath.FromSlash(asset.RelPath))
		generated, err = s.ollama.ImageToPrompt(ctx, usedProvider, baseURL, cfg.OMLXAPIKey, usedModel, sysPrompt, absPath, combinedHint, usedThink)
		if err != nil {
			return "", err
		}
	}

	if generated != "" {
		aid := asset.ID
		_, _ = s.repo.SaveGeneratedPrompt(ctx, &domain.GeneratedPrompt{
			AssetID:           &aid,
			CharacterID:       charPtr,
			TargetProfileSlug: profileSlug,
			SceneDirection:    strings.TrimSpace(userHint),
			OutputPrompt:      generated,
			Provider:          usedProvider,
			ModelName:         fmt.Sprintf("%s (think:%s)", usedModel, usedThink),
		})
	}

	if saveToAsset && generated != "" {
		_ = s.repo.UpdateAssetPrompt(ctx, asset.ID, generated, profileSlug)
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

	if refAssetID > 0 {
		if refAsset, err := s.repo.GetAssetByID(ctx, refAssetID); err == nil && refAsset != nil {
			aid := refAsset.ID
			assetPtr = &aid
			defProv := cfg.Image2PromptProvider
			defMod := cfg.Image2PromptModel
			defThink := cfg.Image2PromptThink
			if defMod == "" {
				defProv = cfg.VisionProvider
				defMod = cfg.VisionModel
				defThink = cfg.VisionThink
			}
			var baseURL string
			usedProvider, usedModel, baseURL, usedThink = resolveExecutionRoute(cfg, defProv, defMod, defThink, override)
			absPath := filepath.Join(s.mediaDir, filepath.FromSlash(refAsset.RelPath))
			generated, err = s.ollama.ImageToPrompt(ctx, usedProvider, baseURL, cfg.OMLXAPIKey, usedModel, sysPrompt, absPath, charSpec, usedThink)
			if err != nil {
				return "", err
			}
		}
	}

	if generated == "" {
		var baseURL string
		usedProvider, usedModel, baseURL, usedThink = resolveExecutionRoute(
			cfg, cfg.Text2PromptProvider, cfg.Text2PromptModel, cfg.Text2PromptThink, override,
		)
		generated, err = s.ollama.TextToPrompt(ctx, usedProvider, baseURL, cfg.OMLXAPIKey, usedModel, sysPrompt, charSpec, usedThink)
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

	provider, model, baseURL, think := resolveExecutionRoute(
		cfg, cfg.Text2PromptProvider, cfg.Text2PromptModel, cfg.Text2PromptThink, override,
	)

	generated, err := s.ollama.TextToPrompt(ctx, provider, baseURL, cfg.OMLXAPIKey, model, sysPrompt, fullUserPrompt, think)
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
