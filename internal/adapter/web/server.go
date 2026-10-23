package web

import (
	"embed"
	"encoding/json"
	"html/template"
	"net/http"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/jonathanhecl/diffusionCatalog/internal/domain"
	"github.com/jonathanhecl/diffusionCatalog/internal/usecase"
)

//go:embed templates/*.html static/*.css
var embeddedAssets embed.FS

// Server handles HTTP routing, template rendering, and static/media file delivery.
type Server struct {
	mediaDir         string
	cacheDir         string
	visionEngineName string
	catalog          *usecase.CatalogService
	scanner          *usecase.ScannerService
	auth             *usecase.AuthService
	tmpl             *template.Template
}

// NewServer constructs the HTTP adapter with embedded HTML templates and CSS.
func NewServer(
	mediaDir, cacheDir, visionEngineName string,
	catalog *usecase.CatalogService,
	scanner *usecase.ScannerService,
	auth *usecase.AuthService,
) (*Server, error) {
	funcs := template.FuncMap{
		"hasTag": func(selected []string, target string) bool {
			for _, s := range selected {
				if strings.EqualFold(s, target) {
					return true
				}
			}
			return false
		},
		"mul100": func(v float64) float64 {
			return v * 100.0
		},
		"joinStrings": func(items []string) string {
			return strings.Join(items, ", ")
		},
		"promptTokens": func(prompt string) []string {
			if strings.TrimSpace(prompt) == "" {
				return nil
			}
			var out []string
			seen := make(map[string]bool)
			for _, part := range strings.Split(prompt, ",") {
				tok := strings.TrimSpace(part)
				if tok == "" || len(tok) > 54 {
					continue
				}
				k := strings.ToLower(tok)
				if !seen[k] {
					seen[k] = true
					out = append(out, tok)
					if len(out) >= 18 {
						break
					}
				}
			}
			return out
		},
		"pageRange": func(total int) []int {
			if total < 1 {
				total = 1
			}
			if total > 10 {
				total = 10
			}
			out := make([]int, total)
			for i := 0; i < total; i++ {
				out[i] = i + 1
			}
			return out
		},
		"scoreRange": func() []int {
			out := make([]int, 11)
			for i := 0; i <= 10; i++ {
				out[i] = i
			}
			return out
		},
	}

	tmpl, err := template.New("catalog.html").Funcs(funcs).ParseFS(embeddedAssets, "templates/*.html")
	if err != nil {
		return nil, err
	}

	return &Server{
		mediaDir:         mediaDir,
		cacheDir:         cacheDir,
		visionEngineName: visionEngineName,
		catalog:          catalog,
		scanner:          scanner,
		auth:             auth,
		tmpl:             tmpl,
	}, nil
}

// Handler returns the root http.Handler with all routes registered.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()

	// Embedded static stylesheet
	mux.Handle("GET /static/", http.FileServer(http.FS(embeddedAssets)))

	// Read-only media folder delivery across multiple roots/disks and local cache crops
	mux.HandleFunc("GET /media/", s.handleServeMedia)
	mux.Handle("GET /cache/", http.StripPrefix("/cache/", http.FileServer(http.Dir(s.cacheDir))))

	// Main Catalog UI
	mux.HandleFunc("GET /", s.handleCatalogIndex)

	// JSON API endpoints
	mux.HandleFunc("GET /api/v1/catalog", s.handleAPICatalog)
	mux.HandleFunc("GET /api/v1/settings", s.handleAPISettings)

	// Admin, Settings & Organization actions
	mux.HandleFunc("POST /admin/login", s.handleLogin)
	mux.HandleFunc("POST /admin/logout", s.handleLogout)
	mux.HandleFunc("POST /admin/scan", s.handleScan)
	mux.HandleFunc("POST /admin/assets/tag", s.handleAddTag)
	mux.HandleFunc("POST /admin/assets/category", s.handleSetAssetCategory)
	mux.HandleFunc("POST /admin/assets/rate", s.handleRateAsset)
	mux.HandleFunc("POST /admin/groups/create", s.handleCreateGroup)
	mux.HandleFunc("POST /admin/groups/attach", s.handleAttachGroup)
	mux.HandleFunc("POST /admin/characters/create", s.handleCreateCharacter)
	mux.HandleFunc("POST /admin/characters/attach", s.handleAttachCharacter)

	// Settings, Multi-Root Media Folders, Default Categories, DB Switcher, Generator System Prompts & Per-Model LoRA Docs
	mux.HandleFunc("POST /admin/settings/save", s.handleSaveSettings)
	mux.HandleFunc("POST /admin/db/switch", s.handleSwitchDB)
	mux.HandleFunc("POST /admin/roots/add", s.handleAddMediaRoot)
	mux.HandleFunc("POST /admin/roots/delete", s.handleDeleteMediaRoot)
	mux.HandleFunc("POST /admin/categories/upsert", s.handleUpsertDefaultCategory)
	mux.HandleFunc("POST /admin/categories/delete", s.handleDeleteDefaultCategory)
	mux.HandleFunc("POST /admin/profiles/upsert", s.handleUpsertProfile)
	mux.HandleFunc("POST /admin/profiles/delete", s.handleDeleteProfile)
	mux.HandleFunc("POST /admin/loras/upsert", s.handleUpsertLoRA)
	mux.HandleFunc("POST /admin/loras/delete", s.handleDeleteLoRA)

	// Ollama AI Actions (Image2Prompt, Character2Prompt & Text2Prompt)
	mux.HandleFunc("POST /admin/ollama/image2prompt", s.handleImage2Prompt)
	mux.HandleFunc("POST /admin/ollama/character2prompt", s.handleCharacter2Prompt)
	mux.HandleFunc("POST /admin/ollama/text2prompt", s.handleText2Prompt)

	return mux
}

type viewData struct {
	ActiveTab         string
	MediaDir          string
	CurrentDBPath     string
	VisionEngineName  string
	AdminUser         string
	Filter            domain.CatalogFilter
	Page              *domain.CatalogPage
	SelectedAsset     *domain.MediaAsset
	RelatedAssets     usecase.RelatedAssetsBundle
	ScanReport        usecase.ScanReport
	Settings          domain.AppSettings
	PromptProfiles    []domain.PromptProfile
	SelectedGenSlug   string
	SelectedProfile   *domain.PromptProfile
	ModelLoRAs        []domain.ModelLoRA
	GeneratedHistory  []domain.GeneratedPrompt
	AIStatus          usecase.AIBackendStatus
	AIResultPrompt    string
	AIResultError     string
	AIInputConcept    string
	AITargetProfile   string
	AICharacterID     int64
	FlashNotice       string
}

func (s *Server) handleServeMedia(w http.ResponseWriter, r *http.Request) {
	rel := strings.TrimPrefix(r.URL.Path, "/media/")
	if rel == "" || strings.Contains(rel, "..") {
		http.NotFound(w, r)
		return
	}

	rootDir := s.mediaDir
	if strings.HasPrefix(rel, "@root_") {
		slash := strings.IndexByte(rel, '/')
		if slash > 6 {
			rootIDStr := rel[6:slash]
			rel = rel[slash+1:]
			if rootID, err := strconv.ParseInt(rootIDStr, 10, 64); err == nil && rootID > 0 {
				if roots, err := s.catalog.ListMediaRoots(r.Context()); err == nil {
					for _, mr := range roots {
						if mr.ID == rootID && mr.Path != "" {
							rootDir = mr.Path
							break
						}
					}
				}
			}
		}
	}

	fullPath := filepath.Join(rootDir, filepath.FromSlash(rel))
	http.ServeFile(w, r, fullPath)
}

func (s *Server) handleCatalogIndex(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	filter := parseCatalogFilter(r)

	tab := r.URL.Query().Get("tab")
	if tab == "" {
		if filter.FaceMode {
			tab = "faces"
		} else {
			tab = "catalog"
		}
	}

	var selectedAsset *domain.MediaAsset
	var relatedAssets usecase.RelatedAssetsBundle
	var assetID int64
	if assetIDStr := r.URL.Query().Get("asset_id"); assetIDStr != "" {
		if id, err := strconv.ParseInt(assetIDStr, 10, 64); err == nil && id > 0 {
			if a, err := s.catalog.GetAsset(ctx, id); err == nil {
				selectedAsset = a
				relatedAssets = s.catalog.GetRelatedAssets(ctx, a)
				assetID = id
				tab = "asset_detail"
			}
		}
	}

	page, err := s.catalog.Query(ctx, filter)
	if err != nil {
		http.Error(w, "Failed to query catalog: "+err.Error(), http.StatusInternalServerError)
		return
	}

	settings, _ := s.catalog.GetSettings(ctx)
	profiles, _ := s.catalog.ListPromptProfiles(ctx)
	loras, _ := s.catalog.ListModelLoRAs(ctx, "")
	history, _ := s.catalog.ListGeneratedPrompts(ctx, assetID, filter.CharacterID, 20)

	selectedGenSlug := r.URL.Query().Get("gen")
	var selectedProfile *domain.PromptProfile
	if selectedGenSlug != "__new__" {
		for i := range profiles {
			if selectedGenSlug == "" || strings.EqualFold(profiles[i].Slug, selectedGenSlug) {
				selectedProfile = &profiles[i]
				selectedGenSlug = profiles[i].Slug
				break
			}
		}
	}

	var aiStatus usecase.AIBackendStatus
	if tab == "settings" || tab == "studio" || tab == "asset_detail" || tab == "characters" {
		aiStatus = s.catalog.InspectAIBackends(ctx)
	}

	defaultTarget := "flux"
	if len(profiles) > 0 {
		defaultTarget = profiles[0].Slug
	}

	data := viewData{
		ActiveTab:        tab,
		MediaDir:         s.mediaDir,
		CurrentDBPath:    s.catalog.CurrentDBPath(),
		VisionEngineName: s.visionEngineName,
		AdminUser:        s.currentAdmin(r),
		Filter:           filter,
		Page:             page,
		SelectedAsset:    selectedAsset,
		RelatedAssets:    relatedAssets,
		ScanReport:       s.scanner.LastReport(),
		Settings:         settings,
		PromptProfiles:   profiles,
		SelectedGenSlug:  selectedGenSlug,
		SelectedProfile:  selectedProfile,
		ModelLoRAs:       loras,
		GeneratedHistory: history,
		AIStatus:         aiStatus,
		AITargetProfile:  defaultTarget,
		FlashNotice:      r.URL.Query().Get("notice"),
	}

	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	_ = s.tmpl.ExecuteTemplate(w, "catalog.html", data)
}

func (s *Server) handleAPICatalog(w http.ResponseWriter, r *http.Request) {
	filter := parseCatalogFilter(r)
	page, err := s.catalog.Query(r.Context(), filter)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(page)
}

func (s *Server) handleAPISettings(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	cfg, _ := s.catalog.GetSettings(ctx)
	profiles, _ := s.catalog.ListPromptProfiles(ctx)
	roots, _ := s.catalog.ListMediaRoots(ctx)
	cats, _ := s.catalog.ListDefaultCategories(ctx)
	aiStatus := s.catalog.InspectAIBackends(ctx)

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]interface{}{
		"current_db_path":    s.catalog.CurrentDBPath(),
		"settings":           cfg,
		"prompt_profiles":    profiles,
		"media_roots":        roots,
		"default_categories": cats,
		"ai_status":          aiStatus,
	})
}

func (s *Server) handleSwitchDB(w http.ResponseWriter, r *http.Request) {
	_ = r.ParseForm()
	newPath := strings.TrimSpace(r.FormValue("db_path"))
	if newPath == "" {
		http.Redirect(w, r, "/?tab=settings&notice=Database+path+cannot+be+empty", http.StatusSeeOther)
		return
	}
	if err := s.catalog.SwitchDatabase(r.Context(), newPath); err != nil {
		http.Redirect(w, r, "/?tab=settings&notice=Failed+to+switch+DB:+"+err.Error(), http.StatusSeeOther)
		return
	}
	if r.FormValue("rescan") == "1" || r.FormValue("rescan") == "on" {
		_, _ = s.scanner.Scan(r.Context())
	}
	http.Redirect(w, r, "/?tab=settings&notice=Switched+active+global+database+to+"+newPath, http.StatusSeeOther)
}

func (s *Server) handleAddMediaRoot(w http.ResponseWriter, r *http.Request) {
	_ = r.ParseForm()
	path := strings.TrimSpace(r.FormValue("path"))
	label := strings.TrimSpace(r.FormValue("label"))
	if path != "" {
		_, _ = s.catalog.AddMediaRoot(r.Context(), path, label)
		_, _ = s.scanner.Scan(r.Context())
	}
	http.Redirect(w, r, "/?tab=settings&notice=Media+folder+registered+and+scanned", http.StatusSeeOther)
}

func (s *Server) handleDeleteMediaRoot(w http.ResponseWriter, r *http.Request) {
	_ = r.ParseForm()
	id, _ := strconv.ParseInt(r.FormValue("id"), 10, 64)
	if id > 0 {
		_ = s.catalog.DeleteMediaRoot(r.Context(), id)
	}
	http.Redirect(w, r, "/?tab=settings&notice=Media+folder+removed+from+index", http.StatusSeeOther)
}

func (s *Server) handleUpsertDefaultCategory(w http.ResponseWriter, r *http.Request) {
	_ = r.ParseForm()
	name := strings.TrimSpace(r.FormValue("name"))
	facet := strings.TrimSpace(r.FormValue("facet"))
	color := strings.TrimSpace(r.FormValue("color"))
	subs := strings.TrimSpace(r.FormValue("subcategories"))
	if name != "" {
		_, _ = s.catalog.UpsertDefaultCategory(r.Context(), name, facet, color, subs)
	}
	http.Redirect(w, r, "/?tab=settings&notice=Default+category+updated+and+synced+to+folder+manifests", http.StatusSeeOther)
}

func (s *Server) handleDeleteDefaultCategory(w http.ResponseWriter, r *http.Request) {
	_ = r.ParseForm()
	id, _ := strconv.ParseInt(r.FormValue("id"), 10, 64)
	if id > 0 {
		_ = s.catalog.DeleteDefaultCategory(r.Context(), id)
	}
	http.Redirect(w, r, "/?tab=settings&notice=Default+category+deleted", http.StatusSeeOther)
}

func (s *Server) handleSetAssetCategory(w http.ResponseWriter, r *http.Request) {
	_ = r.ParseForm()
	assetID, _ := strconv.ParseInt(r.FormValue("asset_id"), 10, 64)
	cat := strings.TrimSpace(r.FormValue("category"))
	sub := strings.TrimSpace(r.FormValue("subcategory"))
	if assetID > 0 {
		_ = s.catalog.SetAssetCategory(r.Context(), assetID, cat, sub)
	}
	http.Redirect(w, r, "/?asset_id="+strconv.FormatInt(assetID, 10)+"&notice=Asset+classification+saved+to+.diffusion-catalog.json", http.StatusSeeOther)
}

func (s *Server) handleRateAsset(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	_ = r.ParseForm()
	assetID, _ := strconv.ParseInt(r.FormValue("asset_id"), 10, 64)
	if assetID <= 0 {
		http.Redirect(w, r, "/", http.StatusSeeOther)
		return
	}

	current, _ := s.catalog.GetAsset(ctx, assetID)
	rating := 0
	favorite := false
	if current != nil {
		rating = current.Rating
		favorite = current.Favorite
	}

	if rStr := strings.TrimSpace(r.FormValue("rating")); rStr != "" {
		if parsed, err := strconv.Atoi(rStr); err == nil {
			rating = parsed
		}
	}
	if fStr := strings.TrimSpace(r.FormValue("favorite")); fStr != "" {
		favorite = fStr == "1" || fStr == "true" || fStr == "on"
	} else if r.FormValue("toggle_favorite") == "1" {
		favorite = !favorite
	}

	_ = s.catalog.RateAsset(ctx, assetID, rating, favorite)

	if ret := strings.TrimSpace(r.FormValue("return_url")); ret != "" && strings.HasPrefix(ret, "/") {
		http.Redirect(w, r, ret, http.StatusSeeOther)
		return
	}
	http.Redirect(w, r, "/?asset_id="+strconv.FormatInt(assetID, 10)+"&notice=Score+and+favorite+saved+to+.diffusion-catalog.json", http.StatusSeeOther)
}

func (s *Server) handleSaveSettings(w http.ResponseWriter, r *http.Request) {
	_ = r.ParseForm()
	cfg := domain.AppSettings{
		OllamaBaseURL:          r.FormValue("ollama_base_url"),
		OMLXBaseURL:            r.FormValue("omlx_base_url"),
		OMLXAPIKey:             r.FormValue("omlx_api_key"),
		VisionProvider:         domain.AIProviderType(r.FormValue("vision_provider")),
		VisionModel:            r.FormValue("vision_model"),
		VisionThink:            r.FormValue("vision_think"),
		Image2PromptProvider:   domain.AIProviderType(r.FormValue("image2prompt_provider")),
		Image2PromptModel:      r.FormValue("image2prompt_model"),
		Image2PromptThink:      r.FormValue("image2prompt_think"),
		Text2PromptProvider:    domain.AIProviderType(r.FormValue("text2prompt_provider")),
		Text2PromptModel:       r.FormValue("text2prompt_model"),
		Text2PromptThink:       r.FormValue("text2prompt_think"),
		DefaultTargetGenerator: r.FormValue("default_target_generator"),
		AutoCaptionOnScan:      r.FormValue("auto_caption_on_scan") == "on" || r.FormValue("auto_caption_on_scan") == "true",
	}
	_ = s.catalog.SaveSettings(r.Context(), cfg)
	http.Redirect(w, r, "/?tab=settings&notice=Settings+saved+successfully", http.StatusSeeOther)
}

func (s *Server) handleUpsertProfile(w http.ResponseWriter, r *http.Request) {
	_ = r.ParseForm()
	p := &domain.PromptProfile{
		Slug:           r.FormValue("slug"),
		Name:           r.FormValue("name"),
		TargetModality: domain.MediaType(r.FormValue("target_modality")),
		Description:    r.FormValue("description"),
		SystemPrompt:   r.FormValue("system_prompt"),
	}
	if p.Name != "" && p.SystemPrompt != "" {
		_, _ = s.catalog.UpsertPromptProfile(r.Context(), p)
	}
	http.Redirect(w, r, "/?tab=generators&gen="+p.Slug+"&notice=Generator+model+profile+saved", http.StatusSeeOther)
}

func (s *Server) handleDeleteProfile(w http.ResponseWriter, r *http.Request) {
	_ = r.ParseForm()
	id, _ := strconv.ParseInt(r.FormValue("id"), 10, 64)
	if id > 0 {
		_ = s.catalog.DeletePromptProfile(r.Context(), id)
	}
	http.Redirect(w, r, "/?tab=generators&notice=Generator+model+deleted", http.StatusSeeOther)
}

func (s *Server) handleUpsertLoRA(w http.ResponseWriter, r *http.Request) {
	_ = r.ParseForm()
	l := &domain.ModelLoRA{
		ProfileSlug:       r.FormValue("profile_slug"),
		Name:              r.FormValue("name"),
		SyntaxTag:         r.FormValue("syntax_tag"),
		TriggerWords:      r.FormValue("trigger_words"),
		RecommendedWeight: r.FormValue("recommended_weight"),
		Description:       r.FormValue("description"),
		UsageExamples:     r.FormValue("usage_examples"),
	}
	if l.ProfileSlug != "" && l.Name != "" {
		_, _ = s.catalog.UpsertModelLoRA(r.Context(), l)
	}
	http.Redirect(w, r, "/?tab=generators&gen="+l.ProfileSlug+"&notice=LoRA+documentation+saved", http.StatusSeeOther)
}

func (s *Server) handleDeleteLoRA(w http.ResponseWriter, r *http.Request) {
	_ = r.ParseForm()
	id, _ := strconv.ParseInt(r.FormValue("id"), 10, 64)
	genSlug := r.FormValue("profile_slug")
	if id > 0 {
		_ = s.catalog.DeleteModelLoRA(r.Context(), id)
	}
	http.Redirect(w, r, "/?tab=generators&gen="+genSlug+"&notice=LoRA+deleted", http.StatusSeeOther)
}

func parseLoRAIDs(vals []string) []int64 {
	var out []int64
	for _, v := range vals {
		if id, err := strconv.ParseInt(strings.TrimSpace(v), 10, 64); err == nil && id > 0 {
			out = append(out, id)
		}
	}
	return out
}

func parseExecutionOverride(r *http.Request) domain.PromptExecutionOverride {
	var ov domain.PromptExecutionOverride
	route := strings.TrimSpace(r.FormValue("override_route"))
	if route != "" {
		if parts := strings.SplitN(route, "|", 2); len(parts) == 2 {
			ov.Provider = domain.AIProviderType(parts[0])
			ov.Model = parts[1]
		} else {
			ov.Model = route
		}
	} else {
		ov.Provider = domain.AIProviderType(strings.TrimSpace(r.FormValue("override_provider")))
		ov.Model = strings.TrimSpace(r.FormValue("override_model"))
	}
	ov.ThinkLevel = strings.TrimSpace(r.FormValue("override_think"))
	return ov
}

func (s *Server) handleImage2Prompt(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	_ = r.ParseForm()
	assetID, _ := strconv.ParseInt(r.FormValue("asset_id"), 10, 64)
	profileSlug := r.FormValue("profile_slug")
	mode := r.FormValue("mode")
	userHint := r.FormValue("user_hint")
	characterID, _ := strconv.ParseInt(r.FormValue("character_id"), 10, 64)
	loraIDs := parseLoRAIDs(r.Form["lora_id"])
	override := parseExecutionOverride(r)
	saveToAsset := r.FormValue("save_to_asset") == "1" || r.FormValue("save_to_asset") == "on"

	outPrompt, runErr := s.catalog.RunImage2Prompt(ctx, assetID, profileSlug, mode, userHint, characterID, loraIDs, override, saveToAsset)

	filter := parseCatalogFilter(r)
	page, _ := s.catalog.Query(ctx, filter)
	selectedAsset, _ := s.catalog.GetAsset(ctx, assetID)
	relatedAssets := s.catalog.GetRelatedAssets(ctx, selectedAsset)
	settings, _ := s.catalog.GetSettings(ctx)
	profiles, _ := s.catalog.ListPromptProfiles(ctx)
	loras, _ := s.catalog.ListModelLoRAs(ctx, "")
	history, _ := s.catalog.ListGeneratedPrompts(ctx, assetID, 0, 20)
	aiStatus := s.catalog.InspectAIBackends(ctx)

	data := viewData{
		ActiveTab:        "asset_detail",
		MediaDir:         s.mediaDir,
		CurrentDBPath:    s.catalog.CurrentDBPath(),
		VisionEngineName: s.visionEngineName,
		AdminUser:        s.currentAdmin(r),
		Filter:           filter,
		Page:             page,
		SelectedAsset:    selectedAsset,
		RelatedAssets:    relatedAssets,
		ScanReport:       s.scanner.LastReport(),
		Settings:         settings,
		PromptProfiles:   profiles,
		ModelLoRAs:       loras,
		GeneratedHistory: history,
		AIStatus:         aiStatus,
		AIResultPrompt:   outPrompt,
		AITargetProfile:  profileSlug,
		AICharacterID:    characterID,
	}
	if runErr != nil {
		data.AIResultError = runErr.Error()
	}

	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	_ = s.tmpl.ExecuteTemplate(w, "catalog.html", data)
}

func (s *Server) handleCharacter2Prompt(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	_ = r.ParseForm()
	characterID, _ := strconv.ParseInt(r.FormValue("character_id"), 10, 64)
	profileSlug := r.FormValue("profile_slug")
	sceneDirection := r.FormValue("scene_direction")
	refAssetID, _ := strconv.ParseInt(r.FormValue("ref_asset_id"), 10, 64)
	loraIDs := parseLoRAIDs(r.Form["lora_id"])
	override := parseExecutionOverride(r)
	returnTab := r.FormValue("return_tab")
	if returnTab == "" {
		returnTab = "characters"
	}

	outPrompt, runErr := s.catalog.RunCharacterPrompt(ctx, characterID, profileSlug, sceneDirection, refAssetID, loraIDs, override)

	filter := parseCatalogFilter(r)
	page, _ := s.catalog.Query(ctx, filter)
	settings, _ := s.catalog.GetSettings(ctx)
	profiles, _ := s.catalog.ListPromptProfiles(ctx)
	loras, _ := s.catalog.ListModelLoRAs(ctx, "")
	history, _ := s.catalog.ListGeneratedPrompts(ctx, 0, 0, 20)
	aiStatus := s.catalog.InspectAIBackends(ctx)

	data := viewData{
		ActiveTab:        returnTab,
		MediaDir:         s.mediaDir,
		CurrentDBPath:    s.catalog.CurrentDBPath(),
		VisionEngineName: s.visionEngineName,
		AdminUser:        s.currentAdmin(r),
		Filter:           filter,
		Page:             page,
		ScanReport:       s.scanner.LastReport(),
		Settings:         settings,
		PromptProfiles:   profiles,
		ModelLoRAs:       loras,
		GeneratedHistory: history,
		AIStatus:         aiStatus,
		AIResultPrompt:   outPrompt,
		AIInputConcept:   sceneDirection,
		AITargetProfile:  profileSlug,
		AICharacterID:    characterID,
	}
	if runErr != nil {
		data.AIResultError = runErr.Error()
	}

	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	_ = s.tmpl.ExecuteTemplate(w, "catalog.html", data)
}

func (s *Server) handleText2Prompt(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	_ = r.ParseForm()
	assetID, _ := strconv.ParseInt(r.FormValue("asset_id"), 10, 64)
	profileSlug := r.FormValue("profile_slug")
	concept := r.FormValue("concept")
	loraIDs := parseLoRAIDs(r.Form["lora_id"])
	override := parseExecutionOverride(r)

	outPrompt, runErr := s.catalog.RunText2Prompt(ctx, concept, profileSlug, loraIDs, override)

	filter := parseCatalogFilter(r)
	page, _ := s.catalog.Query(ctx, filter)
	settings, _ := s.catalog.GetSettings(ctx)
	profiles, _ := s.catalog.ListPromptProfiles(ctx)
	loras, _ := s.catalog.ListModelLoRAs(ctx, "")
	history, _ := s.catalog.ListGeneratedPrompts(ctx, assetID, 0, 20)
	aiStatus := s.catalog.InspectAIBackends(ctx)

	activeTab := "studio"
	var selectedAsset *domain.MediaAsset
	var relatedAssets usecase.RelatedAssetsBundle
	if assetID > 0 {
		selectedAsset, _ = s.catalog.GetAsset(ctx, assetID)
		relatedAssets = s.catalog.GetRelatedAssets(ctx, selectedAsset)
		activeTab = "asset_detail"
	}

	data := viewData{
		ActiveTab:        activeTab,
		MediaDir:         s.mediaDir,
		CurrentDBPath:    s.catalog.CurrentDBPath(),
		VisionEngineName: s.visionEngineName,
		AdminUser:        s.currentAdmin(r),
		Filter:           filter,
		Page:             page,
		SelectedAsset:    selectedAsset,
		RelatedAssets:    relatedAssets,
		ScanReport:       s.scanner.LastReport(),
		Settings:         settings,
		PromptProfiles:   profiles,
		ModelLoRAs:       loras,
		GeneratedHistory: history,
		AIStatus:         aiStatus,
		AIResultPrompt:   outPrompt,
		AIInputConcept:   concept,
		AITargetProfile:  profileSlug,
	}
	if runErr != nil {
		data.AIResultError = runErr.Error()
	}

	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	_ = s.tmpl.ExecuteTemplate(w, "catalog.html", data)
}

func (s *Server) handleLogin(w http.ResponseWriter, r *http.Request) {
	_ = r.ParseForm()
	user := r.FormValue("username")
	pass := r.FormValue("password")

	token, err := s.auth.Authenticate(r.Context(), user, pass)
	if err != nil {
		http.Redirect(w, r, "/?tab=login&error=1", http.StatusSeeOther)
		return
	}

	http.SetCookie(w, &http.Cookie{
		Name:     "dc_admin_session",
		Value:    token,
		Path:     "/",
		HttpOnly: true,
	})
	http.Redirect(w, r, "/", http.StatusSeeOther)
}

func (s *Server) handleLogout(w http.ResponseWriter, r *http.Request) {
	http.SetCookie(w, &http.Cookie{
		Name:   "dc_admin_session",
		Value:  "",
		Path:   "/",
		MaxAge: -1,
	})
	http.Redirect(w, r, "/", http.StatusSeeOther)
}

func (s *Server) handleScan(w http.ResponseWriter, r *http.Request) {
	_, _ = s.scanner.Scan(r.Context())
	http.Redirect(w, r, "/", http.StatusSeeOther)
}

func (s *Server) handleAddTag(w http.ResponseWriter, r *http.Request) {
	_ = r.ParseForm()
	assetID, _ := strconv.ParseInt(r.FormValue("asset_id"), 10, 64)
	name := r.FormValue("name")
	facet := r.FormValue("facet")
	if assetID > 0 && name != "" {
		_ = s.catalog.AddTagToAsset(r.Context(), assetID, name, facet)
	}
	http.Redirect(w, r, "/?asset_id="+strconv.FormatInt(assetID, 10), http.StatusSeeOther)
}

func (s *Server) handleCreateGroup(w http.ResponseWriter, r *http.Request) {
	_ = r.ParseForm()
	name := r.FormValue("name")
	category := r.FormValue("category")
	desc := r.FormValue("description")
	if name != "" {
		_, _ = s.catalog.CreateGroup(r.Context(), name, category, desc, 0)
	}
	http.Redirect(w, r, "/?tab=groups", http.StatusSeeOther)
}

func (s *Server) handleAttachGroup(w http.ResponseWriter, r *http.Request) {
	_ = r.ParseForm()
	assetID, _ := strconv.ParseInt(r.FormValue("asset_id"), 10, 64)
	groupID, _ := strconv.ParseInt(r.FormValue("group_id"), 10, 64)
	if assetID > 0 && groupID > 0 {
		_ = s.catalog.AssignAssetToGroup(r.Context(), groupID, assetID)
	}
	http.Redirect(w, r, "/?asset_id="+strconv.FormatInt(assetID, 10), http.StatusSeeOther)
}

func (s *Server) handleCreateCharacter(w http.ResponseWriter, r *http.Request) {
	_ = r.ParseForm()
	c := &domain.Character{
		Name:                  r.FormValue("name"),
		Archetype:             r.FormValue("archetype"),
		Description:           r.FormValue("description"),
		TriggerWords:          r.FormValue("trigger_words"),
		BasePositivePrompt:    r.FormValue("base_positive_prompt"),
		BaseNegativePrompt:    r.FormValue("base_negative_prompt"),
		RecommendedCheckpoint: r.FormValue("recommended_checkpoint"),
	}
	if c.Name != "" {
		_, _ = s.catalog.CreateCharacter(r.Context(), c, 0)
	}
	http.Redirect(w, r, "/?tab=characters", http.StatusSeeOther)
}

func (s *Server) handleAttachCharacter(w http.ResponseWriter, r *http.Request) {
	_ = r.ParseForm()
	assetID, _ := strconv.ParseInt(r.FormValue("asset_id"), 10, 64)
	charID, _ := strconv.ParseInt(r.FormValue("character_id"), 10, 64)
	if assetID > 0 && charID > 0 {
		_ = s.catalog.AssignAssetToCharacter(r.Context(), charID, assetID)
	}
	http.Redirect(w, r, "/?asset_id="+strconv.FormatInt(assetID, 10), http.StatusSeeOther)
}

func (s *Server) currentAdmin(r *http.Request) string {
	c, err := r.Cookie("dc_admin_session")
	if err != nil {
		return "admin (default)"
	}
	if user, ok := s.auth.VerifyToken(c.Value); ok {
		return user
	}
	return "admin (default)"
}

func parseCatalogFilter(r *http.Request) domain.CatalogFilter {
	q := r.URL.Query()
	page, _ := strconv.Atoi(q.Get("page"))
	if page < 1 {
		page = 1
	}
	perPage, _ := strconv.Atoi(q.Get("per_page"))
	if perPage < 1 {
		perPage = 50
	}
	rootID, _ := strconv.ParseInt(q.Get("root_id"), 10, 64)
	groupID, _ := strconv.ParseInt(q.Get("group_id"), 10, 64)
	charID, _ := strconv.ParseInt(q.Get("character_id"), 10, 64)
	minRating, _ := strconv.Atoi(q.Get("min_rating"))
	favoritesOnly := q.Get("favorites") == "1" || q.Get("favorites") == "true" || q.Get("favorites") == "on"
	faceMode := q.Get("face_mode") == "1" || q.Get("face_mode") == "true"

	return domain.CatalogFilter{
		Query:         q.Get("q"),
		Trigger:       q.Get("trigger"),
		RootID:        rootID,
		Category:      q.Get("category"),
		Subcategory:   q.Get("subcategory"),
		MinRating:     minRating,
		FavoritesOnly: favoritesOnly,
		Engine:        q.Get("engine"),
		ModelName:     q.Get("model"),
		MediaType:     q.Get("media_type"),
		GroupID:       groupID,
		CharacterID:   charID,
		TagNames:      q["tag"],
		FaceMode:      faceMode,
		SortBy:        q.Get("sort"),
		Page:          page,
		PerPage:       perPage,
	}
}
