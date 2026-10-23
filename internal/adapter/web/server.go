package web

import (
	"embed"
	"encoding/json"
	"html/template"
	"net/http"
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

	// Read-only media folder delivery and local cache crops
	mux.Handle("GET /media/", http.StripPrefix("/media/", http.FileServer(http.Dir(s.mediaDir))))
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
	mux.HandleFunc("POST /admin/groups/create", s.handleCreateGroup)
	mux.HandleFunc("POST /admin/groups/attach", s.handleAttachGroup)
	mux.HandleFunc("POST /admin/characters/create", s.handleCreateCharacter)
	mux.HandleFunc("POST /admin/characters/attach", s.handleAttachCharacter)

	// Settings, Generator System Prompts & Per-Model LoRA Docs
	mux.HandleFunc("POST /admin/settings/save", s.handleSaveSettings)
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
	VisionEngineName  string
	AdminUser         string
	Filter            domain.CatalogFilter
	Page              *domain.CatalogPage
	SelectedAsset     *domain.MediaAsset
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
	var assetID int64
	if assetIDStr := r.URL.Query().Get("asset_id"); assetIDStr != "" {
		if id, err := strconv.ParseInt(assetIDStr, 10, 64); err == nil && id > 0 {
			if a, err := s.catalog.GetAsset(ctx, id); err == nil {
				selectedAsset = a
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
		VisionEngineName: s.visionEngineName,
		AdminUser:        s.currentAdmin(r),
		Filter:           filter,
		Page:             page,
		SelectedAsset:    selectedAsset,
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
	aiStatus := s.catalog.InspectAIBackends(ctx)

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]interface{}{
		"settings":        cfg,
		"prompt_profiles": profiles,
		"ai_status":       aiStatus,
	})
}

func (s *Server) handleSaveSettings(w http.ResponseWriter, r *http.Request) {
	_ = r.ParseForm()
	cfg := domain.AppSettings{
		OllamaBaseURL:          r.FormValue("ollama_base_url"),
		OMLXBaseURL:            r.FormValue("omlx_base_url"),
		OMLXAPIKey:             r.FormValue("omlx_api_key"),
		VisionProvider:         domain.AIProviderType(r.FormValue("vision_provider")),
		VisionModel:            r.FormValue("vision_model"),
		Image2PromptProvider:   domain.AIProviderType(r.FormValue("image2prompt_provider")),
		Image2PromptModel:      r.FormValue("image2prompt_model"),
		Text2PromptProvider:    domain.AIProviderType(r.FormValue("text2prompt_provider")),
		Text2PromptModel:       r.FormValue("text2prompt_model"),
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

func (s *Server) handleImage2Prompt(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	_ = r.ParseForm()
	assetID, _ := strconv.ParseInt(r.FormValue("asset_id"), 10, 64)
	profileSlug := r.FormValue("profile_slug")
	mode := r.FormValue("mode")
	userHint := r.FormValue("user_hint")
	characterID, _ := strconv.ParseInt(r.FormValue("character_id"), 10, 64)
	loraIDs := parseLoRAIDs(r.Form["lora_id"])
	saveToAsset := r.FormValue("save_to_asset") == "1" || r.FormValue("save_to_asset") == "on"

	outPrompt, runErr := s.catalog.RunImage2Prompt(ctx, assetID, profileSlug, mode, userHint, characterID, loraIDs, saveToAsset)

	filter := parseCatalogFilter(r)
	page, _ := s.catalog.Query(ctx, filter)
	selectedAsset, _ := s.catalog.GetAsset(ctx, assetID)
	settings, _ := s.catalog.GetSettings(ctx)
	profiles, _ := s.catalog.ListPromptProfiles(ctx)
	loras, _ := s.catalog.ListModelLoRAs(ctx, "")
	history, _ := s.catalog.ListGeneratedPrompts(ctx, assetID, 0, 20)
	aiStatus := s.catalog.InspectAIBackends(ctx)

	data := viewData{
		ActiveTab:        "asset_detail",
		MediaDir:         s.mediaDir,
		VisionEngineName: s.visionEngineName,
		AdminUser:        s.currentAdmin(r),
		Filter:           filter,
		Page:             page,
		SelectedAsset:    selectedAsset,
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
	returnTab := r.FormValue("return_tab")
	if returnTab == "" {
		returnTab = "characters"
	}

	outPrompt, runErr := s.catalog.RunCharacterPrompt(ctx, characterID, profileSlug, sceneDirection, refAssetID, loraIDs)

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

	outPrompt, runErr := s.catalog.RunText2Prompt(ctx, concept, profileSlug, loraIDs)

	filter := parseCatalogFilter(r)
	page, _ := s.catalog.Query(ctx, filter)
	settings, _ := s.catalog.GetSettings(ctx)
	profiles, _ := s.catalog.ListPromptProfiles(ctx)
	loras, _ := s.catalog.ListModelLoRAs(ctx, "")
	history, _ := s.catalog.ListGeneratedPrompts(ctx, assetID, 0, 20)
	aiStatus := s.catalog.InspectAIBackends(ctx)

	activeTab := "studio"
	var selectedAsset *domain.MediaAsset
	if assetID > 0 {
		selectedAsset, _ = s.catalog.GetAsset(ctx, assetID)
		activeTab = "asset_detail"
	}

	data := viewData{
		ActiveTab:        activeTab,
		MediaDir:         s.mediaDir,
		VisionEngineName: s.visionEngineName,
		AdminUser:        s.currentAdmin(r),
		Filter:           filter,
		Page:             page,
		SelectedAsset:    selectedAsset,
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
	groupID, _ := strconv.ParseInt(q.Get("group_id"), 10, 64)
	charID, _ := strconv.ParseInt(q.Get("character_id"), 10, 64)
	faceMode := q.Get("face_mode") == "1" || q.Get("face_mode") == "true"

	return domain.CatalogFilter{
		Query:       q.Get("q"),
		Engine:      q.Get("engine"),
		ModelName:   q.Get("model"),
		MediaType:   q.Get("media_type"),
		GroupID:     groupID,
		CharacterID: charID,
		TagNames:    q["tag"],
		FaceMode:    faceMode,
		SortBy:      q.Get("sort"),
		Page:        page,
		PerPage:     perPage,
	}
}
