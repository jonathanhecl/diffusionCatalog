package web

import (
	"embed"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"html/template"
	"io"
	"log"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

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

// PaginationItem describes a single pagination button or ellipsis gap.
type PaginationItem struct {
	Page     int
	Label    string
	IsActive bool
	IsGap    bool
}

func pageURL(filter domain.CatalogFilter, pageNum int) string {
	q := url.Values{}
	if pageNum > 0 {
		q.Set("page", strconv.Itoa(pageNum))
	}
	if filter.PerPage > 0 && filter.PerPage != 50 {
		q.Set("per_page", strconv.Itoa(filter.PerPage))
	}
	if qStr := strings.TrimSpace(filter.Query); qStr != "" {
		q.Set("q", qStr)
	}
	if trig := strings.TrimSpace(filter.Trigger); trig != "" {
		q.Set("trigger", trig)
	}
	if filter.RootID > 0 {
		q.Set("root_id", strconv.FormatInt(filter.RootID, 10))
	}
	if cat := strings.TrimSpace(filter.Category); cat != "" {
		q.Set("category", cat)
	}
	if subcat := strings.TrimSpace(filter.Subcategory); subcat != "" {
		q.Set("subcategory", subcat)
	}
	if filter.MinRating > 0 {
		q.Set("min_rating", strconv.Itoa(filter.MinRating))
	}
	if filter.FavoritesOnly {
		q.Set("favorites", "1")
	}
	if eng := strings.TrimSpace(filter.Engine); eng != "" {
		q.Set("engine", eng)
	}
	if mod := strings.TrimSpace(filter.ModelName); mod != "" {
		q.Set("model", mod)
	}
	if mt := strings.TrimSpace(filter.MediaType); mt != "" {
		q.Set("media_type", mt)
	}
	if filter.GroupID > 0 {
		q.Set("group_id", strconv.FormatInt(filter.GroupID, 10))
	}
	if filter.CharacterID > 0 {
		q.Set("character_id", strconv.FormatInt(filter.CharacterID, 10))
	}
	for _, tag := range filter.TagNames {
		if t := strings.TrimSpace(tag); t != "" {
			q.Add("tag", t)
		}
	}
	if filter.FaceMode {
		q.Set("face_mode", "1")
	}
	if filter.SortBy != "" && filter.SortBy != "newest" {
		q.Set("sort", filter.SortBy)
	}
	encoded := q.Encode()
	if encoded == "" {
		return "/"
	}
	return "/?" + encoded
}

func sortURL(filter domain.CatalogFilter, sortBy string) string {
	f := filter
	f.SortBy = sortBy
	f.Page = 1
	return pageURL(f, 1)
}

func assetDetailURL(filter domain.CatalogFilter, assetID int64) string {
	pageNum := filter.Page
	if pageNum < 1 {
		pageNum = 1
	}
	base := pageURL(filter, pageNum)
	sep := "?"
	if strings.Contains(base, "?") {
		sep = "&"
	}
	return base + sep + "asset_id=" + strconv.FormatInt(assetID, 10)
}

func paginationList(current, total int) []PaginationItem {
	if total <= 1 {
		return nil
	}
	if current < 1 {
		current = 1
	}
	if current > total {
		current = total
	}

	pageSet := make(map[int]bool)
	pageSet[1] = true
	pageSet[total] = true
	for i := current - 2; i <= current + 2; i++ {
		if i >= 1 && i <= total {
			pageSet[i] = true
		}
	}

	if current <= 3 {
		for i := 1; i <= 5 && i <= total; i++ {
			pageSet[i] = true
		}
	}
	if current >= total-2 {
		for i := total - 4; i <= total; i++ {
			if i >= 1 {
				pageSet[i] = true
			}
		}
	}

	var pages []int
	for p := 1; p <= total; p++ {
		if pageSet[p] {
			pages = append(pages, p)
		}
	}

	var items []PaginationItem
	for i, p := range pages {
		if i > 0 && p > pages[i-1]+1 {
			items = append(items, PaginationItem{
				IsGap: true,
				Label: "…",
			})
		}
		items = append(items, PaginationItem{
			Page:     p,
			Label:    strconv.Itoa(p),
			IsActive: p == current,
		})
	}
	return items
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
		"add": func(a, b int) int {
			return a + b
		},
		"sub": func(a, b int) int {
			return a - b
		},
		"pageURL":        pageURL,
		"sortURL":        sortURL,
		"assetDetailURL": assetDetailURL,
		"paginationList": paginationList,
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
	mux.HandleFunc("POST /admin/assets/tag/remove", s.handleRemoveTag)
	mux.HandleFunc("POST /admin/assets/category", s.handleSetAssetCategory)
	mux.HandleFunc("POST /admin/assets/rate", s.handleRateAsset)
	mux.HandleFunc("POST /admin/groups/create", s.handleCreateGroup)
	mux.HandleFunc("POST /admin/groups/attach", s.handleAttachGroup)
	mux.HandleFunc("POST /admin/groups/detach", s.handleDetachGroup)
	mux.HandleFunc("POST /admin/characters/create", s.handleCreateCharacter)
	mux.HandleFunc("POST /admin/characters/attach", s.handleAttachCharacter)
	mux.HandleFunc("POST /admin/characters/detach", s.handleDetachCharacter)

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

	// Live scan status API endpoint
	mux.HandleFunc("GET /api/scan/status", s.handleScanStatus)

	// Ollama AI Actions (Image2Prompt, Character2Prompt & Text2Prompt)
	mux.HandleFunc("POST /admin/ollama/image2prompt", s.handleImage2Prompt)
	mux.HandleFunc("POST /admin/ollama/character2prompt", s.handleCharacter2Prompt)
	mux.HandleFunc("POST /admin/ollama/text2prompt", s.handleText2Prompt)

	return s.loggingMiddleware(mux)
}

type statusResponseWriter struct {
	http.ResponseWriter
	statusCode int
}

func (rw *statusResponseWriter) WriteHeader(code int) {
	rw.statusCode = code
	rw.ResponseWriter.WriteHeader(code)
}

func (s *Server) loggingMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		rw := &statusResponseWriter{ResponseWriter: w, statusCode: http.StatusOK}
		next.ServeHTTP(rw, r)
		duration := time.Since(start).Round(time.Millisecond)

		// Don't flood console with periodic scan status polling
		if r.URL.Path == "/api/scan/status" {
			return
		}
		log.Printf("[HTTP] %d %s %s (%v) - %s", rw.statusCode, r.Method, r.URL.RequestURI(), duration, r.RemoteAddr)
	})
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
	PrevAssetID       int64
	NextAssetID       int64
	RelatedAssets     usecase.RelatedAssetsBundle
	ScanReport        usecase.ScanReport
	ScanStatus        usecase.ScanStatus
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
	AIMode            string
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

	rawTab := r.URL.Query().Get("tab")
	tab := rawTab
	if tab == "remix" {
		tab = "studio"
	}

	var selectedAsset *domain.MediaAsset
	var relatedAssets usecase.RelatedAssetsBundle
	var assetID int64
	var prevAssetID, nextAssetID int64
	if assetIDStr := r.URL.Query().Get("asset_id"); assetIDStr != "" {
		if id, err := strconv.ParseInt(assetIDStr, 10, 64); err == nil && id > 0 {
			if a, err := s.catalog.GetAsset(ctx, id); err == nil {
				selectedAsset = a
				relatedAssets = s.catalog.GetRelatedAssets(ctx, a)
				assetID = id
				prevAssetID, nextAssetID, _ = s.catalog.GetAdjacentAssetIDs(ctx, id)
			}
		}
	}

	if tab == "" {
		if selectedAsset != nil {
			tab = "asset_detail"
		} else if filter.FaceMode {
			tab = "faces"
		} else {
			tab = "catalog"
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
	if tab == "settings" || tab == "studio" || tab == "asset_detail" || tab == "characters" || tab == "remix" {
		aiStatus = s.catalog.InspectAIBackends(ctx)
	}

	defaultTarget := settings.DefaultTargetGenerator
	if defaultTarget == "" {
		if len(profiles) > 0 {
			defaultTarget = profiles[0].Slug
		} else {
			defaultTarget = "flux"
		}
	}
	if qTarget := r.URL.Query().Get("target_profile"); qTarget != "" {
		defaultTarget = qTarget
	} else if qProfile := r.URL.Query().Get("profile_slug"); qProfile != "" {
		defaultTarget = qProfile
	}

	defaultMode := r.URL.Query().Get("mode")
	if defaultMode == "" {
		if selectedAsset != nil {
			defaultMode = "vision"
		} else {
			defaultMode = "remix"
		}
	}

	inputConcept := r.URL.Query().Get("concept")
	if inputConcept == "" && selectedAsset != nil && selectedAsset.Metadata.PositivePrompt != "" {
		inputConcept = selectedAsset.Metadata.PositivePrompt
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
		PrevAssetID:      prevAssetID,
		NextAssetID:      nextAssetID,
		RelatedAssets:    relatedAssets,
		ScanReport:       s.scanner.LastReport(),
		ScanStatus:       s.scanner.Status(),
		Settings:         settings,
		PromptProfiles:   profiles,
		SelectedGenSlug:  selectedGenSlug,
		SelectedProfile:  selectedProfile,
		ModelLoRAs:       loras,
		GeneratedHistory: history,
		AIStatus:         aiStatus,
		AIInputConcept:   inputConcept,
		AITargetProfile:  defaultTarget,
		AIMode:           defaultMode,
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
		s.scanner.StartScanAsync()
	}
	http.Redirect(w, r, "/?tab=settings&notice=Switched+active+global+database+to+"+newPath, http.StatusSeeOther)
}

func (s *Server) handleAddMediaRoot(w http.ResponseWriter, r *http.Request) {
	_ = r.ParseForm()
	path := strings.TrimSpace(r.FormValue("path"))
	label := strings.TrimSpace(r.FormValue("label"))
	if path != "" {
		log.Printf("[CATALOG] Adding media root folder: %s (label: %s)", path, label)
		_, _ = s.catalog.AddMediaRoot(r.Context(), path, label)
		s.scanner.StartScanAsync()
	}
	if r.Header.Get("Accept") == "application/json" {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]interface{}{"status": "added_and_scanning"})
		return
	}
	http.Redirect(w, r, "/?tab=settings&notice=Media+folder+registered+and+scan+started", http.StatusSeeOther)
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
	http.Redirect(w, r, "/?asset_id="+strconv.FormatInt(assetID, 10)+"&notice=Asset+classification+saved", http.StatusSeeOther)
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
	http.Redirect(w, r, "/?asset_id="+strconv.FormatInt(assetID, 10)+"&notice=Score+and+favorite+saved", http.StatusSeeOther)
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
	if err := r.ParseMultipartForm(32 << 20); err != nil {
		_ = r.ParseForm()
	}
	assetID, _ := strconv.ParseInt(r.FormValue("asset_id"), 10, 64)
	profileSlug := r.FormValue("profile_slug")
	mode := r.FormValue("mode")
	userHint := r.FormValue("user_hint")
	if userHint == "" {
		userHint = r.FormValue("concept")
	}
	characterID, _ := strconv.ParseInt(r.FormValue("character_id"), 10, 64)
	loraIDs := parseLoRAIDs(r.Form["lora_id"])
	override := parseExecutionOverride(r)
	saveToAsset := r.FormValue("save_to_asset") == "1" || r.FormValue("save_to_asset") == "on"

	var frameImagePath string
	rawImagePayload := strings.TrimSpace(r.FormValue("attached_image_data"))
	if rawImagePayload == "" {
		rawImagePayload = strings.TrimSpace(r.FormValue("video_frame_data"))
	}
	if rawImagePayload != "" {
		idx := strings.Index(rawImagePayload, ";base64,")
		b64Content := rawImagePayload
		if idx != -1 {
			b64Content = rawImagePayload[idx+8:]
		}
		if rawDecoded, err := base64.StdEncoding.DecodeString(b64Content); err == nil && len(rawDecoded) > 0 {
			uploadsDir := filepath.Join(s.cacheDir, "uploads")
			_ = os.MkdirAll(uploadsDir, 0755)
			frameImagePath = filepath.Join(uploadsDir, fmt.Sprintf("attached_%d_%d.jpg", assetID, time.Now().UnixNano()))
			if err := os.WriteFile(frameImagePath, rawDecoded, 0644); err != nil {
				log.Printf("[STUDIO] Failed to write attached image: %v", err)
				frameImagePath = ""
			} else {
				log.Printf("[STUDIO] Attached image captured: %s (%d bytes)", frameImagePath, len(rawDecoded))
			}
		}
	}

	if frameImagePath == "" && r.MultipartForm != nil {
		if file, header, err := r.FormFile("attached_file"); err == nil && file != nil {
			defer file.Close()
			uploadsDir := filepath.Join(s.cacheDir, "uploads")
			_ = os.MkdirAll(uploadsDir, 0755)
			ext := filepath.Ext(header.Filename)
			if ext == "" {
				ext = ".jpg"
			}
			destPath := filepath.Join(uploadsDir, fmt.Sprintf("upload_%d_%d%s", assetID, time.Now().UnixNano(), ext))
			if outF, err := os.Create(destPath); err == nil {
				_, _ = io.Copy(outF, file)
				_ = outF.Close()
				frameImagePath = destPath
				log.Printf("[STUDIO] Uploaded file saved: %s", frameImagePath)
			}
		}
	}

	if profileSlug != "" {
		_ = s.catalog.SetDefaultTargetGenerator(ctx, profileSlug)
	}

	outPrompt, runErr := s.catalog.RunImage2Prompt(ctx, assetID, profileSlug, mode, userHint, characterID, loraIDs, override, saveToAsset, frameImagePath)

	filter := parseCatalogFilter(r)
	page, _ := s.catalog.Query(ctx, filter)
	var selectedAsset *domain.MediaAsset
	var prevAssetID, nextAssetID int64
	var relatedAssets usecase.RelatedAssetsBundle
	if assetID > 0 {
		selectedAsset, _ = s.catalog.GetAsset(ctx, assetID)
		prevAssetID, nextAssetID, _ = s.catalog.GetAdjacentAssetIDs(ctx, assetID)
		relatedAssets = s.catalog.GetRelatedAssets(ctx, selectedAsset)
	}
	settings, _ := s.catalog.GetSettings(ctx)
	profiles, _ := s.catalog.ListPromptProfiles(ctx)
	loras, _ := s.catalog.ListModelLoRAs(ctx, "")
	history, _ := s.catalog.ListGeneratedPrompts(ctx, assetID, 0, 20)
	aiStatus := s.catalog.InspectAIBackends(ctx)

	activeTab := "studio"
	if r.FormValue("source_tab") == "asset_detail" {
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
		PrevAssetID:      prevAssetID,
		NextAssetID:      nextAssetID,
		RelatedAssets:    relatedAssets,
		ScanReport:       s.scanner.LastReport(),
		ScanStatus:       s.scanner.Status(),
		Settings:         settings,
		PromptProfiles:   profiles,
		ModelLoRAs:       loras,
		GeneratedHistory: history,
		AIStatus:         aiStatus,
		AIResultPrompt:   outPrompt,
		AIInputConcept:   userHint,
		AITargetProfile:  profileSlug,
		AIMode:           mode,
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

	if profileSlug != "" {
		_ = s.catalog.SetDefaultTargetGenerator(ctx, profileSlug)
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

	if profileSlug != "" {
		_ = s.catalog.SetDefaultTargetGenerator(ctx, profileSlug)
	}

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
		AIMode:           "remix",
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
	log.Printf("[CATALOG] Folder rescan triggered")
	s.scanner.StartScanAsync()
	if r.Header.Get("Accept") == "application/json" {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]interface{}{"status": "started"})
		return
	}
	referer := r.Header.Get("Referer")
	if referer == "" {
		referer = "/"
	}
	http.Redirect(w, r, referer, http.StatusSeeOther)
}

func (s *Server) handleScanStatus(w http.ResponseWriter, r *http.Request) {
	st := s.scanner.Status()
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(st)
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

func (s *Server) handleRemoveTag(w http.ResponseWriter, r *http.Request) {
	_ = r.ParseForm()
	assetID, _ := strconv.ParseInt(r.FormValue("asset_id"), 10, 64)
	tagID, _ := strconv.ParseInt(r.FormValue("tag_id"), 10, 64)
	if assetID > 0 && tagID > 0 {
		_ = s.catalog.RemoveTagFromAsset(r.Context(), assetID, tagID)
	}
	http.Redirect(w, r, "/?asset_id="+strconv.FormatInt(assetID, 10)+"&notice=Tag+removed", http.StatusSeeOther)
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

func (s *Server) handleDetachGroup(w http.ResponseWriter, r *http.Request) {
	_ = r.ParseForm()
	assetID, _ := strconv.ParseInt(r.FormValue("asset_id"), 10, 64)
	groupID, _ := strconv.ParseInt(r.FormValue("group_id"), 10, 64)
	if assetID > 0 && groupID > 0 {
		_ = s.catalog.RemoveAssetFromGroup(r.Context(), groupID, assetID)
	}
	http.Redirect(w, r, "/?asset_id="+strconv.FormatInt(assetID, 10)+"&notice=Removed+from+group", http.StatusSeeOther)
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

func (s *Server) handleDetachCharacter(w http.ResponseWriter, r *http.Request) {
	_ = r.ParseForm()
	assetID, _ := strconv.ParseInt(r.FormValue("asset_id"), 10, 64)
	charID, _ := strconv.ParseInt(r.FormValue("character_id"), 10, 64)
	if assetID > 0 && charID > 0 {
		_ = s.catalog.RemoveAssetFromCharacter(r.Context(), charID, assetID)
	}
	http.Redirect(w, r, "/?asset_id="+strconv.FormatInt(assetID, 10)+"&notice=Removed+from+character", http.StatusSeeOther)
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
