package web

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/jonathanhecl/diffusionCatalog/internal/adapter/metadata"
	"github.com/jonathanhecl/diffusionCatalog/internal/adapter/storage/sqlite"
	"github.com/jonathanhecl/diffusionCatalog/internal/adapter/vision"
	"github.com/jonathanhecl/diffusionCatalog/internal/domain"
	"github.com/jonathanhecl/diffusionCatalog/internal/usecase"
)

func TestPageURL(t *testing.T) {
	tests := []struct {
		name     string
		filter   domain.CatalogFilter
		page     int
		contains []string
	}{
		{
			name:   "basic page 2",
			filter: domain.CatalogFilter{},
			page:   2,
			contains: []string{
				"page=2",
			},
		},
		{
			name: "preserves all query filters",
			filter: domain.CatalogFilter{
				Query:         "cyberpunk warrior",
				Trigger:       "kaelen7",
				RootID:        3,
				Category:      "Characters",
				Subcategory:   "Concept",
				MinRating:     8,
				FavoritesOnly: true,
				Engine:        "Flux",
				ModelName:     "flux1-dev.safetensors",
				MediaType:     "image",
				GroupID:       5,
				CharacterID:   2,
				TagNames:      []string{"cyberpunk", "neon"},
				FaceMode:      true,
				SortBy:        "rating",
				PerPage:       100,
			},
			page: 4,
			contains: []string{
				"page=4",
				"per_page=100",
				"q=cyberpunk+warrior",
				"trigger=kaelen7",
				"root_id=3",
				"category=Characters",
				"subcategory=Concept",
				"min_rating=8",
				"favorites=1",
				"engine=Flux",
				"model=flux1-dev.safetensors",
				"media_type=image",
				"group_id=5",
				"character_id=2",
				"tag=cyberpunk",
				"tag=neon",
				"face_mode=1",
				"sort=rating",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			url := pageURL(tt.filter, tt.page)
			for _, exp := range tt.contains {
				if !strings.Contains(url, exp) {
					t.Errorf("pageURL() = %q, expected to contain %q", url, exp)
				}
			}
		})
	}
}

func TestSortURL(t *testing.T) {
	filter := domain.CatalogFilter{
		Query:   "portrait",
		SortBy:  "rating",
		Page:    5,
		PerPage: 25,
	}

	url := sortURL(filter, "favorites")
	if !strings.Contains(url, "sort=favorites") {
		t.Errorf("sortURL() expected sort=favorites, got %q", url)
	}
	if !strings.Contains(url, "page=1") {
		t.Errorf("sortURL() expected page to reset to 1, got %q", url)
	}
	if !strings.Contains(url, "q=portrait") {
		t.Errorf("sortURL() expected query filter to be retained, got %q", url)
	}
}

func TestAssetDetailURL(t *testing.T) {
	filter := domain.CatalogFilter{
		Query: "scifi",
		Page:  3,
	}
	url := assetDetailURL(filter, 42)
	if !strings.Contains(url, "asset_id=42") {
		t.Errorf("assetDetailURL() expected asset_id=42, got %q", url)
	}
	if !strings.Contains(url, "page=3") {
		t.Errorf("assetDetailURL() expected page=3, got %q", url)
	}
}

func TestPaginationList(t *testing.T) {
	t.Run("single page returns nil", func(t *testing.T) {
		items := paginationList(1, 1)
		if len(items) != 0 {
			t.Errorf("expected 0 items for total=1, got %d", len(items))
		}
	})

	t.Run("small total 3 pages", func(t *testing.T) {
		items := paginationList(2, 3)
		if len(items) != 3 {
			t.Fatalf("expected 3 items, got %d", len(items))
		}
		if items[0].Page != 1 || items[1].Page != 2 || items[2].Page != 3 {
			t.Errorf("unexpected pages: %+v", items)
		}
		if !items[1].IsActive {
			t.Errorf("expected page 2 to be active")
		}
	})

	t.Run("large total 15 pages current 1", func(t *testing.T) {
		items := paginationList(1, 15)
		// Should include 1, 2, 3, 4, 5, gap, 15
		var labels []string
		for _, it := range items {
			labels = append(labels, it.Label)
		}
		expected := []string{"1", "2", "3", "4", "5", "…", "15"}
		if len(labels) != len(expected) {
			t.Fatalf("expected %v, got %v", expected, labels)
		}
		for i := range expected {
			if labels[i] != expected[i] {
				t.Errorf("at index %d expected %q, got %q", i, expected[i], labels[i])
			}
		}
		if !items[0].IsActive {
			t.Errorf("expected page 1 to be active")
		}
	})

	t.Run("large total 15 pages current 8", func(t *testing.T) {
		items := paginationList(8, 15)
		// Should include 1, gap, 6, 7, 8, 9, 10, gap, 15
		var labels []string
		for _, it := range items {
			labels = append(labels, it.Label)
		}
		expected := []string{"1", "…", "6", "7", "8", "9", "10", "…", "15"}
		if len(labels) != len(expected) {
			t.Fatalf("expected %v, got %v", expected, labels)
		}
		for i := range expected {
			if labels[i] != expected[i] {
				t.Errorf("at index %d expected %q, got %q", i, expected[i], labels[i])
			}
		}
	})

	t.Run("large total 15 pages current 15", func(t *testing.T) {
		items := paginationList(15, 15)
		// Should include 1, gap, 11, 12, 13, 14, 15
		var labels []string
		for _, it := range items {
			labels = append(labels, it.Label)
		}
		expected := []string{"1", "…", "11", "12", "13", "14", "15"}
		if len(labels) != len(expected) {
			t.Fatalf("expected %v, got %v", expected, labels)
		}
		for i := range expected {
			if labels[i] != expected[i] {
				t.Errorf("at index %d expected %q, got %q", i, expected[i], labels[i])
			}
		}
		if !items[len(items)-1].IsActive {
			t.Errorf("expected last item to be active")
		}
	})
}

func TestParseCatalogFilter(t *testing.T) {
	req := httptest.NewRequest("GET", "/?page=3&per_page=25&q=robot&tag=scifi&tag=mecha&face_mode=1&sort=rating", nil)
	f := parseCatalogFilter(req)
	if f.Page != 3 {
		t.Errorf("expected Page 3, got %d", f.Page)
	}
	if f.PerPage != 25 {
		t.Errorf("expected PerPage 25, got %d", f.PerPage)
	}
	if f.Query != "robot" {
		t.Errorf("expected Query robot, got %q", f.Query)
	}
	if len(f.TagNames) != 2 || f.TagNames[0] != "scifi" || f.TagNames[1] != "mecha" {
		t.Errorf("expected TagNames [scifi, mecha], got %v", f.TagNames)
	}
	if !f.FaceMode {
		t.Errorf("expected FaceMode true")
	}
	if f.SortBy != "rating" {
		t.Errorf("expected SortBy rating, got %q", f.SortBy)
	}
}

func TestDirectorTabRendering(t *testing.T) {
	ctx := context.Background()
	tmpDir := t.TempDir()
	mediaDir := filepath.Join(tmpDir, "media")
	cacheDir := filepath.Join(tmpDir, "cache")
	dbPath := filepath.Join(tmpDir, "test.db")

	db, err := sqlite.Open(ctx, dbPath)
	if err != nil {
		t.Fatalf("failed to open sqlite: %v", err)
	}
	defer db.Close()

	repo := sqlite.NewRepository(db)
	authSvc, _ := usecase.NewAuthService(ctx, repo, "admin", "admin123")
	extractor := metadata.NewExtractor()
	scannerSvc := usecase.NewScannerService(mediaDir, cacheDir, repo, extractor, vision.NewNoopEngine())
	catalogSvc := usecase.NewCatalogService(mediaDir, repo, nil)

	srv, err := NewServer(mediaDir, cacheDir, "NoopVision", catalogSvc, scannerSvc, authSvc)
	if err != nil {
		t.Fatalf("failed to construct web server: %v", err)
	}

	req := httptest.NewRequest("GET", "/?tab=director", nil)
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, req)

	if rec.Code != 200 {
		t.Fatalf("expected HTTP 200, got %d", rec.Code)
	}

	body := rec.Body.String()
	if !strings.Contains(body, "director-container") {
		t.Errorf("expected body to contain director-container, got:\n%s", body[:min(len(body), 500)])
	}
	if !strings.Contains(body, "directorSvg") {
		t.Errorf("expected body to contain directorSvg canvas element")
	}
	if !strings.Contains(body, "ARTICULATED CHARACTERS") {
		t.Errorf("expected body to contain character palette")
	}
	if !strings.Contains(body, "charCustomizationPanel") {
		t.Errorf("expected body to contain charCustomizationPanel")
	}
	if !strings.Contains(body, "charPoseInput") || !strings.Contains(body, "charAnatomySelect") || !strings.Contains(body, "charExpressionSelect") {
		t.Errorf("expected body to contain character pose input, anatomy select, and expression select")
	}
	if !strings.Contains(body, "btnRandomizeChar") || !strings.Contains(body, "Random Look") {
		t.Errorf("expected body to contain character randomize button")
	}
	if !strings.Contains(body, "East Asian heritage") || !strings.Contains(body, "African heritage") {
		t.Errorf("expected body to contain ethnicity and complexion presets")
	}
	if !strings.Contains(body, "Double Space Buns") || !strings.Contains(body, "Double Ponytail") ||
		!strings.Contains(body, "Messy High Bun") || !strings.Contains(body, "Punk Spiked Mohawk") ||
		!strings.Contains(body, "Textured Pixie Cut") {
		t.Errorf("expected body to contain buns, double ponytail, and punk hairstyles")
	}
	if !strings.Contains(body, "charHairSelect") || !strings.Contains(body, "charClothingSelect") ||
		!strings.Contains(body, "charFeaturesSelect") || !strings.Contains(body, "charDetailsSelect") {
		t.Errorf("expected body to contain character hair, clothing, features, and details preset selects")
	}
	if !strings.Contains(body, "Firefighter Gear") || !strings.Contains(body, "Floral Summer Sundress") ||
		!strings.Contains(body, "Denim Miniskirt") || !strings.Contains(body, "Fast-Food Crew Uniform") {
		t.Errorf("expected body to contain rich professional, formal, and feminine clothing presets")
	}
	if !strings.Contains(body, "charLookingAtSelect") || !strings.Contains(body, "gazeLinesGroup") || !strings.Contains(body, "gazeArrow") {
		t.Errorf("expected body to contain Looking At gaze selector, gaze lines SVG group, and gaze arrowhead marker")
	}
	if !strings.Contains(body, `data-ratio="4:3"`) || !strings.Contains(body, `data-ratio="3:4"`) {
		t.Errorf("expected body to contain 4:3 and 3:4 aspect ratio buttons")
	}
	if !strings.Contains(body, `viewBox="0 0 1024 768"`) {
		t.Errorf("expected default SVG canvas viewBox to be 1024 768 (4:3)")
	}
	if !strings.Contains(body, "propLabelInput") || !strings.Contains(body, "propDescInput") {
		t.Errorf("expected body to contain propLabelInput and propDescInput for customizable props")
	}
	if !strings.Contains(body, `data-subtype="circle"`) {
		t.Errorf("expected body to contain generic circle prop in palette")
	}
	if !strings.Contains(body, `data-pose-name="Hands Up (Surrender / Cheer)"`) ||
		!strings.Contains(body, `data-pose-name="High Kick (Martial Arts)"`) ||
		!strings.Contains(body, `data-pose-name="Shield Advance (Combat Stride)"`) {
		t.Errorf("expected body to contain new pose presets in quick shelf")
	}
	if strings.Contains(body, `data-subtype="mannequin"`) {
		t.Errorf("expected OpenPose mannequin button to be removed from palette")
	}
	if !strings.Contains(body, "sceneEnvironmentBar") || !strings.Contains(body, "sceneEnvironmentInput") || !strings.Contains(body, "sceneTimeOfDayInput") {
		t.Errorf("expected body to contain scene environment bar, environment input, and time of day input")
	}
	if !strings.Contains(body, "Hospital Emergency Room") || !strings.Contains(body, "Auto Repair Garage") ||
		!strings.Contains(body, "Wedding Banquet Hall") || !strings.Contains(body, "Suspension Steel Bridge") {
		t.Errorf("expected body to contain indoor and outdoor environment location presets")
	}
	if !strings.Contains(body, "Golden Hour / Sunset Glow") || !strings.Contains(body, "Deep Night &amp; Moonlight") {
		t.Errorf("expected body to contain time of day and atmosphere presets")
	}
	if !strings.Contains(body, `data-env-type="indoor"`) || !strings.Contains(body, `data-env-type="outdoor"`) || !strings.Contains(body, "btnRandomizeEnvironment") {
		t.Errorf("expected body to contain indoor/outdoor filter buttons and randomize environment button")
	}
	if !strings.Contains(body, `data-random-target="pose"`) || !strings.Contains(body, `data-random-target="anatomy"`) ||
		!strings.Contains(body, `data-random-target="expression"`) || !strings.Contains(body, `data-random-target="hair"`) ||
		!strings.Contains(body, `data-random-target="clothing"`) || !strings.Contains(body, `data-random-target="details"`) ||
		!strings.Contains(body, `data-random-target="environment"`) || !strings.Contains(body, `data-random-target="timeOfDay"`) {
		t.Errorf("expected body to contain individual random buttons for each character and environment field")
	}
	if !strings.Contains(body, "aspect-btn") {
		t.Errorf("expected body to contain aspect ratio buttons")
	}
	if strings.Contains(body, `Search prompts, trigger words, checkpoints`) {
		t.Errorf("director tab should not render the catalog search grid!")
	}
}

func TestDeleteAssetEndpoint(t *testing.T) {
	ctx := context.Background()
	tmpDir := t.TempDir()
	mediaDir := filepath.Join(tmpDir, "media")
	cacheDir := filepath.Join(tmpDir, "cache")
	_ = os.MkdirAll(mediaDir, 0755)
	dbPath := filepath.Join(tmpDir, "test.db")

	db, err := sqlite.Open(ctx, dbPath)
	if err != nil {
		t.Fatalf("failed to open sqlite: %v", err)
	}
	defer db.Close()

	repo := sqlite.NewRepository(db)
	authSvc, _ := usecase.NewAuthService(ctx, repo, "admin", "admin123")
	extractor := metadata.NewExtractor()
	scannerSvc := usecase.NewScannerService(mediaDir, cacheDir, repo, extractor, vision.NewNoopEngine())
	catalogSvc := usecase.NewCatalogService(mediaDir, repo, nil)

	srv, err := NewServer(mediaDir, cacheDir, "NoopVision", catalogSvc, scannerSvc, authSvc)
	if err != nil {
		t.Fatalf("failed to construct web server: %v", err)
	}

	// Create test file on disk
	filePath := filepath.Join(mediaDir, "test_render.png")
	if err := os.WriteFile(filePath, []byte("fake image bytes"), 0644); err != nil {
		t.Fatalf("failed to write test file: %v", err)
	}

	assetID, err := repo.UpsertAsset(ctx, &domain.MediaAsset{
		RelPath:   "test_render.png",
		FileName:  "test_render.png",
		FileHash:  "11223344",
		MediaType: domain.MediaTypeImage,
		Width:     512,
		Height:    512,
	})
	if err != nil || assetID <= 0 {
		t.Fatalf("failed to upsert asset: %v", err)
	}

	// 1. Issue POST /admin/assets/delete
	form := url.Values{}
	form.Set("asset_id", strconv.FormatInt(assetID, 10))
	form.Set("return_url", "/?notice=Media+deleted+successfully")

	req := httptest.NewRequest("POST", "/admin/assets/delete", strings.NewReader(form.Encode()))
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	rec := httptest.NewRecorder()

	srv.Handler().ServeHTTP(rec, req)

	if rec.Code != 303 {
		t.Errorf("expected HTTP 303 SeeOther redirect, got %d", rec.Code)
	}

	// 2. Verify file is gone from disk
	if _, err := os.Stat(filePath); !os.IsNotExist(err) {
		t.Errorf("expected file to be deleted from disk")
	}

	// 3. Verify record is gone from DB
	retrieved, err := repo.GetAssetByID(ctx, assetID)
	if err == nil && retrieved != nil {
		t.Errorf("expected asset to be removed from DB")
	}
}

func TestCleanModelName(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		{
			input:    "/mnt/resources/Default/urn:air:ltx2:diffusion_model:huggingface:Kijal%2FLTX2.3_comfy@main%2Fdiffusion_models%2Fltx-2.3-22b-dev_transformer_only_fp8_scaled.safetensors",
			expected: "ltx-2.3-22b-dev_transformer_only_fp8_scaled.safetensors",
		},
		{
			input:    "C:\\AI\\models\\Stable-diffusion\\sd_xl_base_1.0.safetensors",
			expected: "sd_xl_base_1.0.safetensors",
		},
		{
			input:    "https://huggingface.co/repo/model/resolve/main/flux1-dev.safetensors",
			expected: "flux1-dev.safetensors",
		},
		{
			input:    "urn:air:sdxl:checkpoint:civitai:101055@128078",
			expected: "101055@128078",
		},
		{
			input:    "v1-5-pruned-emaonly",
			expected: "v1-5-pruned-emaonly",
		},
		{
			input:    "",
			expected: "Unknown",
		},
	}

	for _, tc := range tests {
		actual := cleanModelName(tc.input)
		if actual != tc.expected {
			t.Errorf("cleanModelName(%q) = %q, expected %q", tc.input, actual, tc.expected)
		}
	}
}

func TestRecentPromptsStudioRendering(t *testing.T) {
	ctx := context.Background()
	tmpDir := t.TempDir()
	mediaDir := filepath.Join(tmpDir, "media")
	cacheDir := filepath.Join(tmpDir, "cache")
	dbPath := filepath.Join(tmpDir, "test.db")

	db, err := sqlite.Open(ctx, dbPath)
	if err != nil {
		t.Fatalf("failed to open sqlite: %v", err)
	}
	defer db.Close()

	repo := sqlite.NewRepository(db)
	authSvc, _ := usecase.NewAuthService(ctx, repo, "admin", "admin123")
	extractor := metadata.NewExtractor()
	scannerSvc := usecase.NewScannerService(mediaDir, cacheDir, repo, extractor, vision.NewNoopEngine())
	catalogSvc := usecase.NewCatalogService(mediaDir, repo, nil)

	// Save a generated prompt with SceneDirection
	_, err = repo.SaveGeneratedPrompt(ctx, &domain.GeneratedPrompt{
		TargetProfileSlug: "flux",
		SceneDirection:    "cyberpunk street at twilight with neon rain",
		OutputPrompt:      "A highly detailed photograph of a cyberpunk street at twilight, reflective puddles",
		Provider:          domain.ProviderOllama,
		ModelName:         "gemma3:12b",
	})
	if err != nil {
		t.Fatalf("failed to save generated prompt: %v", err)
	}

	srv, err := NewServer(mediaDir, cacheDir, "NoopVision", catalogSvc, scannerSvc, authSvc)
	if err != nil {
		t.Fatalf("failed to construct web server: %v", err)
	}

	req := httptest.NewRequest("GET", "/?tab=studio", nil)
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, req)

	if rec.Code != 200 {
		t.Fatalf("expected HTTP 200, got %d", rec.Code)
	}

	body := rec.Body.String()
	if !strings.Contains(body, "Recent Generated Prompts") {
		t.Errorf("expected body to contain 'Recent Generated Prompts'")
	}
	if !strings.Contains(body, "cyberpunk street at twilight with neon rain") {
		t.Errorf("expected body to display input concept text")
	}
	if !strings.Contains(body, "A highly detailed photograph of a cyberpunk street at twilight") {
		t.Errorf("expected body to display output prompt text")
	}
	if !strings.Contains(body, "btn-copy-recent") {
		t.Errorf("expected copy button for recent prompt")
	}
	if !strings.Contains(body, "clearSessionPromptHistory") {
		t.Errorf("expected clearSessionPromptHistory handler")
	}
}

func TestAPIAIStatusEndpoint(t *testing.T) {
	ctx := context.Background()
	mediaDir := t.TempDir()
	cacheDir := t.TempDir()
	dbPath := filepath.Join(t.TempDir(), "test_api.db")

	db, err := sqlite.Open(ctx, dbPath)
	if err != nil {
		t.Fatalf("failed to open sqlite: %v", err)
	}
	defer db.Close()

	repo := sqlite.NewRepository(db)
	authSvc, _ := usecase.NewAuthService(ctx, repo, "admin", "admin123")
	extractor := metadata.NewExtractor()
	scannerSvc := usecase.NewScannerService(mediaDir, cacheDir, repo, extractor, vision.NewNoopEngine())
	catalogSvc := usecase.NewCatalogService(mediaDir, repo, nil)

	srv, err := NewServer(mediaDir, cacheDir, "NoopVision", catalogSvc, scannerSvc, authSvc)
	if err != nil {
		t.Fatalf("failed to construct web server: %v", err)
	}

	req := httptest.NewRequest("GET", "/api/v1/ai-status", nil)
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected HTTP 200, got %d", rec.Code)
	}
	if ct := rec.Header().Get("Content-Type"); !strings.Contains(ct, "application/json") {
		t.Errorf("expected application/json content-type, got %s", ct)
	}

	var status usecase.AIBackendStatus
	if err := json.Unmarshal(rec.Body.Bytes(), &status); err != nil {
		t.Fatalf("failed to parse json response: %v", err)
	}

	// Test with refresh parameter
	reqRefresh := httptest.NewRequest("GET", "/api/v1/ai-status?refresh=1", nil)
	recRefresh := httptest.NewRecorder()
	srv.Handler().ServeHTTP(recRefresh, reqRefresh)
	if recRefresh.Code != http.StatusOK {
		t.Fatalf("expected HTTP 200 on refresh, got %d", recRefresh.Code)
	}
}

func TestClearPromptsEndpoint(t *testing.T) {
	ctx := context.Background()
	mediaDir := t.TempDir()
	cacheDir := t.TempDir()
	dbPath := filepath.Join(t.TempDir(), "test_prompts.db")

	db, err := sqlite.Open(ctx, dbPath)
	if err != nil {
		t.Fatalf("failed to open sqlite: %v", err)
	}
	defer db.Close()

	repo := sqlite.NewRepository(db)
	authSvc, _ := usecase.NewAuthService(ctx, repo, "admin", "admin123")
	extractor := metadata.NewExtractor()
	scannerSvc := usecase.NewScannerService(mediaDir, cacheDir, repo, extractor, vision.NewNoopEngine())
	catalogSvc := usecase.NewCatalogService(mediaDir, repo, nil)

	// Insert a prompt into DB
	_, err = repo.SaveGeneratedPrompt(ctx, &domain.GeneratedPrompt{
		TargetProfileSlug: "flux",
		SceneDirection:    "sample concept",
		OutputPrompt:      "generated prompt output",
		Provider:          domain.ProviderOllama,
		ModelName:         "llama3:latest",
	})
	if err != nil {
		t.Fatalf("failed to save generated prompt: %v", err)
	}

	historyBefore, _ := catalogSvc.ListGeneratedPrompts(ctx, 0, 0, 10)
	if len(historyBefore) != 1 {
		t.Fatalf("expected 1 prompt in history, got %d", len(historyBefore))
	}

	srv, err := NewServer(mediaDir, cacheDir, "NoopVision", catalogSvc, scannerSvc, authSvc)
	if err != nil {
		t.Fatalf("failed to construct web server: %v", err)
	}

	// Call POST /admin/prompts/clear with JSON accept header
	req := httptest.NewRequest("POST", "/admin/prompts/clear", strings.NewReader("all=1"))
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	req.Header.Set("Accept", "application/json")
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected HTTP 200, got %d", rec.Code)
	}

	// Verify history is now empty in DB
	historyAfter, _ := catalogSvc.ListGeneratedPrompts(ctx, 0, 0, 10)
	if len(historyAfter) != 0 {
		t.Fatalf("expected 0 prompts in history after clear, got %d", len(historyAfter))
	}
}
