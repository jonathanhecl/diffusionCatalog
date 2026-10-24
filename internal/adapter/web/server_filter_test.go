package web_test

import (
	"context"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/jonathanhecl/diffusionCatalog/internal/adapter/metadata"
	"github.com/jonathanhecl/diffusionCatalog/internal/adapter/storage/sqlite"
	"github.com/jonathanhecl/diffusionCatalog/internal/adapter/vision"
	"github.com/jonathanhecl/diffusionCatalog/internal/adapter/web"
	"github.com/jonathanhecl/diffusionCatalog/internal/domain"
	"github.com/jonathanhecl/diffusionCatalog/internal/usecase"
)

func setupTestServer(t *testing.T) (*web.Server, *sqlite.Repository, int64, int64) {
	ctx := context.Background()
	tmpDir := t.TempDir()
	mediaDir := filepath.Join(tmpDir, "media")
	cacheDir := filepath.Join(tmpDir, "cache")
	dbPath := filepath.Join(tmpDir, "test_web.db")

	db, err := sqlite.Open(ctx, dbPath)
	if err != nil {
		t.Fatalf("sqlite.Open error: %v", err)
	}

	repo := sqlite.NewRepository(db)
	authSvc, _ := usecase.NewAuthService(ctx, repo, "admin", "admin123")
	extractor := metadata.NewExtractor()
	scannerSvc := usecase.NewScannerService(mediaDir, cacheDir, repo, extractor, vision.NewNoopEngine())
	catalogSvc := usecase.NewCatalogService(mediaDir, repo, nil)

	now := time.Now().UTC()
	a1 := &domain.MediaAsset{
		RootID:    1,
		RelPath:   "cyber_a.png",
		FileName:  "cyber_a.png",
		MediaType: domain.MediaTypeImage,
		Category:  "Cyberpunk",
		Rating:    9,
		FileMTime: now.Add(-10 * time.Minute),
	}
	a2 := &domain.MediaAsset{
		RootID:    1,
		RelPath:   "cyber_b.png",
		FileName:  "cyber_b.png",
		MediaType: domain.MediaTypeImage,
		Category:  "Cyberpunk",
		Rating:    5,
		FileMTime: now.Add(-20 * time.Minute),
	}
	id1, _ := repo.UpsertAsset(ctx, a1)
	id2, _ := repo.UpsertAsset(ctx, a2)

	server, err := web.NewServer(mediaDir, cacheDir, "noop", catalogSvc, scannerSvc, authSvc)
	if err != nil {
		t.Fatalf("web.NewServer error: %v", err)
	}

	return server, repo, id1, id2
}

func TestCatalogFilterPersistenceAndMediaNavigation(t *testing.T) {
	server, _, id1, id2 := setupTestServer(t)
	handler := server.Handler()

	// 1. Visit catalog with filter: category=Cyberpunk
	req := httptest.NewRequest("GET", "/?category=Cyberpunk&sort=rating", nil)
	w := httptest.NewRecorder()
	handler.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", w.Code)
	}

	// Verify cookie dc_catalog_filter was set
	cookies := w.Result().Cookies()
	var filterCookie *http.Cookie
	for _, c := range cookies {
		if c.Name == "dc_catalog_filter" {
			filterCookie = c
			break
		}
	}
	if filterCookie == nil {
		t.Fatalf("expected cookie dc_catalog_filter to be set")
	}
	if !strings.Contains(filterCookie.Value, "category=Cyberpunk") {
		t.Errorf("cookie does not contain category=Cyberpunk: %s", filterCookie.Value)
	}

	// 2. Return to root "/" with that cookie -> should redirect to saved filter
	reqReturn := httptest.NewRequest("GET", "/", nil)
	reqReturn.AddCookie(filterCookie)
	wReturn := httptest.NewRecorder()
	handler.ServeHTTP(wReturn, reqReturn)

	if wReturn.Code != http.StatusFound {
		t.Fatalf("expected redirect 302, got %d", wReturn.Code)
	}
	loc := wReturn.Header().Get("Location")
	if !strings.Contains(loc, "category=Cyberpunk") {
		t.Errorf("expected redirect location to contain category=Cyberpunk, got %q", loc)
	}

	// 3. Reset filter via /?reset=1 -> cookie cleared and redirects to /
	reqReset := httptest.NewRequest("GET", "/?reset=1", nil)
	reqReset.AddCookie(filterCookie)
	wReset := httptest.NewRecorder()
	handler.ServeHTTP(wReset, reqReset)

	if wReset.Code != http.StatusFound {
		t.Fatalf("expected reset redirect 302, got %d", wReset.Code)
	}
	var resetCookie *http.Cookie
	for _, c := range wReset.Result().Cookies() {
		if c.Name == "dc_catalog_filter" {
			resetCookie = c
			break
		}
	}
	if resetCookie == nil || resetCookie.MaxAge > 0 {
		t.Errorf("expected cookie to be cleared (MaxAge < 0)")
	}

	// 4. Inspect media detail navigation: view id1 with category=Cyberpunk
	reqDetail := httptest.NewRequest("GET", "/?asset_id="+itoa(id1)+"&category=Cyberpunk", nil)
	wDetail := httptest.NewRecorder()
	handler.ServeHTTP(wDetail, reqDetail)

	if wDetail.Code != http.StatusOK {
		t.Fatalf("expected detail view status 200, got %d", wDetail.Code)
	}
	body := wDetail.Body.String()

	// Verify "Prev" and "Next" navigation buttons exist
	if !strings.Contains(body, "Prev") {
		t.Errorf("expected body to contain 'Prev' button")
	}
	if !strings.Contains(body, "Next") {
		t.Errorf("expected body to contain 'Next' button")
	}

	// Next media in category Cyberpunk is id2
	expectedNextLink := "asset_id=" + itoa(id2)
	if !strings.Contains(body, expectedNextLink) {
		t.Errorf("expected body to contain Next link pointing to %s", expectedNextLink)
	}

	// Back to Catalog should preserve category=Cyberpunk
	if !strings.Contains(body, "category=Cyberpunk") {
		t.Errorf("expected body to preserve category=Cyberpunk in links")
	}
}

func itoa(v int64) string {
	return strconv.FormatInt(v, 10)
}
