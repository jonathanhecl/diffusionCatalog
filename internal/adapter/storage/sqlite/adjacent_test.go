package sqlite_test

import (
	"context"
	"path/filepath"
	"testing"
	"time"

	"github.com/jonathanhecl/diffusionCatalog/internal/adapter/storage/sqlite"
	"github.com/jonathanhecl/diffusionCatalog/internal/domain"
)

func TestGetAdjacentAssetIDsRespectsFiltersAndOrder(t *testing.T) {
	ctx := context.Background()
	tmpDir := t.TempDir()
	dbPath := filepath.Join(tmpDir, "test_adjacent.db")

	db, err := sqlite.Open(ctx, dbPath)
	if err != nil {
		t.Fatalf("sqlite.Open error: %v", err)
	}
	defer db.Close()

	repo := sqlite.NewRepository(db)

	// Insert 4 assets with different categories, ratings, and timestamps
	now := time.Now().UTC()
	a1 := &domain.MediaAsset{
		RootID:    1,
		RelPath:   "cyber_1.png",
		FileName:  "cyber_1.png",
		MediaType: domain.MediaTypeImage,
		Category:  "Cyberpunk",
		Rating:    9,
		FileMTime: now.Add(-10 * time.Minute),
	}
	a2 := &domain.MediaAsset{
		RootID:    1,
		RelPath:   "land_1.png",
		FileName:  "land_1.png",
		MediaType: domain.MediaTypeImage,
		Category:  "Landscape",
		Rating:    7,
		FileMTime: now.Add(-20 * time.Minute),
	}
	a3 := &domain.MediaAsset{
		RootID:    1,
		RelPath:   "cyber_2.png",
		FileName:  "cyber_2.png",
		MediaType: domain.MediaTypeImage,
		Category:  "Cyberpunk",
		Rating:    4,
		FileMTime: now.Add(-30 * time.Minute),
	}
	a4 := &domain.MediaAsset{
		RootID:    1,
		RelPath:   "cyber_3.png",
		FileName:  "cyber_3.png",
		MediaType: domain.MediaTypeImage,
		Category:  "Cyberpunk",
		Rating:    8,
		FileMTime: now.Add(-40 * time.Minute),
	}

	id1, _ := repo.UpsertAsset(ctx, a1)
	id2, _ := repo.UpsertAsset(ctx, a2)
	id3, _ := repo.UpsertAsset(ctx, a3)
	id4, _ := repo.UpsertAsset(ctx, a4)

	// Test 1: Category filter = "Cyberpunk" (order default: newest)
	// Matching set ordered newest first:
	// a1 (id1, -10m), a3 (id3, -30m), a4 (id4, -40m)
	// a2 (Landscape) is NOT in this set!
	cyberFilter := domain.CatalogFilter{Category: "Cyberpunk"}

	// When on a3: Prev should be a1 (newer), Next should be a4 (older)
	prev, next, err := repo.GetAdjacentAssetIDs(ctx, id3, cyberFilter)
	if err != nil {
		t.Fatalf("GetAdjacentAssetIDs error: %v", err)
	}
	if prev != id1 {
		t.Errorf("expected prev = %d (a1), got %d", id1, prev)
	}
	if next != id4 {
		t.Errorf("expected next = %d (a4), got %d", id4, next)
	}

	// When on a1 (first in Cyberpunk): Prev should be 0, Next should be a3
	prev, next, err = repo.GetAdjacentAssetIDs(ctx, id1, cyberFilter)
	if err != nil {
		t.Fatalf("GetAdjacentAssetIDs error: %v", err)
	}
	if prev != 0 {
		t.Errorf("expected prev = 0 on first item, got %d", prev)
	}
	if next != id3 {
		t.Errorf("expected next = %d (a3), got %d", id3, next)
	}

	// When on a4 (last in Cyberpunk): Prev should be a3, Next should be 0
	prev, next, err = repo.GetAdjacentAssetIDs(ctx, id4, cyberFilter)
	if err != nil {
		t.Fatalf("GetAdjacentAssetIDs error: %v", err)
	}
	if prev != id3 {
		t.Errorf("expected prev = %d (a3), got %d", id3, prev)
	}
	if next != 0 {
		t.Errorf("expected next = 0 on last item, got %d", next)
	}

	// Test 2: Category filter = "Cyberpunk" with Sort = "rating"
	// Ratings: a1=9, a4=8, a3=4
	// Ordered by rating desc: a1, a4, a3
	ratingFilter := domain.CatalogFilter{Category: "Cyberpunk", SortBy: "rating"}

	// When on a4: Prev should be a1 (rating 9), Next should be a3 (rating 4)
	prev, next, err = repo.GetAdjacentAssetIDs(ctx, id4, ratingFilter)
	if err != nil {
		t.Fatalf("GetAdjacentAssetIDs with rating sort error: %v", err)
	}
	if prev != id1 {
		t.Errorf("expected prev = %d (a1 with rating 9), got %d", id1, prev)
	}
	if next != id3 {
		t.Errorf("expected next = %d (a3 with rating 4), got %d", id3, next)
	}

	// Test 3: Landscape filter
	// Only 1 item (a2)
	landFilter := domain.CatalogFilter{Category: "Landscape"}
	prev, next, err = repo.GetAdjacentAssetIDs(ctx, id2, landFilter)
	if err != nil {
		t.Fatalf("GetAdjacentAssetIDs error: %v", err)
	}
	if prev != 0 || next != 0 {
		t.Errorf("expected prev=0 and next=0 for single item in filter, got prev=%d, next=%d", prev, next)
	}
}
