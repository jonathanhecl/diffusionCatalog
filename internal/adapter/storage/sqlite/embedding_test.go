package sqlite_test

import (
	"context"
	"path/filepath"
	"testing"

	"github.com/jonathanhecl/diffusionCatalog/internal/adapter/storage/sqlite"
	"github.com/jonathanhecl/diffusionCatalog/internal/domain"
)

func TestAssetEmbeddingPersistenceAndSimilarity(t *testing.T) {
	ctx := context.Background()
	tmpDir := t.TempDir()
	dbPath := filepath.Join(tmpDir, "test_emb.db")

	db, err := sqlite.Open(ctx, dbPath)
	if err != nil {
		t.Fatalf("failed to open sqlite: %v", err)
	}
	defer db.Close()

	repo := sqlite.NewRepository(db, dbPath)

	// Create root and sample assets
	rootID, err := repo.UpsertMediaRoot(ctx, &domain.MediaRoot{
		Path:    filepath.Join(tmpDir, "media"),
		Label:   "Media",
		Enabled: true,
	})
	if err != nil {
		t.Fatalf("failed to create media root: %v", err)
	}

	asset1ID, _ := repo.UpsertAsset(ctx, &domain.MediaAsset{
		RootID:    rootID,
		RelPath:   "cat.png",
		FileName:  "cat.png",
		MediaType: domain.MediaTypeImage,
	})
	asset2ID, _ := repo.UpsertAsset(ctx, &domain.MediaAsset{
		RootID:    rootID,
		RelPath:   "dog.png",
		FileName:  "dog.png",
		MediaType: domain.MediaTypeImage,
	})

	// Check initially no embeddings
	has1, err := repo.HasAssetEmbedding(ctx, asset1ID, "embeddinggemma-2")
	if err != nil || has1 {
		t.Errorf("expected has1=false, got %v, err=%v", has1, err)
	}

	// Save embedding vectors:
	// asset1 is oriented along dimension 0 [1.0, 0.0, 0.0] (e.g. feline)
	// asset2 is oriented along dimension 1 [0.0, 1.0, 0.0] (e.g. canine)
	if err := repo.SaveAssetEmbedding(ctx, asset1ID, "embeddinggemma-2", []float32{1.0, 0.0, 0.0}); err != nil {
		t.Fatalf("failed to save asset1 embedding: %v", err)
	}
	if err := repo.SaveAssetEmbedding(ctx, asset2ID, "embeddinggemma-2", []float32{0.0, 1.0, 0.0}); err != nil {
		t.Fatalf("failed to save asset2 embedding: %v", err)
	}

	// Check HasAssetEmbedding now returns true
	has1, err = repo.HasAssetEmbedding(ctx, asset1ID, "embeddinggemma-2")
	if err != nil || !has1 {
		t.Errorf("expected has1=true, got %v, err=%v", has1, err)
	}
	// With different model, it should be false
	hasOld, _ := repo.HasAssetEmbedding(ctx, asset1ID, "old-model")
	if hasOld {
		t.Errorf("expected hasOld=false for old-model")
	}

	// Query with a vector close to asset1 [0.9, 0.1, 0.0]
	queryVec := []float32{0.9, 0.1, 0.0}
	matches, err := repo.FindSimilarAssetIDs(ctx, queryVec, "embeddinggemma-2", 0.5, 10)
	if err != nil {
		t.Fatalf("FindSimilarAssetIDs failed: %v", err)
	}
	if len(matches) != 1 || matches[0] != asset1ID {
		t.Errorf("expected match [asset1], got %v", matches)
	}

	// Test InvalidateEmbeddingsExcept:
	// Save an embedding with a different model on asset2
	_ = repo.SaveAssetEmbedding(ctx, asset2ID, "clip-vit", []float32{0.5, 0.5, 0.5})
	// Invalidate everything except embeddinggemma-2
	if err := repo.InvalidateEmbeddingsExcept(ctx, "embeddinggemma-2"); err != nil {
		t.Fatalf("InvalidateEmbeddingsExcept failed: %v", err)
	}

	// Verify clip-vit on asset2 was removed and embeddinggemma-2 on asset1 remains
	hasClip, _ := repo.HasAssetEmbedding(ctx, asset2ID, "clip-vit")
	if hasClip {
		t.Errorf("expected clip-vit embedding to be removed")
	}
	hasGemma, _ := repo.HasAssetEmbedding(ctx, asset1ID, "embeddinggemma-2")
	if !hasGemma {
		t.Errorf("expected embeddinggemma-2 to be kept")
	}

	// Verify QueryCatalog ranks by EmbeddingAssetIDs order when no explicit sort is chosen
	page, err := repo.QueryCatalog(ctx, domain.CatalogFilter{
		Query:             "test",
		EmbeddingAssetIDs: []int64{asset2ID, asset1ID},
	})
	if err != nil {
		t.Fatalf("QueryCatalog failed: %v", err)
	}
	if len(page.Assets) != 2 {
		t.Fatalf("expected 2 assets, got %d", len(page.Assets))
	}
	if page.Assets[0].ID != asset2ID || page.Assets[1].ID != asset1ID {
		t.Errorf("expected assets in order [%d, %d], got [%d, %d]", asset2ID, asset1ID, page.Assets[0].ID, page.Assets[1].ID)
	}
}

func TestHybridSearchCaptionRelevanceOverEmbeddings(t *testing.T) {
	ctx := context.Background()
	tmpDir := t.TempDir()
	dbPath := filepath.Join(tmpDir, "test_hybrid.db")

	db, err := sqlite.Open(ctx, dbPath)
	if err != nil {
		t.Fatalf("failed to open sqlite: %v", err)
	}
	defer db.Close()

	repo := sqlite.NewRepository(db, dbPath)
	rootID, _ := repo.UpsertMediaRoot(ctx, &domain.MediaRoot{
		Path:    filepath.Join(tmpDir, "media"),
		Label:   "Media",
		Enabled: true,
	})

	// Create 5 generic assets (semantic embedding matches that do NOT contain the search term in text)
	var embeddingIDs []int64
	for i := 1; i <= 5; i++ {
		id, err := repo.UpsertAsset(ctx, &domain.MediaAsset{
			RootID:         rootID,
			RelPath:        filepath.Join("generic", string(rune('a'+i))+".png"),
			FileName:       string(rune('a'+i)) + ".png",
			MediaType:      domain.MediaTypeImage,
			Metadata: domain.GenerationParams{
				PositivePrompt: "A professional character reference photograph of a woman in an office",
			},
		})
		if err != nil {
			t.Fatalf("failed to insert generic asset: %v", err)
		}
		embeddingIDs = append(embeddingIDs, id)
	}

	// Create 1 asset that actually has the searched word "LenaX" in its caption / positive prompt
	captionAssetID, err := repo.UpsertAsset(ctx, &domain.MediaAsset{
		RootID:    rootID,
		RelPath:   "target/lenax_portrait.png",
		FileName:  "lenax_portrait.png",
		MediaType: domain.MediaTypeImage,
		Metadata: domain.GenerationParams{
			PositivePrompt: "LenaX, Blonde hair. An empty train platform, lit only by scattered candles: a 21-year-old adult woman",
		},
	})
	if err != nil {
		t.Fatalf("failed to insert caption asset: %v", err)
	}

	// Search for "LenaX" with embedding results simulating a semantic search
	page, err := repo.QueryCatalog(ctx, domain.CatalogFilter{
		Query:             "LenaX",
		EmbeddingAssetIDs: embeddingIDs,
	})
	if err != nil {
		t.Fatalf("QueryCatalog failed: %v", err)
	}

	// Total should be 6: 1 direct caption match + 5 semantic matches
	if page.TotalCount != 6 {
		t.Fatalf("expected 6 total matches, got %d", page.TotalCount)
	}

	// The asset with "LenaX" in the caption MUST be ranked FIRST (#1)!
	if page.Assets[0].ID != captionAssetID {
		t.Fatalf("expected asset with caption match (ID %d) to be #1, but got ID %d (prompt: %q)",
			captionAssetID, page.Assets[0].ID, page.Assets[0].Metadata.PositivePrompt)
	}

	// Subsequent assets should be the embedding IDs in order
	for i := 0; i < 5; i++ {
		if page.Assets[i+1].ID != embeddingIDs[i] {
			t.Errorf("expected position %d to be embedding ID %d, got %d", i+2, embeddingIDs[i], page.Assets[i+1].ID)
		}
	}

	// Test multi-word caption search "empty train candles": should also prioritize caption match
	pageMulti, err := repo.QueryCatalog(ctx, domain.CatalogFilter{
		Query:             "empty train candles",
		EmbeddingAssetIDs: embeddingIDs,
	})
	if err != nil {
		t.Fatalf("QueryCatalog multi-word failed: %v", err)
	}
	if len(pageMulti.Assets) == 0 || pageMulti.Assets[0].ID != captionAssetID {
		t.Fatalf("expected multi-word query to rank caption match #1, got ID %d", pageMulti.Assets[0].ID)
	}
}
