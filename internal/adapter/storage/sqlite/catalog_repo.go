package sqlite

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/jonathanhecl/diffusionCatalog/internal/domain"
)

// Repository implements domain.CatalogRepository using SQLite and FTS5.
type Repository struct {
	mu     sync.RWMutex
	db     *sql.DB
	dbPath string
}

// NewRepository creates a SQLite-backed CatalogRepository.
func NewRepository(db *sql.DB, dbPath ...string) *Repository {
	p := "./catalog.db"
	if len(dbPath) > 0 && strings.TrimSpace(dbPath[0]) != "" {
		p = strings.TrimSpace(dbPath[0])
	}
	return &Repository{db: db, dbPath: p}
}

func (r *Repository) conn() *sql.DB {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.db
}

// CurrentDBPath returns the active SQLite database file path.
func (r *Repository) CurrentDBPath() string {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.dbPath
}

// SwitchDB opens, migrates, and hot-swaps the active SQLite database file at runtime.
func (r *Repository) SwitchDB(ctx context.Context, newDBPath string) error {
	newDBPath = strings.TrimSpace(newDBPath)
	if newDBPath == "" {
		return fmt.Errorf("database path cannot be empty")
	}
	if abs, err := filepath.Abs(newDBPath); err == nil {
		newDBPath = abs
	}
	if dir := filepath.Dir(newDBPath); dir != "" && dir != "." {
		_ = os.MkdirAll(dir, 0755)
	}
	newDB, err := Open(ctx, newDBPath)
	if err != nil {
		return err
	}

	r.mu.Lock()
	oldDB := r.db
	r.db = newDB
	r.dbPath = newDBPath
	r.mu.Unlock()

	if oldDB != nil {
		_ = oldDB.Close()
	}
	return nil
}

// EnsureDefaultAdmin seeds the default admin user if the users table is empty.
func (r *Repository) EnsureDefaultAdmin(ctx context.Context, username, passwordHash string) error {
	db := r.conn()
	var count int
	if err := db.QueryRowContext(ctx, `SELECT COUNT(*) FROM users`).Scan(&count); err != nil {
		return err
	}
	if count > 0 {
		return nil
	}
	_, err := db.ExecContext(ctx,
		`INSERT INTO users (username, password_hash, role) VALUES (?, ?, 'admin')`,
		username, passwordHash,
	)
	return err
}

// GetUserByUsername fetches a user record by username.
func (r *Repository) GetUserByUsername(ctx context.Context, username string) (*domain.User, error) {
	db := r.conn()
	var u domain.User
	err := db.QueryRowContext(ctx,
		`SELECT id, username, password_hash, role, created_at FROM users WHERE username = ?`,
		username,
	).Scan(&u.ID, &u.Username, &u.PasswordHash, &u.Role, &u.CreatedAt)
	if err != nil {
		return nil, err
	}
	return &u, nil
}

// UpsertMediaRoot registers or updates a watched media folder (local path or external drive).
func (r *Repository) UpsertMediaRoot(ctx context.Context, root *domain.MediaRoot) (int64, error) {
	db := r.conn()
	cleanPath := strings.TrimSpace(root.Path)
	if abs, err := filepath.Abs(cleanPath); err == nil {
		cleanPath = abs
	}
	label := strings.TrimSpace(root.Label)
	if label == "" {
		label = filepath.Base(cleanPath)
	}
	enabled := 1
	if !root.Enabled {
		enabled = 0
	}
	var id int64
	err := db.QueryRowContext(ctx, `
		INSERT INTO media_roots (path, label, enabled)
		VALUES (?, ?, ?)
		ON CONFLICT(path) DO UPDATE SET
			label = excluded.label,
			enabled = excluded.enabled
		RETURNING id`,
		cleanPath, label, enabled,
	).Scan(&id)
	if err != nil {
		return 0, err
	}
	root.ID = id
	root.Path = cleanPath
	root.Label = label
	return id, nil
}

// DeleteMediaRoot removes a media root and its indexed assets from SQLite (without touching disk files).
func (r *Repository) DeleteMediaRoot(ctx context.Context, id int64) error {
	db := r.conn()
	_, _ = db.ExecContext(ctx, `DELETE FROM assets WHERE root_id = ?`, id)
	_, err := db.ExecContext(ctx, `DELETE FROM media_roots WHERE id = ?`, id)
	_ = r.PruneOrphanedTags(ctx)
	return err
}

// GetMediaRootByID fetches a MediaRoot by its ID.
func (r *Repository) GetMediaRootByID(ctx context.Context, id int64) (*domain.MediaRoot, error) {
	db := r.conn()
	var mr domain.MediaRoot
	var enabled int
	err := db.QueryRowContext(ctx, `
		SELECT r.id, r.path, r.label, r.enabled, r.created_at, COUNT(a.id)
		FROM media_roots r
		LEFT JOIN assets a ON a.root_id = r.id
		WHERE r.id = ?
		GROUP BY r.id`, id,
	).Scan(&mr.ID, &mr.Path, &mr.Label, &enabled, &mr.CreatedAt, &mr.AssetCount)
	if err != nil {
		return nil, err
	}
	mr.Enabled = enabled == 1
	if info, statErr := os.Stat(mr.Path); statErr == nil && info.IsDir() {
		mr.Accessible = true
	}
	return &mr, nil
}

// ListMediaRoots returns all registered media root folders with their indexed asset counts and live accessibility status.
func (r *Repository) ListMediaRoots(ctx context.Context) ([]domain.MediaRoot, error) {
	db := r.conn()
	rows, err := db.QueryContext(ctx, `
		SELECT r.id, r.path, r.label, r.enabled, r.created_at, COUNT(a.id)
		FROM media_roots r
		LEFT JOIN assets a ON a.root_id = r.id
		GROUP BY r.id
		ORDER BY r.id ASC`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var out []domain.MediaRoot
	for rows.Next() {
		var mr domain.MediaRoot
		var enabled int
		if err := rows.Scan(&mr.ID, &mr.Path, &mr.Label, &enabled, &mr.CreatedAt, &mr.AssetCount); err != nil {
			return nil, err
		}
		mr.Enabled = enabled == 1
		if info, statErr := os.Stat(mr.Path); statErr == nil && info.IsDir() {
			mr.Accessible = true
		}
		out = append(out, mr)
	}
	return out, nil
}

// UpsertDefaultCategory creates or updates a global default category and its subcategories in SQLite.
func (r *Repository) UpsertDefaultCategory(ctx context.Context, cat *domain.DefaultCategory) (int64, error) {
	db := r.conn()
	name := strings.TrimSpace(cat.Name)
	facet := strings.TrimSpace(cat.Facet)
	if facet == "" {
		facet = "category"
	}
	color := strings.TrimSpace(cat.Color)
	if color == "" {
		color = "#2563eb"
	}
	subsJSON, _ := json.Marshal(cat.Subcategories)
	if len(subsJSON) == 0 {
		subsJSON = []byte("[]")
	}
	var id int64
	err := db.QueryRowContext(ctx, `
		INSERT INTO default_categories (name, facet, color, subcategories_json)
		VALUES (?, ?, ?, ?)
		ON CONFLICT(name) DO UPDATE SET
			facet = excluded.facet,
			color = excluded.color,
			subcategories_json = excluded.subcategories_json
		RETURNING id`,
		name, facet, color, string(subsJSON),
	).Scan(&id)
	if err != nil {
		return 0, err
	}
	cat.ID = id
	return id, nil
}

// DeleteDefaultCategory removes a default category from SQLite by ID.
func (r *Repository) DeleteDefaultCategory(ctx context.Context, id int64) error {
	db := r.conn()
	_, err := db.ExecContext(ctx, `DELETE FROM default_categories WHERE id = ?`, id)
	return err
}

// ListDefaultCategories returns all global default categories and their subcategories.
func (r *Repository) ListDefaultCategories(ctx context.Context) ([]domain.DefaultCategory, error) {
	db := r.conn()
	rows, err := db.QueryContext(ctx, `
		SELECT id, name, facet, color, subcategories_json
		FROM default_categories
		ORDER BY id ASC`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var out []domain.DefaultCategory
	for rows.Next() {
		var c domain.DefaultCategory
		var subsStr string
		if err := rows.Scan(&c.ID, &c.Name, &c.Facet, &c.Color, &subsStr); err != nil {
			return nil, err
		}
		_ = json.Unmarshal([]byte(subsStr), &c.Subcategories)
		out = append(out, c)
	}
	return out, nil
}

// UpsertAsset inserts or updates a cataloged media asset by its relative path.
func (r *Repository) UpsertAsset(ctx context.Context, a *domain.MediaAsset) (int64, error) {
	db := r.conn()
	if a.RootID <= 0 {
		a.RootID = 1
	}
	lorasBytes, _ := json.Marshal(a.Metadata.LoRAs)
	if len(lorasBytes) == 0 {
		lorasBytes = []byte("[]")
	}
	favInt := 0
	if a.Favorite {
		favInt = 1
	}

	var fileMTimeVal interface{} = nil
	if !a.FileMTime.IsZero() {
		fileMTimeVal = a.FileMTime.UTC()
	}

	var metaExtractedVal interface{} = nil
	if a.MetadataExtractedAt != nil && !a.MetadataExtractedAt.IsZero() {
		metaExtractedVal = a.MetadataExtractedAt.UTC()
	} else if a.Metadata.PositivePrompt != "" || a.Metadata.RawMetadata != "" {
		metaExtractedVal = time.Now().UTC()
	}

	var facesScannedVal interface{}
	if a.FacesScannedAt != nil && !a.FacesScannedAt.IsZero() {
		facesScannedVal = a.FacesScannedAt.UTC()
	}

	query := `
	INSERT INTO assets (
		root_id, rel_path, file_name, file_hash, media_type, width, height, duration_sec, file_size,
		view_count, rating, favorite, category, subcategory, engine, positive_prompt, negative_prompt, model_name, model_hash,
		sampler, scheduler, steps, cfg_scale, seed, loras_json, raw_metadata, file_mtime, metadata_extracted_at, faces_scanned_at
	) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
	ON CONFLICT(rel_path) DO UPDATE SET
		root_id = excluded.root_id,
		file_hash = excluded.file_hash,
		width = excluded.width,
		height = excluded.height,
		file_size = excluded.file_size,
		file_mtime = excluded.file_mtime,
		category = CASE WHEN excluded.category != '' THEN excluded.category ELSE assets.category END,
		subcategory = CASE WHEN excluded.subcategory != '' THEN excluded.subcategory ELSE assets.subcategory END,
		rating = CASE WHEN excluded.rating > 0 THEN excluded.rating ELSE assets.rating END,
		favorite = CASE WHEN excluded.favorite > 0 THEN excluded.favorite ELSE assets.favorite END,
		view_count = CASE WHEN excluded.view_count > assets.view_count THEN excluded.view_count ELSE assets.view_count END,
		engine = excluded.engine,
		positive_prompt = excluded.positive_prompt,
		negative_prompt = excluded.negative_prompt,
		model_name = excluded.model_name,
		model_hash = excluded.model_hash,
		sampler = excluded.sampler,
		scheduler = excluded.scheduler,
		steps = excluded.steps,
		cfg_scale = excluded.cfg_scale,
		seed = excluded.seed,
		loras_json = excluded.loras_json,
		raw_metadata = excluded.raw_metadata,
		metadata_extracted_at = CASE WHEN excluded.metadata_extracted_at IS NOT NULL THEN excluded.metadata_extracted_at ELSE assets.metadata_extracted_at END,
		faces_scanned_at = CASE WHEN excluded.faces_scanned_at IS NOT NULL THEN excluded.faces_scanned_at ELSE assets.faces_scanned_at END
	RETURNING id;
	`
	var id int64
	err := db.QueryRowContext(ctx, query,
		a.RootID, a.RelPath, a.FileName, a.FileHash, string(a.MediaType), a.Width, a.Height, a.DurationSec, a.FileSize,
		a.ViewCount, a.Rating, favInt, a.Category, a.Subcategory, string(a.Metadata.Engine), a.Metadata.PositivePrompt, a.Metadata.NegativePrompt,
		a.Metadata.ModelName, a.Metadata.ModelHash, a.Metadata.Sampler, a.Metadata.Scheduler,
		a.Metadata.Steps, a.Metadata.CFGScale, a.Metadata.Seed, string(lorasBytes), a.Metadata.RawMetadata, fileMTimeVal, metaExtractedVal, facesScannedVal,
	).Scan(&id)
	if err != nil {
		return 0, err
	}
	a.ID = id
	return id, nil
}

// GetAssetByID retrieves a full MediaAsset along with its tags, groups, characters, and detected faces.
func (r *Repository) GetAssetByID(ctx context.Context, id int64) (*domain.MediaAsset, error) {
	db := r.conn()
	row := db.QueryRowContext(ctx, `
		SELECT a.id, a.root_id, COALESCE(mr.path, ''), COALESCE(mr.label, ''),
		       a.rel_path, a.file_name, a.file_hash, a.media_type, a.width, a.height, a.duration_sec, a.file_size,
		       a.view_count, a.rating, a.favorite, a.category, a.subcategory, a.engine, a.positive_prompt, a.negative_prompt,
		       a.model_name, a.model_hash, a.sampler, a.scheduler, a.steps, a.cfg_scale, a.seed,
		       a.loras_json, a.raw_metadata, a.file_mtime, a.indexed_at, a.metadata_extracted_at, a.faces_scanned_at
		FROM assets a
		LEFT JOIN media_roots mr ON mr.id = a.root_id
		WHERE a.id = ?`, id)

	asset, err := scanAssetRow(row)
	if err != nil {
		return nil, err
	}

	asset.Tags, _ = r.getTagsForAsset(ctx, asset.ID)
	asset.Groups, _ = r.getGroupsForAsset(ctx, asset.ID)
	asset.Characters, _ = r.getCharactersForAsset(ctx, asset.ID)
	asset.Faces, _ = r.getFacesForAsset(ctx, asset.ID)
	return asset, nil
}

// DeleteAsset removes a single media asset by ID.
func (r *Repository) DeleteAsset(ctx context.Context, id int64) error {
	return r.DeleteAssetsByIDs(ctx, []int64{id})
}

// DeleteAssetsByIDs removes multiple media assets by ID in batches.
// Triggers and foreign keys cascade deletions into assets_fts, asset_tags, group_assets, character_examples, and faces.
func (r *Repository) DeleteAssetsByIDs(ctx context.Context, ids []int64) error {
	if len(ids) == 0 {
		return nil
	}
	db := r.conn()

	const batchSize = 500
	for i := 0; i < len(ids); i += batchSize {
		end := i + batchSize
		if end > len(ids) {
			end = len(ids)
		}
		chunk := ids[i:end]

		query := "DELETE FROM assets WHERE id IN (?" + strings.Repeat(",?", len(chunk)-1) + ")"
		args := make([]any, len(chunk))
		for j, id := range chunk {
			args[j] = id
		}
		if _, err := db.ExecContext(ctx, query, args...); err != nil {
			return err
		}
	}
	_ = r.PruneOrphanedTags(ctx)
	return nil
}

// GetExistingAssetFingerprints returns a lookup map keyed by rel_path containing basic metadata
// (id, file_size, file_mtime, file_hash, rating, favorite, category, subcategory) to allow lightning-fast differential scanning.
func (r *Repository) GetExistingAssetFingerprints(ctx context.Context, rootID int64) (map[string]domain.MediaAsset, error) {
	db := r.conn()
	query := `
		SELECT id, rel_path, file_size, file_hash, file_mtime, indexed_at, rating, favorite, category, subcategory, positive_prompt
		FROM assets`
	var rows *sql.Rows
	var err error
	if rootID > 0 {
		query += ` WHERE root_id = ?`
		rows, err = db.QueryContext(ctx, query, rootID)
	} else {
		rows, err = db.QueryContext(ctx, query)
	}
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	out := make(map[string]domain.MediaAsset)
	for rows.Next() {
		var a domain.MediaAsset
		var relPath, posPrompt string
		var mTime, idxTime sql.NullTime
		var favInt int
		if err := rows.Scan(&a.ID, &relPath, &a.FileSize, &a.FileHash, &mTime, &idxTime, &a.Rating, &favInt, &a.Category, &a.Subcategory, &posPrompt); err != nil {
			return nil, err
		}
		a.RelPath = relPath
		a.Metadata.PositivePrompt = posPrompt
		if mTime.Valid {
			a.FileMTime = mTime.Time
		} else if idxTime.Valid {
			a.FileMTime = idxTime.Time
		}
		a.Favorite = favInt == 1
		out[relPath] = a
	}
	return out, nil
}

// buildCatalogFilterWhereAndOrder converts a domain.CatalogFilter into a WHERE clause, argument slice, and ORDER BY clause for the assets table.
func buildCatalogFilterWhereAndOrder(filter domain.CatalogFilter) (whereClause string, args []interface{}, orderBy string) {
	var where []string

	if q := strings.TrimSpace(filter.Query); q != "" {
		like := "%" + q + "%"
		words := strings.Fields(q)

		var textConditions []string
		textConditions = append(textConditions, `(a.positive_prompt LIKE ? OR a.raw_metadata LIKE ? OR a.file_name LIKE ? OR a.model_name LIKE ? OR a.category LIKE ? OR a.subcategory LIKE ? OR EXISTS (
			SELECT 1 FROM asset_tags at JOIN tags t ON t.id = at.tag_id WHERE at.asset_id = a.id AND t.name LIKE ?
		))`)
		args = append(args, like, like, like, like, like, like, like)

		if len(words) > 1 && len(words) <= 8 {
			var wordClauses []string
			for _, w := range words {
				wLike := "%" + w + "%"
				wordClauses = append(wordClauses, `(a.positive_prompt LIKE ? OR a.raw_metadata LIKE ? OR a.file_name LIKE ?)`)
				args = append(args, wLike, wLike, wLike)
			}
			textConditions = append(textConditions, fmt.Sprintf("(%s)", strings.Join(wordClauses, " AND ")))
		}

		if len(filter.EmbeddingAssetIDs) > 0 {
			var inClauses []string
			for _, id := range filter.EmbeddingAssetIDs {
				inClauses = append(inClauses, strconv.FormatInt(id, 10))
			}
			where = append(where, fmt.Sprintf(`((%s) OR a.id IN (%s))`, strings.Join(textConditions, " OR "), strings.Join(inClauses, ",")))
		} else {
			where = append(where, fmt.Sprintf(`(%s)`, strings.Join(textConditions, " OR ")))
		}
	} else if len(filter.EmbeddingAssetIDs) > 0 {
		var inClauses []string
		for _, id := range filter.EmbeddingAssetIDs {
			inClauses = append(inClauses, strconv.FormatInt(id, 10))
		}
		where = append(where, fmt.Sprintf(`a.id IN (%s)`, strings.Join(inClauses, ",")))
	}
	if trig := strings.TrimSpace(filter.Trigger); trig != "" {
		tlike := "%" + trig + "%"
		where = append(where, `(a.positive_prompt LIKE ? OR a.loras_json LIKE ? OR EXISTS (
			SELECT 1 FROM character_examples ce JOIN characters c ON c.id = ce.character_id WHERE ce.asset_id = a.id AND c.trigger_words LIKE ?
		) OR EXISTS (
			SELECT 1 FROM asset_tags at JOIN tags t ON t.id = at.tag_id WHERE at.asset_id = a.id AND t.name LIKE ?
		))`)
		args = append(args, tlike, tlike, tlike, tlike)
	}
	rootIDs := filter.GetRootIDs()
	if len(rootIDs) == 1 {
		where = append(where, `a.root_id = ?`)
		args = append(args, rootIDs[0])
	} else if len(rootIDs) > 1 {
		var inPlaceholders []string
		for _, rid := range rootIDs {
			inPlaceholders = append(inPlaceholders, "?")
			args = append(args, rid)
		}
		where = append(where, fmt.Sprintf(`a.root_id IN (%s)`, strings.Join(inPlaceholders, ",")))
	}
	categories := filter.GetCategories()
	if len(categories) == 1 {
		where = append(where, `LOWER(a.category) = LOWER(?)`)
		args = append(args, strings.TrimSpace(categories[0]))
	} else if len(categories) > 1 {
		var inPlaceholders []string
		for _, cat := range categories {
			inPlaceholders = append(inPlaceholders, "LOWER(?)")
			args = append(args, strings.TrimSpace(cat))
		}
		where = append(where, fmt.Sprintf(`LOWER(a.category) IN (%s)`, strings.Join(inPlaceholders, ",")))
	}
	if filter.Subcategory != "" {
		where = append(where, `LOWER(a.subcategory) = LOWER(?)`)
		args = append(args, strings.TrimSpace(filter.Subcategory))
	}
	if filter.MinRating > 0 {
		where = append(where, `a.rating >= ?`)
		args = append(args, filter.MinRating)
	}
	if filter.FavoritesOnly {
		where = append(where, `a.favorite = 1`)
	}
	if filter.Engine != "" {
		where = append(where, `a.engine = ?`)
		args = append(args, filter.Engine)
	}
	if filter.ModelName != "" {
		where = append(where, `a.model_name = ?`)
		args = append(args, filter.ModelName)
	}
	if filter.MediaType != "" {
		where = append(where, `a.media_type = ?`)
		args = append(args, filter.MediaType)
	}
	if filter.GroupID > 0 {
		where = append(where, `EXISTS (SELECT 1 FROM group_assets ga WHERE ga.asset_id = a.id AND ga.group_id = ?)`)
		args = append(args, filter.GroupID)
	}
	if filter.CharacterID > 0 {
		where = append(where, `EXISTS (SELECT 1 FROM character_examples ce WHERE ce.asset_id = a.id AND ce.character_id = ?)`)
		args = append(args, filter.CharacterID)
	}
	for _, tagName := range filter.TagNames {
		tagName = strings.TrimSpace(tagName)
		if tagName == "" {
			continue
		}
		where = append(where, `EXISTS (
			SELECT 1 FROM asset_tags at JOIN tags t ON t.id = at.tag_id
			WHERE at.asset_id = a.id AND LOWER(t.name) = LOWER(?)
		)`)
		args = append(args, tagName)
	}

	if len(where) > 0 {
		whereClause = "WHERE " + strings.Join(where, " AND ")
	}

	orderBy = "COALESCE(a.file_mtime, a.indexed_at) DESC, a.id DESC"
	switch filter.SortBy {
	case "oldest":
		orderBy = "COALESCE(a.file_mtime, a.indexed_at) ASC, a.id ASC"
	case "rating", "popular":
		orderBy = "a.rating DESC, a.favorite DESC, a.view_count DESC, a.id DESC"
	case "favorites":
		orderBy = "a.favorite DESC, a.rating DESC, a.view_count DESC, a.id DESC"
	case "views":
		orderBy = "a.view_count DESC, a.id DESC"
	case "seed":
		orderBy = "a.seed ASC, a.id DESC"
	default:
		q := strings.TrimSpace(filter.Query)
		if q != "" {
			escapedQ := strings.ReplaceAll(q, "'", "''")
			startsLike := "'" + escapedQ + "%'"
			containsLike := "'%" + escapedQ + "%'"
			words := strings.Fields(q)

			var multiWordClause string
			if len(words) > 1 && len(words) <= 8 {
				var wordClauses []string
				for _, w := range words {
					ew := strings.ReplaceAll(w, "'", "''")
					wordClauses = append(wordClauses, fmt.Sprintf("(LOWER(a.positive_prompt) LIKE LOWER('%%%s%%') OR LOWER(a.raw_metadata) LIKE LOWER('%%%s%%'))", ew, ew))
				}
				multiWordClause = fmt.Sprintf("WHEN %s THEN 3", strings.Join(wordClauses, " AND "))
			}

			var embedOrderClause string
			if len(filter.EmbeddingAssetIDs) > 0 {
				var embedCases []string
				for idx, id := range filter.EmbeddingAssetIDs {
					embedCases = append(embedCases, fmt.Sprintf("WHEN %d THEN %d", id, idx+1))
				}
				embedOrderClause = fmt.Sprintf("(CASE a.id %s ELSE 999999 END) ASC, ", strings.Join(embedCases, " "))
			}

			orderBy = fmt.Sprintf(`(CASE 
				WHEN LOWER(a.positive_prompt) LIKE LOWER(%s) THEN 1
				WHEN LOWER(a.positive_prompt) LIKE LOWER(%s) THEN 2
				%s
				WHEN LOWER(a.raw_metadata) LIKE LOWER(%s) THEN 4
				WHEN LOWER(a.file_name) LIKE LOWER(%s) THEN 5
				WHEN LOWER(a.model_name) LIKE LOWER(%s) OR LOWER(a.category) LIKE LOWER(%s) OR LOWER(a.subcategory) LIKE LOWER(%s) THEN 6
				ELSE 7
			END) ASC, %sCOALESCE(a.file_mtime, a.indexed_at) DESC, a.id DESC`,
				startsLike, containsLike,
				multiWordClause,
				containsLike, containsLike, containsLike, containsLike, containsLike,
				embedOrderClause,
			)
		} else if len(filter.EmbeddingAssetIDs) > 0 {
			var caseClauses []string
			for idx, id := range filter.EmbeddingAssetIDs {
				caseClauses = append(caseClauses, fmt.Sprintf("WHEN %d THEN %d", id, idx+1))
			}
			orderBy = fmt.Sprintf("(CASE a.id %s ELSE 999999 END) ASC, COALESCE(a.file_mtime, a.indexed_at) DESC, a.id DESC", strings.Join(caseClauses, " "))
		}
	}

	return whereClause, args, orderBy
}

// GetAdjacentAssetIDs returns the previous and next asset IDs relative to the specified asset ID,
// respecting the specified filter criteria and sort order.
func (r *Repository) GetAdjacentAssetIDs(ctx context.Context, id int64, filter domain.CatalogFilter) (int64, int64, error) {
	db := r.conn()

	whereClause, args, orderBy := buildCatalogFilterWhereAndOrder(filter)

	// Window query: LAG() gives previous item in the ordered list, LEAD() gives next item.
	query := fmt.Sprintf(`
		WITH filtered AS (
			SELECT a.id,
			       LAG(a.id) OVER (ORDER BY %s) AS prev_id,
			       LEAD(a.id) OVER (ORDER BY %s) AS next_id
			FROM assets a
			%s
		)
		SELECT COALESCE(prev_id, 0), COALESCE(next_id, 0)
		FROM filtered
		WHERE id = ?`, orderBy, orderBy, whereClause)

	var prevID, nextID int64
	queryArgs := append(append([]interface{}{}, args...), id)
	err := db.QueryRowContext(ctx, query, queryArgs...).Scan(&prevID, &nextID)
	if err == nil {
		return prevID, nextID, nil
	}

	// Fallback to global adjacent if the asset is not within the filtered set
	fallbackQuery := `
		WITH all_assets AS (
			SELECT a.id,
			       LAG(a.id) OVER (ORDER BY COALESCE(a.file_mtime, a.indexed_at) DESC, a.id DESC) AS prev_id,
			       LEAD(a.id) OVER (ORDER BY COALESCE(a.file_mtime, a.indexed_at) DESC, a.id DESC) AS next_id
			FROM assets a
		)
		SELECT COALESCE(prev_id, 0), COALESCE(next_id, 0)
		FROM all_assets
		WHERE id = ?`
	_ = db.QueryRowContext(ctx, fallbackQuery, id).Scan(&prevID, &nextID)
	return prevID, nextID, nil
}

// IncrementAssetView increments the view counter of an asset.
func (r *Repository) IncrementAssetView(ctx context.Context, id int64) error {
	db := r.conn()
	_, err := db.ExecContext(ctx, `UPDATE assets SET view_count = view_count + 1 WHERE id = ?`, id)
	return err
}

// UpdateAssetPrompt updates an asset's positive prompt (e.g. from Ollama Image2Prompt or Text2Prompt).
func (r *Repository) UpdateAssetPrompt(ctx context.Context, assetID int64, positivePrompt, engine string) error {
	db := r.conn()
	if engine != "" {
		_, err := db.ExecContext(ctx, `UPDATE assets SET positive_prompt = ?, engine = ? WHERE id = ?`, positivePrompt, engine, assetID)
		return err
	}
	_, err := db.ExecContext(ctx, `UPDATE assets SET positive_prompt = ? WHERE id = ?`, positivePrompt, assetID)
	return err
}

// UpdateAssetCategory assigns a category and subcategory to an asset.
func (r *Repository) UpdateAssetCategory(ctx context.Context, assetID int64, category, subcategory string) error {
	db := r.conn()
	_, err := db.ExecContext(ctx, `UPDATE assets SET category = ?, subcategory = ? WHERE id = ?`,
		strings.TrimSpace(category), strings.TrimSpace(subcategory), assetID)
	return err
}

// UpdateAssetRating updates an asset's 0-10 score and favorite status in SQLite.
func (r *Repository) UpdateAssetRating(ctx context.Context, assetID int64, rating int, favorite bool) error {
	db := r.conn()
	if rating < 0 {
		rating = 0
	}
	if rating > 10 {
		rating = 10
	}
	favInt := 0
	if favorite {
		favInt = 1
	}
	_, err := db.ExecContext(ctx, `UPDATE assets SET rating = ?, favorite = ? WHERE id = ?`, rating, favInt, assetID)
	return err
}

// GetSettings retrieves the application, Ollama, and OpenAI configuration from SQLite.
func (r *Repository) GetSettings(ctx context.Context) (domain.AppSettings, error) {
	db := r.conn()
	s := domain.AppSettings{
		OllamaEnabled:          true,
		OllamaBaseURL:          "http://127.0.0.1:11434",
		OllamaDefaultModel:     "llama3.2-vision:latest",
		OllamaDefaultThink:     "off",
		OpenAIEnabled:          true,
		OpenAIBaseURL:          "http://127.0.0.1:8000",
		OpenAIAPIKey:           "",
		OpenAIDefaultModel:     "SuperQwen3.8-27b-abliterated-MLX-4bit",
		OpenAIDefaultThink:     "low",
		RemixDefaultProvider:   domain.ProviderOllama,
		StudioDefaultProvider:  domain.ProviderOpenAI,
		DefaultTargetGenerator: "flux",
		EmbeddingProvider:      domain.ProviderOllama,

		// Compatibility sync
		OMLXBaseURL:          "http://127.0.0.1:8000",
		OMLXAPIKey:           "",
		VisionProvider:       domain.ProviderOllama,
		VisionModel:          "llama3.2-vision:latest",
		VisionThink:          "off",
		Image2PromptProvider: domain.ProviderOllama,
		Image2PromptModel:    "llama3.2-vision:latest",
		Image2PromptThink:    "off",
		Text2PromptProvider:  domain.ProviderOpenAI,
		Text2PromptModel:     "SuperQwen3.8-27b-abliterated-MLX-4bit",
		Text2PromptThink:     "low",
	}
	rows, err := db.QueryContext(ctx, `SELECT key, value FROM app_settings`)
	if err != nil {
		return s, err
	}
	defer rows.Close()

	for rows.Next() {
		var k, v string
		if err := rows.Scan(&k, &v); err == nil {
			v = strings.Trim(strings.TrimSpace(v), "\"'")
			switch k {
			case "ollama_enabled":
				s.OllamaEnabled = (v == "true" || v == "1")
			case "ollama_base_url":
				s.OllamaBaseURL = v
			case "ollama_default_model":
				s.OllamaDefaultModel = v
			case "ollama_default_think":
				s.OllamaDefaultThink = v
			case "openai_enabled":
				s.OpenAIEnabled = (v == "true" || v == "1")
			case "openai_base_url":
				s.OpenAIBaseURL = v
				s.OMLXBaseURL = v
			case "openai_api_key":
				s.OpenAIAPIKey = v
				s.OMLXAPIKey = v
			case "openai_default_model":
				s.OpenAIDefaultModel = v
			case "openai_default_think":
				s.OpenAIDefaultThink = v
			case "remix_default_provider":
				s.RemixDefaultProvider = domain.AIProviderType(v)
			case "studio_default_provider":
				s.StudioDefaultProvider = domain.AIProviderType(v)
			case "omlx_base_url":
				if s.OpenAIBaseURL == "" {
					s.OpenAIBaseURL = v
				}
				s.OMLXBaseURL = v
			case "omlx_api_key":
				if s.OpenAIAPIKey == "" {
					s.OpenAIAPIKey = v
				}
				s.OMLXAPIKey = v
			case "vision_provider":
				s.VisionProvider = domain.AIProviderType(v)
			case "vision_model":
				s.VisionModel = v
				if s.OllamaDefaultModel == "" {
					s.OllamaDefaultModel = v
				}
			case "vision_think":
				s.VisionThink = v
				if s.OllamaDefaultThink == "" {
					s.OllamaDefaultThink = v
				}
			case "image2prompt_provider":
				s.Image2PromptProvider = domain.AIProviderType(v)
				if s.RemixDefaultProvider == "" {
					s.RemixDefaultProvider = domain.AIProviderType(v)
				}
			case "image2prompt_model":
				s.Image2PromptModel = v
			case "image2prompt_think":
				s.Image2PromptThink = v
			case "text2prompt_provider":
				s.Text2PromptProvider = domain.AIProviderType(v)
				if s.StudioDefaultProvider == "" {
					s.StudioDefaultProvider = domain.AIProviderType(v)
				}
			case "text2prompt_model":
				s.Text2PromptModel = v
				if s.OpenAIDefaultModel == "" {
					s.OpenAIDefaultModel = v
				}
			case "text2prompt_think":
				s.Text2PromptThink = v
				if s.OpenAIDefaultThink == "" {
					s.OpenAIDefaultThink = v
				}
			case "default_target_generator":
				s.DefaultTargetGenerator = v
			case "auto_caption_on_scan":
				s.AutoCaptionOnScan = (v == "true" || v == "1")
			case "embedding_enabled":
				s.EmbeddingEnabled = (v == "true" || v == "1")
			case "embedding_provider":
				s.EmbeddingProvider = domain.AIProviderType(v)
			case "embedding_model":
				s.EmbeddingModel = v
			}
		}
	}
	return s, nil
}

// SaveSettings persists application, Ollama, and OpenAI settings into SQLite.
func (r *Repository) SaveSettings(ctx context.Context, s domain.AppSettings) error {
	db := r.conn()
	autoCap := "false"
	if s.AutoCaptionOnScan {
		autoCap = "true"
	}
	ollamaEnabled := "false"
	if s.OllamaEnabled {
		ollamaEnabled = "true"
	}
	openaiEnabled := "false"
	if s.OpenAIEnabled {
		openaiEnabled = "true"
	}
	embeddingEnabled := "false"
	if s.EmbeddingEnabled {
		embeddingEnabled = "true"
	}
	if s.OllamaDefaultThink == "" {
		s.OllamaDefaultThink = "off"
	}
	if s.OpenAIDefaultThink == "" {
		s.OpenAIDefaultThink = "low"
	}
	if s.RemixDefaultProvider == "" {
		s.RemixDefaultProvider = domain.ProviderOllama
	}
	if s.EmbeddingProvider == "" {
		s.EmbeddingProvider = domain.ProviderOllama
	}
	if s.StudioDefaultProvider == "" {
		s.StudioDefaultProvider = domain.ProviderOpenAI
	}

	openaiBase := s.OpenAIBaseURL
	if openaiBase == "" {
		openaiBase = s.OMLXBaseURL
	}
	openaiKey := s.OpenAIAPIKey
	if openaiKey == "" {
		openaiKey = s.OMLXAPIKey
	}

	pairs := map[string]string{
		"ollama_enabled":           ollamaEnabled,
		"ollama_base_url":          strings.TrimSpace(s.OllamaBaseURL),
		"ollama_default_model":     strings.TrimSpace(s.OllamaDefaultModel),
		"ollama_default_think":     strings.TrimSpace(s.OllamaDefaultThink),
		"openai_enabled":           openaiEnabled,
		"openai_base_url":          strings.TrimSpace(openaiBase),
		"openai_api_key":           strings.TrimSpace(openaiKey),
		"openai_default_model":     strings.TrimSpace(s.OpenAIDefaultModel),
		"openai_default_think":     strings.TrimSpace(s.OpenAIDefaultThink),
		"remix_default_provider":   string(s.RemixDefaultProvider),
		"studio_default_provider":  string(s.StudioDefaultProvider),
		"default_target_generator": strings.TrimSpace(s.DefaultTargetGenerator),
		"auto_caption_on_scan":     autoCap,
		"embedding_enabled":        embeddingEnabled,
		"embedding_provider":       string(s.EmbeddingProvider),
		"embedding_model":          strings.TrimSpace(s.EmbeddingModel),

		// Compatibility sync
		"omlx_base_url":         strings.TrimSpace(openaiBase),
		"omlx_api_key":          strings.TrimSpace(openaiKey),
		"vision_provider":       string(s.RemixDefaultProvider),
		"vision_model":          strings.TrimSpace(s.OllamaDefaultModel),
		"vision_think":          strings.TrimSpace(s.OllamaDefaultThink),
		"image2prompt_provider": string(s.RemixDefaultProvider),
		"image2prompt_model":    strings.TrimSpace(s.OllamaDefaultModel),
		"image2prompt_think":    strings.TrimSpace(s.OllamaDefaultThink),
		"text2prompt_provider":  string(s.StudioDefaultProvider),
		"text2prompt_model":     strings.TrimSpace(s.OpenAIDefaultModel),
		"text2prompt_think":     strings.TrimSpace(s.OpenAIDefaultThink),
	}
	for k, v := range pairs {
		if _, err := db.ExecContext(ctx, `
			INSERT INTO app_settings (key, value) VALUES (?, ?)
			ON CONFLICT(key) DO UPDATE SET value = excluded.value`, k, v); err != nil {
			return err
		}
	}
	return nil
}

// SetDefaultTargetGenerator updates the active default target generator slug in app_settings.
func (r *Repository) SetDefaultTargetGenerator(ctx context.Context, slug string) error {
	slug = strings.TrimSpace(slug)
	if slug == "" {
		return nil
	}
	db := r.conn()
	_, err := db.ExecContext(ctx, `
		INSERT INTO app_settings (key, value) VALUES ('default_target_generator', ?)
		ON CONFLICT(key) DO UPDATE SET value = excluded.value`, slug)
	return err
}

// UpsertPromptProfile creates or updates a generator system prompt profile (flux, krea2, moss, minimaxh3, etc.).
func (r *Repository) UpsertPromptProfile(ctx context.Context, p *domain.PromptProfile) (int64, error) {
	db := r.conn()
	slug := slugify(p.Slug)
	if slug == "item" && p.Name != "" {
		slug = slugify(p.Name)
	}
	modality := string(p.TargetModality)
	if modality != "video" {
		modality = "image"
	}
	var id int64
	err := db.QueryRowContext(ctx, `
		INSERT INTO prompt_profiles (slug, name, target_modality, description, system_prompt, updated_at)
		VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
		ON CONFLICT(slug) DO UPDATE SET
			name = excluded.name,
			target_modality = excluded.target_modality,
			description = excluded.description,
			system_prompt = excluded.system_prompt,
			updated_at = CURRENT_TIMESTAMP
		RETURNING id`,
		slug, strings.TrimSpace(p.Name), modality, strings.TrimSpace(p.Description), strings.TrimSpace(p.SystemPrompt),
	).Scan(&id)
	if err != nil {
		return 0, err
	}
	p.ID = id
	p.Slug = slug
	return id, nil
}

// DeletePromptProfile removes a generator system prompt profile by ID.
func (r *Repository) DeletePromptProfile(ctx context.Context, id int64) error {
	db := r.conn()
	_, err := db.ExecContext(ctx, `DELETE FROM prompt_profiles WHERE id = ?`, id)
	return err
}

// GetPromptProfileBySlug fetches a generator system prompt profile by its slug.
func (r *Repository) GetPromptProfileBySlug(ctx context.Context, slug string) (*domain.PromptProfile, error) {
	db := r.conn()
	var p domain.PromptProfile
	var mod string
	err := db.QueryRowContext(ctx, `
		SELECT id, slug, name, target_modality, description, system_prompt, updated_at
		FROM prompt_profiles WHERE LOWER(slug) = LOWER(?)`, strings.TrimSpace(slug),
	).Scan(&p.ID, &p.Slug, &p.Name, &mod, &p.Description, &p.SystemPrompt, &p.UpdatedAt)
	if err != nil {
		return nil, err
	}
	p.TargetModality = domain.MediaType(mod)
	return &p, nil
}

// ListPromptProfiles returns all generator system prompt profiles ordered by slug.
func (r *Repository) ListPromptProfiles(ctx context.Context) ([]domain.PromptProfile, error) {
	db := r.conn()
	rows, err := db.QueryContext(ctx, `
		SELECT id, slug, name, target_modality, description, system_prompt, updated_at
		FROM prompt_profiles
		ORDER BY slug ASC`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var out []domain.PromptProfile
	for rows.Next() {
		var p domain.PromptProfile
		var mod string
		if err := rows.Scan(&p.ID, &p.Slug, &p.Name, &mod, &p.Description, &p.SystemPrompt, &p.UpdatedAt); err != nil {
			return nil, err
		}
		p.TargetModality = domain.MediaType(mod)
		out = append(out, p)
	}
	return out, nil
}

// UpsertModelLoRA creates or updates a model-scoped LoRA documentation entry.
func (r *Repository) UpsertModelLoRA(ctx context.Context, l *domain.ModelLoRA) (int64, error) {
	db := r.conn()
	if l.RecommendedWeight == "" {
		l.RecommendedWeight = "0.8"
	}
	var id int64
	err := db.QueryRowContext(ctx, `
		INSERT INTO model_loras (
			profile_slug, name, syntax_tag, trigger_words, recommended_weight, description, usage_examples
		) VALUES (?, ?, ?, ?, ?, ?, ?)
		ON CONFLICT(profile_slug, name) DO UPDATE SET
			syntax_tag = excluded.syntax_tag,
			trigger_words = excluded.trigger_words,
			recommended_weight = excluded.recommended_weight,
			description = excluded.description,
			usage_examples = excluded.usage_examples
		RETURNING id`,
		strings.TrimSpace(l.ProfileSlug),
		strings.TrimSpace(l.Name),
		strings.TrimSpace(l.SyntaxTag),
		strings.TrimSpace(l.TriggerWords),
		strings.TrimSpace(l.RecommendedWeight),
		strings.TrimSpace(l.Description),
		strings.TrimSpace(l.UsageExamples),
	).Scan(&id)
	if err != nil {
		return 0, err
	}
	l.ID = id
	return id, nil
}

// DeleteModelLoRA removes a model-scoped LoRA documentation entry by ID.
func (r *Repository) DeleteModelLoRA(ctx context.Context, id int64) error {
	db := r.conn()
	_, err := db.ExecContext(ctx, `DELETE FROM model_loras WHERE id = ?`, id)
	return err
}

// ListModelLoRAs lists LoRA documentation entries, optionally filtered by profileSlug.
func (r *Repository) ListModelLoRAs(ctx context.Context, profileSlug string) ([]domain.ModelLoRA, error) {
	db := r.conn()
	var rows *sql.Rows
	var err error
	if strings.TrimSpace(profileSlug) != "" {
		rows, err = db.QueryContext(ctx, `
			SELECT id, profile_slug, name, syntax_tag, trigger_words, recommended_weight, description, usage_examples, created_at
			FROM model_loras
			WHERE LOWER(profile_slug) = LOWER(?)
			ORDER BY name ASC`, strings.TrimSpace(profileSlug))
	} else {
		rows, err = db.QueryContext(ctx, `
			SELECT id, profile_slug, name, syntax_tag, trigger_words, recommended_weight, description, usage_examples, created_at
			FROM model_loras
			ORDER BY profile_slug ASC, name ASC`)
	}
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var out []domain.ModelLoRA
	for rows.Next() {
		var l domain.ModelLoRA
		if err := rows.Scan(
			&l.ID, &l.ProfileSlug, &l.Name, &l.SyntaxTag, &l.TriggerWords,
			&l.RecommendedWeight, &l.Description, &l.UsageExamples, &l.CreatedAt,
		); err != nil {
			return nil, err
		}
		out = append(out, l)
	}
	return out, nil
}

// UpsertKnowledgeSnippet creates or updates a global contextual knowledge snippet.
func (r *Repository) UpsertKnowledgeSnippet(ctx context.Context, k *domain.KnowledgeSnippet) (int64, error) {
	db := r.conn()
	slug := strings.TrimSpace(k.Slug)
	if slug == "" {
		slug = strings.ToLower(strings.ReplaceAll(strings.TrimSpace(k.Title), " ", "-"))
	}
	var id int64
	err := db.QueryRowContext(ctx, `
		INSERT INTO knowledge_snippets (
			slug, title, description, content, updated_at
		) VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
		ON CONFLICT(slug) DO UPDATE SET
			title = excluded.title,
			description = excluded.description,
			content = excluded.content,
			updated_at = CURRENT_TIMESTAMP
		RETURNING id`,
		slug,
		strings.TrimSpace(k.Title),
		strings.TrimSpace(k.Description),
		k.Content,
	).Scan(&id)
	if err != nil {
		return 0, err
	}
	k.ID = id
	k.Slug = slug
	return id, nil
}

// DeleteKnowledgeSnippet deletes a knowledge snippet by ID.
func (r *Repository) DeleteKnowledgeSnippet(ctx context.Context, id int64) error {
	db := r.conn()
	_, err := db.ExecContext(ctx, `DELETE FROM knowledge_snippets WHERE id = ?`, id)
	return err
}

// ListKnowledgeSnippets lists all knowledge snippets ordered by title ASC.
func (r *Repository) ListKnowledgeSnippets(ctx context.Context) ([]domain.KnowledgeSnippet, error) {
	db := r.conn()
	rows, err := db.QueryContext(ctx, `
		SELECT id, slug, title, description, content, created_at, updated_at
		FROM knowledge_snippets
		ORDER BY title ASC`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var out []domain.KnowledgeSnippet
	for rows.Next() {
		var k domain.KnowledgeSnippet
		if err := rows.Scan(&k.ID, &k.Slug, &k.Title, &k.Description, &k.Content, &k.CreatedAt, &k.UpdatedAt); err != nil {
			return nil, err
		}
		out = append(out, k)
	}
	return out, nil
}

// GetKnowledgeSnippetByID fetches a single knowledge snippet by ID.
func (r *Repository) GetKnowledgeSnippetByID(ctx context.Context, id int64) (*domain.KnowledgeSnippet, error) {
	db := r.conn()
	var k domain.KnowledgeSnippet
	err := db.QueryRowContext(ctx, `
		SELECT id, slug, title, description, content, created_at, updated_at
		FROM knowledge_snippets
		WHERE id = ?`, id).Scan(&k.ID, &k.Slug, &k.Title, &k.Description, &k.Content, &k.CreatedAt, &k.UpdatedAt)
	if err == sql.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	return &k, nil
}

// GetKnowledgeSnippetsByIDs fetches specific KnowledgeSnippet entries by their IDs.
func (r *Repository) GetKnowledgeSnippetsByIDs(ctx context.Context, ids []int64) ([]domain.KnowledgeSnippet, error) {
	db := r.conn()
	if len(ids) == 0 {
		return nil, nil
	}
	var placeholders []string
	var args []interface{}
	for _, id := range ids {
		if id > 0 {
			placeholders = append(placeholders, "?")
			args = append(args, id)
		}
	}
	if len(placeholders) == 0 {
		return nil, nil
	}
	query := fmt.Sprintf(`
		SELECT id, slug, title, description, content, created_at, updated_at
		FROM knowledge_snippets
		WHERE id IN (%s)
		ORDER BY title ASC`, strings.Join(placeholders, ","))
	rows, err := db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var out []domain.KnowledgeSnippet
	for rows.Next() {
		var k domain.KnowledgeSnippet
		if err := rows.Scan(&k.ID, &k.Slug, &k.Title, &k.Description, &k.Content, &k.CreatedAt, &k.UpdatedAt); err != nil {
			return nil, err
		}
		out = append(out, k)
	}
	return out, nil
}

// GetModelLoRAsByIDs fetches specific ModelLoRA entries by their IDs.
func (r *Repository) GetModelLoRAsByIDs(ctx context.Context, ids []int64) ([]domain.ModelLoRA, error) {
	db := r.conn()
	if len(ids) == 0 {
		return nil, nil
	}
	var placeholders []string
	var args []interface{}
	for _, id := range ids {
		if id > 0 {
			placeholders = append(placeholders, "?")
			args = append(args, id)
		}
	}
	if len(placeholders) == 0 {
		return nil, nil
	}
	query := fmt.Sprintf(`
		SELECT id, profile_slug, name, syntax_tag, trigger_words, recommended_weight, description, usage_examples, created_at
		FROM model_loras
		WHERE id IN (%s)
		ORDER BY id ASC`, strings.Join(placeholders, ","))
	rows, err := db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var out []domain.ModelLoRA
	for rows.Next() {
		var l domain.ModelLoRA
		if err := rows.Scan(
			&l.ID, &l.ProfileSlug, &l.Name, &l.SyntaxTag, &l.TriggerWords,
			&l.RecommendedWeight, &l.Description, &l.UsageExamples, &l.CreatedAt,
		); err != nil {
			return nil, err
		}
		out = append(out, l)
	}
	return out, nil
}

// SaveGeneratedPrompt stores a newly generated prompt linked to an Asset and/or Character.
func (r *Repository) SaveGeneratedPrompt(ctx context.Context, gp *domain.GeneratedPrompt) (int64, error) {
	db := r.conn()
	var id int64
	err := db.QueryRowContext(ctx, `
		INSERT INTO generated_prompts (
			asset_id, character_id, target_profile_slug, scene_direction, output_prompt, provider, model_name, duration_sec
		) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
		RETURNING id`,
		gp.AssetID, gp.CharacterID, gp.TargetProfileSlug, gp.SceneDirection, gp.OutputPrompt, string(gp.Provider), gp.ModelName, gp.DurationSec,
	).Scan(&id)
	if err != nil {
		return 0, err
	}
	gp.ID = id
	return id, nil
}

// ListGeneratedPrompts returns recent AI-generated prompts filtered optionally by assetID or characterID.
func (r *Repository) ListGeneratedPrompts(ctx context.Context, assetID, characterID int64, limit int) ([]domain.GeneratedPrompt, error) {
	db := r.conn()
	if limit < 1 {
		limit = 20
	}
	var where []string
	var args []interface{}
	if assetID > 0 {
		where = append(where, `gp.asset_id = ?`)
		args = append(args, assetID)
	}
	if characterID > 0 {
		where = append(where, `gp.character_id = ?`)
		args = append(args, characterID)
	}
	whereClause := ""
	if len(where) > 0 {
		whereClause = "WHERE " + strings.Join(where, " AND ")
	}
	args = append(args, limit)

	query := fmt.Sprintf(`
		SELECT gp.id, gp.asset_id, gp.character_id, COALESCE(c.name, ''),
		       gp.target_profile_slug, gp.scene_direction, gp.output_prompt,
		       gp.provider, gp.model_name, gp.duration_sec, gp.created_at
		FROM generated_prompts gp
		LEFT JOIN characters c ON c.id = gp.character_id
		%s
		ORDER BY gp.id DESC
		LIMIT ?`, whereClause)

	rows, err := db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var out []domain.GeneratedPrompt
	for rows.Next() {
		var item domain.GeneratedPrompt
		var aID, cID sql.NullInt64
		var prov string
		if err := rows.Scan(
			&item.ID, &aID, &cID, &item.CharacterName,
			&item.TargetProfileSlug, &item.SceneDirection, &item.OutputPrompt,
			&prov, &item.ModelName, &item.DurationSec, &item.CreatedAt,
		); err != nil {
			return nil, err
		}
		if aID.Valid {
			v := aID.Int64
			item.AssetID = &v
		}
		if cID.Valid {
			v := cID.Int64
			item.CharacterID = &v
		}
		item.Provider = domain.AIProviderType(prov)
		out = append(out, item)
	}
	return out, nil
}

// ClearGeneratedPrompts deletes generated prompts from the database.
func (r *Repository) ClearGeneratedPrompts(ctx context.Context, assetID, characterID int64) error {
	db := r.conn()
	var (
		query string
		args  []interface{}
	)
	if assetID > 0 && characterID > 0 {
		query = "DELETE FROM generated_prompts WHERE asset_id = ? OR character_id = ?"
		args = []interface{}{assetID, characterID}
	} else if assetID > 0 {
		query = "DELETE FROM generated_prompts WHERE asset_id = ?"
		args = []interface{}{assetID}
	} else if characterID > 0 {
		query = "DELETE FROM generated_prompts WHERE character_id = ?"
		args = []interface{}{characterID}
	} else {
		query = "DELETE FROM generated_prompts"
	}
	_, err := db.ExecContext(ctx, query, args...)
	return err
}

// SaveStudioComposition creates or updates a scene composition in SQLite.
func (r *Repository) SaveStudioComposition(ctx context.Context, comp *domain.StudioComposition) (int64, error) {
	db := r.conn()
	if comp.ID > 0 {
		_, err := db.ExecContext(ctx, `
			UPDATE studio_compositions
			SET name = ?, description = ?, scene_json = ?, thumbnail_data = ?, prompt_output = ?, updated_at = CURRENT_TIMESTAMP
			WHERE id = ?`,
			comp.Name, comp.Description, comp.SceneJSON, comp.ThumbnailData, comp.PromptOutput, comp.ID,
		)
		return comp.ID, err
	}

	var id int64
	err := db.QueryRowContext(ctx, `
		INSERT INTO studio_compositions (name, description, scene_json, thumbnail_data, prompt_output)
		VALUES (?, ?, ?, ?, ?)
		RETURNING id`,
		comp.Name, comp.Description, comp.SceneJSON, comp.ThumbnailData, comp.PromptOutput,
	).Scan(&id)
	if err != nil {
		return 0, err
	}
	comp.ID = id
	return id, nil
}

// GetStudioCompositionByID retrieves a single composition by ID.
func (r *Repository) GetStudioCompositionByID(ctx context.Context, id int64) (*domain.StudioComposition, error) {
	db := r.conn()
	var comp domain.StudioComposition
	err := db.QueryRowContext(ctx, `
		SELECT id, name, description, scene_json, thumbnail_data, prompt_output, created_at, updated_at
		FROM studio_compositions
		WHERE id = ?`, id,
	).Scan(&comp.ID, &comp.Name, &comp.Description, &comp.SceneJSON, &comp.ThumbnailData, &comp.PromptOutput, &comp.CreatedAt, &comp.UpdatedAt)
	if err != nil {
		return nil, err
	}
	return &comp, nil
}

// ListStudioCompositions retrieves all saved scene compositions ordered by newest.
func (r *Repository) ListStudioCompositions(ctx context.Context) ([]domain.StudioComposition, error) {
	db := r.conn()
	rows, err := db.QueryContext(ctx, `
		SELECT id, name, description, scene_json, thumbnail_data, prompt_output, created_at, updated_at
		FROM studio_compositions
		ORDER BY updated_at DESC`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var out []domain.StudioComposition
	for rows.Next() {
		var comp domain.StudioComposition
		if err := rows.Scan(&comp.ID, &comp.Name, &comp.Description, &comp.SceneJSON, &comp.ThumbnailData, &comp.PromptOutput, &comp.CreatedAt, &comp.UpdatedAt); err != nil {
			return nil, err
		}
		out = append(out, comp)
	}
	return out, nil
}

// DeleteStudioComposition removes a composition by ID.
func (r *Repository) DeleteStudioComposition(ctx context.Context, id int64) error {
	db := r.conn()
	_, err := db.ExecContext(ctx, `DELETE FROM studio_compositions WHERE id = ?`, id)
	return err
}

// UpsertStudioPose creates or updates a skeletal character pose preset.
func (r *Repository) UpsertStudioPose(ctx context.Context, pose *domain.StudioPose) (int64, error) {
	db := r.conn()
	if pose.ID > 0 {
		_, err := db.ExecContext(ctx, `
			UPDATE studio_poses
			SET name = ?, category = ?, pose_json = ?, description = ?
			WHERE id = ?`,
			pose.Name, pose.Category, pose.PoseJSON, pose.Description, pose.ID,
		)
		return pose.ID, err
	}

	var id int64
	err := db.QueryRowContext(ctx, `
		INSERT INTO studio_poses (name, category, pose_json, description)
		VALUES (?, ?, ?, ?)
		ON CONFLICT(name) DO UPDATE SET
			category = excluded.category,
			pose_json = excluded.pose_json,
			description = excluded.description
		RETURNING id`,
		pose.Name, pose.Category, pose.PoseJSON, pose.Description,
	).Scan(&id)
	if err != nil {
		return 0, err
	}
	pose.ID = id
	return id, nil
}

// ListStudioPoses returns all saved pose presets.
func (r *Repository) ListStudioPoses(ctx context.Context) ([]domain.StudioPose, error) {
	db := r.conn()
	rows, err := db.QueryContext(ctx, `
		SELECT id, name, category, pose_json, description, created_at
		FROM studio_poses
		ORDER BY category ASC, name ASC`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var out []domain.StudioPose
	for rows.Next() {
		var p domain.StudioPose
		if err := rows.Scan(&p.ID, &p.Name, &p.Category, &p.PoseJSON, &p.Description, &p.CreatedAt); err != nil {
			return nil, err
		}
		out = append(out, p)
	}
	return out, nil
}

// DeleteStudioPose removes a pose preset by ID.
func (r *Repository) DeleteStudioPose(ctx context.Context, id int64) error {
	db := r.conn()
	_, err := db.ExecContext(ctx, `DELETE FROM studio_poses WHERE id = ?`, id)
	return err
}

// QueryCatalog executes a filtered, faceted, and paginated search across assets (or faces when FaceMode is active).
func (r *Repository) QueryCatalog(ctx context.Context, filter domain.CatalogFilter) (*domain.CatalogPage, error) {
	db := r.conn()
	if filter.Page < 1 {
		filter.Page = 1
	}
	if filter.PerPage < 1 || filter.PerPage > 200 {
		filter.PerPage = 50
	}

	page := &domain.CatalogPage{
		Page:      filter.Page,
		PerPage:   filter.PerPage,
		FacetTags: make(map[string][]domain.Tag),
	}

	// Populate global facets
	page.MediaRoots, _ = r.ListMediaRoots(ctx)
	page.DefaultCategories, _ = r.ListDefaultCategories(ctx)
	existingCats := make(map[string]bool)
	for _, c := range page.DefaultCategories {
		existingCats[strings.ToLower(strings.TrimSpace(c.Name))] = true
	}
	assetCats, _ := r.distinctStrings(ctx, `SELECT DISTINCT category FROM assets WHERE category != '' ORDER BY category ASC`)
	for _, ac := range assetCats {
		acTrim := strings.TrimSpace(ac)
		if acTrim != "" && !existingCats[strings.ToLower(acTrim)] {
			existingCats[strings.ToLower(acTrim)] = true
			page.DefaultCategories = append(page.DefaultCategories, domain.DefaultCategory{
				Name:  acTrim,
				Facet: "category",
				Color: "#2563eb",
			})
		}
	}
	tags, _ := r.ListTags(ctx)
	page.AvailableTags = tags
	for _, t := range tags {
		page.FacetTags[t.Facet] = append(page.FacetTags[t.Facet], t)
	}
	page.Engines, _ = r.distinctStrings(ctx, `SELECT DISTINCT engine FROM assets WHERE engine != '' ORDER BY engine`)
	page.Models, _ = r.distinctStrings(ctx, `SELECT DISTINCT model_name FROM assets WHERE model_name != '' ORDER BY model_name`)
	page.Groups, _ = r.ListGroups(ctx)
	page.Characters, _ = r.ListCharacters(ctx)
	page.AvailableTriggers = r.collectTriggerWords(ctx, page.Characters)

	if filter.FaceMode {
		if filter.Cluster == "" {
			clusters, err := r.ListFaceClusters(ctx, filter)
			if err != nil {
				return nil, err
			}
			page.FaceClusters = clusters
			page.TotalCount = len(clusters)
			page.TotalPages = 1
			page.Page = 1
			return page, nil
		}

		faces, total, err := r.ListDetectedFaces(ctx, filter)
		if err != nil {
			return nil, err
		}
		page.Faces = faces
		page.TotalCount = total
		page.TotalPages = (total + filter.PerPage - 1) / filter.PerPage
		if page.TotalPages < 1 {
			page.TotalPages = 1
		}
		if page.TotalCount > 0 && filter.Page > page.TotalPages {
			filter.Page = page.TotalPages
		}
		if filter.Page < 1 {
			filter.Page = 1
		}
		page.Page = filter.Page
		return page, nil
	}

	whereClause, args, orderBy := buildCatalogFilterWhereAndOrder(filter)

	countSQL := fmt.Sprintf(`SELECT COUNT(*) FROM assets a %s`, whereClause)
	if err := db.QueryRowContext(ctx, countSQL, args...).Scan(&page.TotalCount); err != nil {
		return nil, err
	}

	page.TotalPages = (page.TotalCount + filter.PerPage - 1) / filter.PerPage
	if page.TotalPages < 1 {
		page.TotalPages = 1
	}
	if page.TotalCount > 0 && filter.Page > page.TotalPages {
		filter.Page = page.TotalPages
	}
	if filter.Page < 1 {
		filter.Page = 1
	}
	page.Page = filter.Page

	offset := (filter.Page - 1) * filter.PerPage
	selectSQL := fmt.Sprintf(`
		SELECT a.id, a.root_id, COALESCE(mr.path, ''), COALESCE(mr.label, ''),
		       a.rel_path, a.file_name, a.file_hash, a.media_type, a.width, a.height, a.duration_sec, a.file_size,
		       a.view_count, a.rating, a.favorite, a.category, a.subcategory, a.engine, a.positive_prompt, a.negative_prompt,
		       a.model_name, a.model_hash, a.sampler, a.scheduler, a.steps, a.cfg_scale, a.seed,
		       a.loras_json, a.raw_metadata, a.file_mtime, a.indexed_at, a.metadata_extracted_at, a.faces_scanned_at
		FROM assets a
		LEFT JOIN media_roots mr ON mr.id = a.root_id
		%s
		ORDER BY %s
		LIMIT ? OFFSET ?`, whereClause, orderBy)

	queryArgs := append(append([]interface{}{}, args...), filter.PerPage, offset)
	rows, err := db.QueryContext(ctx, selectSQL, queryArgs...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	for rows.Next() {
		a, err := scanAssetRow(rows)
		if err != nil {
			return nil, err
		}
		page.Assets = append(page.Assets, *a)
	}

	for i := range page.Assets {
		page.Assets[i].Tags, _ = r.getTagsForAsset(ctx, page.Assets[i].ID)
		page.Assets[i].Characters, _ = r.getCharactersForAsset(ctx, page.Assets[i].ID)
		page.Assets[i].Faces, _ = r.getFacesForAsset(ctx, page.Assets[i].ID)
	}

	return page, nil
}

// UpsertTag creates or updates an agnostic tag under a specified facet.
func (r *Repository) UpsertTag(ctx context.Context, name, facet, color string) (*domain.Tag, error) {
	db := r.conn()
	name = strings.TrimSpace(name)
	if facet == "" {
		facet = "general"
	}
	if color == "" {
		color = "#2563eb"
	}
	var t domain.Tag
	err := db.QueryRowContext(ctx, `
		INSERT INTO tags (name, facet, color) VALUES (?, ?, ?)
		ON CONFLICT(name) DO UPDATE SET facet = excluded.facet
		RETURNING id, name, facet, color`,
		name, facet, color,
	).Scan(&t.ID, &t.Name, &t.Facet, &t.Color)
	if err != nil {
		return nil, err
	}
	return &t, nil
}

// AttachTagToAsset links a tag to an asset.
func (r *Repository) AttachTagToAsset(ctx context.Context, assetID, tagID int64) error {
	db := r.conn()
	_, err := db.ExecContext(ctx, `INSERT OR IGNORE INTO asset_tags (asset_id, tag_id) VALUES (?, ?)`, assetID, tagID)
	return err
}

// DetachTagFromAsset unlinks a tag from an asset.
func (r *Repository) DetachTagFromAsset(ctx context.Context, assetID, tagID int64) error {
	db := r.conn()
	_, err := db.ExecContext(ctx, `DELETE FROM asset_tags WHERE asset_id = ? AND tag_id = ?`, assetID, tagID)
	if err == nil {
		_ = r.PruneOrphanedTags(ctx)
	}
	return err
}

// ListTags returns all tags ordered by usage count and name.
func (r *Repository) ListTags(ctx context.Context) ([]domain.Tag, error) {
	db := r.conn()
	rows, err := db.QueryContext(ctx, `
		SELECT t.id, t.name, t.facet, t.color, COUNT(at.asset_id) AS cnt
		FROM tags t
		JOIN asset_tags at ON at.tag_id = t.id
		GROUP BY t.id
		HAVING cnt > 0
		ORDER BY cnt DESC, t.name ASC`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var out []domain.Tag
	for rows.Next() {
		var t domain.Tag
		if err := rows.Scan(&t.ID, &t.Name, &t.Facet, &t.Color, &t.AssetCount); err != nil {
			return nil, err
		}
		out = append(out, t)
	}
	return out, nil
}

// PruneOrphanedTags removes any tags that are not associated with any asset.
func (r *Repository) PruneOrphanedTags(ctx context.Context) error {
	db := r.conn()
	_, err := db.ExecContext(ctx, `DELETE FROM tags WHERE id NOT IN (SELECT DISTINCT tag_id FROM asset_tags)`)
	return err
}

// CreateGroup creates or updates an agnostic asset collection.
func (r *Repository) CreateGroup(ctx context.Context, g *domain.Group) (int64, error) {
	db := r.conn()
	slug := slugify(g.Name)
	if g.Category == "" {
		g.Category = "Collection"
	}
	var id int64
	err := db.QueryRowContext(ctx, `
		INSERT INTO groups (name, slug, description, category, cover_asset_id)
		VALUES (?, ?, ?, ?, ?)
		ON CONFLICT(name) DO UPDATE SET
			description = excluded.description,
			category = excluded.category,
			cover_asset_id = COALESCE(excluded.cover_asset_id, groups.cover_asset_id)
		RETURNING id`,
		g.Name, slug, g.Description, g.Category, g.CoverAssetID,
	).Scan(&id)
	if err != nil {
		return 0, err
	}
	g.ID = id
	g.Slug = slug
	return id, nil
}

// UpdateGroup updates the name, slug, and description of a group.
func (r *Repository) UpdateGroup(ctx context.Context, id int64, name, description string) error {
	db := r.conn()
	slug := slugify(name)
	_, err := db.ExecContext(ctx, `
		UPDATE groups
		SET name = ?, slug = ?, description = ?
		WHERE id = ?`,
		name, slug, description, id,
	)
	return err
}

// DeleteGroup deletes a group and cascades deletion to group_assets.
func (r *Repository) DeleteGroup(ctx context.Context, id int64) error {
	db := r.conn()
	_, err := db.ExecContext(ctx, `DELETE FROM groups WHERE id = ?`, id)
	return err
}

// AttachAssetToGroup links an asset to a group.
func (r *Repository) AttachAssetToGroup(ctx context.Context, groupID, assetID int64) error {
	db := r.conn()
	_, err := db.ExecContext(ctx, `INSERT OR IGNORE INTO group_assets (group_id, asset_id) VALUES (?, ?)`, groupID, assetID)
	return err
}

// DetachAssetFromGroup unlinks an asset from a group.
func (r *Repository) DetachAssetFromGroup(ctx context.Context, groupID, assetID int64) error {
	db := r.conn()
	_, err := db.ExecContext(ctx, `DELETE FROM group_assets WHERE group_id = ? AND asset_id = ?`, groupID, assetID)
	return err
}

// ListGroups lists all groups with their member counts.
func (r *Repository) ListGroups(ctx context.Context) ([]domain.Group, error) {
	db := r.conn()
	rows, err := db.QueryContext(ctx, `
		SELECT g.id, g.name, g.slug, g.description, g.category, g.cover_asset_id, COUNT(ga.asset_id)
		FROM groups g
		LEFT JOIN group_assets ga ON ga.group_id = g.id
		GROUP BY g.id
		ORDER BY g.name ASC`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var out []domain.Group
	for rows.Next() {
		var g domain.Group
		var coverID sql.NullInt64
		if err := rows.Scan(&g.ID, &g.Name, &g.Slug, &g.Description, &g.Category, &coverID, &g.AssetCount); err != nil {
			return nil, err
		}
		if coverID.Valid {
			v := coverID.Int64
			g.CoverAssetID = &v
		}
		out = append(out, g)
	}

	if len(out) > 0 {
		pRows, pErr := db.QueryContext(ctx, `
			SELECT group_id, id, rel_path, file_name, media_type, rating
			FROM (
				SELECT ga.group_id, a.id, a.rel_path, a.file_name, a.media_type, a.rating,
				       ROW_NUMBER() OVER (PARTITION BY ga.group_id ORDER BY a.rating DESC, a.file_mtime DESC, a.id DESC) as rn
				FROM group_assets ga
				JOIN assets a ON a.id = ga.asset_id
			)
			WHERE rn <= 3
			ORDER BY group_id, rn ASC`)
		if pErr == nil {
			defer pRows.Close()
			previewsByGroup := make(map[int64][]domain.GroupPreviewAsset)
			for pRows.Next() {
				var gid int64
				var pa domain.GroupPreviewAsset
				if err := pRows.Scan(&gid, &pa.ID, &pa.RelPath, &pa.FileName, &pa.MediaType, &pa.Rating); err == nil {
					previewsByGroup[gid] = append(previewsByGroup[gid], pa)
				}
			}
			for i := range out {
				if previews, ok := previewsByGroup[out[i].ID]; ok {
					out[i].PreviewAssets = previews
				}
			}
		}
	}

	return out, nil
}

// CreateCharacter creates or updates a character profile folder with its canonical prompts and example links.
func (r *Repository) CreateCharacter(ctx context.Context, c *domain.Character) (int64, error) {
	db := r.conn()
	slug := slugify(c.Name)
	if c.Archetype == "" {
		c.Archetype = "Character"
	}
	lorasJSON, _ := json.Marshal(c.LoRAs)
	if len(lorasJSON) == 0 {
		lorasJSON = []byte("[]")
	}
	refImagesJSON, _ := json.Marshal(c.ReferenceImages)
	if len(refImagesJSON) == 0 {
		refImagesJSON = []byte("[]")
	}

	var id int64
	err := db.QueryRowContext(ctx, `
		INSERT INTO characters (
			name, slug, archetype, description, trigger_words,
			base_positive_prompt, base_negative_prompt, recommended_checkpoint, loras_json, avatar_asset_id,
			reference_images_json
		) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
		ON CONFLICT(name) DO UPDATE SET
			archetype = excluded.archetype,
			description = excluded.description,
			trigger_words = excluded.trigger_words,
			base_positive_prompt = excluded.base_positive_prompt,
			base_negative_prompt = excluded.base_negative_prompt,
			recommended_checkpoint = excluded.recommended_checkpoint,
			loras_json = excluded.loras_json,
			avatar_asset_id = COALESCE(excluded.avatar_asset_id, characters.avatar_asset_id),
			reference_images_json = excluded.reference_images_json
		RETURNING id`,
		c.Name, slug, c.Archetype, c.Description, c.TriggerWords,
		c.BasePositivePrompt, c.BaseNegativePrompt, c.RecommendedCheckpoint, string(lorasJSON), c.AvatarAssetID,
		string(refImagesJSON),
	).Scan(&id)
	if err != nil {
		return 0, err
	}
	c.ID = id
	c.Slug = slug
	return id, nil
}

// UpdateCharacter updates an existing character sheet details and reference images.
func (r *Repository) UpdateCharacter(ctx context.Context, c *domain.Character) error {
	db := r.conn()
	slug := c.Slug
	if slug == "" {
		slug = strings.ToLower(strings.ReplaceAll(strings.TrimSpace(c.Name), " ", "-"))
	}
	refImagesJSON, err := json.Marshal(c.ReferenceImages)
	if err != nil || len(refImagesJSON) == 0 {
		refImagesJSON = []byte("[]")
	}

	_, err = db.ExecContext(ctx, `
		UPDATE characters SET
			name = ?,
			slug = ?,
			description = ?,
			reference_images_json = ?
		WHERE id = ?`,
		c.Name, slug, c.Description, string(refImagesJSON), c.ID,
	)
	return err
}

// DeleteCharacter removes a character sheet by ID and cascades references.
func (r *Repository) DeleteCharacter(ctx context.Context, id int64) error {
	db := r.conn()
	_, err := db.ExecContext(ctx, `DELETE FROM characters WHERE id = ?`, id)
	return err
}

// GetCharacterByID returns a character profile along with its linked example assets.
func (r *Repository) GetCharacterByID(ctx context.Context, id int64) (*domain.Character, error) {
	db := r.conn()
	var c domain.Character
	var lorasStr string
	var refImagesStr string
	var avatarID sql.NullInt64

	err := db.QueryRowContext(ctx, `
		SELECT id, name, slug, archetype, description, trigger_words,
		       base_positive_prompt, base_negative_prompt, recommended_checkpoint, loras_json, avatar_asset_id,
		       reference_images_json, created_at
		FROM characters WHERE id = ?`, id,
	).Scan(
		&c.ID, &c.Name, &c.Slug, &c.Archetype, &c.Description, &c.TriggerWords,
		&c.BasePositivePrompt, &c.BaseNegativePrompt, &c.RecommendedCheckpoint, &lorasStr, &avatarID,
		&refImagesStr, &c.CreatedAt,
	)
	if err != nil {
		return nil, err
	}
	_ = json.Unmarshal([]byte(lorasStr), &c.LoRAs)
	_ = json.Unmarshal([]byte(refImagesStr), &c.ReferenceImages)
	if avatarID.Valid {
		v := avatarID.Int64
		c.AvatarAssetID = &v
	}

	page, err := r.QueryCatalog(ctx, domain.CatalogFilter{CharacterID: c.ID, PerPage: 100})
	if err == nil {
		c.Examples = page.Assets
		c.ExampleCount = len(c.Examples)
	}
	return &c, nil
}

// AttachAssetToCharacter links an example asset to a character folder.
func (r *Repository) AttachAssetToCharacter(ctx context.Context, characterID, assetID int64) error {
	db := r.conn()
	_, err := db.ExecContext(ctx, `INSERT OR IGNORE INTO character_examples (character_id, asset_id) VALUES (?, ?)`, characterID, assetID)
	return err
}

// DetachAssetFromCharacter unlinks an example asset from a character folder.
func (r *Repository) DetachAssetFromCharacter(ctx context.Context, characterID, assetID int64) error {
	db := r.conn()
	_, err := db.ExecContext(ctx, `DELETE FROM character_examples WHERE character_id = ? AND asset_id = ?`, characterID, assetID)
	return err
}

// ListCharacters returns all character profiles with their example counts and reference images.
func (r *Repository) ListCharacters(ctx context.Context) ([]domain.Character, error) {
	db := r.conn()
	rows, err := db.QueryContext(ctx, `
		SELECT c.id, c.name, c.slug, c.archetype, c.description, c.trigger_words,
		       c.base_positive_prompt, c.base_negative_prompt, c.recommended_checkpoint, c.loras_json,
		       c.avatar_asset_id, COALESCE(c.reference_images_json, '[]'), c.created_at, COUNT(ce.asset_id)
		FROM characters c
		LEFT JOIN character_examples ce ON ce.character_id = c.id
		GROUP BY c.id
		ORDER BY c.name ASC`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var out []domain.Character
	for rows.Next() {
		var c domain.Character
		var lorasStr string
		var refImagesStr string
		var avatarID sql.NullInt64
		if err := rows.Scan(
			&c.ID, &c.Name, &c.Slug, &c.Archetype, &c.Description, &c.TriggerWords,
			&c.BasePositivePrompt, &c.BaseNegativePrompt, &c.RecommendedCheckpoint, &lorasStr,
			&avatarID, &refImagesStr, &c.CreatedAt, &c.ExampleCount,
		); err != nil {
			return nil, err
		}
		_ = json.Unmarshal([]byte(lorasStr), &c.LoRAs)
		_ = json.Unmarshal([]byte(refImagesStr), &c.ReferenceImages)
		if avatarID.Valid {
			v := avatarID.Int64
			c.AvatarAssetID = &v
		}
		out = append(out, c)
	}
	return out, nil
}

// SaveDetectedFace inserts a cropped face/subject region linked to an asset.
func (r *Repository) SaveDetectedFace(ctx context.Context, face *domain.DetectedFace) (int64, error) {
	db := r.conn()
	embJSON, _ := json.Marshal(face.Embedding)
	if len(embJSON) == 0 {
		embJSON = []byte("[]")
	}
	var id int64
	err := db.QueryRowContext(ctx, `
		INSERT INTO faces (
			asset_id, bbox_x, bbox_y, bbox_w, bbox_h, crop_rel_path,
			confidence, embedding_json, cluster_label, expression_tag, character_id
		) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
		RETURNING id`,
		face.AssetID, face.Box.X, face.Box.Y, face.Box.Width, face.Box.Height,
		face.CropRelPath, face.Confidence, string(embJSON), face.ClusterLabel, face.ExpressionTag, face.CharacterID,
	).Scan(&id)
	if err != nil {
		return 0, err
	}
	face.ID = id
	return id, nil
}

// ListDetectedFaces queries detected face regions for the optional Face Explorer view.
func (r *Repository) ListDetectedFaces(ctx context.Context, filter domain.CatalogFilter) ([]domain.DetectedFace, int, error) {
	db := r.conn()
	var where []string
	var args []interface{}

	if q := strings.TrimSpace(filter.Query); q != "" {
		like := "%" + q + "%"
		where = append(where, `(f.cluster_label LIKE ? OR f.expression_tag LIKE ? OR a.positive_prompt LIKE ?)`)
		args = append(args, like, like, like)
	}
	rootIDs := filter.GetRootIDs()
	if len(rootIDs) == 1 {
		where = append(where, `a.root_id = ?`)
		args = append(args, rootIDs[0])
	} else if len(rootIDs) > 1 {
		var inPlaceholders []string
		for _, rid := range rootIDs {
			inPlaceholders = append(inPlaceholders, "?")
			args = append(args, rid)
		}
		where = append(where, fmt.Sprintf(`a.root_id IN (%s)`, strings.Join(inPlaceholders, ",")))
	}
	categories := filter.GetCategories()
	if len(categories) == 1 {
		where = append(where, `LOWER(a.category) = LOWER(?)`)
		args = append(args, strings.TrimSpace(categories[0]))
	} else if len(categories) > 1 {
		var inPlaceholders []string
		for _, cat := range categories {
			inPlaceholders = append(inPlaceholders, "LOWER(?)")
			args = append(args, strings.TrimSpace(cat))
		}
		where = append(where, fmt.Sprintf(`LOWER(a.category) IN (%s)`, strings.Join(inPlaceholders, ",")))
	}
	if filter.CharacterID > 0 {
		where = append(where, `f.character_id = ?`)
		args = append(args, filter.CharacterID)
	}
	if filter.Cluster != "" {
		where = append(where, `f.cluster_label = ?`)
		args = append(args, filter.Cluster)
	}
	if filter.Engine != "" {
		where = append(where, `a.engine = ?`)
		args = append(args, filter.Engine)
	}

	whereClause := ""
	if len(where) > 0 {
		whereClause = "WHERE " + strings.Join(where, " AND ")
	}

	var total int
	countSQL := fmt.Sprintf(`
		SELECT COUNT(*) FROM faces f
		JOIN assets a ON a.id = f.asset_id
		%s`, whereClause)
	if err := db.QueryRowContext(ctx, countSQL, args...).Scan(&total); err != nil {
		return nil, 0, err
	}

	totalPages := (total + filter.PerPage - 1) / filter.PerPage
	if totalPages < 1 {
		totalPages = 1
	}
	if total > 0 && filter.Page > totalPages {
		filter.Page = totalPages
	}
	if filter.Page < 1 {
		filter.Page = 1
	}

	offset := (filter.Page - 1) * filter.PerPage
	querySQL := fmt.Sprintf(`
		SELECT f.id, f.asset_id, a.root_id, a.rel_path, a.positive_prompt, a.engine, a.model_name,
		       f.bbox_x, f.bbox_y, f.bbox_w, f.bbox_h, f.crop_rel_path, f.confidence,
		       f.cluster_label, f.expression_tag, f.character_id, COALESCE(c.name, ''), f.created_at
		FROM faces f
		JOIN assets a ON a.id = f.asset_id
		LEFT JOIN characters c ON c.id = f.character_id
		%s
		ORDER BY f.id DESC
		LIMIT ? OFFSET ?`, whereClause)

	queryArgs := append(append([]interface{}{}, args...), filter.PerPage, offset)
	rows, err := db.QueryContext(ctx, querySQL, queryArgs...)
	if err != nil {
		return nil, 0, err
	}
	defer rows.Close()

	var out []domain.DetectedFace
	for rows.Next() {
		var f domain.DetectedFace
		var engineStr string
		var charID sql.NullInt64
		if err := rows.Scan(
			&f.ID, &f.AssetID, &f.AssetRootID, &f.AssetRelPath, &f.AssetPrompt, &engineStr, &f.AssetModel,
			&f.Box.X, &f.Box.Y, &f.Box.Width, &f.Box.Height, &f.CropRelPath, &f.Confidence,
			&f.ClusterLabel, &f.ExpressionTag, &charID, &f.CharacterName, &f.CreatedAt,
		); err != nil {
			return nil, 0, err
		}
		f.AssetEngine = domain.EngineType(engineStr)
		if charID.Valid {
			v := charID.Int64
			f.CharacterID = &v
		}
		out = append(out, f)
	}
	return out, total, nil
}

// ListFaceClusters groups detected faces by cluster label, picking the best cropped face preview per cluster.
func (r *Repository) ListFaceClusters(ctx context.Context, filter domain.CatalogFilter) ([]domain.FaceCluster, error) {
	db := r.conn()
	var where []string
	var args []interface{}

	if q := strings.TrimSpace(filter.Query); q != "" {
		like := "%" + q + "%"
		where = append(where, `(f.cluster_label LIKE ? OR f.expression_tag LIKE ? OR a.positive_prompt LIKE ?)`)
		args = append(args, like, like, like)
	}
	rootIDs := filter.GetRootIDs()
	if len(rootIDs) == 1 {
		where = append(where, `a.root_id = ?`)
		args = append(args, rootIDs[0])
	} else if len(rootIDs) > 1 {
		var inPlaceholders []string
		for _, rid := range rootIDs {
			inPlaceholders = append(inPlaceholders, "?")
			args = append(args, rid)
		}
		where = append(where, fmt.Sprintf(`a.root_id IN (%s)`, strings.Join(inPlaceholders, ",")))
	}
	categories := filter.GetCategories()
	if len(categories) == 1 {
		where = append(where, `LOWER(a.category) = LOWER(?)`)
		args = append(args, strings.TrimSpace(categories[0]))
	} else if len(categories) > 1 {
		var inPlaceholders []string
		for _, cat := range categories {
			inPlaceholders = append(inPlaceholders, "LOWER(?)")
			args = append(args, strings.TrimSpace(cat))
		}
		where = append(where, fmt.Sprintf(`LOWER(a.category) IN (%s)`, strings.Join(inPlaceholders, ",")))
	}
	if filter.CharacterID > 0 {
		where = append(where, `f.character_id = ?`)
		args = append(args, filter.CharacterID)
	}
	if filter.Engine != "" {
		where = append(where, `a.engine = ?`)
		args = append(args, filter.Engine)
	}

	whereClause := "WHERE f.cluster_label != ''"
	if len(where) > 0 {
		whereClause += " AND " + strings.Join(where, " AND ")
	}

	query := fmt.Sprintf(`
		SELECT rf.cluster_label,
		       rf.crop_rel_path,
		       rf.character_id,
		       rf.char_name,
		       stats.face_count,
		       stats.asset_count
		FROM (
			SELECT f.cluster_label,
			       f.crop_rel_path,
			       f.character_id,
			       COALESCE(c.name, '') AS char_name,
			       ROW_NUMBER() OVER (
			           PARTITION BY f.cluster_label
			           ORDER BY (f.bbox_w * f.bbox_h) DESC, f.confidence DESC, f.id ASC
			       ) AS rn
			FROM faces f
			JOIN assets a ON a.id = f.asset_id
			LEFT JOIN characters c ON c.id = f.character_id
			%s
		) rf
		JOIN (
			SELECT f.cluster_label,
			       COUNT(f.id) AS face_count,
			       COUNT(DISTINCT f.asset_id) AS asset_count
			FROM faces f
			JOIN assets a ON a.id = f.asset_id
			%s
			GROUP BY f.cluster_label
		) stats ON stats.cluster_label = rf.cluster_label
		WHERE rf.rn = 1
		ORDER BY stats.face_count DESC, rf.cluster_label ASC
	`, whereClause, whereClause)

	doubleArgs := append(append([]interface{}{}, args...), args...)
	rows, err := db.QueryContext(ctx, query, doubleArgs...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var clusters []domain.FaceCluster
	for rows.Next() {
		var fc domain.FaceCluster
		var charID sql.NullInt64
		if err := rows.Scan(
			&fc.Label,
			&fc.BestCropPath,
			&charID,
			&fc.CharacterName,
			&fc.FaceCount,
			&fc.AssetCount,
		); err != nil {
			return nil, err
		}
		if charID.Valid {
			v := charID.Int64
			fc.CharacterID = &v
		}
		clusters = append(clusters, fc)
	}
	return clusters, nil
}

type rowScanner interface {
	Scan(dest ...interface{}) error
}

func scanAssetRow(s rowScanner) (*domain.MediaAsset, error) {
	var a domain.MediaAsset
	var mediaTypeStr, engineStr, lorasStr string
	var favInt int
	var fileMTime sql.NullTime
	var metaExtracted sql.NullTime
	var facesScanned sql.NullTime
	err := s.Scan(
		&a.ID, &a.RootID, &a.RootPath, &a.RootLabel,
		&a.RelPath, &a.FileName, &a.FileHash, &mediaTypeStr,
		&a.Width, &a.Height, &a.DurationSec, &a.FileSize,
		&a.ViewCount, &a.Rating, &favInt, &a.Category, &a.Subcategory, &engineStr,
		&a.Metadata.PositivePrompt, &a.Metadata.NegativePrompt,
		&a.Metadata.ModelName, &a.Metadata.ModelHash,
		&a.Metadata.Sampler, &a.Metadata.Scheduler,
		&a.Metadata.Steps, &a.Metadata.CFGScale, &a.Metadata.Seed,
		&lorasStr, &a.Metadata.RawMetadata, &fileMTime, &a.IndexedAt, &metaExtracted, &facesScanned,
	)
	if err != nil {
		return nil, err
	}
	if fileMTime.Valid {
		a.FileMTime = fileMTime.Time
	} else {
		a.FileMTime = a.IndexedAt
	}
	if metaExtracted.Valid {
		t := metaExtracted.Time
		a.MetadataExtractedAt = &t
	}
	if facesScanned.Valid {
		t := facesScanned.Time
		a.FacesScannedAt = &t
	}
	a.Favorite = favInt == 1
	a.MediaType = domain.MediaType(mediaTypeStr)
	a.Metadata.Engine = domain.EngineType(engineStr)
	_ = json.Unmarshal([]byte(lorasStr), &a.Metadata.LoRAs)
	return &a, nil
}

func (r *Repository) getTagsForAsset(ctx context.Context, assetID int64) ([]domain.Tag, error) {
	db := r.conn()
	rows, err := db.QueryContext(ctx, `
		SELECT t.id, t.name, t.facet, t.color
		FROM tags t
		JOIN asset_tags at ON at.tag_id = t.id
		WHERE at.asset_id = ?
		ORDER BY t.facet, t.name`, assetID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []domain.Tag
	for rows.Next() {
		var t domain.Tag
		if err := rows.Scan(&t.ID, &t.Name, &t.Facet, &t.Color); err == nil {
			out = append(out, t)
		}
	}
	return out, nil
}

func (r *Repository) getGroupsForAsset(ctx context.Context, assetID int64) ([]domain.Group, error) {
	db := r.conn()
	rows, err := db.QueryContext(ctx, `
		SELECT g.id, g.name, g.slug, g.description, g.category
		FROM groups g
		JOIN group_assets ga ON ga.group_id = g.id
		WHERE ga.asset_id = ?`, assetID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []domain.Group
	for rows.Next() {
		var g domain.Group
		if err := rows.Scan(&g.ID, &g.Name, &g.Slug, &g.Description, &g.Category); err == nil {
			out = append(out, g)
		}
	}
	return out, nil
}

func (r *Repository) getCharactersForAsset(ctx context.Context, assetID int64) ([]domain.Character, error) {
	db := r.conn()
	rows, err := db.QueryContext(ctx, `
		SELECT c.id, c.name, c.slug, c.archetype, c.trigger_words
		FROM characters c
		JOIN character_examples ce ON ce.character_id = c.id
		WHERE ce.asset_id = ?`, assetID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []domain.Character
	for rows.Next() {
		var c domain.Character
		if err := rows.Scan(&c.ID, &c.Name, &c.Slug, &c.Archetype, &c.TriggerWords); err == nil {
			out = append(out, c)
		}
	}
	return out, nil
}

func (r *Repository) getFacesForAsset(ctx context.Context, assetID int64) ([]domain.DetectedFace, error) {
	db := r.conn()
	rows, err := db.QueryContext(ctx, `
		SELECT id, asset_id, bbox_x, bbox_y, bbox_w, bbox_h, crop_rel_path, confidence, cluster_label, expression_tag
		FROM faces WHERE asset_id = ?`, assetID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []domain.DetectedFace
	for rows.Next() {
		var f domain.DetectedFace
		if err := rows.Scan(&f.ID, &f.AssetID, &f.Box.X, &f.Box.Y, &f.Box.Width, &f.Box.Height, &f.CropRelPath, &f.Confidence, &f.ClusterLabel, &f.ExpressionTag); err == nil {
			out = append(out, f)
		}
	}
	return out, nil
}

func (r *Repository) distinctStrings(ctx context.Context, query string) ([]string, error) {
	db := r.conn()
	rows, err := db.QueryContext(ctx, query)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []string
	for rows.Next() {
		var s string
		if err := rows.Scan(&s); err == nil && s != "" {
			out = append(out, s)
		}
	}
	return out, nil
}

func (r *Repository) collectTriggerWords(ctx context.Context, chars []domain.Character) []string {
	var out []string
	seen := make(map[string]bool)
	addToken := func(tok string) {
		tok = strings.TrimSpace(tok)
		if tok == "" || len(tok) > 48 {
			return
		}
		key := strings.ToLower(tok)
		if !seen[key] {
			seen[key] = true
			out = append(out, tok)
		}
	}

	for _, c := range chars {
		for _, part := range strings.Split(c.TriggerWords, ",") {
			addToken(part)
		}
	}

	loras, _ := r.ListModelLoRAs(ctx, "")
	for _, l := range loras {
		for _, part := range strings.Split(l.TriggerWords, ",") {
			addToken(part)
		}
	}
	return out
}

func slugify(s string) string {
	s = strings.ToLower(strings.TrimSpace(s))
	var b strings.Builder
	for _, r := range s {
		if (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') {
			b.WriteRune(r)
		} else if r == ' ' || r == '-' || r == '_' {
			b.WriteRune('-')
		}
	}
	res := strings.Trim(b.String(), "-")
	if res == "" {
		return "item"
	}
	return res
}

// HasAssetEmbedding checks whether an embedding exists for the given asset and model.
func (r *Repository) HasAssetEmbedding(ctx context.Context, assetID int64, model string) (bool, error) {
	db := r.conn()
	var exists int
	err := db.QueryRowContext(ctx, `SELECT 1 FROM asset_embeddings WHERE asset_id = ? AND model = ? LIMIT 1`, assetID, model).Scan(&exists)
	if err == sql.ErrNoRows {
		return false, nil
	}
	if err != nil {
		return false, err
	}
	return exists == 1, nil
}

// SaveAssetEmbedding inserts or updates an asset's vector embedding.
func (r *Repository) SaveAssetEmbedding(ctx context.Context, assetID int64, model string, embedding []float32) error {
	db := r.conn()
	rawJSON, err := json.Marshal(embedding)
	if err != nil {
		return err
	}
	_, err = db.ExecContext(ctx, `
		INSERT INTO asset_embeddings (asset_id, model, embedding_json, created_at)
		VALUES (?, ?, ?, CURRENT_TIMESTAMP)
		ON CONFLICT(asset_id) DO UPDATE SET
			model = excluded.model,
			embedding_json = excluded.embedding_json,
			created_at = excluded.created_at`, assetID, model, string(rawJSON))
	return err
}

// InvalidateEmbeddingsExcept deletes all asset embeddings that were generated by any model other than currentModel.
func (r *Repository) InvalidateEmbeddingsExcept(ctx context.Context, currentModel string) error {
	currentModel = strings.TrimSpace(currentModel)
	if currentModel == "" {
		return nil
	}
	db := r.conn()
	_, err := db.ExecContext(ctx, `DELETE FROM asset_embeddings WHERE model != ?`, currentModel)
	return err
}

type scoredAsset struct {
	id    int64
	score float32
}

// FindSimilarAssetIDs retrieves asset IDs with vector embeddings similar to queryVec using cosine similarity.
func (r *Repository) FindSimilarAssetIDs(ctx context.Context, queryVec []float32, model string, minScore float32, limit int) ([]int64, error) {
	if len(queryVec) == 0 || strings.TrimSpace(model) == "" || limit <= 0 {
		return nil, nil
	}
	db := r.conn()
	rows, err := db.QueryContext(ctx, `SELECT asset_id, embedding_json FROM asset_embeddings WHERE model = ?`, model)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	// Compute query vector norm once
	var queryNorm float32
	for _, v := range queryVec {
		queryNorm += v * v
	}
	if queryNorm <= 0 {
		return nil, nil
	}
	queryNormSqrt := float32(math.Sqrt(float64(queryNorm)))

	var scored []scoredAsset
	for rows.Next() {
		var (
			assetID int64
			rawJSON string
		)
		if err := rows.Scan(&assetID, &rawJSON); err != nil {
			continue
		}
		var vec []float32
		if err := json.Unmarshal([]byte(rawJSON), &vec); err != nil || len(vec) != len(queryVec) {
			continue
		}

		var dot, vecNorm float32
		for i := 0; i < len(queryVec); i++ {
			dot += queryVec[i] * vec[i]
			vecNorm += vec[i] * vec[i]
		}
		if vecNorm <= 0 {
			continue
		}
		sim := dot / (queryNormSqrt * float32(math.Sqrt(float64(vecNorm))))
		if sim >= minScore {
			scored = append(scored, scoredAsset{id: assetID, score: sim})
		}
	}

	sort.Slice(scored, func(i, j int) bool {
		return scored[i].score > scored[j].score
	})

	if len(scored) == 0 {
		return nil, nil
	}

	topScore := scored[0].score
	cutoff := minScore
	if rel := topScore - 0.08; rel > cutoff {
		cutoff = rel
	}

	var res []int64
	for _, s := range scored {
		if s.score < cutoff {
			break
		}
		res = append(res, s.id)
		if len(res) >= limit {
			break
		}
	}
	return res, nil
}

// UpdateAssetMetadata updates the extracted generation parameters and metadata timestamp for an asset.
func (r *Repository) UpdateAssetMetadata(ctx context.Context, assetID int64, meta domain.GenerationParams, extractedAt time.Time) error {
	db := r.conn()
	lorasBytes, _ := json.Marshal(meta.LoRAs)
	if len(lorasBytes) == 0 {
		lorasBytes = []byte("[]")
	}
	_, err := db.ExecContext(ctx, `
		UPDATE assets SET
			engine = ?,
			positive_prompt = ?,
			negative_prompt = ?,
			model_name = ?,
			model_hash = ?,
			sampler = ?,
			scheduler = ?,
			steps = ?,
			cfg_scale = ?,
			seed = ?,
			loras_json = ?,
			raw_metadata = ?,
			metadata_extracted_at = ?
		WHERE id = ?`,
		string(meta.Engine), meta.PositivePrompt, meta.NegativePrompt,
		meta.ModelName, meta.ModelHash, meta.Sampler, meta.Scheduler,
		meta.Steps, meta.CFGScale, meta.Seed, string(lorasBytes), meta.RawMetadata,
		extractedAt.UTC(), assetID,
	)
	return err
}

// ListAssetsMissingMetadata queries all assets that have no positive prompt and no raw metadata, or null extraction date.
func (r *Repository) ListAssetsMissingMetadata(ctx context.Context) ([]domain.MediaAsset, error) {
	db := r.conn()
	rows, err := db.QueryContext(ctx, `
		SELECT a.id, a.root_id, COALESCE(mr.path, ''), COALESCE(mr.label, ''),
		       a.rel_path, a.file_name, a.file_hash, a.media_type, a.width, a.height, a.duration_sec, a.file_size,
		       a.view_count, a.rating, a.favorite, a.category, a.subcategory, a.engine, a.positive_prompt, a.negative_prompt,
		       a.model_name, a.model_hash, a.sampler, a.scheduler, a.steps, a.cfg_scale, a.seed,
		       a.loras_json, a.raw_metadata, a.file_mtime, a.indexed_at, a.metadata_extracted_at, a.faces_scanned_at
		FROM assets a
		LEFT JOIN media_roots mr ON mr.id = a.root_id
		WHERE (a.positive_prompt = '' AND a.raw_metadata = '') OR a.metadata_extracted_at IS NULL
		ORDER BY a.id ASC`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var out []domain.MediaAsset
	for rows.Next() {
		a, err := scanAssetRow(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, *a)
	}
	return out, nil
}

// ListImageAssets returns all cataloged media assets with media_type = 'image'.
func (r *Repository) ListImageAssets(ctx context.Context) ([]domain.MediaAsset, error) {
	db := r.conn()
	rows, err := db.QueryContext(ctx, `
		SELECT a.id, a.root_id, COALESCE(mr.path, ''), COALESCE(mr.label, ''),
		       a.rel_path, a.file_name, a.file_hash, a.media_type, a.width, a.height, a.duration_sec, a.file_size,
		       a.view_count, a.rating, a.favorite, a.category, a.subcategory, a.engine, a.positive_prompt, a.negative_prompt,
		       a.model_name, a.model_hash, a.sampler, a.scheduler, a.steps, a.cfg_scale, a.seed,
		       a.loras_json, a.raw_metadata, a.file_mtime, a.indexed_at, a.metadata_extracted_at, a.faces_scanned_at
		FROM assets a
		LEFT JOIN media_roots mr ON mr.id = a.root_id
		WHERE a.media_type = 'image'
		ORDER BY a.id ASC`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var out []domain.MediaAsset
	for rows.Next() {
		a, err := scanAssetRow(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, *a)
	}
	return out, nil
}

// ListImageAssetsMissingFaces returns image assets that either haven't been scanned for faces
// or were processed previously with 0 faces found.
func (r *Repository) ListImageAssetsMissingFaces(ctx context.Context) ([]domain.MediaAsset, error) {
	db := r.conn()
	rows, err := db.QueryContext(ctx, `
		SELECT a.id, a.root_id, COALESCE(mr.path, ''), COALESCE(mr.label, ''),
		       a.rel_path, a.file_name, a.file_hash, a.media_type, a.width, a.height, a.duration_sec, a.file_size,
		       a.view_count, a.rating, a.favorite, a.category, a.subcategory, a.engine, a.positive_prompt, a.negative_prompt,
		       a.model_name, a.model_hash, a.sampler, a.scheduler, a.steps, a.cfg_scale, a.seed,
		       a.loras_json, a.raw_metadata, a.file_mtime, a.indexed_at, a.metadata_extracted_at, a.faces_scanned_at
		FROM assets a
		LEFT JOIN media_roots mr ON mr.id = a.root_id
		LEFT JOIN faces f ON f.asset_id = a.id
		WHERE a.media_type = 'image' AND (a.faces_scanned_at IS NULL OR f.id IS NULL)
		GROUP BY a.id
		ORDER BY a.id ASC`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var out []domain.MediaAsset
	for rows.Next() {
		a, err := scanAssetRow(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, *a)
	}
	return out, nil
}

// MarkAssetFacesScanned updates the timestamp when an asset was processed for facial recognition.
func (r *Repository) MarkAssetFacesScanned(ctx context.Context, assetID int64, at time.Time) error {
	db := r.conn()
	_, err := db.ExecContext(ctx, `UPDATE assets SET faces_scanned_at = ? WHERE id = ?`, at.UTC(), assetID)
	return err
}

// ListFacesWithEmbeddings retrieves all detected faces that have a non-empty embedding vector.
func (r *Repository) ListFacesWithEmbeddings(ctx context.Context) ([]domain.DetectedFace, error) {
	db := r.conn()
	rows, err := db.QueryContext(ctx, `
		SELECT f.id, f.asset_id, a.root_id, a.rel_path, a.positive_prompt, a.engine, a.model_name,
		       f.bbox_x, f.bbox_y, f.bbox_w, f.bbox_h, f.crop_rel_path, f.confidence,
		       f.cluster_label, f.expression_tag, f.character_id, COALESCE(c.name, ''), f.embedding_json, f.created_at
		FROM faces f
		JOIN assets a ON a.id = f.asset_id
		LEFT JOIN characters c ON c.id = f.character_id
		WHERE f.embedding_json != '' AND f.embedding_json != '[]'
		ORDER BY f.id ASC`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var out []domain.DetectedFace
	for rows.Next() {
		var f domain.DetectedFace
		var engineStr, embJSON string
		var charID sql.NullInt64
		if err := rows.Scan(
			&f.ID, &f.AssetID, &f.AssetRootID, &f.AssetRelPath, &f.AssetPrompt, &engineStr, &f.AssetModel,
			&f.Box.X, &f.Box.Y, &f.Box.Width, &f.Box.Height, &f.CropRelPath, &f.Confidence,
			&f.ClusterLabel, &f.ExpressionTag, &charID, &f.CharacterName, &embJSON, &f.CreatedAt,
		); err != nil {
			return nil, err
		}
		f.AssetEngine = domain.EngineType(engineStr)
		if charID.Valid {
			v := charID.Int64
			f.CharacterID = &v
		}
		_ = json.Unmarshal([]byte(embJSON), &f.Embedding)
		out = append(out, f)
	}
	return out, nil
}

// UpdateFaceCluster sets the cluster label and optional linked character for a detected face.
func (r *Repository) UpdateFaceCluster(ctx context.Context, faceID int64, clusterLabel string, characterID *int64) error {
	db := r.conn()
	_, err := db.ExecContext(ctx, `
		UPDATE faces SET cluster_label = ?, character_id = ? WHERE id = ?`,
		clusterLabel, characterID, faceID)
	return err
}

// RenameFaceCluster updates all face records sharing an old cluster label to a new label,
// optionally linking them to a Character record.
func (r *Repository) RenameFaceCluster(ctx context.Context, oldLabel, newLabel string, characterID *int64) error {
	db := r.conn()
	if characterID != nil && *characterID > 0 {
		_, err := db.ExecContext(ctx, `
			UPDATE faces SET cluster_label = ?, character_id = ? WHERE cluster_label = ?`,
			newLabel, *characterID, oldLabel)
		return err
	}
	_, err := db.ExecContext(ctx, `
		UPDATE faces SET cluster_label = ? WHERE cluster_label = ?`,
		newLabel, oldLabel)
	return err
}

// DeleteFacesForAsset removes all detected face records associated with an asset.
func (r *Repository) DeleteFacesForAsset(ctx context.Context, assetID int64) error {
	db := r.conn()
	_, err := db.ExecContext(ctx, `DELETE FROM faces WHERE asset_id = ?`, assetID)
	return err
}
