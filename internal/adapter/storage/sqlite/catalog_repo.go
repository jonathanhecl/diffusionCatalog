package sqlite

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"

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

	query := `
	INSERT INTO assets (
		root_id, rel_path, file_name, file_hash, media_type, width, height, duration_sec, file_size,
		view_count, rating, favorite, category, subcategory, engine, positive_prompt, negative_prompt, model_name, model_hash,
		sampler, scheduler, steps, cfg_scale, seed, loras_json, raw_metadata, file_mtime
	) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
		raw_metadata = excluded.raw_metadata
	RETURNING id;
	`
	var id int64
	err := db.QueryRowContext(ctx, query,
		a.RootID, a.RelPath, a.FileName, a.FileHash, string(a.MediaType), a.Width, a.Height, a.DurationSec, a.FileSize,
		a.ViewCount, a.Rating, favInt, a.Category, a.Subcategory, string(a.Metadata.Engine), a.Metadata.PositivePrompt, a.Metadata.NegativePrompt,
		a.Metadata.ModelName, a.Metadata.ModelHash, a.Metadata.Sampler, a.Metadata.Scheduler,
		a.Metadata.Steps, a.Metadata.CFGScale, a.Metadata.Seed, string(lorasBytes), a.Metadata.RawMetadata, fileMTimeVal,
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
		       a.loras_json, a.raw_metadata, a.file_mtime, a.indexed_at
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
	return nil
}

// GetExistingAssetFingerprints returns a lookup map keyed by rel_path containing basic metadata
// (id, file_size, file_mtime, file_hash, rating, favorite, category, subcategory) to allow lightning-fast differential scanning.
func (r *Repository) GetExistingAssetFingerprints(ctx context.Context, rootID int64) (map[string]domain.MediaAsset, error) {
	db := r.conn()
	query := `
		SELECT id, rel_path, file_size, file_hash, file_mtime, indexed_at, rating, favorite, category, subcategory
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
		var relPath string
		var mTime, idxTime sql.NullTime
		var favInt int
		if err := rows.Scan(&a.ID, &relPath, &a.FileSize, &a.FileHash, &mTime, &idxTime, &a.Rating, &favInt, &a.Category, &a.Subcategory); err != nil {
			return nil, err
		}
		a.RelPath = relPath
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
		where = append(where, `(a.positive_prompt LIKE ? OR a.file_name LIKE ? OR a.model_name LIKE ? OR a.category LIKE ? OR a.subcategory LIKE ? OR EXISTS (
			SELECT 1 FROM asset_tags at JOIN tags t ON t.id = at.tag_id WHERE at.asset_id = a.id AND t.name LIKE ?
		))`)
		args = append(args, like, like, like, like, like, like)
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
	if filter.RootID > 0 {
		where = append(where, `a.root_id = ?`)
		args = append(args, filter.RootID)
	}
	if filter.Category != "" {
		where = append(where, `LOWER(a.category) = LOWER(?)`)
		args = append(args, strings.TrimSpace(filter.Category))
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

// GetSettings retrieves the application, Ollama, and oMLX configuration from SQLite.
func (r *Repository) GetSettings(ctx context.Context) (domain.AppSettings, error) {
	db := r.conn()
	s := domain.AppSettings{
		OllamaBaseURL:          "http://127.0.0.1:11434",
		OMLXBaseURL:            "http://127.0.0.1:8000",
		OMLXAPIKey:             "",
		VisionProvider:         domain.ProviderOllama,
		VisionModel:            "llama3.2-vision:latest",
		VisionThink:            "off",
		Image2PromptProvider:   domain.ProviderOllama,
		Image2PromptModel:      "llama3.2-vision:latest",
		Image2PromptThink:      "off",
		Text2PromptProvider:    domain.ProviderOpenAI,
		Text2PromptModel:       "SuperQwen3.8-27b-abliterated-MLX-4bit",
		Text2PromptThink:       "low",
		DefaultTargetGenerator: "flux",
	}
	rows, err := db.QueryContext(ctx, `SELECT key, value FROM app_settings`)
	if err != nil {
		return s, err
	}
	defer rows.Close()

	for rows.Next() {
		var k, v string
		if err := rows.Scan(&k, &v); err == nil {
			switch k {
			case "ollama_base_url":
				s.OllamaBaseURL = v
			case "omlx_base_url":
				s.OMLXBaseURL = v
			case "omlx_api_key":
				s.OMLXAPIKey = v
			case "vision_provider":
				s.VisionProvider = domain.AIProviderType(v)
			case "vision_model":
				s.VisionModel = v
			case "vision_think":
				s.VisionThink = v
			case "image2prompt_provider":
				s.Image2PromptProvider = domain.AIProviderType(v)
			case "image2prompt_model":
				s.Image2PromptModel = v
			case "image2prompt_think":
				s.Image2PromptThink = v
			case "text2prompt_provider":
				s.Text2PromptProvider = domain.AIProviderType(v)
			case "text2prompt_model":
				s.Text2PromptModel = v
			case "text2prompt_think":
				s.Text2PromptThink = v
			case "default_target_generator":
				s.DefaultTargetGenerator = v
			case "auto_caption_on_scan":
				s.AutoCaptionOnScan = (v == "true" || v == "1")
			}
		}
	}
	return s, nil
}

// SaveSettings persists application, Ollama, and oMLX settings into SQLite.
func (r *Repository) SaveSettings(ctx context.Context, s domain.AppSettings) error {
	db := r.conn()
	autoCap := "false"
	if s.AutoCaptionOnScan {
		autoCap = "true"
	}
	if s.VisionProvider == "" {
		s.VisionProvider = domain.ProviderOllama
	}
	if s.VisionThink == "" {
		s.VisionThink = "off"
	}
	if s.Image2PromptProvider == "" {
		s.Image2PromptProvider = domain.ProviderOllama
	}
	if s.Image2PromptThink == "" {
		s.Image2PromptThink = "off"
	}
	if s.Text2PromptProvider == "" {
		s.Text2PromptProvider = domain.ProviderOllama
	}
	if s.Text2PromptThink == "" {
		s.Text2PromptThink = "low"
	}
	pairs := map[string]string{
		"ollama_base_url":          strings.TrimSpace(s.OllamaBaseURL),
		"omlx_base_url":            strings.TrimSpace(s.OMLXBaseURL),
		"omlx_api_key":             strings.TrimSpace(s.OMLXAPIKey),
		"vision_provider":          string(s.VisionProvider),
		"vision_model":             strings.TrimSpace(s.VisionModel),
		"vision_think":             strings.TrimSpace(s.VisionThink),
		"image2prompt_provider":    string(s.Image2PromptProvider),
		"image2prompt_model":       strings.TrimSpace(s.Image2PromptModel),
		"image2prompt_think":       strings.TrimSpace(s.Image2PromptThink),
		"text2prompt_provider":     string(s.Text2PromptProvider),
		"text2prompt_model":        strings.TrimSpace(s.Text2PromptModel),
		"text2prompt_think":        strings.TrimSpace(s.Text2PromptThink),
		"default_target_generator": strings.TrimSpace(s.DefaultTargetGenerator),
		"auto_caption_on_scan":     autoCap,
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
			asset_id, character_id, target_profile_slug, scene_direction, output_prompt, provider, model_name
		) VALUES (?, ?, ?, ?, ?, ?, ?)
		RETURNING id`,
		gp.AssetID, gp.CharacterID, gp.TargetProfileSlug, gp.SceneDirection, gp.OutputPrompt, string(gp.Provider), gp.ModelName,
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
		       gp.provider, gp.model_name, gp.created_at
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
			&prov, &item.ModelName, &item.CreatedAt,
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
		       a.loras_json, a.raw_metadata, a.file_mtime, a.indexed_at
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
	return err
}

// ListTags returns all tags ordered by usage count and name.
func (r *Repository) ListTags(ctx context.Context) ([]domain.Tag, error) {
	db := r.conn()
	rows, err := db.QueryContext(ctx, `
		SELECT t.id, t.name, t.facet, t.color, COUNT(at.asset_id) AS cnt
		FROM tags t
		LEFT JOIN asset_tags at ON at.tag_id = t.id
		GROUP BY t.id
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

	var id int64
	err := db.QueryRowContext(ctx, `
		INSERT INTO characters (
			name, slug, archetype, description, trigger_words,
			base_positive_prompt, base_negative_prompt, recommended_checkpoint, loras_json, avatar_asset_id
		) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
		ON CONFLICT(name) DO UPDATE SET
			archetype = excluded.archetype,
			description = excluded.description,
			trigger_words = excluded.trigger_words,
			base_positive_prompt = excluded.base_positive_prompt,
			base_negative_prompt = excluded.base_negative_prompt,
			recommended_checkpoint = excluded.recommended_checkpoint,
			loras_json = excluded.loras_json,
			avatar_asset_id = COALESCE(excluded.avatar_asset_id, characters.avatar_asset_id)
		RETURNING id`,
		c.Name, slug, c.Archetype, c.Description, c.TriggerWords,
		c.BasePositivePrompt, c.BaseNegativePrompt, c.RecommendedCheckpoint, string(lorasJSON), c.AvatarAssetID,
	).Scan(&id)
	if err != nil {
		return 0, err
	}
	c.ID = id
	c.Slug = slug
	return id, nil
}

// GetCharacterByID returns a character profile along with its linked example assets.
func (r *Repository) GetCharacterByID(ctx context.Context, id int64) (*domain.Character, error) {
	db := r.conn()
	var c domain.Character
	var lorasStr string
	var avatarID sql.NullInt64

	err := db.QueryRowContext(ctx, `
		SELECT id, name, slug, archetype, description, trigger_words,
		       base_positive_prompt, base_negative_prompt, recommended_checkpoint, loras_json, avatar_asset_id, created_at
		FROM characters WHERE id = ?`, id,
	).Scan(
		&c.ID, &c.Name, &c.Slug, &c.Archetype, &c.Description, &c.TriggerWords,
		&c.BasePositivePrompt, &c.BaseNegativePrompt, &c.RecommendedCheckpoint, &lorasStr, &avatarID, &c.CreatedAt,
	)
	if err != nil {
		return nil, err
	}
	_ = json.Unmarshal([]byte(lorasStr), &c.LoRAs)
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

// ListCharacters returns all character profiles with their example counts.
func (r *Repository) ListCharacters(ctx context.Context) ([]domain.Character, error) {
	db := r.conn()
	rows, err := db.QueryContext(ctx, `
		SELECT c.id, c.name, c.slug, c.archetype, c.description, c.trigger_words,
		       c.base_positive_prompt, c.base_negative_prompt, c.recommended_checkpoint, c.loras_json,
		       c.avatar_asset_id, c.created_at, COUNT(ce.asset_id)
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
		var avatarID sql.NullInt64
		if err := rows.Scan(
			&c.ID, &c.Name, &c.Slug, &c.Archetype, &c.Description, &c.TriggerWords,
			&c.BasePositivePrompt, &c.BaseNegativePrompt, &c.RecommendedCheckpoint, &lorasStr,
			&avatarID, &c.CreatedAt, &c.ExampleCount,
		); err != nil {
			return nil, err
		}
		_ = json.Unmarshal([]byte(lorasStr), &c.LoRAs)
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
	if filter.RootID > 0 {
		where = append(where, `a.root_id = ?`)
		args = append(args, filter.RootID)
	}
	if filter.CharacterID > 0 {
		where = append(where, `f.character_id = ?`)
		args = append(args, filter.CharacterID)
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

type rowScanner interface {
	Scan(dest ...interface{}) error
}

func scanAssetRow(s rowScanner) (*domain.MediaAsset, error) {
	var a domain.MediaAsset
	var mediaTypeStr, engineStr, lorasStr string
	var favInt int
	var fileMTime sql.NullTime
	err := s.Scan(
		&a.ID, &a.RootID, &a.RootPath, &a.RootLabel,
		&a.RelPath, &a.FileName, &a.FileHash, &mediaTypeStr,
		&a.Width, &a.Height, &a.DurationSec, &a.FileSize,
		&a.ViewCount, &a.Rating, &favInt, &a.Category, &a.Subcategory, &engineStr,
		&a.Metadata.PositivePrompt, &a.Metadata.NegativePrompt,
		&a.Metadata.ModelName, &a.Metadata.ModelHash,
		&a.Metadata.Sampler, &a.Metadata.Scheduler,
		&a.Metadata.Steps, &a.Metadata.CFGScale, &a.Metadata.Seed,
		&lorasStr, &a.Metadata.RawMetadata, &fileMTime, &a.IndexedAt,
	)
	if err != nil {
		return nil, err
	}
	if fileMTime.Valid {
		a.FileMTime = fileMTime.Time
	} else {
		a.FileMTime = a.IndexedAt
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
