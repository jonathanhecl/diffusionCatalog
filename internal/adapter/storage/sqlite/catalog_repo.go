package sqlite

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/jonathanhecl/diffusionCatalog/internal/domain"
)

// Repository implements domain.CatalogRepository using SQLite and FTS5.
type Repository struct {
	db *sql.DB
}

// NewRepository creates a SQLite-backed CatalogRepository.
func NewRepository(db *sql.DB) *Repository {
	return &Repository{db: db}
}

// EnsureDefaultAdmin seeds the default admin user if the users table is empty.
func (r *Repository) EnsureDefaultAdmin(ctx context.Context, username, passwordHash string) error {
	var count int
	if err := r.db.QueryRowContext(ctx, `SELECT COUNT(*) FROM users`).Scan(&count); err != nil {
		return err
	}
	if count > 0 {
		return nil
	}
	_, err := r.db.ExecContext(ctx,
		`INSERT INTO users (username, password_hash, role) VALUES (?, ?, 'admin')`,
		username, passwordHash,
	)
	return err
}

// GetUserByUsername fetches a user record by username.
func (r *Repository) GetUserByUsername(ctx context.Context, username string) (*domain.User, error) {
	var u domain.User
	err := r.db.QueryRowContext(ctx,
		`SELECT id, username, password_hash, role, created_at FROM users WHERE username = ?`,
		username,
	).Scan(&u.ID, &u.Username, &u.PasswordHash, &u.Role, &u.CreatedAt)
	if err != nil {
		return nil, err
	}
	return &u, nil
}

// UpsertAsset inserts or updates a cataloged media asset by its relative path.
func (r *Repository) UpsertAsset(ctx context.Context, a *domain.MediaAsset) (int64, error) {
	lorasBytes, _ := json.Marshal(a.Metadata.LoRAs)
	if len(lorasBytes) == 0 {
		lorasBytes = []byte("[]")
	}

	query := `
	INSERT INTO assets (
		rel_path, file_name, file_hash, media_type, width, height, duration_sec, file_size,
		view_count, rating, engine, positive_prompt, negative_prompt, model_name, model_hash,
		sampler, scheduler, steps, cfg_scale, seed, loras_json, raw_metadata
	) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
	ON CONFLICT(rel_path) DO UPDATE SET
		file_hash = excluded.file_hash,
		width = excluded.width,
		height = excluded.height,
		file_size = excluded.file_size,
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
	err := r.db.QueryRowContext(ctx, query,
		a.RelPath, a.FileName, a.FileHash, string(a.MediaType), a.Width, a.Height, a.DurationSec, a.FileSize,
		a.ViewCount, a.Rating, string(a.Metadata.Engine), a.Metadata.PositivePrompt, a.Metadata.NegativePrompt,
		a.Metadata.ModelName, a.Metadata.ModelHash, a.Metadata.Sampler, a.Metadata.Scheduler,
		a.Metadata.Steps, a.Metadata.CFGScale, a.Metadata.Seed, string(lorasBytes), a.Metadata.RawMetadata,
	).Scan(&id)
	if err != nil {
		return 0, err
	}
	a.ID = id
	return id, nil
}

// GetAssetByID retrieves a full MediaAsset along with its tags, groups, characters, and detected faces.
func (r *Repository) GetAssetByID(ctx context.Context, id int64) (*domain.MediaAsset, error) {
	row := r.db.QueryRowContext(ctx, `
		SELECT id, rel_path, file_name, file_hash, media_type, width, height, duration_sec, file_size,
		       view_count, rating, engine, positive_prompt, negative_prompt, model_name, model_hash,
		       sampler, scheduler, steps, cfg_scale, seed, loras_json, raw_metadata, indexed_at
		FROM assets WHERE id = ?`, id)

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

// IncrementAssetView increments the view counter of an asset.
func (r *Repository) IncrementAssetView(ctx context.Context, id int64) error {
	_, err := r.db.ExecContext(ctx, `UPDATE assets SET view_count = view_count + 1 WHERE id = ?`, id)
	return err
}

// UpdateAssetPrompt updates an asset's positive prompt (e.g. from Ollama Image2Prompt or Text2Prompt).
func (r *Repository) UpdateAssetPrompt(ctx context.Context, assetID int64, positivePrompt, engine string) error {
	if engine != "" {
		_, err := r.db.ExecContext(ctx, `UPDATE assets SET positive_prompt = ?, engine = ? WHERE id = ?`, positivePrompt, engine, assetID)
		return err
	}
	_, err := r.db.ExecContext(ctx, `UPDATE assets SET positive_prompt = ? WHERE id = ?`, positivePrompt, assetID)
	return err
}

// GetSettings retrieves the application, Ollama, and oMLX configuration from SQLite.
func (r *Repository) GetSettings(ctx context.Context) (domain.AppSettings, error) {
	s := domain.AppSettings{
		OllamaBaseURL:          "http://mac-mini.local:11434",
		OMLXBaseURL:            "http://mac-mini.local:8000",
		OMLXAPIKey:             "omlx",
		VisionProvider:         domain.ProviderOllama,
		VisionModel:            "llama3.2-vision:latest",
		Image2PromptProvider:   domain.ProviderOllama,
		Image2PromptModel:      "llama3.2-vision:latest",
		Text2PromptProvider:    domain.ProviderOMLX,
		Text2PromptModel:       "SuperQwen3.8-27b-abliterated-MLX-4bit",
		DefaultTargetGenerator: "flux",
	}
	rows, err := r.db.QueryContext(ctx, `SELECT key, value FROM app_settings`)
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
			case "image2prompt_provider":
				s.Image2PromptProvider = domain.AIProviderType(v)
			case "image2prompt_model":
				s.Image2PromptModel = v
			case "text2prompt_provider":
				s.Text2PromptProvider = domain.AIProviderType(v)
			case "text2prompt_model":
				s.Text2PromptModel = v
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
	autoCap := "false"
	if s.AutoCaptionOnScan {
		autoCap = "true"
	}
	if s.VisionProvider == "" {
		s.VisionProvider = domain.ProviderOllama
	}
	if s.Image2PromptProvider == "" {
		s.Image2PromptProvider = domain.ProviderOllama
	}
	if s.Text2PromptProvider == "" {
		s.Text2PromptProvider = domain.ProviderOllama
	}
	pairs := map[string]string{
		"ollama_base_url":          strings.TrimSpace(s.OllamaBaseURL),
		"omlx_base_url":            strings.TrimSpace(s.OMLXBaseURL),
		"omlx_api_key":             strings.TrimSpace(s.OMLXAPIKey),
		"vision_provider":          string(s.VisionProvider),
		"vision_model":             strings.TrimSpace(s.VisionModel),
		"image2prompt_provider":    string(s.Image2PromptProvider),
		"image2prompt_model":       strings.TrimSpace(s.Image2PromptModel),
		"text2prompt_provider":     string(s.Text2PromptProvider),
		"text2prompt_model":        strings.TrimSpace(s.Text2PromptModel),
		"default_target_generator": strings.TrimSpace(s.DefaultTargetGenerator),
		"auto_caption_on_scan":     autoCap,
	}
	for k, v := range pairs {
		if _, err := r.db.ExecContext(ctx, `
			INSERT INTO app_settings (key, value) VALUES (?, ?)
			ON CONFLICT(key) DO UPDATE SET value = excluded.value`, k, v); err != nil {
			return err
		}
	}
	return nil
}

// UpsertPromptProfile creates or updates a generator system prompt profile (flux, krea2, moss, minimaxh3, etc.).
func (r *Repository) UpsertPromptProfile(ctx context.Context, p *domain.PromptProfile) (int64, error) {
	slug := slugify(p.Slug)
	if slug == "item" && p.Name != "" {
		slug = slugify(p.Name)
	}
	modality := string(p.TargetModality)
	if modality != "video" {
		modality = "image"
	}
	var id int64
	err := r.db.QueryRowContext(ctx, `
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
	_, err := r.db.ExecContext(ctx, `DELETE FROM prompt_profiles WHERE id = ?`, id)
	return err
}

// GetPromptProfileBySlug fetches a generator system prompt profile by its slug.
func (r *Repository) GetPromptProfileBySlug(ctx context.Context, slug string) (*domain.PromptProfile, error) {
	var p domain.PromptProfile
	var mod string
	err := r.db.QueryRowContext(ctx, `
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
	rows, err := r.db.QueryContext(ctx, `
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

// SaveGeneratedPrompt stores a newly generated prompt linked to an Asset and/or Character.
func (r *Repository) SaveGeneratedPrompt(ctx context.Context, gp *domain.GeneratedPrompt) (int64, error) {
	var id int64
	err := r.db.QueryRowContext(ctx, `
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

	rows, err := r.db.QueryContext(ctx, query, args...)
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

// QueryCatalog executes a filtered, faceted, and paginated search across assets (or faces when FaceMode is active).
func (r *Repository) QueryCatalog(ctx context.Context, filter domain.CatalogFilter) (*domain.CatalogPage, error) {
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
	tags, _ := r.ListTags(ctx)
	page.AvailableTags = tags
	for _, t := range tags {
		page.FacetTags[t.Facet] = append(page.FacetTags[t.Facet], t)
	}
	page.Engines, _ = r.distinctStrings(ctx, `SELECT DISTINCT engine FROM assets WHERE engine != '' ORDER BY engine`)
	page.Models, _ = r.distinctStrings(ctx, `SELECT DISTINCT model_name FROM assets WHERE model_name != '' ORDER BY model_name`)
	page.Groups, _ = r.ListGroups(ctx)
	page.Characters, _ = r.ListCharacters(ctx)

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
		return page, nil
	}

	var where []string
	var args []interface{}

	if q := strings.TrimSpace(filter.Query); q != "" {
		like := "%" + q + "%"
		where = append(where, `(a.positive_prompt LIKE ? OR a.file_name LIKE ? OR a.model_name LIKE ? OR EXISTS (
			SELECT 1 FROM asset_tags at JOIN tags t ON t.id = at.tag_id WHERE at.asset_id = a.id AND t.name LIKE ?
		))`)
		args = append(args, like, like, like, like)
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

	whereClause := ""
	if len(where) > 0 {
		whereClause = "WHERE " + strings.Join(where, " AND ")
	}

	countSQL := fmt.Sprintf(`SELECT COUNT(*) FROM assets a %s`, whereClause)
	if err := r.db.QueryRowContext(ctx, countSQL, args...).Scan(&page.TotalCount); err != nil {
		return nil, err
	}

	page.TotalPages = (page.TotalCount + filter.PerPage - 1) / filter.PerPage
	if page.TotalPages < 1 {
		page.TotalPages = 1
	}

	orderBy := "a.indexed_at DESC, a.id DESC"
	switch filter.SortBy {
	case "popular":
		orderBy = "a.rating DESC, a.view_count DESC, a.id DESC"
	case "views":
		orderBy = "a.view_count DESC, a.id DESC"
	case "seed":
		orderBy = "a.seed ASC, a.id DESC"
	}

	offset := (filter.Page - 1) * filter.PerPage
	selectSQL := fmt.Sprintf(`
		SELECT a.id, a.rel_path, a.file_name, a.file_hash, a.media_type, a.width, a.height, a.duration_sec, a.file_size,
		       a.view_count, a.rating, a.engine, a.positive_prompt, a.negative_prompt, a.model_name, a.model_hash,
		       a.sampler, a.scheduler, a.steps, a.cfg_scale, a.seed, a.loras_json, a.raw_metadata, a.indexed_at
		FROM assets a
		%s
		ORDER BY %s
		LIMIT ? OFFSET ?`, whereClause, orderBy)

	queryArgs := append(append([]interface{}{}, args...), filter.PerPage, offset)
	rows, err := r.db.QueryContext(ctx, selectSQL, queryArgs...)
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
	name = strings.TrimSpace(name)
	if facet == "" {
		facet = "general"
	}
	if color == "" {
		color = "#2563eb"
	}
	var t domain.Tag
	err := r.db.QueryRowContext(ctx, `
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
	_, err := r.db.ExecContext(ctx, `INSERT OR IGNORE INTO asset_tags (asset_id, tag_id) VALUES (?, ?)`, assetID, tagID)
	return err
}

// ListTags returns all tags ordered by usage count and name.
func (r *Repository) ListTags(ctx context.Context) ([]domain.Tag, error) {
	rows, err := r.db.QueryContext(ctx, `
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
	slug := slugify(g.Name)
	if g.Category == "" {
		g.Category = "Collection"
	}
	var id int64
	err := r.db.QueryRowContext(ctx, `
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
	_, err := r.db.ExecContext(ctx, `INSERT OR IGNORE INTO group_assets (group_id, asset_id) VALUES (?, ?)`, groupID, assetID)
	return err
}

// ListGroups lists all groups with their member counts.
func (r *Repository) ListGroups(ctx context.Context) ([]domain.Group, error) {
	rows, err := r.db.QueryContext(ctx, `
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
	slug := slugify(c.Name)
	if c.Archetype == "" {
		c.Archetype = "Character"
	}
	lorasJSON, _ := json.Marshal(c.LoRAs)
	if len(lorasJSON) == 0 {
		lorasJSON = []byte("[]")
	}

	var id int64
	err := r.db.QueryRowContext(ctx, `
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
	var c domain.Character
	var lorasStr string
	var avatarID sql.NullInt64

	err := r.db.QueryRowContext(ctx, `
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
	_, err := r.db.ExecContext(ctx, `INSERT OR IGNORE INTO character_examples (character_id, asset_id) VALUES (?, ?)`, characterID, assetID)
	return err
}

// ListCharacters returns all character profiles with their example counts.
func (r *Repository) ListCharacters(ctx context.Context) ([]domain.Character, error) {
	rows, err := r.db.QueryContext(ctx, `
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
	embJSON, _ := json.Marshal(face.Embedding)
	if len(embJSON) == 0 {
		embJSON = []byte("[]")
	}
	var id int64
	err := r.db.QueryRowContext(ctx, `
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
	var where []string
	var args []interface{}

	if q := strings.TrimSpace(filter.Query); q != "" {
		like := "%" + q + "%"
		where = append(where, `(f.cluster_label LIKE ? OR f.expression_tag LIKE ? OR a.positive_prompt LIKE ?)`)
		args = append(args, like, like, like)
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
	if err := r.db.QueryRowContext(ctx, countSQL, args...).Scan(&total); err != nil {
		return nil, 0, err
	}

	offset := (filter.Page - 1) * filter.PerPage
	querySQL := fmt.Sprintf(`
		SELECT f.id, f.asset_id, a.rel_path, a.positive_prompt, a.engine, a.model_name,
		       f.bbox_x, f.bbox_y, f.bbox_w, f.bbox_h, f.crop_rel_path, f.confidence,
		       f.cluster_label, f.expression_tag, f.character_id, COALESCE(c.name, ''), f.created_at
		FROM faces f
		JOIN assets a ON a.id = f.asset_id
		LEFT JOIN characters c ON c.id = f.character_id
		%s
		ORDER BY f.id DESC
		LIMIT ? OFFSET ?`, whereClause)

	queryArgs := append(append([]interface{}{}, args...), filter.PerPage, offset)
	rows, err := r.db.QueryContext(ctx, querySQL, queryArgs...)
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
			&f.ID, &f.AssetID, &f.AssetRelPath, &f.AssetPrompt, &engineStr, &f.AssetModel,
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
	err := s.Scan(
		&a.ID, &a.RelPath, &a.FileName, &a.FileHash, &mediaTypeStr,
		&a.Width, &a.Height, &a.DurationSec, &a.FileSize,
		&a.ViewCount, &a.Rating, &engineStr,
		&a.Metadata.PositivePrompt, &a.Metadata.NegativePrompt,
		&a.Metadata.ModelName, &a.Metadata.ModelHash,
		&a.Metadata.Sampler, &a.Metadata.Scheduler,
		&a.Metadata.Steps, &a.Metadata.CFGScale, &a.Metadata.Seed,
		&lorasStr, &a.Metadata.RawMetadata, &a.IndexedAt,
	)
	if err != nil {
		return nil, err
	}
	a.MediaType = domain.MediaType(mediaTypeStr)
	a.Metadata.Engine = domain.EngineType(engineStr)
	_ = json.Unmarshal([]byte(lorasStr), &a.Metadata.LoRAs)
	return &a, nil
}

func (r *Repository) getTagsForAsset(ctx context.Context, assetID int64) ([]domain.Tag, error) {
	rows, err := r.db.QueryContext(ctx, `
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
	rows, err := r.db.QueryContext(ctx, `
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
	rows, err := r.db.QueryContext(ctx, `
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
	rows, err := r.db.QueryContext(ctx, `
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
	rows, err := r.db.QueryContext(ctx, query)
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
