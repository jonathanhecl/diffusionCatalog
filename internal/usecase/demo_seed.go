package usecase

import (
	"bytes"
	"context"
	"encoding/binary"
	"hash/crc32"
	"image"
	"image/color"
	"image/png"
	"math"
	"os"
	"path/filepath"

	"github.com/jonathanhecl/diffusionCatalog/internal/domain"
)

type demoSpec struct {
	Subfolder      string
	FileName       string
	TextChunkKey   string
	TextChunkValue string
	PalettePrimary color.RGBA
	PaletteAccent  color.RGBA
	PatternMode    int
	Tags           []struct{ Name, Facet string }
	CharacterName  string
	ExpressionTag  string
	ClusterLabel   string
	Views          int64
	Rating         int
}

// SeedDemoFolderIfEmpty checks if mediaDir contains any media files; if empty, it creates
// synthetic PNG files with real embedded PNG tEXt chunks (A1111, ComfyUI, Fooocus, SwarmUI)
// and seeds agnostic tags, groups, character profiles, and sample face crops.
func SeedDemoFolderIfEmpty(ctx context.Context, mediaDir, cacheDir string, repo domain.CatalogRepository, scanner *ScannerService) error {
	_ = os.MkdirAll(mediaDir, 0755)
	entries, err := os.ReadDir(mediaDir)
	if err != nil {
		return err
	}
	for _, e := range entries {
		if !e.IsDir() && e.Name() != ".DS_Store" {
			return nil
		}
		if e.IsDir() && e.Name() != "." && e.Name() != ".." {
			return nil
		}
	}

	specs := []demoSpec{
		{
			Subfolder:    "Cyberpunk_Chronicles",
			FileName:     "01_kaelen_neon_visor.png",
			TextChunkKey: "parameters",
			TextChunkValue: `masterpiece, best quality, cinematic portrait of Kaelen-7, cybernetic obsidian visor, rain-slicked Neo-Veridia alley, volumetric cyan and magenta lighting, <lora:kaelen_v3:0.85> <lora:anamorphe_lens:0.5>
Negative prompt: lowres, blurry, bad anatomy, oversaturated, watermark
Steps: 30, Sampler: DPM++ 2M Karras, Schedule type: Karras, CFG scale: 6.5, Seed: 884120951, Size: 768x960, Model hash: 9f1a2b3c, Model: juggernautXL_v9`,
			PalettePrimary: color.RGBA{14, 29, 64, 255},
			PaletteAccent:  color.RGBA{56, 189, 248, 255},
			PatternMode:    0,
			Tags: []struct{ Name, Facet string }{
				{"cyberpunk", "style"},
				{"volumetric-neon", "lighting"},
				{"close-up", "composition"},
				{"visor", "subject"},
			},
			CharacterName: "Kaelen-7 (Netrunner)",
			ExpressionTag: "Focused Gaze",
			ClusterLabel:  "Cluster #01 (Kaelen)",
			Views:         1420,
			Rating:        5,
		},
		{
			Subfolder:    "Cyberpunk_Chronicles",
			FileName:     "02_kaelen_terminal_deck.png",
			TextChunkKey: "prompt",
			TextChunkValue: `{
				"1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "flux1-dev-fp8.safetensors"}},
				"2": {"class_type": "LoraLoader", "inputs": {"lora_name": "kaelen_v3.safetensors"}},
				"3": {"class_type": "CLIPTextEncode", "inputs": {"text": "Kaelen-7 operating a holographic neural deck, dark server vault, emerald rim light, crisp 85mm depth of field"}},
				"4": {"class_type": "CLIPTextEncode", "inputs": {"text": "worst quality, low quality, deformed hands, extra digits"}},
				"5": {"class_type": "KSampler", "inputs": {"seed": 441928371, "steps": 24, "cfg": 3.5, "sampler_name": "euler", "scheduler": "simple"}}
			}`,
			PalettePrimary: color.RGBA{12, 38, 52, 255},
			PaletteAccent:  color.RGBA{16, 185, 129, 255},
			PatternMode:    1,
			Tags: []struct{ Name, Facet string }{
				{"cyberpunk", "style"},
				{"rim-light", "lighting"},
				{"medium-shot", "composition"},
				{"hologram", "subject"},
			},
			CharacterName: "Kaelen-7 (Netrunner)",
			ExpressionTag: "Analytical",
			ClusterLabel:  "Cluster #01 (Kaelen)",
			Views:         980,
			Rating:        5,
		},
		{
			Subfolder:    "Astral_Archivists",
			FileName:     "03_lyra_observatory_dome.png",
			TextChunkKey: "parameters",
			TextChunkValue: `editorial fantasy illustration, Lyra Vance inside brass celestial astrolabe dome, floating starlight motes, golden hour bounce light, <lora:lyra_astral_v2:0.9>
Negative prompt: ugly, low contrast, flat shading, text, signature
Steps: 34, Sampler: Euler a, Schedule type: Automatic, CFG scale: 7.0, Seed: 771029384, Size: 768x960, Model hash: 4d5e6f7a, Model: animagineXL_v31`,
			PalettePrimary: color.RGBA{30, 27, 75, 255},
			PaletteAccent:  color.RGBA{251, 191, 36, 255},
			PatternMode:    2,
			Tags: []struct{ Name, Facet string }{
				{"celestial-fantasy", "style"},
				{"golden-hour", "lighting"},
				{"portrait", "composition"},
				{"astrolabe", "subject"},
			},
			CharacterName: "Lyra Vance (Astrolabe Chronicler)",
			ExpressionTag: "Wonder",
			ClusterLabel:  "Cluster #02 (Lyra)",
			Views:         1650,
			Rating:        5,
		},
		{
			Subfolder:    "Astral_Archivists",
			FileName:     "04_lyra_ancient_codex.png",
			TextChunkKey: "sui_image_params",
			TextChunkValue: `{
				"prompt": "Lyra Vance reading a luminescent glass codex, library of Alexandria sci-fi hybrid, soft bioluminescent blue glow, intricate filigree robes",
				"negative_prompt": "blurry, lowres, watermark, cropped",
				"model": "sdxl_albedoBase_v21",
				"steps": 28,
				"cfg_scale": 5.5,
				"seed": 192837465,
				"sampler": "dpmpp_2m_sde",
				"scheduler": "karras"
			}`,
			PalettePrimary: color.RGBA{23, 37, 84, 255},
			PaletteAccent:  color.RGBA{147, 197, 253, 255},
			PatternMode:    3,
			Tags: []struct{ Name, Facet string }{
				{"celestial-fantasy", "style"},
				{"bioluminescent", "lighting"},
				{"close-up", "composition"},
				{"codex", "subject"},
			},
			CharacterName: "Lyra Vance (Astrolabe Chronicler)",
			ExpressionTag: "Serene Smile",
			ClusterLabel:  "Cluster #02 (Lyra)",
			Views:         1120,
			Rating:        4,
		},
		{
			Subfolder:    "Mecha_Architecture",
			FileName:     "05_solaris_hangar_bay.png",
			TextChunkKey: "parameters",
			TextChunkValue: `architectural concept art, Unit-09 Solaris heavy orbital frame docked in subterranean gantry, dramatic volumetric god rays, industrial sparks, brutalist concrete, <lora:hard_surface_mech:0.75>
Negative prompt: blurry, low detail, cartoon, oversaturated
Steps: 36, Sampler: DPM++ SDE Karras, Schedule type: Karras, CFG scale: 7.5, Seed: 556677889, Size: 768x960, Model hash: 8c9d0e1f, Model: realvisXL_v40, Version: Fooocus v2.4.3`,
			PalettePrimary: color.RGBA{39, 39, 42, 255},
			PaletteAccent:  color.RGBA{249, 115, 22, 255},
			PatternMode:    4,
			Tags: []struct{ Name, Facet string }{
				{"brutalist-scifi", "style"},
				{"god-rays", "lighting"},
				{"wide-angle", "composition"},
				{"mecha-frame", "subject"},
			},
			CharacterName: "Unit-09 Solaris (Orbital Frame)",
			ExpressionTag: "Optic Array Active",
			ClusterLabel:  "Cluster #03 (Solaris)",
			Views:         840,
			Rating:        4,
		},
		{
			Subfolder:    "Mecha_Architecture",
			FileName:     "06_solaris_orbital_drop.png",
			TextChunkKey: "invokeai_metadata",
			TextChunkValue: `{
				"positive_prompt": "Unit-09 Solaris descending through stratospheric clouds, plasma re-entry trail, high contrast telemetry HUD overlay, cinematic IMAX frame",
				"negative_prompt": "low quality, jpeg artifacts, distorted horizon",
				"model": "flux1-schnell-fp8",
				"steps": 12,
				"cfg_scale": 2.0,
				"seed": 334455667,
				"sampler": "euler",
				"scheduler": "beta"
			}`,
			PalettePrimary: color.RGBA{49, 16, 47, 255},
			PaletteAccent:  color.RGBA{244, 63, 94, 255},
			PatternMode:    5,
			Tags: []struct{ Name, Facet string }{
				{"brutalist-scifi", "style"},
				{"high-contrast", "lighting"},
				{"dynamic-angle", "composition"},
				{"orbital-drop", "subject"},
			},
			CharacterName: "Unit-09 Solaris (Orbital Frame)",
			ExpressionTag: "Combat Telemetry",
			ClusterLabel:  "Cluster #03 (Solaris)",
			Views:         1310,
			Rating:        5,
		},
	}

	cropsDir := filepath.Join(cacheDir, "crops")
	_ = os.MkdirAll(cropsDir, 0755)

	// Generate synthetic PNG files with embedded metadata chunks
	for _, sp := range specs {
		dir := filepath.Join(mediaDir, sp.Subfolder)
		_ = os.MkdirAll(dir, 0755)
		fullPath := filepath.Join(dir, sp.FileName)

		img := renderSyntheticArtwork(480, 600, sp.PalettePrimary, sp.PaletteAccent, sp.PatternMode)
		if err := writePNGWithTEXTChunk(fullPath, img, sp.TextChunkKey, sp.TextChunkValue); err != nil {
			return err
		}
	}

	// Scan the folder using ScannerService
	if _, err := scanner.Scan(ctx); err != nil {
		return err
	}

	// Create Character Profiles with canonical prompts & attach examples + face crops
	charDefs := map[string]*domain.Character{
		"Kaelen-7 (Netrunner)": {
			Name:                  "Kaelen-7 (Netrunner)",
			Archetype:             "Cyberpunk Protagonist",
			Description:           "Neural-interface archivist operating in Neo-Veridia's deep-grid sectors. Consistent obsidian visor and high-collar tactical coat.",
			TriggerWords:          "kaelen7, cybernetic obsidian visor, silver undercut hair, high-collar tactical coat",
			BasePositivePrompt:    "masterpiece, best quality, cinematic portrait of kaelen7, cybernetic obsidian visor, silver undercut hair, high-collar tactical coat, volumetric neon lighting, <lora:kaelen_v3:0.85>",
			BaseNegativePrompt:    "lowres, blurry, bad anatomy, extra fingers, watermark, oversaturated",
			RecommendedCheckpoint: "juggernautXL_v9 / flux1-dev-fp8",
			LoRAs:                 []string{"kaelen_v3", "anamorphe_lens"},
		},
		"Lyra Vance (Astrolabe Chronicler)": {
			Name:                  "Lyra Vance (Astrolabe Chronicler)",
			Archetype:             "Astral Scholar",
			Description:           "Keeper of celestial cartography and luminescent glass codices. Warm amber rim lighting and ornate brass instruments.",
			TriggerWords:          "lyra_vance, astral scholar robes, brass astrolabe halo, amber eyes",
			BasePositivePrompt:    "editorial fantasy illustration, lyra_vance, astral scholar robes, brass astrolabe halo, floating starlight motes, golden hour bounce light, <lora:lyra_astral_v2:0.9>",
			BaseNegativePrompt:    "ugly, flat shading, text, watermark, deformed hands",
			RecommendedCheckpoint: "animagineXL_v31 / sdxl_albedoBase_v21",
			LoRAs:                 []string{"lyra_astral_v2"},
		},
		"Unit-09 Solaris (Orbital Frame)": {
			Name:                  "Unit-09 Solaris (Orbital Frame)",
			Archetype:             "Heavy Mecha Subject",
			Description:           "Hard-surface orbital reconnaissance frame with dual optic sensor array and weathered ceramic armor plating.",
			TriggerWords:          "unit09_solaris, heavy orbital mecha frame, orange optic array, weathered white ceramic armor",
			BasePositivePrompt:    "architectural concept art, unit09_solaris, heavy orbital mecha frame, orange optic array, dramatic volumetric lighting, <lora:hard_surface_mech:0.75>",
			BaseNegativePrompt:    "blurry, cartoon, low detail, asymmetrical optics",
			RecommendedCheckpoint: "realvisXL_v40 / flux1-schnell-fp8",
			LoRAs:                 []string{"hard_surface_mech"},
		},
	}

	charIDs := make(map[string]int64)
	for name, cdef := range charDefs {
		cid, err := repo.CreateCharacter(ctx, cdef)
		if err == nil {
			charIDs[name] = cid
		}
	}

	page, err := repo.QueryCatalog(ctx, domain.CatalogFilter{PerPage: 50})
	if err != nil {
		return err
	}

	for _, asset := range page.Assets {
		for _, sp := range specs {
			if asset.FileName == sp.FileName {
				asset.ViewCount = sp.Views
				asset.Rating = sp.Rating
				_, _ = repo.UpsertAsset(ctx, &asset)

				for _, tg := range sp.Tags {
					if tagObj, err := repo.UpsertTag(ctx, tg.Name, tg.Facet, "#2563eb"); err == nil {
						_ = repo.AttachTagToAsset(ctx, asset.ID, tagObj.ID)
					}
				}

				if cid, ok := charIDs[sp.CharacterName]; ok {
					_ = repo.AttachAssetToCharacter(ctx, cid, asset.ID)
					cdef := charDefs[sp.CharacterName]
					if cdef.AvatarAssetID == nil {
						idCopy := asset.ID
						cdef.AvatarAssetID = &idCopy
						_, _ = repo.CreateCharacter(ctx, cdef)
					}

					// Create a cropped focus/face thumbnail in .cache/crops for the optional Face Explorer view
					cropName := "crop_" + sp.FileName
					cropAbs := filepath.Join(cropsDir, cropName)
					cropImg := renderSyntheticArtwork(320, 320, sp.PalettePrimary, sp.PaletteAccent, sp.PatternMode+10)
					if f, err := os.Create(cropAbs); err == nil {
						_ = png.Encode(f, cropImg)
						_ = f.Close()
					}
					cidCopy := cid
					_, _ = repo.SaveDetectedFace(ctx, &domain.DetectedFace{
						AssetID:       asset.ID,
						Box:           domain.BoundingBox{X: 110, Y: 70, Width: 260, Height: 260},
						CropRelPath:   "crops/" + cropName,
						Confidence:    0.97,
						ClusterLabel:  sp.ClusterLabel,
						ExpressionTag: sp.ExpressionTag,
						CharacterID:   &cidCopy,
					})
				}
			}
		}
	}

	return nil
}

func renderSyntheticArtwork(w, h int, primary, accent color.RGBA, mode int) *image.RGBA {
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	cx, cy := float64(w)/2.0, float64(h)*0.42

	for y := 0; y < h; y++ {
		fy := float64(y) / float64(h)
		for x := 0; x < w; x++ {
			fx := float64(x) / float64(w)

			// Base dark gradient
			r := float64(primary.R)*(1.0-fy*0.5) + float64(accent.R)*(fy*0.18)
			g := float64(primary.G)*(1.0-fy*0.5) + float64(accent.G)*(fy*0.18)
			b := float64(primary.B)*(1.0-fy*0.5) + float64(accent.B)*(fy*0.18)

			// Geometric focal subject (halo / visor / emblem)
			dx := float64(x) - cx
			dy := float64(y) - cy
			dist := math.Sqrt(dx*dx + dy*dy)

			ringRadius := float64(w) * (0.24 + 0.03*float64(mode%3))
			if math.Abs(dist-ringRadius) < 4.5 {
				r = float64(accent.R)
				g = float64(accent.G)
				b = float64(accent.B)
			} else if dist < ringRadius*0.78 {
				glow := 1.0 - (dist / (ringRadius * 0.78))
				r += float64(accent.R) * glow * 0.55
				g += float64(accent.G) * glow * 0.55
				b += float64(accent.B) * glow * 0.55

				// Horizontal visor / optic band
				if math.Abs(dy) < 16 && math.Abs(dx) < ringRadius*0.52 {
					r = math.Min(255, float64(accent.R)*1.25)
					g = math.Min(255, float64(accent.G)*1.25)
					b = math.Min(255, float64(accent.B)*1.25)
				}
			}

			// Subtle grid lines
			if x%32 == 0 || y%32 == 0 {
				r += 9
				g += 12
				b += 18
			}
			_ = fx

			img.SetRGBA(x, y, color.RGBA{
				R: clampByte(r),
				G: clampByte(g),
				B: clampByte(b),
				A: 255,
			})
		}
	}
	return img
}

func clampByte(v float64) uint8 {
	if v < 0 {
		return 0
	}
	if v > 255 {
		return 255
	}
	return uint8(v)
}

func writePNGWithTEXTChunk(path string, img image.Image, key, value string) error {
	var rawPNG bytes.Buffer
	if err := png.Encode(&rawPNG, img); err != nil {
		return err
	}
	data := rawPNG.Bytes()
	// Insert tEXt chunk right before IEND (last 12 bytes of a standard PNG)
	if len(data) < 20 {
		return os.WriteFile(path, data, 0644)
	}
	iendPos := len(data) - 12

	var textPayload bytes.Buffer
	textPayload.WriteString(key)
	textPayload.WriteByte(0)
	textPayload.WriteString(value)
	payloadBytes := textPayload.Bytes()

	var chunkBuf bytes.Buffer
	_ = binary.Write(&chunkBuf, binary.BigEndian, uint32(len(payloadBytes)))
	chunkBuf.WriteString("tEXt")
	chunkBuf.Write(payloadBytes)

	crc := crc32.NewIEEE()
	crc.Write([]byte("tEXt"))
	crc.Write(payloadBytes)
	_ = binary.Write(&chunkBuf, binary.BigEndian, crc.Sum32())

	var out bytes.Buffer
	out.Write(data[:iendPos])
	out.Write(chunkBuf.Bytes())
	out.Write(data[iendPos:])

	return os.WriteFile(path, out.Bytes(), 0644)
}
