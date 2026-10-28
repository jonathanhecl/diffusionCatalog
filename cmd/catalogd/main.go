package main

import (
	"context"
	"errors"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/jonathanhecl/diffusionCatalog/internal/adapter/metadata"
	"github.com/jonathanhecl/diffusionCatalog/internal/adapter/ollama"
	"github.com/jonathanhecl/diffusionCatalog/internal/adapter/storage/sqlite"
	"github.com/jonathanhecl/diffusionCatalog/internal/adapter/vision"
	"github.com/jonathanhecl/diffusionCatalog/internal/adapter/web"
	"github.com/jonathanhecl/diffusionCatalog/internal/bootstrap"
	"github.com/jonathanhecl/diffusionCatalog/internal/config"
	"github.com/jonathanhecl/diffusionCatalog/internal/domain"
	"github.com/jonathanhecl/diffusionCatalog/internal/usecase"
)

func main() {
	cfg := config.Load()

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	bootRes := bootstrap.EnsureEnvironment(ctx, cfg.MediaDir, cfg.CacheDir, cfg.ForceSetup, cfg.NonInteractive)
	if bootRes.Cleanup != nil {
		defer bootRes.Cleanup()
	}
	if cfg.VisionWorkerURL == "" && bootRes.VisionWorkerURL != "" {
		cfg.VisionWorkerURL = bootRes.VisionWorkerURL
	}

	db, err := sqlite.Open(ctx, cfg.DBPath)
	if err != nil {
		log.Fatalf("failed to initialize sqlite database: %v", err)
	}
	defer db.Close()

	repo := sqlite.NewRepository(db, cfg.DBPath)

	var visionEngine domain.VisionEngine
	if cfg.VisionWorkerURL != "" {
		visionEngine = vision.NewHTTPWorkerEngine(cfg.VisionWorkerURL)
	} else {
		visionEngine = vision.NewNoopEngine()
	}

	ollamaClient := ollama.NewClient()
	catalogService := usecase.NewCatalogService(cfg.MediaDir, repo, ollamaClient)
	extractor := metadata.NewExtractor()
	scannerService := usecase.NewScannerService(cfg.MediaDir, cfg.CacheDir, repo, extractor, visionEngine)
	scannerService.SetOllamaProvider(ollamaClient)

	authService, err := usecase.NewAuthService(ctx, repo, cfg.AdminUser, cfg.AdminPass)
	if err != nil {
		log.Fatalf("failed to initialize auth service: %v", err)
	}

	if cfg.SeedDemoIfEmpty {
		if err := usecase.SeedDemoFolderIfEmpty(ctx, cfg.MediaDir, cfg.CacheDir, repo, scannerService); err != nil {
			log.Printf("warning: demo seed error: %v", err)
		}
	}

	if cfg.AutoScanOnBoot {
		go func() {
			// Initial scan on startup
			if _, err := scannerService.Scan(ctx); err != nil && !errors.Is(err, context.Canceled) {
				log.Printf("warning: initial background scan error: %v", err)
			}

			if cfg.AutoScanInterval <= 0 {
				return
			}

			log.Printf("[SCAN] Background media folder auto-sync active (interval: %v)", cfg.AutoScanInterval)
			ticker := time.NewTicker(cfg.AutoScanInterval)
			defer ticker.Stop()

			for {
				select {
				case <-ctx.Done():
					return
				case <-ticker.C:
					if !scannerService.IsScanning() {
						if _, err := scannerService.Scan(ctx); err != nil && !errors.Is(err, context.Canceled) {
							log.Printf("warning: periodic background scan error: %v", err)
						}
					}
				}
			}
		}()
	}

	server, err := web.NewServer(cfg.MediaDir, cfg.CacheDir, visionEngine.Name(), catalogService, scannerService, authService)
	if err != nil {
		log.Fatalf("failed to initialize web server: %v", err)
	}

	httpServer := &http.Server{
		Addr:    cfg.Addr,
		Handler: server.Handler(),
	}

	go func() {
		log.Printf("DiffusionCatalog server running on http://%s (config: %s, media: %s, db: %s)", cfg.Addr, cfg.ConfigFile, cfg.MediaDir, cfg.DBPath)
		if err := httpServer.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			if strings.Contains(err.Error(), "permission denied") {
				log.Printf("\n[!] Error: Unable to bind to %s (permission denied).\n"+
					"    Note: On Linux, ports below 1024 require root privileges or CAP_NET_BIND_SERVICE.\n"+
					"    To resolve this:\n"+
					"      1) Run with sudo: sudo ./catalogd\n"+
					"      2) Grant capability: sudo setcap 'cap_net_bind_service=+ep' ./catalogd\n"+
					"      3) Change port in %s (e.g. \"port\": 1080 or \"addr\": \":1080\")\n"+
					"      4) Or pass a custom port flag: ./catalogd --port=1080\n\n", cfg.Addr, cfg.ConfigFile)
			}
			log.Fatalf("http server error: %v", err)
		}
	}()

	<-ctx.Done()
	log.Println("Shutting down DiffusionCatalog...")

	shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if err := httpServer.Shutdown(shutdownCtx); err != nil {
		log.Printf("http shutdown error: %v", err)
	}
	log.Println("DiffusionCatalog stopped cleanly.")
}
