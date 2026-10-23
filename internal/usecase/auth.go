package usecase

import (
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"errors"

	"github.com/jonathanhecl/diffusionCatalog/internal/domain"
)

// AuthService manages administrator authentication and default account seeding.
type AuthService struct {
	repo   domain.CatalogRepository
	secret []byte
}

// NewAuthService creates a new AuthService and ensures the default administrator exists.
func NewAuthService(ctx context.Context, repo domain.CatalogRepository, defaultUser, defaultPass string) (*AuthService, error) {
	s := &AuthService{
		repo:   repo,
		secret: []byte("diffusion-catalog-local-secret-" + defaultUser),
	}
	hash := s.hashPassword(defaultPass)
	if err := repo.EnsureDefaultAdmin(ctx, defaultUser, hash); err != nil {
		return nil, err
	}
	return s, nil
}

// Authenticate verifies username and password, returning a signed session token.
func (s *AuthService) Authenticate(ctx context.Context, username, password string) (string, error) {
	u, err := s.repo.GetUserByUsername(ctx, username)
	if err != nil {
		return "", errors.New("invalid credentials")
	}
	if u.PasswordHash != s.hashPassword(password) {
		return "", errors.New("invalid credentials")
	}
	return s.SignToken(u.Username), nil
}

// SignToken creates an HMAC-signed token for cookie session validation.
func (s *AuthService) SignToken(username string) string {
	mac := hmac.New(sha256.New, s.secret)
	mac.Write([]byte(username))
	sig := hex.EncodeToString(mac.Sum(nil))
	return username + ":" + sig
}

// VerifyToken checks if a cookie token is valid and returns the authenticated username.
func (s *AuthService) VerifyToken(token string) (string, bool) {
	for i := 0; i < len(token); i++ {
		if token[i] == ':' {
			user := token[:i]
			if s.SignToken(user) == token {
				return user, true
			}
			return "", false
		}
	}
	return "", false
}

func (s *AuthService) hashPassword(raw string) string {
	sum := sha256.Sum256([]byte("dc-salt:" + raw))
	return hex.EncodeToString(sum[:])
}
