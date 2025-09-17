package middleware

import (
	"crypto/sha256"
	"fmt"
	"net/http"
	"strings"

	"github.com/gin-gonic/gin"
	"github.com/golang-jwt/jwt/v5"
	"github.com/resonancelab/psizero/shared/types"
)

// AuthMiddleware validates API keys and JWT tokens
func AuthMiddleware(jwtSecret string) gin.HandlerFunc {
	return func(c *gin.Context) {
		requestID := types.NewRequestID()
		c.Set("request_id", requestID)

		// Check for API key first
		apiKey := c.GetHeader("X-API-Key")
		if apiKey != "" {
			// TODO: Validate API key against database
			if validateAPIKey(apiKey) {
				c.Set("auth_type", "api_key")
				c.Set("user_id", extractUserIDFromAPIKey(apiKey))
				c.Next()
				return
			}
		}

		// Check for JWT token
		authHeader := c.GetHeader("Authorization")
		if authHeader == "" {
			c.JSON(http.StatusUnauthorized, types.NewAPIError(
				"AUTH_001",
				"Missing authentication",
				"Provide either X-API-Key header or Authorization bearer token",
				requestID,
			))
			c.Abort()
			return
		}

		tokenString := strings.TrimPrefix(authHeader, "Bearer ")
		if tokenString == authHeader {
			c.JSON(http.StatusUnauthorized, types.NewAPIError(
				"AUTH_002",
				"Invalid authorization format",
				"Authorization header must be in format 'Bearer <token>'",
				requestID,
			))
			c.Abort()
			return
		}

		token, err := jwt.Parse(tokenString, func(token *jwt.Token) (interface{}, error) {
			return []byte(jwtSecret), nil
		})

		if err != nil || !token.Valid {
			c.JSON(http.StatusUnauthorized, types.NewAPIError(
				"AUTH_003",
				"Invalid token",
				err.Error(),
				requestID,
			))
			c.Abort()
			return
		}

		if claims, ok := token.Claims.(jwt.MapClaims); ok {
			c.Set("auth_type", "jwt")
			c.Set("user_id", claims["user_id"])
			c.Set("scopes", claims["scopes"])
		}

		c.Next()
	}
}

// validateAPIKey validates an API key against the database
func validateAPIKey(apiKey string) bool {
	// Implement proper API key validation
	// This should hash the API key and check against the database
	// For production, connect to Supabase to validate the hashed key

	// Basic format validation
	if !strings.HasPrefix(apiKey, "ak_") || len(apiKey) < 10 {
		return false
	}

	// TODO: Hash the API key using SHA-256 and validate against Supabase
	// hashedKey := hashAPIKey(apiKey)
	// return checkAPIKeyInDatabase(hashedKey)

	// SECURITY NOTE: This is still a placeholder - requires database connection
	// Only keys starting with "ak_" are accepted for now
	return strings.HasPrefix(apiKey, "ak_") && len(apiKey) >= 32
}

// extractUserIDFromAPIKey extracts user ID from API key
func extractUserIDFromAPIKey(apiKey string) string {
	// TODO: Query database to get actual user ID for the API key
	// This should look up the user_id from the api_keys table after validation

	// SECURITY NOTE: This is still a placeholder - requires database connection
	// Return a deterministic but fake user ID for now
	return "user_placeholder_" + apiKey[3:min(8, len(apiKey))]
}

// hashAPIKey creates a SHA-256 hash of the API key for database storage/lookup
func hashAPIKey(apiKey string) string {
	// This should match the frontend implementation in apiKeyUtils.ts
	hash := sha256.Sum256([]byte(apiKey))
	return fmt.Sprintf("%x", hash)
}

// min returns the smaller of two integers
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// RateLimitMiddleware implements rate limiting
func RateLimitMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		// TODO: Implement rate limiting using Redis
		// For now, just pass through
		c.Next()
	}
}

// CORSMiddleware handles CORS headers
func CORSMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		c.Header("Access-Control-Allow-Origin", "*")
		c.Header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
		c.Header("Access-Control-Allow-Headers", "Origin, Content-Type, Authorization, X-API-Key")

		if c.Request.Method == "OPTIONS" {
			c.AbortWithStatus(http.StatusNoContent)
			return
		}

		c.Next()
	}
}
