package middleware

import (
	"crypto/sha256"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"time"

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
	// Basic format validation
	if !strings.HasPrefix(apiKey, "ak_") || len(apiKey) < 32 {
		return false
	}

	// Hash the API key for database lookup
	hashedKey := hashAPIKey(apiKey)

	// Check against database (placeholder implementation)
	// In production, this would query Supabase or another database
	return checkAPIKeyInDatabase(hashedKey)
}

// extractUserIDFromAPIKey extracts user ID from API key
func extractUserIDFromAPIKey(apiKey string) string {
	// Hash the API key for database lookup
	hashedKey := hashAPIKey(apiKey)

	// Query database to get user ID
	// TODO: Implement database connection to Supabase
	// This should query the api_keys table to get the user_id associated with the hashed key

	// Placeholder implementation - in production this would:
	// 1. Connect to Supabase database
	// 2. Query api_keys table for the hashed key
	// 3. Return the associated user_id
	// 4. Return empty string if not found

	// For now, return a deterministic user ID based on the hash
	// This is NOT secure and should be replaced with proper database lookup
	if len(hashedKey) >= 8 {
		return "user_" + hashedKey[:8]
	}
	return "user_unknown"
}

// hashAPIKey creates a SHA-256 hash of the API key for database storage/lookup
func hashAPIKey(apiKey string) string {
	// This should match the frontend implementation in apiKeyUtils.ts
	hash := sha256.Sum256([]byte(apiKey))
	return fmt.Sprintf("%x", hash)
}

// checkAPIKeyInDatabase validates the hashed API key against the database
func checkAPIKeyInDatabase(hashedKey string) bool {
	// TODO: Implement database connection to Supabase
	// This should query the api_keys table to check if the hashed key exists and is active

	// Placeholder implementation - in production this would:
	// 1. Connect to Supabase database
	// 2. Query api_keys table for the hashed key
	// 3. Check if the key is active and not expired
	// 4. Return true if valid, false otherwise

	// For now, accept any properly formatted hashed key
	// This is NOT secure and should be replaced with proper database validation
	return len(hashedKey) == 64 // SHA-256 produces 64 character hex string
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
	// Simple in-memory rate limiter (for production, use Redis)
	limiter := NewInMemoryRateLimiter()

	return func(c *gin.Context) {
		// Get client identifier (IP address or user ID)
		clientID := c.ClientIP()
		if userID, exists := c.Get("user_id"); exists {
			clientID = userID.(string)
		}

		// Check rate limit
		if !limiter.Allow(clientID) {
			requestID := c.GetString("request_id")
			c.JSON(http.StatusTooManyRequests, types.NewAPIError(
				"RATE_001",
				"Rate limit exceeded",
				"Too many requests. Please try again later.",
				requestID,
			))
			c.Abort()
			return
		}

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

// InMemoryRateLimiter implements a simple in-memory rate limiter
type InMemoryRateLimiter struct {
	clients map[string]*ClientLimiter
	mu      sync.RWMutex
}

// ClientLimiter tracks requests for a specific client
type ClientLimiter struct {
	requests    []time.Time
	lastCleanup time.Time
}

// NewInMemoryRateLimiter creates a new in-memory rate limiter
func NewInMemoryRateLimiter() *InMemoryRateLimiter {
	return &InMemoryRateLimiter{
		clients: make(map[string]*ClientLimiter),
	}
}

// Allow checks if a request from the client should be allowed
func (rl *InMemoryRateLimiter) Allow(clientID string) bool {
	rl.mu.Lock()
	defer rl.mu.Unlock()

	now := time.Now()
	client, exists := rl.clients[clientID]

	if !exists {
		client = &ClientLimiter{
			requests:    make([]time.Time, 0),
			lastCleanup: now,
		}
		rl.clients[clientID] = client
	}

	// Clean up old requests (older than 1 minute)
	rl.cleanupOldRequests(client, now)

	// Check rate limit: 100 requests per minute
	if len(client.requests) >= 100 {
		return false
	}

	// Add current request
	client.requests = append(client.requests, now)
	return true
}

// cleanupOldRequests removes requests older than 1 minute
func (rl *InMemoryRateLimiter) cleanupOldRequests(client *ClientLimiter, now time.Time) {
	cutoff := now.Add(-time.Minute)

	// Find first request that's still valid
	validStart := 0
	for i, reqTime := range client.requests {
		if reqTime.After(cutoff) {
			validStart = i
			break
		}
	}

	// Keep only valid requests
	if validStart > 0 {
		client.requests = client.requests[validStart:]
	}

	client.lastCleanup = now
}
