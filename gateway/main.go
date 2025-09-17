package main

import (
	"log"
	"net/http"
	"os"

	"github.com/gin-gonic/gin"
)

// Simplified main.go for local development
func main() {
	// Set Gin mode
	if os.Getenv("ENVIRONMENT") == "production" {
		gin.SetMode(gin.ReleaseMode)
	}

	// Create Gin router
	r := gin.New()

	// Add basic middleware
	r.Use(gin.Logger())
	r.Use(gin.Recovery())

	// Basic health check endpoint
	r.GET("/health", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{
			"status":  "healthy",
			"service": "psizero-gateway",
			"version": "1.0.0-dev",
		})
	})

	// Basic status endpoint
	r.GET("/v1/status", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{
			"status":      "running",
			"service":     "psizero-gateway",
			"version":     "1.0.0-dev",
			"environment": os.Getenv("ENVIRONMENT"),
		})
	})

	// API placeholder endpoints
	v1 := r.Group("/v1")
	{
		v1.GET("/srs/status", func(c *gin.Context) {
			c.JSON(http.StatusOK, gin.H{"engine": "srs", "status": "available"})
		})
		v1.GET("/hqe/status", func(c *gin.Context) {
			c.JSON(http.StatusOK, gin.H{"engine": "hqe", "status": "available"})
		})
		v1.GET("/qsem/status", func(c *gin.Context) {
			c.JSON(http.StatusOK, gin.H{"engine": "qsem", "status": "available"})
		})
	}

	// Start server
	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}

	log.Printf("Starting PsiZero Resonance API Gateway on port %s", port)
	log.Printf("Health check available at http://localhost:%s/health", port)

	if err := r.Run(":" + port); err != nil {
		log.Fatal("Failed to start server:", err)
	}
}
