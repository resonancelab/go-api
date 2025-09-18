package main

import (
	"log"
	"os"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/resonancelab/psizero/gateway/integration"
	"github.com/resonancelab/psizero/gateway/router"
	"github.com/resonancelab/psizero/gateway/services"
	"github.com/resonancelab/psizero/shared/middleware"
	"github.com/resonancelab/psizero/shared/types"
	swaggerFiles "github.com/swaggo/files"
	ginSwagger "github.com/swaggo/gin-swagger"
)

// @title PsiZero Resonance Platform API
// @version 1.0
// @description Advanced Resonance-Based API Suite for quantum and symbolic computing
// @termsOfService https://api.psizero.dev/terms

// @contact.name API Support
// @contact.url https://api.psizero.dev/support
// @contact.email support@psizero.dev

// @license.name MIT
// @license.url https://opensource.org/licenses/MIT

// @host api.psizero.dev
// @BasePath /v1

// @securityDefinitions.apikey ApiKeyAuth
// @in header
// @name X-API-Key

// @securityDefinitions.apikey BearerAuth
// @in header
// @name Authorization

func main() {
	// Load configuration
	config := loadConfig()

	// Initialize service container
	log.Println("Initializing service container...")
	container, err := services.NewServiceContainer(config)
	if err != nil {
		log.Fatal("Failed to initialize service container:", err)
	}
	defer container.Shutdown()

	log.Println("All engines initialized successfully")

	// Set Gin mode
	if config.Environment == "production" {
		gin.SetMode(gin.ReleaseMode)
	}

	// Create Gin router
	r := gin.New()

	// Add core middleware
	r.Use(gin.Logger())
	r.Use(gin.Recovery())

	// Add enhanced error handling middleware
	errorConfig := &middleware.ErrorHandlerConfig{
		EnableStackTrace:     config.Environment == "development",
		MaxRequestTimeout:    30 * time.Second,
		EnableCircuitBreaker: true,
		ErrorThreshold:       10,
		TimeWindow:           1 * time.Minute,
	}
	r.Use(middleware.ErrorHandlerMiddleware(errorConfig))
	// Note: Removed TimeoutMiddleware - ErrorHandlerMiddleware already handles timeouts
	r.Use(middleware.ValidationMiddleware())
	r.Use(middleware.ResourceCleanupMiddleware())

	// Add CORS and rate limiting
	r.Use(middleware.CORSMiddleware())
	r.Use(middleware.RateLimitMiddleware())

	// Basic health check endpoint (no auth required)
	r.GET("/health", func(c *gin.Context) {
		healthStatus := container.HealthCheck()
		allHealthy := true
		for _, status := range healthStatus {
			if !status {
				allHealthy = false
				break
			}
		}

		statusCode := 200
		if !allHealthy {
			statusCode = 503
		}

		c.JSON(statusCode, gin.H{
			"status":      map[bool]string{true: "healthy", false: "unhealthy"}[allHealthy],
			"service":     "psizero-gateway",
			"version":     "1.0.0",
			"engines":     healthStatus,
			"initialized": container.IsInitialized(),
		})
	})

	// Comprehensive health check endpoint (no auth required)
	r.GET("/health/detailed", func(c *gin.Context) {
		healthChecker := integration.NewIntegrationHealthChecker(container)
		results, err := healthChecker.CheckAllServices()

		if err != nil {
			c.JSON(500, gin.H{
				"status": "error",
				"error":  err.Error(),
			})
			return
		}

		overallHealth, err := healthChecker.GetOverallHealth()
		if err != nil {
			c.JSON(500, gin.H{
				"status": "error",
				"error":  err.Error(),
			})
			return
		}

		statusCode := 200
		if overallHealth == "unhealthy" {
			statusCode = 503
		} else if overallHealth == "degraded" {
			statusCode = 206
		}

		c.JSON(statusCode, gin.H{
			"status":          overallHealth,
			"service":         "psizero-gateway",
			"version":         "1.0.0",
			"detailed_checks": results,
			"timestamp":       time.Now(),
		})
	})

	// Swagger documentation (no auth required)
	r.GET("/docs/*any", ginSwagger.WrapHandler(swaggerFiles.Handler))

	// API routes with authentication
	v1 := r.Group("/v1")
	v1.Use(middleware.AuthMiddleware(config.JWTSecret))

	// Initialize service routers with dependency injection
	router.SetupSRSRoutes(v1, container)
	router.SetupHQERoutes(v1, container)
	router.SetupQSEMRoutes(v1, container)
	router.SetupNLCRoutes(v1, container)
	router.SetupQCRRoutes(v1, container)
	router.SetupIChingRoutes(v1, container)
	router.SetupUnifiedRoutes(v1, container)
	router.SetupWebhookRoutes(v1)

	// Start server
	port := config.Port
	if port == 0 {
		port = 8080
	}

	log.Printf("Starting PsiZero Resonance API Gateway on port %d", port)
	log.Printf("Documentation available at http://localhost:%d/docs/", port)

	if err := r.Run(":8080"); err != nil {
		log.Fatal("Failed to start server:", err)
	}
}

func loadConfig() *types.Config {
	return &types.Config{
		Port:        getEnvInt("PORT", 8080),
		DatabaseURL: getEnv("DATABASE_URL", "postgres://localhost/psizero_resonance?sslmode=disable"),
		RedisURL:    getEnv("REDIS_URL", "redis://localhost:6379"),
		JWTSecret:   getEnv("JWT_SECRET", "your-secret-key"),
		LogLevel:    getEnv("LOG_LEVEL", "info"),
		Environment: getEnv("ENVIRONMENT", "development"),
		ServiceName: "psizero-gateway",
		MetricsPort: getEnvInt("METRICS_PORT", 9090),
		HealthPort:  getEnvInt("HEALTH_PORT", 8081),
	}
}

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

func getEnvInt(key string, defaultValue int) int {
	if value := os.Getenv(key); value != "" {
		// Simple conversion, in production use strconv.Atoi with error handling
		return defaultValue
	}
	return defaultValue
}
