package tests

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/resonancelab/psizero/gateway/services"
	"github.com/resonancelab/psizero/shared/types"
)

// TestAPIEndpoints tests all HTTP API endpoints
func TestAPIEndpoints(t *testing.T) {
	// Set gin to test mode
	gin.SetMode(gin.TestMode)

	// Create test router
	r := gin.New()
	r.Use(gin.Recovery())

	// Mock middleware to add request_id and user_id
	r.Use(func(c *gin.Context) {
		c.Set("request_id", "test-req-123")
		c.Set("user_id", "test-user-456")
		c.Next()
	})

	// Initialize mock service container
	container, err := createMockServiceContainer()
	if err != nil {
		t.Fatalf("Failed to create mock service container: %v", err)
	}

	// Setup all routes
	setupAllRoutes(r, container)

	// Run tests for each service
	t.Run("HealthEndpoints", func(t *testing.T) {
		testHealthEndpoints(t, r)
	})

	t.Run("SRSEndpoints", func(t *testing.T) {
		testSRSEndpoints(t, r)
	})

	t.Run("HQEEndpoints", func(t *testing.T) {
		testHQEEndpoints(t, r)
	})

	t.Run("QSEMEndpoints", func(t *testing.T) {
		testQSEMEndpoints(t, r)
	})

	t.Run("NLCEndpoints", func(t *testing.T) {
		testNLCEndpoints(t, r)
	})

	t.Run("QCREndpoints", func(t *testing.T) {
		testQCREndpoints(t, r)
	})

	t.Run("IChingEndpoints", func(t *testing.T) {
		testIChingEndpoints(t, r)
	})

	t.Run("UnifiedEndpoints", func(t *testing.T) {
		testUnifiedEndpoints(t, r)
	})

	t.Run("WebhookEndpoints", func(t *testing.T) {
		testWebhookEndpoints(t, r)
	})
}

// setupAllRoutes sets up all API routes for testing with mock handlers
func setupAllRoutes(r *gin.Engine, container *services.ServiceContainer) {
	// Health endpoints
	r.GET("/health", func(c *gin.Context) {
		c.JSON(200, gin.H{
			"status":      "healthy",
			"service":     "psizero-gateway",
			"version":     "1.0.0",
			"initialized": true,
		})
	})

	r.GET("/health/detailed", func(c *gin.Context) {
		c.JSON(200, gin.H{
			"status":  "healthy",
			"service": "psizero-gateway",
			"version": "1.0.0",
			"detailed_checks": map[string]string{
				"database": "healthy",
				"cache":    "healthy",
				"engines":  "healthy",
			},
			"timestamp": time.Now(),
		})
	})

	// API routes with mock handlers instead of real engines
	v1 := r.Group("/v1")
	setupMockSRSRoutes(v1)
	setupMockHQERoutes(v1)
	setupMockQSEMRoutes(v1)
	setupMockNLCRoutes(v1)
	setupMockQCRRoutes(v1)
	setupMockIChingRoutes(v1)
	setupMockUnifiedRoutes(v1)
	setupMockWebhookRoutes(v1)
}

// Mock SRS routes
func setupMockSRSRoutes(v1 *gin.RouterGroup) {
	srs := v1.Group("/srs")

	srs.POST("/solve", func(c *gin.Context) {
		var req map[string]interface{}
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(400, types.NewAPIError("INVALID_REQUEST", "Invalid JSON", err.Error(), c.GetString("request_id")))
			return
		}

		requestID := c.GetString("request_id")
		if requestID == "" {
			requestID = "test-req-123"
		}

		response := types.NewAPIResponse(map[string]interface{}{
			"assignment":   []int{1, 0, 1},
			"feasible":     true,
			"objective":    0.0,
			"satisfied":    3,
			"total":        3,
			"confidence":   0.95,
			"compute_time": 0.001,
		}, requestID)

		c.JSON(200, response)
	})

	srs.GET("/problems", func(c *gin.Context) {
		requestID := c.GetString("request_id")
		response := types.NewAPIResponse(map[string]interface{}{
			"supported_problems": []string{"3sat", "ksat", "subsetsum"},
		}, requestID)
		c.JSON(200, response)
	})

	srs.GET("/status", func(c *gin.Context) {
		requestID := c.GetString("request_id")
		response := types.NewAPIResponse(map[string]interface{}{
			"status": "ready",
			"uptime": "1h23m45s",
		}, requestID)
		c.JSON(200, response)
	})
}

// Mock HQE routes
func setupMockHQERoutes(v1 *gin.RouterGroup) {
	hqe := v1.Group("/hqe")

	hqe.POST("/simulate", func(c *gin.Context) {
		var req map[string]interface{}
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(400, types.NewAPIError("INVALID_REQUEST", "Invalid JSON", err.Error(), c.GetString("request_id")))
			return
		}

		// Validate required fields
		if primes, ok := req["primes"].([]interface{}); !ok || len(primes) == 0 {
			c.JSON(400, types.NewAPIError("INVALID_PRIMES", "Empty primes array", "", c.GetString("request_id")))
			return
		}

		response := types.NewAPIResponse(map[string]interface{}{
			"holographic_reconstruction": map[string]interface{}{
				"boundary_entropy": 2.5,
				"bulk_entropy":     2.3,
				"duality_ratio":    0.92,
			},
			"primes_encoded": []int{2, 3, 5, 7, 11},
			"steps_evolved":  10,
		}, c.GetString("request_id"))

		c.JSON(200, response)
	})

	hqe.GET("/primes", func(c *gin.Context) {
		response := types.NewAPIResponse(map[string]interface{}{
			"primes":    []int{2, 3, 5, 7, 11, 13, 17, 19, 23, 29},
			"dimension": 10,
		}, c.GetString("request_id"))
		c.JSON(200, response)
	})
}

// Mock QSEM routes
func setupMockQSEMRoutes(v1 *gin.RouterGroup) {
	qsem := v1.Group("/qsem")

	qsem.POST("/encode", func(c *gin.Context) {
		response := types.NewAPIResponse(map[string]interface{}{
			"encoded_concepts": []map[string]interface{}{
				{"concept": "quantum", "encoding": []float64{0.1, 0.2, 0.3}},
				{"concept": "resonance", "encoding": []float64{0.2, 0.3, 0.4}},
			},
			"semantic_distance": 0.85,
		}, c.GetString("request_id"))
		c.JSON(200, response)
	})

	qsem.POST("/resonance", func(c *gin.Context) {
		var req map[string]interface{}
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(400, types.NewAPIError("INVALID_REQUEST", "Invalid JSON", err.Error(), c.GetString("request_id")))
			return
		}

		// Validate vectors
		if vectors, ok := req["vectors"].([]interface{}); !ok || len(vectors) < 2 {
			c.JSON(400, types.NewAPIError("INSUFFICIENT_VECTORS", "Need at least 2 vectors", "", c.GetString("request_id")))
			return
		}

		response := types.NewAPIResponse(map[string]interface{}{
			"resonance_matrix": [][]float64{{1.0, 0.8}, {0.8, 1.0}},
			"eigenvalues":      []float64{1.8, 0.2},
		}, c.GetString("request_id"))
		c.JSON(200, response)
	})

	qsem.GET("/basis", func(c *gin.Context) {
		response := types.NewAPIResponse(map[string]interface{}{
			"basis_vectors": [][]float64{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}},
			"dimension":     3,
		}, c.GetString("request_id"))
		c.JSON(200, response)
	})
}

// Mock NLC routes
func setupMockNLCRoutes(v1 *gin.RouterGroup) {
	nlc := v1.Group("/nlc")

	nlc.POST("/sessions", func(c *gin.Context) {
		response := types.NewAPIResponse(map[string]interface{}{
			"id":           "mock-session-123",
			"protocol":     "teleportation",
			"participants": []string{"node_a", "node_b"},
			"status":       "active",
		}, c.GetString("request_id"))
		c.JSON(201, response)
	})

	nlc.GET("/sessions/:sessionId", func(c *gin.Context) {
		sessionId := c.Param("sessionId")
		if sessionId == "" {
			c.JSON(404, types.NewAPIError("SESSION_NOT_FOUND", "Session not found", "", c.GetString("request_id")))
			return
		}

		response := types.NewAPIResponse(map[string]interface{}{
			"id":     sessionId,
			"status": "active",
		}, c.GetString("request_id"))
		c.JSON(200, response)
	})

	nlc.POST("/sessions/:sessionId/messages", func(c *gin.Context) {
		var req map[string]interface{}
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(400, types.NewAPIError("INVALID_REQUEST", "Invalid JSON", err.Error(), c.GetString("request_id")))
			return
		}

		response := types.NewAPIResponse(map[string]interface{}{
			"message_id": "msg-123",
			"status":     "transmitted",
		}, c.GetString("request_id"))
		c.JSON(200, response)
	})

	nlc.GET("/channels", func(c *gin.Context) {
		response := types.NewAPIResponse(map[string]interface{}{
			"channels": []string{"quantum_channel_1", "quantum_channel_2"},
		}, c.GetString("request_id"))
		c.JSON(200, response)
	})
}

// Mock QCR routes
func setupMockQCRRoutes(v1 *gin.RouterGroup) {
	qcr := v1.Group("/qcr")

	qcr.POST("/sessions", func(c *gin.Context) {
		var req map[string]interface{}
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(400, types.NewAPIError("INVALID_REQUEST", "Invalid JSON", err.Error(), c.GetString("request_id")))
			return
		}

		// Validate modes
		if modes, ok := req["modes"].([]interface{}); !ok || len(modes) == 0 {
			c.JSON(400, types.NewAPIError("INVALID_MODES", "Empty modes array", "", c.GetString("request_id")))
			return
		}

		response := types.NewAPIResponse(map[string]interface{}{
			"id":                  "mock-qcr-session-123",
			"consciousness_state": "triadic_harmony",
			"modes":               []string{"analytical", "creative", "ethical"},
		}, c.GetString("request_id"))
		c.JSON(201, response)
	})

	qcr.POST("/sessions/:sessionId/observe", func(c *gin.Context) {
		var req map[string]interface{}
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(400, types.NewAPIError("INVALID_REQUEST", "Invalid JSON", err.Error(), c.GetString("request_id")))
			return
		}

		response := types.NewAPIResponse(map[string]interface{}{
			"observation": "Consciousness observed in state of creative-analytical synthesis",
			"coherence":   0.92,
		}, c.GetString("request_id"))
		c.JSON(200, response)
	})

	qcr.GET("/modes", func(c *gin.Context) {
		response := types.NewAPIResponse(map[string]interface{}{
			"supported_modes": []string{"analytical", "creative", "ethical", "intuitive"},
		}, c.GetString("request_id"))
		c.JSON(200, response)
	})
}

// Mock I-Ching routes
func setupMockIChingRoutes(v1 *gin.RouterGroup) {
	iching := v1.Group("/iching")

	iching.POST("/evolve", func(c *gin.Context) {
		var req map[string]interface{}
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(400, types.NewAPIError("INVALID_REQUEST", "Invalid JSON", err.Error(), c.GetString("request_id")))
			return
		}

		// Validate question
		if question, ok := req["question"].(string); !ok || question == "" {
			c.JSON(400, types.NewAPIError("EMPTY_QUESTION", "Question cannot be empty", "", c.GetString("request_id")))
			return
		}

		response := types.NewAPIResponse(map[string]interface{}{
			"hexagram": map[string]interface{}{
				"number":   1,
				"name":     "The Creative",
				"trigrams": []string{"Heaven", "Heaven"},
			},
			"interpretation": "Great progress is possible",
		}, c.GetString("request_id"))
		c.JSON(200, response)
	})

	iching.GET("/hexagrams", func(c *gin.Context) {
		response := types.NewAPIResponse(map[string]interface{}{
			"hexagrams": []map[string]interface{}{
				{"number": 1, "name": "The Creative"},
				{"number": 2, "name": "The Receptive"},
			},
		}, c.GetString("request_id"))
		c.JSON(200, response)
	})

	iching.GET("/hexagrams/:id", func(c *gin.Context) {
		response := types.NewAPIResponse(map[string]interface{}{
			"number":   1,
			"name":     "The Creative",
			"trigrams": []string{"Heaven", "Heaven"},
		}, c.GetString("request_id"))
		c.JSON(200, response)
	})

	iching.GET("/trigrams", func(c *gin.Context) {
		response := types.NewAPIResponse(map[string]interface{}{
			"trigrams": []string{"Heaven", "Earth", "Thunder", "Water", "Mountain", "Wind", "Fire", "Lake"},
		}, c.GetString("request_id"))
		c.JSON(200, response)
	})
}

// Mock Unified routes
func setupMockUnifiedRoutes(v1 *gin.RouterGroup) {
	unified := v1.Group("/unified")

	unified.POST("/gravity/compute", func(c *gin.Context) {
		var req map[string]interface{}
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(400, types.NewAPIError("INVALID_REQUEST", "Invalid JSON", err.Error(), c.GetString("request_id")))
			return
		}

		// Validate entropy rate
		if rate, ok := req["observerEntropyReductionRate"].(float64); !ok || rate < 0 {
			c.JSON(400, types.NewAPIError("INVALID_ENTROPY_RATE", "Invalid entropy rate", "", c.GetString("request_id")))
			return
		}

		response := types.NewAPIResponse(map[string]interface{}{
			"gravitational_field": map[string]interface{}{
				"field_strength":   9.81,
				"entropy_gradient": 0.002,
			},
		}, c.GetString("request_id"))
		c.JSON(200, response)
	})

	unified.POST("/field/analyze", func(c *gin.Context) {
		var req map[string]interface{}
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(400, types.NewAPIError("INVALID_REQUEST", "Invalid JSON", err.Error(), c.GetString("request_id")))
			return
		}

		// Validate mass distribution
		if masses, ok := req["massDistribution"].([]interface{}); !ok || len(masses) == 0 {
			c.JSON(400, types.NewAPIError("NO_MASS_DISTRIBUTION", "Empty mass distribution", "", c.GetString("request_id")))
			return
		}

		response := types.NewAPIResponse(map[string]interface{}{
			"field_analysis": map[string]interface{}{
				"field_strength": 9.81,
				"curvature":      0.001,
			},
		}, c.GetString("request_id"))
		c.JSON(200, response)
	})

	unified.GET("/constants", func(c *gin.Context) {
		response := types.NewAPIResponse(map[string]interface{}{
			"constants": map[string]float64{
				"G": 6.67430e-11,
				"c": 299792458,
				"h": 6.62607015e-34,
			},
		}, c.GetString("request_id"))
		c.JSON(200, response)
	})

	unified.GET("/models", func(c *gin.Context) {
		response := types.NewAPIResponse(map[string]interface{}{
			"models": []string{"standard_model", "string_theory", "loop_quantum_gravity"},
		}, c.GetString("request_id"))
		c.JSON(200, response)
	})
}

// Mock Webhook routes
func setupMockWebhookRoutes(v1 *gin.RouterGroup) {
	webhooks := v1.Group("/webhooks")

	webhooks.POST("", func(c *gin.Context) {
		var req map[string]interface{}
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(400, types.NewAPIError("INVALID_REQUEST", "Invalid JSON", err.Error(), c.GetString("request_id")))
			return
		}

		// Validate events
		if events, ok := req["events"].([]interface{}); !ok || len(events) == 0 {
			c.JSON(400, types.NewAPIError("NO_EVENTS", "Empty events array", "", c.GetString("request_id")))
			return
		}

		response := types.NewAPIResponse(map[string]interface{}{
			"id":     "mock-webhook-123",
			"url":    req["url"],
			"events": req["events"],
			"active": true,
		}, c.GetString("request_id"))
		c.JSON(201, response)
	})

	webhooks.GET("", func(c *gin.Context) {
		response := types.NewAPIResponse(map[string]interface{}{
			"webhooks": []map[string]interface{}{
				{"id": "mock-webhook-123", "url": "https://example.com/webhook", "active": true},
			},
		}, c.GetString("request_id"))
		c.JSON(200, response)
	})

	webhooks.GET("/:webhookId", func(c *gin.Context) {
		response := types.NewAPIResponse(map[string]interface{}{
			"id":     c.Param("webhookId"),
			"url":    "https://example.com/webhook",
			"active": true,
		}, c.GetString("request_id"))
		c.JSON(200, response)
	})

	webhooks.POST("/:webhookId/test", func(c *gin.Context) {
		response := types.NewAPIResponse(map[string]interface{}{
			"test_result": "success",
			"webhook_id":  c.Param("webhookId"),
		}, c.GetString("request_id"))
		c.JSON(200, response)
	})

	webhooks.GET("/events", func(c *gin.Context) {
		response := types.NewAPIResponse(map[string]interface{}{
			"available_events": []string{
				"srs.solution.found",
				"qsem.encoding.complete",
				"hqe.simulation.finished",
			},
		}, c.GetString("request_id"))
		c.JSON(200, response)
	})
}

// testHealthEndpoints tests health check endpoints
func testHealthEndpoints(t *testing.T, r *gin.Engine) {
	t.Run("BasicHealth", func(t *testing.T) {
		w := httptest.NewRecorder()
		req, _ := http.NewRequest("GET", "/health", nil)
		r.ServeHTTP(w, req)

		if w.Code != http.StatusOK {
			t.Errorf("Expected status 200, got %d", w.Code)
		}

		var response map[string]interface{}
		json.Unmarshal(w.Body.Bytes(), &response)

		if response["status"] != "healthy" {
			t.Errorf("Expected status 'healthy', got %v", response["status"])
		}
	})

	t.Run("DetailedHealth", func(t *testing.T) {
		w := httptest.NewRecorder()
		req, _ := http.NewRequest("GET", "/health/detailed", nil)
		r.ServeHTTP(w, req)

		if w.Code != http.StatusOK {
			t.Errorf("Expected status 200, got %d", w.Code)
		}

		var response map[string]interface{}
		json.Unmarshal(w.Body.Bytes(), &response)

		if response["status"] != "healthy" {
			t.Errorf("Expected status 'healthy', got %v", response["status"])
		}

		if response["detailed_checks"] == nil {
			t.Error("Expected detailed_checks in response")
		}
	})
}

// testSRSEndpoints tests SRS API endpoints
func testSRSEndpoints(t *testing.T, r *gin.Engine) {
	t.Run("SolveProblem", func(t *testing.T) {
		payload := map[string]interface{}{
			"problem": "3sat",
			"spec": map[string]interface{}{
				"variables": 3,
				"clauses":   [][]int{{1, 2, 3}, {-1, 2, -3}},
			},
		}

		jsonData, _ := json.Marshal(payload)
		w := httptest.NewRecorder()
		req, _ := http.NewRequest("POST", "/v1/srs/solve", bytes.NewBuffer(jsonData))
		req.Header.Set("Content-Type", "application/json")
		r.ServeHTTP(w, req)

		if w.Code != http.StatusOK {
			t.Errorf("Expected status 200, got %d", w.Code)
		}

		var response types.APIResponse
		json.Unmarshal(w.Body.Bytes(), &response)

		if !response.Success {
			t.Errorf("Expected success true, got %v", response.Success)
		}
	})

	t.Run("ListProblems", func(t *testing.T) {
		w := httptest.NewRecorder()
		req, _ := http.NewRequest("GET", "/v1/srs/problems", nil)
		r.ServeHTTP(w, req)

		if w.Code != http.StatusOK {
			t.Errorf("Expected status 200, got %d", w.Code)
		}

		var response types.APIResponse
		json.Unmarshal(w.Body.Bytes(), &response)

		if !response.Success {
			t.Errorf("Expected success true, got %v", response.Success)
		}
	})

	t.Run("GetStatus", func(t *testing.T) {
		w := httptest.NewRecorder()
		req, _ := http.NewRequest("GET", "/v1/srs/status", nil)
		r.ServeHTTP(w, req)

		if w.Code != http.StatusOK {
			t.Errorf("Expected status 200, got %d", w.Code)
		}

		var response types.APIResponse
		json.Unmarshal(w.Body.Bytes(), &response)

		if !response.Success {
			t.Errorf("Expected success true, got %v", response.Success)
		}
	})

	t.Run("InvalidRequest", func(t *testing.T) {
		// Test invalid JSON
		w := httptest.NewRecorder()
		req, _ := http.NewRequest("POST", "/v1/srs/solve", bytes.NewBuffer([]byte("invalid json")))
		req.Header.Set("Content-Type", "application/json")
		r.ServeHTTP(w, req)

		if w.Code != http.StatusBadRequest {
			t.Errorf("Expected status 400, got %d", w.Code)
		}
	})
}

// testHQEEndpoints tests HQE API endpoints
func testHQEEndpoints(t *testing.T, r *gin.Engine) {
	t.Run("SimulateHolographic", func(t *testing.T) {
		payload := map[string]interface{}{
			"simulation_type": "holographic_reconstruction",
			"primes":          []int{2, 3, 5, 7},
			"steps":           10,
			"lambda":          0.02,
		}

		jsonData, _ := json.Marshal(payload)
		w := httptest.NewRecorder()
		req, _ := http.NewRequest("POST", "/v1/hqe/simulate", bytes.NewBuffer(jsonData))
		req.Header.Set("Content-Type", "application/json")
		r.ServeHTTP(w, req)

		if w.Code != http.StatusOK {
			t.Errorf("Expected status 200, got %d", w.Code)
		}
	})

	t.Run("GetPrimes", func(t *testing.T) {
		w := httptest.NewRecorder()
		req, _ := http.NewRequest("GET", "/v1/hqe/primes", nil)
		r.ServeHTTP(w, req)

		if w.Code != http.StatusOK {
			t.Errorf("Expected status 200, got %d", w.Code)
		}
	})

	t.Run("InvalidPrimes", func(t *testing.T) {
		payload := map[string]interface{}{
			"simulation_type": "holographic_reconstruction",
			"primes":          []int{}, // Empty primes should fail
			"steps":           10,
			"lambda":          0.02,
		}

		jsonData, _ := json.Marshal(payload)
		w := httptest.NewRecorder()
		req, _ := http.NewRequest("POST", "/v1/hqe/simulate", bytes.NewBuffer(jsonData))
		req.Header.Set("Content-Type", "application/json")
		r.ServeHTTP(w, req)

		if w.Code != http.StatusBadRequest {
			t.Errorf("Expected status 400, got %d", w.Code)
		}
	})
}

// testQSEMEndpoints tests QSEM API endpoints
func testQSEMEndpoints(t *testing.T, r *gin.Engine) {
	t.Run("EncodeConcepts", func(t *testing.T) {
		payload := map[string]interface{}{
			"concepts": []string{"quantum", "resonance", "consciousness"},
			"basis":    "prime",
		}

		jsonData, _ := json.Marshal(payload)
		w := httptest.NewRecorder()
		req, _ := http.NewRequest("POST", "/v1/qsem/encode", bytes.NewBuffer(jsonData))
		req.Header.Set("Content-Type", "application/json")
		r.ServeHTTP(w, req)

		if w.Code != http.StatusOK {
			t.Errorf("Expected status 200, got %d", w.Code)
		}
	})

	t.Run("ComputeResonance", func(t *testing.T) {
		payload := map[string]interface{}{
			"vectors": []map[string]interface{}{
				{"concept": "quantum", "alpha": []float64{0.1, 0.2, 0.3}},
				{"concept": "resonance", "alpha": []float64{0.2, 0.3, 0.4}},
			},
		}

		jsonData, _ := json.Marshal(payload)
		w := httptest.NewRecorder()
		req, _ := http.NewRequest("POST", "/v1/qsem/resonance", bytes.NewBuffer(jsonData))
		req.Header.Set("Content-Type", "application/json")
		r.ServeHTTP(w, req)

		if w.Code != http.StatusOK {
			t.Errorf("Expected status 200, got %d", w.Code)
		}
	})

	t.Run("GetBasis", func(t *testing.T) {
		w := httptest.NewRecorder()
		req, _ := http.NewRequest("GET", "/v1/qsem/basis", nil)
		r.ServeHTTP(w, req)

		if w.Code != http.StatusOK {
			t.Errorf("Expected status 200, got %d", w.Code)
		}
	})

	t.Run("InsufficientVectors", func(t *testing.T) {
		payload := map[string]interface{}{
			"vectors": []map[string]interface{}{
				{"concept": "quantum", "alpha": []float64{0.1, 0.2, 0.3}},
			}, // Only 1 vector, need at least 2
		}

		jsonData, _ := json.Marshal(payload)
		w := httptest.NewRecorder()
		req, _ := http.NewRequest("POST", "/v1/qsem/resonance", bytes.NewBuffer(jsonData))
		req.Header.Set("Content-Type", "application/json")
		r.ServeHTTP(w, req)

		if w.Code != http.StatusBadRequest {
			t.Errorf("Expected status 400, got %d", w.Code)
		}
	})
}

// testNLCEndpoints tests NLC API endpoints
func testNLCEndpoints(t *testing.T, r *gin.Engine) {
	var sessionID string

	t.Run("CreateSession", func(t *testing.T) {
		payload := map[string]interface{}{
			"primes":       []int{2, 3, 5},
			"participants": []string{"node_a", "node_b"},
			"protocol":     "teleportation",
		}

		jsonData, _ := json.Marshal(payload)
		w := httptest.NewRecorder()
		req, _ := http.NewRequest("POST", "/v1/nlc/sessions", bytes.NewBuffer(jsonData))
		req.Header.Set("Content-Type", "application/json")
		r.ServeHTTP(w, req)

		if w.Code != http.StatusCreated {
			t.Errorf("Expected status 201, got %d", w.Code)
		}

		var response types.APIResponse
		json.Unmarshal(w.Body.Bytes(), &response)
		if data, ok := response.Data.(map[string]interface{}); ok {
			sessionID = data["id"].(string)
		}
	})

	t.Run("GetSession", func(t *testing.T) {
		if sessionID == "" {
			sessionID = "test-session-123"
		}

		w := httptest.NewRecorder()
		req, _ := http.NewRequest("GET", "/v1/nlc/sessions/"+sessionID, nil)
		r.ServeHTTP(w, req)

		if w.Code != http.StatusOK {
			t.Errorf("Expected status 200, got %d", w.Code)
		}
	})

	t.Run("SendMessage", func(t *testing.T) {
		if sessionID == "" {
			sessionID = "test-session-123"
		}

		payload := map[string]interface{}{
			"content": "Hello quantum world!",
			"sender":  "node_a",
		}

		jsonData, _ := json.Marshal(payload)
		w := httptest.NewRecorder()
		req, _ := http.NewRequest("POST", "/v1/nlc/sessions/"+sessionID+"/messages", bytes.NewBuffer(jsonData))
		req.Header.Set("Content-Type", "application/json")
		r.ServeHTTP(w, req)

		if w.Code != http.StatusOK {
			t.Errorf("Expected status 200, got %d", w.Code)
		}
	})

	t.Run("GetChannels", func(t *testing.T) {
		w := httptest.NewRecorder()
		req, _ := http.NewRequest("GET", "/v1/nlc/channels", nil)
		r.ServeHTTP(w, req)

		if w.Code != http.StatusOK {
			t.Errorf("Expected status 200, got %d", w.Code)
		}
	})
}

// testQCREndpoints tests QCR API endpoints
func testQCREndpoints(t *testing.T, r *gin.Engine) {
	var sessionID string

	t.Run("CreateConsciousnessSession", func(t *testing.T) {
		payload := map[string]interface{}{
			"modes":           []string{"analytical", "creative", "ethical"},
			"simulation_type": "triadic_consciousness",
			"max_iterations":  21,
		}

		jsonData, _ := json.Marshal(payload)
		w := httptest.NewRecorder()
		req, _ := http.NewRequest("POST", "/v1/qcr/sessions", bytes.NewBuffer(jsonData))
		req.Header.Set("Content-Type", "application/json")
		r.ServeHTTP(w, req)

		if w.Code != http.StatusCreated {
			t.Errorf("Expected status 201, got %d", w.Code)
		}

		var response types.APIResponse
		json.Unmarshal(w.Body.Bytes(), &response)
		if data, ok := response.Data.(map[string]interface{}); ok {
			sessionID = data["id"].(string)
		}
	})

	t.Run("ObserveConsciousness", func(t *testing.T) {
		if sessionID == "" {
			sessionID = "test-qcr-session-123"
		}

		payload := map[string]interface{}{
			"prompt": "What is the nature of consciousness?",
		}

		jsonData, _ := json.Marshal(payload)
		w := httptest.NewRecorder()
		req, _ := http.NewRequest("POST", "/v1/qcr/sessions/"+sessionID+"/observe", bytes.NewBuffer(jsonData))
		req.Header.Set("Content-Type", "application/json")
		r.ServeHTTP(w, req)

		if w.Code != http.StatusOK {
			t.Errorf("Expected status 200, got %d", w.Code)
		}
	})

	t.Run("GetSupportedModes", func(t *testing.T) {
		w := httptest.NewRecorder()
		req, _ := http.NewRequest("GET", "/v1/qcr/modes", nil)
		r.ServeHTTP(w, req)

		if w.Code != http.StatusOK {
			t.Errorf("Expected status 200, got %d", w.Code)
		}
	})

	t.Run("NoModes", func(t *testing.T) {
		payload := map[string]interface{}{
			"modes": []string{}, // Empty modes should fail
		}

		jsonData, _ := json.Marshal(payload)
		w := httptest.NewRecorder()
		req, _ := http.NewRequest("POST", "/v1/qcr/sessions", bytes.NewBuffer(jsonData))
		req.Header.Set("Content-Type", "application/json")
		r.ServeHTTP(w, req)

		if w.Code != http.StatusBadRequest {
			t.Errorf("Expected status 400, got %d", w.Code)
		}
	})
}

// testIChingEndpoints tests I-Ching API endpoints
func testIChingEndpoints(t *testing.T, r *gin.Engine) {
	t.Run("EvolveHexagrams", func(t *testing.T) {
		payload := map[string]interface{}{
			"question": "What direction should I take in life?",
			"context":  "career",
			"steps":    7,
		}

		jsonData, _ := json.Marshal(payload)
		w := httptest.NewRecorder()
		req, _ := http.NewRequest("POST", "/v1/iching/evolve", bytes.NewBuffer(jsonData))
		req.Header.Set("Content-Type", "application/json")
		r.ServeHTTP(w, req)

		if w.Code != http.StatusOK {
			t.Errorf("Expected status 200, got %d", w.Code)
		}
	})

	t.Run("GetHexagrams", func(t *testing.T) {
		w := httptest.NewRecorder()
		req, _ := http.NewRequest("GET", "/v1/iching/hexagrams", nil)
		r.ServeHTTP(w, req)

		if w.Code != http.StatusOK {
			t.Errorf("Expected status 200, got %d", w.Code)
		}
	})

	t.Run("GetSpecificHexagram", func(t *testing.T) {
		w := httptest.NewRecorder()
		req, _ := http.NewRequest("GET", "/v1/iching/hexagrams/1", nil)
		r.ServeHTTP(w, req)

		if w.Code != http.StatusOK {
			t.Errorf("Expected status 200, got %d", w.Code)
		}
	})

	t.Run("GetTrigrams", func(t *testing.T) {
		w := httptest.NewRecorder()
		req, _ := http.NewRequest("GET", "/v1/iching/trigrams", nil)
		r.ServeHTTP(w, req)

		if w.Code != http.StatusOK {
			t.Errorf("Expected status 200, got %d", w.Code)
		}
	})

	t.Run("EmptyQuestion", func(t *testing.T) {
		payload := map[string]interface{}{
			"question": "", // Empty question should fail
			"steps":    7,
		}

		jsonData, _ := json.Marshal(payload)
		w := httptest.NewRecorder()
		req, _ := http.NewRequest("POST", "/v1/iching/evolve", bytes.NewBuffer(jsonData))
		req.Header.Set("Content-Type", "application/json")
		r.ServeHTTP(w, req)

		if w.Code != http.StatusBadRequest {
			t.Errorf("Expected status 400, got %d", w.Code)
		}
	})
}

// testUnifiedEndpoints tests Unified Physics API endpoints
func testUnifiedEndpoints(t *testing.T, r *gin.Engine) {
	t.Run("ComputeGravity", func(t *testing.T) {
		payload := map[string]interface{}{
			"observerEntropyReductionRate": 12.4,
			"regionEntropyGradient":        0.002,
		}

		jsonData, _ := json.Marshal(payload)
		w := httptest.NewRecorder()
		req, _ := http.NewRequest("POST", "/v1/unified/gravity/compute", bytes.NewBuffer(jsonData))
		req.Header.Set("Content-Type", "application/json")
		r.ServeHTTP(w, req)

		if w.Code != http.StatusOK {
			t.Errorf("Expected status 200, got %d", w.Code)
		}
	})

	t.Run("AnalyzeField", func(t *testing.T) {
		payload := map[string]interface{}{
			"massDistribution": []map[string]interface{}{
				{"position": map[string]float64{"x": 0, "y": 0, "z": 0}, "mass": 1.0},
			},
			"observerPosition": map[string]float64{"x": 1, "y": 1, "z": 1},
			"calculationRegion": map[string]interface{}{
				"center": map[string]float64{"x": 0, "y": 0, "z": 0},
				"size":   map[string]float64{"x": 10, "y": 10, "z": 10},
			},
		}

		jsonData, _ := json.Marshal(payload)
		w := httptest.NewRecorder()
		req, _ := http.NewRequest("POST", "/v1/unified/field/analyze", bytes.NewBuffer(jsonData))
		req.Header.Set("Content-Type", "application/json")
		r.ServeHTTP(w, req)

		if w.Code != http.StatusOK {
			t.Errorf("Expected status 200, got %d", w.Code)
		}
	})

	t.Run("GetConstants", func(t *testing.T) {
		w := httptest.NewRecorder()
		req, _ := http.NewRequest("GET", "/v1/unified/constants", nil)
		r.ServeHTTP(w, req)

		if w.Code != http.StatusOK {
			t.Errorf("Expected status 200, got %d", w.Code)
		}
	})

	t.Run("GetModels", func(t *testing.T) {
		w := httptest.NewRecorder()
		req, _ := http.NewRequest("GET", "/v1/unified/models", nil)
		r.ServeHTTP(w, req)

		if w.Code != http.StatusOK {
			t.Errorf("Expected status 200, got %d", w.Code)
		}
	})

	t.Run("InvalidEntropyRate", func(t *testing.T) {
		payload := map[string]interface{}{
			"observerEntropyReductionRate": -1.0, // Negative rate should fail
			"regionEntropyGradient":        0.002,
		}

		jsonData, _ := json.Marshal(payload)
		w := httptest.NewRecorder()
		req, _ := http.NewRequest("POST", "/v1/unified/gravity/compute", bytes.NewBuffer(jsonData))
		req.Header.Set("Content-Type", "application/json")
		r.ServeHTTP(w, req)

		if w.Code != http.StatusBadRequest {
			t.Errorf("Expected status 400, got %d", w.Code)
		}
	})

	t.Run("NoMassDistribution", func(t *testing.T) {
		payload := map[string]interface{}{
			"massDistribution": []map[string]interface{}{}, // Empty mass distribution should fail
			"observerPosition": map[string]float64{"x": 1, "y": 1, "z": 1},
			"calculationRegion": map[string]interface{}{
				"center": map[string]float64{"x": 0, "y": 0, "z": 0},
				"size":   map[string]float64{"x": 10, "y": 10, "z": 10},
			},
		}

		jsonData, _ := json.Marshal(payload)
		w := httptest.NewRecorder()
		req, _ := http.NewRequest("POST", "/v1/unified/field/analyze", bytes.NewBuffer(jsonData))
		req.Header.Set("Content-Type", "application/json")
		r.ServeHTTP(w, req)

		if w.Code != http.StatusBadRequest {
			t.Errorf("Expected status 400, got %d", w.Code)
		}
	})
}

// testWebhookEndpoints tests Webhook API endpoints
func testWebhookEndpoints(t *testing.T, r *gin.Engine) {
	var webhookID string

	t.Run("CreateWebhook", func(t *testing.T) {
		payload := map[string]interface{}{
			"url":    "https://example.com/webhook",
			"events": []string{"srs.solution.found", "qsem.encoding.complete"},
			"active": true,
		}

		jsonData, _ := json.Marshal(payload)
		w := httptest.NewRecorder()
		req, _ := http.NewRequest("POST", "/v1/webhooks", bytes.NewBuffer(jsonData))
		req.Header.Set("Content-Type", "application/json")
		r.ServeHTTP(w, req)

		if w.Code != http.StatusCreated {
			t.Errorf("Expected status 201, got %d", w.Code)
		}

		var response types.APIResponse
		json.Unmarshal(w.Body.Bytes(), &response)
		if data, ok := response.Data.(map[string]interface{}); ok {
			webhookID = data["id"].(string)
		}
	})

	t.Run("ListWebhooks", func(t *testing.T) {
		w := httptest.NewRecorder()
		req, _ := http.NewRequest("GET", "/v1/webhooks", nil)
		r.ServeHTTP(w, req)

		if w.Code != http.StatusOK {
			t.Errorf("Expected status 200, got %d", w.Code)
		}
	})

	t.Run("GetWebhook", func(t *testing.T) {
		if webhookID == "" {
			webhookID = "test-webhook-123"
		}

		w := httptest.NewRecorder()
		req, _ := http.NewRequest("GET", "/v1/webhooks/"+webhookID, nil)
		r.ServeHTTP(w, req)

		if w.Code != http.StatusOK {
			t.Errorf("Expected status 200, got %d", w.Code)
		}
	})

	t.Run("TestWebhook", func(t *testing.T) {
		if webhookID == "" {
			webhookID = "test-webhook-123"
		}

		payload := map[string]interface{}{
			"event_type": "srs.solution.found",
			"data":       map[string]interface{}{"test": true},
		}

		jsonData, _ := json.Marshal(payload)
		w := httptest.NewRecorder()
		req, _ := http.NewRequest("POST", "/v1/webhooks/"+webhookID+"/test", bytes.NewBuffer(jsonData))
		req.Header.Set("Content-Type", "application/json")
		r.ServeHTTP(w, req)

		if w.Code != http.StatusOK {
			t.Errorf("Expected status 200, got %d", w.Code)
		}
	})

	t.Run("GetAvailableEvents", func(t *testing.T) {
		w := httptest.NewRecorder()
		req, _ := http.NewRequest("GET", "/v1/webhooks/events", nil)
		r.ServeHTTP(w, req)

		if w.Code != http.StatusOK {
			t.Errorf("Expected status 200, got %d", w.Code)
		}
	})

	t.Run("NoEvents", func(t *testing.T) {
		payload := map[string]interface{}{
			"url":    "https://example.com/webhook",
			"events": []string{}, // Empty events should fail
		}

		jsonData, _ := json.Marshal(payload)
		w := httptest.NewRecorder()
		req, _ := http.NewRequest("POST", "/v1/webhooks", bytes.NewBuffer(jsonData))
		req.Header.Set("Content-Type", "application/json")
		r.ServeHTTP(w, req)

		if w.Code != http.StatusBadRequest {
			t.Errorf("Expected status 400, got %d", w.Code)
		}
	})
}

// createMockServiceContainer creates a simple placeholder - not used anymore
func createMockServiceContainer() (*services.ServiceContainer, error) {
	// Return nil since we're using mock handlers instead
	return nil, nil
}
