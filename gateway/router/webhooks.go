package router

import (
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
	"github.com/resonancelab/psizero/shared/types"
)

// Webhook API types
type WebhookCreateRequest struct {
	URL    string         `json:"url" binding:"required"`
	Events []string       `json:"events" binding:"required"`
	Secret string         `json:"secret,omitempty"`
	Active bool           `json:"active,omitempty"`
	Config *WebhookConfig `json:"config,omitempty"`
}

type WebhookUpdateRequest struct {
	URL    string         `json:"url,omitempty"`
	Events []string       `json:"events,omitempty"`
	Secret string         `json:"secret,omitempty"`
	Active *bool          `json:"active,omitempty"`
	Config *WebhookConfig `json:"config,omitempty"`
}

type WebhookConfig struct {
	ContentType    string `json:"contentType,omitempty" example:"application/json"`
	InsecureSSL    bool   `json:"insecureSSL,omitempty"`
	RetryAttempts  int    `json:"retryAttempts,omitempty" example:"3"`
	RetryDelay     int    `json:"retryDelay,omitempty" example:"5"`
	TimeoutSeconds int    `json:"timeoutSeconds,omitempty" example:"30"`
}

type WebhookResponse struct {
	ID        string         `json:"id"`
	URL       string         `json:"url"`
	Events    []string       `json:"events"`
	Active    bool           `json:"active"`
	Config    *WebhookConfig `json:"config"`
	CreatedAt time.Time      `json:"created_at"`
	UpdatedAt time.Time      `json:"updated_at"`
	LastPing  *time.Time     `json:"last_ping,omitempty"`
	Status    string         `json:"status"`
}

type WebhookDelivery struct {
	ID           string              `json:"id"`
	WebhookID    string              `json:"webhook_id"`
	Event        *types.WebhookEvent `json:"event"`
	Success      bool                `json:"success"`
	StatusCode   int                 `json:"status_code,omitempty"`
	Response     string              `json:"response,omitempty"`
	Error        string              `json:"error,omitempty"`
	Duration     float64             `json:"duration_ms"`
	AttemptCount int                 `json:"attempt_count"`
	DeliveredAt  time.Time           `json:"delivered_at"`
}

type WebhookTestRequest struct {
	EventType string                 `json:"event_type" example:"srs.solution.found"`
	Data      map[string]interface{} `json:"data,omitempty"`
}

// SetupWebhookRoutes configures webhook management routes
func SetupWebhookRoutes(rg *gin.RouterGroup) {
	webhooks := rg.Group("/webhooks")
	{
		webhooks.POST("", createWebhook)
		webhooks.GET("", listWebhooks)
		webhooks.GET("/:id", getWebhook)
		webhooks.PUT("/:id", updateWebhook)
		webhooks.DELETE("/:id", deleteWebhook)
		webhooks.POST("/:id/test", testWebhook)
		webhooks.GET("/:id/deliveries", getWebhookDeliveries)
		webhooks.GET("/:id/deliveries/:delivery_id", getWebhookDelivery)
		webhooks.POST("/:id/deliveries/:delivery_id/redeliver", redeliverWebhook)
		webhooks.GET("/events", getAvailableEvents)
	}
}

// createWebhook handles webhook creation
// @Summary Create a new webhook
// @Description Create a new webhook to receive real-time notifications for platform events
// @Tags Webhooks
// @Accept json
// @Produce json
// @Param request body WebhookCreateRequest true "Webhook configuration"
// @Success 201 {object} types.APIResponse{data=WebhookResponse}
// @Failure 400 {object} types.APIResponse
// @Failure 401 {object} types.APIResponse
// @Security ApiKeyAuth
// @Security BearerAuth
// @Router /v1/webhooks [post]
func createWebhook(c *gin.Context) {
	requestID := c.GetString("request_id")
	var req WebhookCreateRequest

	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, types.NewAPIError(
			"WEBHOOK_001",
			"Invalid request format",
			err.Error(),
			requestID,
		))
		return
	}

	if len(req.Events) == 0 {
		c.JSON(http.StatusBadRequest, types.NewAPIError(
			"WEBHOOK_002",
			"No events specified",
			"At least one event type must be specified",
			requestID,
		))
		return
	}

	webhookID := "wh_" + uuid.New().String()[:16]
	now := time.Now()

	response := WebhookResponse{
		ID:     webhookID,
		URL:    req.URL,
		Events: req.Events,
		Active: true,
		Config: &WebhookConfig{
			ContentType:    "application/json",
			InsecureSSL:    false,
			RetryAttempts:  3,
			RetryDelay:     5,
			TimeoutSeconds: 30,
		},
		CreatedAt: now,
		UpdatedAt: now,
		Status:    "active",
	}

	if req.Config != nil {
		response.Config = req.Config
	}

	c.JSON(http.StatusCreated, types.NewAPIResponse(response, requestID))
}

// listWebhooks returns all webhooks for the user
// @Summary List webhooks
// @Description Get a list of all webhooks configured for the authenticated user
// @Tags Webhooks
// @Produce json
// @Success 200 {object} types.APIResponse{data=[]WebhookResponse}
// @Security ApiKeyAuth
// @Security BearerAuth
// @Router /v1/webhooks [get]
func listWebhooks(c *gin.Context) {
	requestID := c.GetString("request_id")

	// Mock webhook list
	webhooks := []WebhookResponse{
		{
			ID:     "wh_abc123def456",
			URL:    "https://api.example.com/webhooks/psizero",
			Events: []string{"srs.solution.found", "qsem.encoding.complete"},
			Active: true,
			Config: &WebhookConfig{
				ContentType:    "application/json",
				RetryAttempts:  3,
				TimeoutSeconds: 30,
			},
			CreatedAt: time.Now().Add(-24 * time.Hour),
			UpdatedAt: time.Now().Add(-1 * time.Hour),
			LastPing:  &[]time.Time{time.Now().Add(-10 * time.Minute)}[0],
			Status:    "active",
		},
	}

	c.JSON(http.StatusOK, types.NewAPIResponse(webhooks, requestID))
}

// getWebhook returns specific webhook details
// @Summary Get webhook details
// @Description Get detailed information about a specific webhook
// @Tags Webhooks
// @Produce json
// @Param id path string true "Webhook ID"
// @Success 200 {object} types.APIResponse{data=WebhookResponse}
// @Failure 404 {object} types.APIResponse
// @Security ApiKeyAuth
// @Security BearerAuth
// @Router /v1/webhooks/{id} [get]
func getWebhook(c *gin.Context) {
	requestID := c.GetString("request_id")
	webhookID := c.Param("id")

	// Mock webhook retrieval
	webhook := WebhookResponse{
		ID:     webhookID,
		URL:    "https://api.example.com/webhooks/psizero",
		Events: []string{"srs.solution.found", "qsem.encoding.complete"},
		Active: true,
		Config: &WebhookConfig{
			ContentType:    "application/json",
			RetryAttempts:  3,
			TimeoutSeconds: 30,
		},
		CreatedAt: time.Now().Add(-24 * time.Hour),
		UpdatedAt: time.Now().Add(-1 * time.Hour),
		LastPing:  &[]time.Time{time.Now().Add(-10 * time.Minute)}[0],
		Status:    "active",
	}

	c.JSON(http.StatusOK, types.NewAPIResponse(webhook, requestID))
}

// updateWebhook handles webhook updates
// @Summary Update webhook
// @Description Update an existing webhook configuration
// @Tags Webhooks
// @Accept json
// @Produce json
// @Param id path string true "Webhook ID"
// @Param request body WebhookUpdateRequest true "Updated webhook configuration"
// @Success 200 {object} types.APIResponse{data=WebhookResponse}
// @Failure 404 {object} types.APIResponse
// @Security ApiKeyAuth
// @Security BearerAuth
// @Router /v1/webhooks/{id} [put]
func updateWebhook(c *gin.Context) {
	requestID := c.GetString("request_id")
	webhookID := c.Param("id")

	var req WebhookUpdateRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, types.NewAPIError(
			"WEBHOOK_003",
			"Invalid request format",
			err.Error(),
			requestID,
		))
		return
	}

	// Mock webhook update
	webhook := WebhookResponse{
		ID:        webhookID,
		URL:       req.URL,
		Events:    req.Events,
		Active:    *req.Active,
		Config:    req.Config,
		CreatedAt: time.Now().Add(-24 * time.Hour),
		UpdatedAt: time.Now(),
		Status:    "active",
	}

	c.JSON(http.StatusOK, types.NewAPIResponse(webhook, requestID))
}

// deleteWebhook removes a webhook
// @Summary Delete webhook
// @Description Delete an existing webhook
// @Tags Webhooks
// @Produce json
// @Param id path string true "Webhook ID"
// @Success 200 {object} types.APIResponse
// @Failure 404 {object} types.APIResponse
// @Security ApiKeyAuth
// @Security BearerAuth
// @Router /v1/webhooks/{id} [delete]
func deleteWebhook(c *gin.Context) {
	requestID := c.GetString("request_id")
	webhookID := c.Param("id")

	result := map[string]interface{}{
		"webhook_id": webhookID,
		"deleted":    true,
		"deleted_at": time.Now(),
	}

	c.JSON(http.StatusOK, types.NewAPIResponse(result, requestID))
}

// testWebhook sends a test webhook delivery
// @Summary Test webhook
// @Description Send a test event to the webhook URL to verify configuration
// @Tags Webhooks
// @Accept json
// @Produce json
// @Param id path string true "Webhook ID"
// @Param request body WebhookTestRequest true "Test event data"
// @Success 200 {object} types.APIResponse{data=WebhookDelivery}
// @Failure 404 {object} types.APIResponse
// @Security ApiKeyAuth
// @Security BearerAuth
// @Router /v1/webhooks/{id}/test [post]
func testWebhook(c *gin.Context) {
	requestID := c.GetString("request_id")
	webhookID := c.Param("id")

	var req WebhookTestRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, types.NewAPIError(
			"WEBHOOK_004",
			"Invalid test request",
			err.Error(),
			requestID,
		))
		return
	}

	// Mock test delivery
	delivery := WebhookDelivery{
		ID:        "del_" + uuid.New().String()[:8],
		WebhookID: webhookID,
		Event: &types.WebhookEvent{
			ID:        uuid.New().String(),
			Type:      req.EventType,
			Service:   "test",
			UserID:    c.GetString("user_id"),
			Data:      req.Data,
			Timestamp: time.Now(),
		},
		Success:      true,
		StatusCode:   200,
		Response:     "OK",
		Duration:     125.5,
		AttemptCount: 1,
		DeliveredAt:  time.Now(),
	}

	c.JSON(http.StatusOK, types.NewAPIResponse(delivery, requestID))
}

// getWebhookDeliveries returns delivery history
// @Summary Get webhook deliveries
// @Description Get delivery history for a webhook
// @Tags Webhooks
// @Produce json
// @Param id path string true "Webhook ID"
// @Success 200 {object} types.APIResponse{data=[]WebhookDelivery}
// @Security ApiKeyAuth
// @Security BearerAuth
// @Router /v1/webhooks/{id}/deliveries [get]
func getWebhookDeliveries(c *gin.Context) {
	requestID := c.GetString("request_id")
	webhookID := c.Param("id")

	// Mock delivery history
	deliveries := []WebhookDelivery{
		{
			ID:        "del_abc123",
			WebhookID: webhookID,
			Event: &types.WebhookEvent{
				Type:    "srs.solution.found",
				Service: "srs",
				Data:    map[string]interface{}{"feasible": true},
			},
			Success:      true,
			StatusCode:   200,
			Duration:     89.2,
			AttemptCount: 1,
			DeliveredAt:  time.Now().Add(-2 * time.Hour),
		},
	}

	result := map[string]interface{}{
		"webhook_id":  webhookID,
		"deliveries":  deliveries,
		"total_count": len(deliveries),
	}

	c.JSON(http.StatusOK, types.NewAPIResponse(result, requestID))
}

// getWebhookDelivery returns specific delivery details
// @Summary Get webhook delivery details
// @Description Get detailed information about a specific webhook delivery
// @Tags Webhooks
// @Produce json
// @Param id path string true "Webhook ID"
// @Param delivery_id path string true "Delivery ID"
// @Success 200 {object} types.APIResponse{data=WebhookDelivery}
// @Security ApiKeyAuth
// @Security BearerAuth
// @Router /v1/webhooks/{id}/deliveries/{delivery_id} [get]
func getWebhookDelivery(c *gin.Context) {
	requestID := c.GetString("request_id")
	webhookID := c.Param("id")
	deliveryID := c.Param("delivery_id")

	delivery := WebhookDelivery{
		ID:        deliveryID,
		WebhookID: webhookID,
		Event: &types.WebhookEvent{
			Type:    "srs.solution.found",
			Service: "srs",
			Data:    map[string]interface{}{"feasible": true, "iterations": 42},
		},
		Success:      true,
		StatusCode:   200,
		Response:     "OK",
		Duration:     89.2,
		AttemptCount: 1,
		DeliveredAt:  time.Now().Add(-2 * time.Hour),
	}

	c.JSON(http.StatusOK, types.NewAPIResponse(delivery, requestID))
}

// redeliverWebhook retries a failed delivery
// @Summary Redeliver webhook
// @Description Retry delivery of a previously failed webhook
// @Tags Webhooks
// @Produce json
// @Param id path string true "Webhook ID"
// @Param delivery_id path string true "Delivery ID"
// @Success 200 {object} types.APIResponse{data=WebhookDelivery}
// @Security ApiKeyAuth
// @Security BearerAuth
// @Router /v1/webhooks/{id}/deliveries/{delivery_id}/redeliver [post]
func redeliverWebhook(c *gin.Context) {
	requestID := c.GetString("request_id")
	webhookID := c.Param("id")
	deliveryID := c.Param("delivery_id")

	newDelivery := WebhookDelivery{
		ID:        "del_" + uuid.New().String()[:8],
		WebhookID: webhookID,
		Event: &types.WebhookEvent{
			Type:    "srs.solution.found",
			Service: "srs",
			Data:    map[string]interface{}{"redelivery": true, "original_id": deliveryID},
		},
		Success:      true,
		StatusCode:   200,
		Response:     "OK",
		Duration:     95.1,
		AttemptCount: 2,
		DeliveredAt:  time.Now(),
	}

	c.JSON(http.StatusOK, types.NewAPIResponse(newDelivery, requestID))
}

// getAvailableEvents returns available webhook events
// @Summary Get available webhook events
// @Description Get a list of all available webhook event types
// @Tags Webhooks
// @Produce json
// @Success 200 {object} types.APIResponse
// @Security ApiKeyAuth
// @Security BearerAuth
// @Router /v1/webhooks/events [get]
func getAvailableEvents(c *gin.Context) {
	requestID := c.GetString("request_id")

	events := map[string]interface{}{
		"srs": []string{
			"srs.solution.found",
			"srs.solution.failed",
			"srs.computation.started",
			"srs.plateau.detected",
		},
		"hqe": []string{
			"hqe.simulation.complete",
			"hqe.convergence.achieved",
			"hqe.quantum.state.updated",
		},
		"qsem": []string{
			"qsem.encoding.complete",
			"qsem.resonance.computed",
			"qsem.vectors.stored",
		},
		"nlc": []string{
			"nlc.session.created",
			"nlc.session.closed",
			"nlc.message.transmitted",
			"nlc.channel.quality.changed",
		},
		"qcr": []string{
			"qcr.session.stabilized",
			"qcr.observation.complete",
			"qcr.consciousness.evolved",
		},
		"iching": []string{
			"iching.evolution.complete",
			"iching.hexagram.stabilized",
			"iching.guidance.generated",
		},
		"unified": []string{
			"unified.gravity.computed",
			"unified.field.analyzed",
			"unified.constants.updated",
		},
		"system": []string{
			"user.quota.exceeded",
			"api.rate_limit.reached",
			"system.maintenance.scheduled",
		},
	}

	c.JSON(http.StatusOK, types.NewAPIResponse(events, requestID))
}
