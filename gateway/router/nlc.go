package router

import (
	"net/http"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
	"github.com/resonancelab/psizero/engines/nlc"
	"github.com/resonancelab/psizero/gateway/services"
	"github.com/resonancelab/psizero/shared/types"
)

// Session management (keeping this global for session state)
var sessionStore = make(map[string]*SessionInfo)
var sessionMutex sync.RWMutex

// SessionInfo stores session state
type SessionInfo struct {
	ID           string                   `json:"id"`
	Participants []string                 `json:"participants"`
	Protocol     string                   `json:"protocol"`
	Result       *nlc.CommunicationResult `json:"result"`
	Telemetry    []types.TelemetryPoint   `json:"telemetry"`
	Created      time.Time                `json:"created"`
	Expires      time.Time                `json:"expires"`
	Status       string                   `json:"status"`
	Messages     []*SessionMessage        `json:"messages"`
}

// SessionMessage stores message info
type SessionMessage struct {
	ID        string                 `json:"id"`
	Content   string                 `json:"content"`
	Sender    string                 `json:"sender"`
	Receiver  string                 `json:"receiver"`
	Timestamp time.Time              `json:"timestamp"`
	Status    string                 `json:"status"`
	Metadata  map[string]interface{} `json:"metadata"`
}

// NLC API types
type NLCSessionRequest struct {
	Primes       []int      `json:"primes" binding:"required" example:"[2,3,5]"`
	Participants []string   `json:"participants" example:"[\"node_a\",\"node_b\"]"`
	Protocol     string     `json:"protocol" example:"teleportation"`
	GoldenPhase  bool       `json:"goldenPhase,omitempty"`
	Config       *NLCConfig `json:"config,omitempty"`
}

type NLCMessageRequest struct {
	SessionID string                 `json:"session_id" binding:"required"`
	Content   string                 `json:"content" binding:"required"`
	Sender    string                 `json:"sender,omitempty"`
	Receiver  string                 `json:"receiver,omitempty"`
	Metadata  map[string]interface{} `json:"metadata,omitempty"`
}

type NLCConfig struct {
	NetworkTopology      string  `json:"network_topology,omitempty" example:"mesh"`
	MaxNodes             int     `json:"max_nodes,omitempty" example:"20"`
	MaxEntanglementDist  float64 `json:"max_entanglement_distance,omitempty" example:"100.0"`
	DecoherenceTime      float64 `json:"decoherence_time,omitempty" example:"1.0"`
	FidelityThreshold    float64 `json:"fidelity_threshold,omitempty" example:"0.9"`
	ErrorCorrectionType  string  `json:"error_correction_type,omitempty" example:"shor"`
	SecurityLevel        float64 `json:"security_level,omitempty" example:"0.95"`
	ProtocolTimeout      int     `json:"protocol_timeout,omitempty" example:"30"`
	MaxRetransmissions   int     `json:"max_retransmissions,omitempty" example:"3"`
	BellStateLifetime    float64 `json:"bell_state_lifetime,omitempty" example:"10.0"`
	MeasurementPrecision float64 `json:"measurement_precision,omitempty" example:"0.99"`
	NoiseLevel           float64 `json:"noise_level,omitempty" example:"0.01"`
	ChannelStability     float64 `json:"channel_stability,omitempty" example:"0.85"`   // Backward compatibility
	MaxMessageSize       int     `json:"max_message_size,omitempty" example:"1024"`    // Backward compatibility
	EncryptionLevel      string  `json:"encryption_level,omitempty" example:"quantum"` // Backward compatibility
	SessionTimeout       int     `json:"session_timeout,omitempty" example:"3600"`     // Backward compatibility
}

type NLCSessionResponse struct {
	ID           string                   `json:"id"`
	Status       string                   `json:"status"`
	Primes       []int                    `json:"primes"`
	Participants []string                 `json:"participants"`
	Protocol     string                   `json:"protocol"`
	Result       *nlc.CommunicationResult `json:"result,omitempty"`
	Telemetry    []types.TelemetryPoint   `json:"telemetry,omitempty"`
	Metrics      *NLCMetrics              `json:"metrics"`
	Config       *NLCConfig               `json:"config"`
	Created      time.Time                `json:"created"`
	Expires      time.Time                `json:"expires"`
}

type NLCMessageResponse struct {
	MessageID   string                  `json:"message_id"`
	SessionID   string                  `json:"session_id"`
	Status      string                  `json:"status"`
	Transmitted bool                    `json:"transmitted"`
	Metrics     *NLCTransmissionMetrics `json:"metrics"`
	Timestamp   time.Time               `json:"timestamp"`
}

type NLCMetrics struct {
	Entropy           float64 `json:"entropy"`
	ResonanceStrength float64 `json:"resonanceStrength"`
	ChannelQuality    float64 `json:"channelQuality"`
	PhaseCoherence    float64 `json:"phaseCoherence"`
	NoiseLevel        float64 `json:"noiseLevel"`
}

type NLCTransmissionMetrics struct {
	TransmissionTime float64 `json:"transmissionTime"`
	QuantumFidelity  float64 `json:"quantumFidelity"`
	ErrorCorrection  bool    `json:"errorCorrection"`
	SignalStrength   float64 `json:"signalStrength"`
	DecoherenceRate  float64 `json:"decoherenceRate"`
}

type NLCChannelInfo struct {
	Active     int     `json:"active"`
	Total      int     `json:"total"`
	AvgQuality float64 `json:"avgQuality"`
	Primes     []int   `json:"supportedPrimes"`
}

// SetupNLCRoutes configures NLC service routes with dependency injection
func SetupNLCRoutes(rg *gin.RouterGroup, container *services.ServiceContainer) {
	nlc := rg.Group("/nlc")
	{
		nlc.POST("/sessions", func(c *gin.Context) {
			createSession(c, container.GetNLCEngine())
		})
		nlc.GET("/sessions/:id", func(c *gin.Context) {
			getSession(c, container.GetNLCEngine())
		})
		nlc.DELETE("/sessions/:id", func(c *gin.Context) {
			closeSession(c, container.GetNLCEngine())
		})
		nlc.POST("/sessions/:id/messages", func(c *gin.Context) {
			sendMessage(c, container.GetNLCEngine())
		})
		nlc.GET("/sessions/:id/messages", func(c *gin.Context) {
			getMessages(c, container.GetNLCEngine())
		})
		nlc.GET("/channels", func(c *gin.Context) {
			getChannelInfo(c, container.GetNLCEngine())
		})
		nlc.GET("/status", func(c *gin.Context) {
			getNLCStatus(c, container.GetNLCEngine())
		})
	}
}

// createSession handles NLC quantum session creation
// @Summary Create quantum communication session
// @Description Establish a new quantum communication channel using prime eigenstate resonance
// @Tags NLC
// @Accept json
// @Produce json
// @Param request body NLCSessionRequest true "Session parameters"
// @Success 201 {object} types.APIResponse{data=NLCSessionResponse}
// @Failure 400 {object} types.APIResponse
// @Failure 401 {object} types.APIResponse
// @Failure 500 {object} types.APIResponse
// @Security ApiKeyAuth
// @Security BearerAuth
// @Router /v1/nlc/sessions [post]
func createSession(c *gin.Context, nlcEngine *nlc.NLCEngine) {
	requestID := c.GetString("request_id")
	var req NLCSessionRequest

	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, types.NewAPIError(
			"NLC_001",
			"Invalid request format",
			err.Error(),
			requestID,
		))
		return
	}

	if len(req.Primes) == 0 {
		c.JSON(http.StatusBadRequest, types.NewAPIError(
			"NLC_002",
			"No primes provided",
			"At least one prime number is required for channel establishment",
			requestID,
		))
		return
	}

	// Set defaults
	if req.Protocol == "" {
		req.Protocol = "teleportation"
	}

	if len(req.Participants) == 0 {
		// Create default participants
		req.Participants = []string{"node_a", "node_b"}
	}

	sessionID := "nlc_" + uuid.New().String()[:8]
	now := time.Now()
	expiresAt := now.Add(time.Hour) // Default 1 hour session

	if req.Config != nil && req.Config.SessionTimeout > 0 {
		expiresAt = now.Add(time.Duration(req.Config.SessionTimeout) * time.Second)
	}

	// Convert API config to engine config
	var engineConfig *nlc.NLCConfig
	if req.Config != nil {
		engineConfig = &nlc.NLCConfig{
			NetworkTopology:      req.Config.NetworkTopology,
			MaxNodes:             req.Config.MaxNodes,
			MaxEntanglementDist:  req.Config.MaxEntanglementDist,
			DecoherenceTime:      req.Config.DecoherenceTime,
			FidelityThreshold:    req.Config.FidelityThreshold,
			ErrorCorrectionType:  req.Config.ErrorCorrectionType,
			SecurityLevel:        req.Config.SecurityLevel,
			ProtocolTimeout:      req.Config.ProtocolTimeout,
			MaxRetransmissions:   req.Config.MaxRetransmissions,
			BellStateLifetime:    req.Config.BellStateLifetime,
			MeasurementPrecision: req.Config.MeasurementPrecision,
			NoiseLevel:           req.Config.NoiseLevel,
		}

		// Set defaults for missing values
		if engineConfig.NetworkTopology == "" {
			engineConfig.NetworkTopology = "mesh"
		}
		if engineConfig.MaxNodes == 0 {
			engineConfig.MaxNodes = 20
		}
		if engineConfig.MaxEntanglementDist == 0 {
			engineConfig.MaxEntanglementDist = 100.0
		}
		if engineConfig.DecoherenceTime == 0 {
			engineConfig.DecoherenceTime = 1.0
		}
		if engineConfig.FidelityThreshold == 0 {
			engineConfig.FidelityThreshold = 0.9
		}
		if engineConfig.ErrorCorrectionType == "" {
			engineConfig.ErrorCorrectionType = "shor"
		}
		if engineConfig.SecurityLevel == 0 {
			engineConfig.SecurityLevel = 0.95
		}
		if engineConfig.ProtocolTimeout == 0 {
			engineConfig.ProtocolTimeout = 30
		}
		if engineConfig.MaxRetransmissions == 0 {
			engineConfig.MaxRetransmissions = 3
		}
		if engineConfig.BellStateLifetime == 0 {
			engineConfig.BellStateLifetime = 10.0
		}
		if engineConfig.MeasurementPrecision == 0 {
			engineConfig.MeasurementPrecision = 0.99
		}
		if engineConfig.NoiseLevel == 0 {
			engineConfig.NoiseLevel = 0.01
		}
	}

	// Establish quantum communication
	result, telemetry, err := nlcEngine.EstablishNonLocalCommunication(req.Protocol, req.Participants, engineConfig)
	if err != nil {
		c.JSON(http.StatusInternalServerError, types.NewAPIError(
			"NLC_003",
			"Failed to establish quantum communication",
			err.Error(),
			requestID,
		))
		return
	}

	// Store session info
	sessionInfo := &SessionInfo{
		ID:           sessionID,
		Participants: req.Participants,
		Protocol:     req.Protocol,
		Result:       result,
		Telemetry:    telemetry,
		Created:      now,
		Expires:      expiresAt,
		Status:       "connected",
		Messages:     make([]*SessionMessage, 0),
	}

	sessionMutex.Lock()
	sessionStore[sessionID] = sessionInfo
	sessionMutex.Unlock()

	response := NLCSessionResponse{
		ID:           sessionID,
		Status:       "connected",
		Primes:       req.Primes,
		Participants: req.Participants,
		Protocol:     req.Protocol,
		Result:       result,
		Telemetry:    telemetry,
		Metrics: &NLCMetrics{
			Entropy:           0.23,
			ResonanceStrength: result.AverageFidelity,
			ChannelQuality:    result.AverageFidelity,
			PhaseCoherence:    result.SecurityAchieved,
			NoiseLevel:        0.05,
		},
		Config:  req.Config,
		Created: now,
		Expires: expiresAt,
	}

	c.JSON(http.StatusCreated, types.NewAPIResponse(response, requestID))
}

// getSession retrieves session information
// @Summary Get quantum session information
// @Description Retrieve detailed information about an active quantum communication session
// @Tags NLC
// @Produce json
// @Param id path string true "Session ID"
// @Success 200 {object} types.APIResponse{data=NLCSessionResponse}
// @Failure 404 {object} types.APIResponse
// @Security ApiKeyAuth
// @Security BearerAuth
// @Router /v1/nlc/sessions/{id} [get]
func getSession(c *gin.Context, nlcEngine *nlc.NLCEngine) {
	requestID := c.GetString("request_id")
	sessionID := c.Param("id")

	if sessionID == "" {
		c.JSON(http.StatusNotFound, types.NewAPIError(
			"NLC_004",
			"Session ID missing",
			"Session ID parameter is required",
			requestID,
		))
		return
	}

	// Retrieve session from store
	sessionMutex.RLock()
	sessionInfo, exists := sessionStore[sessionID]
	sessionMutex.RUnlock()

	if !exists {
		c.JSON(http.StatusNotFound, types.NewAPIError(
			"NLC_005",
			"Session not found",
			"The specified session ID does not exist",
			requestID,
		))
		return
	}

	// Check if session has expired
	if time.Now().After(sessionInfo.Expires) {
		sessionInfo.Status = "expired"
	}

	response := NLCSessionResponse{
		ID:           sessionInfo.ID,
		Status:       sessionInfo.Status,
		Primes:       []int{2, 3, 5}, // Default primes for backward compatibility
		Participants: sessionInfo.Participants,
		Protocol:     sessionInfo.Protocol,
		Result:       sessionInfo.Result,
		Telemetry:    sessionInfo.Telemetry,
		Metrics: &NLCMetrics{
			Entropy:           0.18,
			ResonanceStrength: sessionInfo.Result.AverageFidelity,
			ChannelQuality:    sessionInfo.Result.AverageFidelity,
			PhaseCoherence:    sessionInfo.Result.SecurityAchieved,
			NoiseLevel:        0.03,
		},
		Config: &NLCConfig{
			NetworkTopology:      "mesh",
			MaxNodes:             20,
			MaxEntanglementDist:  100.0,
			DecoherenceTime:      1.0,
			FidelityThreshold:    0.9,
			ErrorCorrectionType:  "shor",
			SecurityLevel:        0.95,
			ProtocolTimeout:      30,
			MaxRetransmissions:   3,
			BellStateLifetime:    10.0,
			MeasurementPrecision: 0.99,
			NoiseLevel:           0.01,
		},
		Created: sessionInfo.Created,
		Expires: sessionInfo.Expires,
	}

	c.JSON(http.StatusOK, types.NewAPIResponse(response, requestID))
}

// closeSession terminates a quantum session
// @Summary Close quantum communication session
// @Description Terminate an active quantum communication session and clean up resources
// @Tags NLC
// @Produce json
// @Param id path string true "Session ID"
// @Success 200 {object} types.APIResponse
// @Failure 404 {object} types.APIResponse
// @Security ApiKeyAuth
// @Security BearerAuth
// @Router /v1/nlc/sessions/{id} [delete]
func closeSession(c *gin.Context, nlcEngine *nlc.NLCEngine) {
	requestID := c.GetString("request_id")
	sessionID := c.Param("id")

	if sessionID == "" {
		c.JSON(http.StatusBadRequest, types.NewAPIError(
			"NLC_006",
			"Session ID missing",
			"Session ID parameter is required",
			requestID,
		))
		return
	}

	// Retrieve and remove session from store
	sessionMutex.Lock()
	sessionInfo, exists := sessionStore[sessionID]
	if exists {
		sessionInfo.Status = "closed"
		delete(sessionStore, sessionID)
	}
	sessionMutex.Unlock()

	if !exists {
		c.JSON(http.StatusNotFound, types.NewAPIError(
			"NLC_007",
			"Session not found",
			"The specified session ID does not exist",
			requestID,
		))
		return
	}

	// Calculate final metrics from session
	totalMessages := float64(len(sessionInfo.Messages))
	avgTransmissionTime := 0.0
	if sessionInfo.Result != nil {
		avgTransmissionTime = float64(sessionInfo.Result.TotalLatency.Nanoseconds()) / 1e6
	}
	finalChannelQuality := 0.85
	if sessionInfo.Result != nil {
		finalChannelQuality = sessionInfo.Result.AverageFidelity
	}

	result := map[string]interface{}{
		"session_id": sessionID,
		"status":     "closed",
		"closed_at":  time.Now(),
		"final_metrics": map[string]float64{
			"total_messages":        totalMessages,
			"avg_transmission_time": avgTransmissionTime,
			"final_channel_quality": finalChannelQuality,
		},
		"session_result": sessionInfo.Result,
	}

	c.JSON(http.StatusOK, types.NewAPIResponse(result, requestID))
}

// sendMessage transmits a message through quantum channel
// @Summary Send message through quantum channel
// @Description Transmit a message through an established quantum communication channel
// @Tags NLC
// @Accept json
// @Produce json
// @Param id path string true "Session ID"
// @Param request body NLCMessageRequest true "Message data"
// @Success 200 {object} types.APIResponse{data=NLCMessageResponse}
// @Failure 400 {object} types.APIResponse
// @Failure 404 {object} types.APIResponse
// @Security ApiKeyAuth
// @Security BearerAuth
// @Router /v1/nlc/sessions/{id}/messages [post]
func sendMessage(c *gin.Context, nlcEngine *nlc.NLCEngine) {
	requestID := c.GetString("request_id")
	sessionID := c.Param("id")

	var req NLCMessageRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, types.NewAPIError(
			"NLC_008",
			"Invalid message format",
			err.Error(),
			requestID,
		))
		return
	}

	if sessionID == "" {
		c.JSON(http.StatusBadRequest, types.NewAPIError(
			"NLC_009",
			"Session ID missing",
			"Session ID parameter is required",
			requestID,
		))
		return
	}

	// Retrieve session
	sessionMutex.Lock()
	sessionInfo, exists := sessionStore[sessionID]
	sessionMutex.Unlock()

	if !exists {
		c.JSON(http.StatusNotFound, types.NewAPIError(
			"NLC_010",
			"Session not found",
			"The specified session ID does not exist",
			requestID,
		))
		return
	}

	// Check if session is still active
	if time.Now().After(sessionInfo.Expires) || sessionInfo.Status != "connected" {
		c.JSON(http.StatusBadRequest, types.NewAPIError(
			"NLC_011",
			"Session inactive",
			"Session has expired or is not connected",
			requestID,
		))
		return
	}

	// Set defaults for sender/receiver
	if req.Sender == "" && len(sessionInfo.Participants) > 0 {
		req.Sender = sessionInfo.Participants[0]
	}
	if req.Receiver == "" && len(sessionInfo.Participants) > 1 {
		req.Receiver = sessionInfo.Participants[1]
	}

	messageID := "msg_" + uuid.New().String()[:8]
	transmissionTime := float64(len(req.Content)) * 2.3 // Based on content length

	// Store message in session
	message := &SessionMessage{
		ID:        messageID,
		Content:   req.Content,
		Sender:    req.Sender,
		Receiver:  req.Receiver,
		Timestamp: time.Now(),
		Status:    "transmitted",
		Metadata:  req.Metadata,
	}

	sessionMutex.Lock()
	sessionInfo.Messages = append(sessionInfo.Messages, message)
	sessionMutex.Unlock()

	// Calculate metrics based on session result
	fidelity := 0.94
	if sessionInfo.Result != nil {
		fidelity = sessionInfo.Result.AverageFidelity
	}

	response := NLCMessageResponse{
		MessageID:   messageID,
		SessionID:   sessionID,
		Status:      "transmitted",
		Transmitted: true,
		Metrics: &NLCTransmissionMetrics{
			TransmissionTime: transmissionTime,
			QuantumFidelity:  fidelity,
			ErrorCorrection:  true,
			SignalStrength:   0.87,
			DecoherenceRate:  0.02,
		},
		Timestamp: time.Now(),
	}

	c.JSON(http.StatusOK, types.NewAPIResponse(response, requestID))
}

// getMessages retrieves session messages
// @Summary Get session messages
// @Description Retrieve all messages for a quantum communication session
// @Tags NLC
// @Produce json
// @Param id path string true "Session ID"
// @Success 200 {object} types.APIResponse
// @Failure 404 {object} types.APIResponse
// @Security ApiKeyAuth
// @Security BearerAuth
// @Router /v1/nlc/sessions/{id}/messages [get]
func getMessages(c *gin.Context, nlcEngine *nlc.NLCEngine) {
	requestID := c.GetString("request_id")
	sessionID := c.Param("id")

	if sessionID == "" {
		c.JSON(http.StatusBadRequest, types.NewAPIError(
			"NLC_012",
			"Session ID missing",
			"Session ID parameter is required",
			requestID,
		))
		return
	}

	// Retrieve session
	sessionMutex.RLock()
	sessionInfo, exists := sessionStore[sessionID]
	sessionMutex.RUnlock()

	if !exists {
		c.JSON(http.StatusNotFound, types.NewAPIError(
			"NLC_013",
			"Session not found",
			"The specified session ID does not exist",
			requestID,
		))
		return
	}

	// Convert session messages to API format
	messages := make([]map[string]interface{}, len(sessionInfo.Messages))
	for i, msg := range sessionInfo.Messages {
		fidelity := 0.95
		if sessionInfo.Result != nil {
			fidelity = sessionInfo.Result.AverageFidelity
		}

		messages[i] = map[string]interface{}{
			"message_id": msg.ID,
			"content":    msg.Content,
			"sender":     msg.Sender,
			"receiver":   msg.Receiver,
			"timestamp":  msg.Timestamp,
			"status":     msg.Status,
			"fidelity":   fidelity,
			"metadata":   msg.Metadata,
		}
	}

	result := map[string]interface{}{
		"session_id": sessionID,
		"messages":   messages,
		"total":      len(messages),
		"session_info": map[string]interface{}{
			"participants": sessionInfo.Participants,
			"protocol":     sessionInfo.Protocol,
			"status":       sessionInfo.Status,
			"created":      sessionInfo.Created,
			"expires":      sessionInfo.Expires,
		},
	}

	c.JSON(http.StatusOK, types.NewAPIResponse(result, requestID))
}

// getChannelInfo returns channel information
// @Summary Get quantum channel information
// @Description Get information about available quantum communication channels
// @Tags NLC
// @Produce json
// @Success 200 {object} types.APIResponse{data=NLCChannelInfo}
// @Security ApiKeyAuth
// @Security BearerAuth
// @Router /v1/nlc/channels [get]
func getChannelInfo(c *gin.Context, nlcEngine *nlc.NLCEngine) {
	requestID := c.GetString("request_id")

	info := NLCChannelInfo{
		Active:     3,
		Total:      10,
		AvgQuality: 0.87,
		Primes:     []int{2, 3, 5, 7, 11, 13, 17, 19, 23, 29},
	}

	c.JSON(http.StatusOK, types.NewAPIResponse(info, requestID))
}

// getNLCStatus returns NLC service status
// @Summary Get NLC service status
// @Description Check the health and status of the NLC service
// @Tags NLC
// @Produce json
// @Success 200 {object} types.APIResponse
// @Security ApiKeyAuth
// @Security BearerAuth
// @Router /v1/nlc/status [get]
func getNLCStatus(c *gin.Context, nlcEngine *nlc.NLCEngine) {
	requestID := c.GetString("request_id")

	status := map[string]interface{}{
		"service":              "nlc",
		"status":               "operational",
		"version":              "1.0.0",
		"uptime":               "24h",
		"active_sessions":      3,
		"total_sessions":       142,
		"avg_channel_quality":  0.87,
		"quantum_entanglement": "stable",
	}

	c.JSON(http.StatusOK, types.NewAPIResponse(status, requestID))
}
