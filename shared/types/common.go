package types

import (
	"time"

	"github.com/google/uuid"
)

// Common request/response structures

// APIResponse is the standard response wrapper
type APIResponse struct {
	Success   bool        `json:"success"`
	Data      interface{} `json:"data,omitempty"`
	Error     *APIError   `json:"error,omitempty"`
	RequestID string      `json:"request_id"`
	Timestamp time.Time   `json:"timestamp"`
}

// APIError represents an API error
type APIError struct {
	Code    string `json:"code"`
	Message string `json:"message"`
	Details string `json:"details,omitempty"`
}

// TelemetryPoint represents a single telemetry data point
type TelemetryPoint struct {
	Step              int       `json:"t"`
	SymbolicEntropy   float64   `json:"S"`
	LyapunovMetric    float64   `json:"L"`
	SatisfactionRate  float64   `json:"satRate"`
	ResonanceStrength float64   `json:"resonanceStrength,omitempty"`
	Dominance         float64   `json:"dominance,omitempty"`
	Timestamp         time.Time `json:"timestamp"`
}

// Metrics represents common computation metrics
type Metrics struct {
	Entropy           float64 `json:"entropy"`
	PlateauDetected   bool    `json:"plateauDetected"`
	Dominance         float64 `json:"dominance,omitempty"`
	ResonanceStrength float64 `json:"resonanceStrength,omitempty"`
	ConvergenceTime   float64 `json:"convergenceTime,omitempty"`
	Iterations        int     `json:"iterations,omitempty"`
}

// SessionInfo represents session metadata
type SessionInfo struct {
	ID        string    `json:"id"`
	UserID    string    `json:"user_id"`
	CreatedAt time.Time `json:"created_at"`
	UpdatedAt time.Time `json:"updated_at"`
	Status    string    `json:"status"`
	ExpiresAt time.Time `json:"expires_at,omitempty"`
}

// WebhookEvent represents a webhook event
type WebhookEvent struct {
	ID        string                 `json:"id"`
	Type      string                 `json:"type"`
	Service   string                 `json:"service"`
	UserID    string                 `json:"user_id"`
	Data      map[string]interface{} `json:"data"`
	Timestamp time.Time              `json:"timestamp"`
}

// Config represents common service configuration
type Config struct {
	Port        int    `json:"port"`
	DatabaseURL string `json:"database_url"`
	RedisURL    string `json:"redis_url"`
	JWTSecret   string `json:"jwt_secret"`
	LogLevel    string `json:"log_level"`
	Environment string `json:"environment"`
	ServiceName string `json:"service_name"`
	MetricsPort int    `json:"metrics_port"`
	HealthPort  int    `json:"health_port"`
}

// NewRequestID generates a new request ID
func NewRequestID() string {
	return uuid.New().String()
}

// NewAPIResponse creates a successful API response
func NewAPIResponse(data interface{}, requestID string) *APIResponse {
	return &APIResponse{
		Success:   true,
		Data:      data,
		RequestID: requestID,
		Timestamp: time.Now(),
	}
}

// NewAPIError creates an error API response
func NewAPIError(code, message, details, requestID string) *APIResponse {
	return &APIResponse{
		Success: false,
		Error: &APIError{
			Code:    code,
			Message: message,
			Details: details,
		},
		RequestID: requestID,
		Timestamp: time.Now(),
	}
}

// EngineState represents the state of a resonance engine
type EngineState struct {
	ID              string                 `json:"id"`
	Type            string                 `json:"type"`
	Status          string                 `json:"status"`
	LastUpdate      time.Time              `json:"last_update"`
	Metrics         map[string]interface{} `json:"metrics"`
	Configuration   map[string]interface{} `json:"configuration"`
	ResonanceLevel  float64                `json:"resonance_level"`
	Coherence       float64                `json:"coherence"`
	EntanglementMap map[string]interface{} `json:"entanglement_map"`
}

// GlobalResonanceState represents the global resonance state of the system
type GlobalResonanceState struct {
	GlobalResonance   float64                 `json:"global_resonance"`
	SystemCoherence   float64                 `json:"system_coherence"`
	UnificationDegree float64                 `json:"unification_degree"`
	SyncTimestamp     time.Time               `json:"sync_timestamp"`
	AggregatedMetrics map[string]interface{}  `json:"aggregated_metrics"`
	GlobalConfig      map[string]interface{}  `json:"global_config"`
	EngineStates      map[string]*EngineState `json:"engine_states"`
	TelemetryHistory  []TelemetryPoint        `json:"telemetry_history"`
	SharedPrimes      []int                   `json:"shared_primes"`
}

// PerformanceMetric represents a performance metric
type PerformanceMetric struct {
	Name         string      `json:"name"`
	CurrentValue float64     `json:"current_value"`
	MinValue     float64     `json:"min_value"`
	MaxValue     float64     `json:"max_value"`
	AverageValue float64     `json:"average_value"`
	Values       []float64   `json:"values"`
	Timestamps   []time.Time `json:"timestamps"`
}
