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
	Step              int     `json:"t"`
	SymbolicEntropy   float64 `json:"S"`
	LyapunovMetric    float64 `json:"L"`
	SatisfactionRate  float64 `json:"satRate"`
	ResonanceStrength float64 `json:"resonanceStrength,omitempty"`
	Dominance         float64 `json:"dominance,omitempty"`
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
	Port         int    `json:"port"`
	DatabaseURL  string `json:"database_url"`
	RedisURL     string `json:"redis_url"`
	JWTSecret    string `json:"jwt_secret"`
	LogLevel     string `json:"log_level"`
	Environment  string `json:"environment"`
	ServiceName  string `json:"service_name"`
	MetricsPort  int    `json:"metrics_port"`
	HealthPort   int    `json:"health_port"`
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