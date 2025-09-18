package middleware

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"os"
	"time"

	"github.com/gin-gonic/gin"
)

// LoggingConfig configures request logging behavior
type LoggingConfig struct {
	EnableRequestBody  bool     `json:"enable_request_body"`
	EnableResponseBody bool     `json:"enable_response_body"`
	MaxBodySize        int      `json:"max_body_size"`
	SkipPaths          []string `json:"skip_paths"`
}

// RequestLogEntry represents a logged request
type RequestLogEntry struct {
	RequestID    string        `json:"request_id"`
	Method       string        `json:"method"`
	Path         string        `json:"path"`
	StatusCode   int           `json:"status_code"`
	Duration     time.Duration `json:"duration"`
	ClientIP     string        `json:"client_ip"`
	UserAgent    string        `json:"user_agent"`
	RequestSize  int64         `json:"request_size"`
	ResponseSize int           `json:"response_size"`
	RequestBody  string        `json:"request_body,omitempty"`
	ResponseBody string        `json:"response_body,omitempty"`
	UserID       string        `json:"user_id,omitempty"`
	AuthType     string        `json:"auth_type,omitempty"`
	Timestamp    time.Time     `json:"timestamp"`
	Error        string        `json:"error,omitempty"`
}

// bodyLogWriter wraps gin.ResponseWriter to capture response body
type bodyLogWriter struct {
	gin.ResponseWriter
	body *bytes.Buffer
}

func (w bodyLogWriter) Write(b []byte) (int, error) {
	w.body.Write(b)
	return w.ResponseWriter.Write(b)
}

// RequestLoggingMiddleware logs all requests with configurable detail level
func RequestLoggingMiddleware(config *LoggingConfig) gin.HandlerFunc {
	if config == nil {
		config = &LoggingConfig{
			EnableRequestBody:  false,
			EnableResponseBody: false,
			MaxBodySize:        1024 * 4, // 4KB
			SkipPaths:          []string{"/health", "/metrics"},
		}
	}

	return func(c *gin.Context) {
		// Skip logging for certain paths
		for _, skipPath := range config.SkipPaths {
			if c.Request.URL.Path == skipPath {
				c.Next()
				return
			}
		}

		start := time.Now()
		requestID := c.GetString("request_id")
		if requestID == "" {
			requestID = "unknown"
		}

		// Capture request body if enabled
		var requestBody string
		if config.EnableRequestBody && c.Request.Body != nil {
			bodyBytes, _ := io.ReadAll(c.Request.Body)
			if len(bodyBytes) <= config.MaxBodySize {
				requestBody = string(bodyBytes)
			} else {
				requestBody = fmt.Sprintf("[TRUNCATED - %d bytes]", len(bodyBytes))
			}
			// Restore the body for downstream handlers
			c.Request.Body = io.NopCloser(bytes.NewBuffer(bodyBytes))
		}

		// Set up response body capture if enabled
		var responseBody string
		var blw *bodyLogWriter
		if config.EnableResponseBody {
			blw = &bodyLogWriter{
				ResponseWriter: c.Writer,
				body:           bytes.NewBufferString(""),
			}
			c.Writer = blw
		}

		// Process request
		c.Next()

		// Capture response body if enabled
		if config.EnableResponseBody && blw != nil {
			if blw.body.Len() <= config.MaxBodySize {
				responseBody = blw.body.String()
			} else {
				responseBody = fmt.Sprintf("[TRUNCATED - %d bytes]", blw.body.Len())
			}
		}

		// Get user info from context if available
		userID, _ := c.Get("user_id")
		authType, _ := c.Get("auth_type")

		// Get error if any
		var errorMsg string
		if len(c.Errors) > 0 {
			errorMsg = c.Errors.Last().Error()
		}

		// Create log entry
		logEntry := RequestLogEntry{
			RequestID:    requestID,
			Method:       c.Request.Method,
			Path:         c.Request.URL.Path,
			StatusCode:   c.Writer.Status(),
			Duration:     time.Since(start),
			ClientIP:     c.ClientIP(),
			UserAgent:    c.Request.UserAgent(),
			RequestSize:  c.Request.ContentLength,
			ResponseSize: c.Writer.Size(),
			RequestBody:  requestBody,
			ResponseBody: responseBody,
			Timestamp:    start,
			Error:        errorMsg,
		}

		if userID != nil {
			logEntry.UserID = fmt.Sprintf("%v", userID)
		}
		if authType != nil {
			logEntry.AuthType = fmt.Sprintf("%v", authType)
		}

		// Log the request
		logRequest(logEntry)
	}
}

// logRequest outputs the request log entry in structured format
func logRequest(entry RequestLogEntry) {
	// Determine log format based on environment
	logFormat := os.Getenv("LOG_FORMAT")
	if logFormat == "" {
		logFormat = "text" // Default to human-readable
	}

	switch logFormat {
	case "json":
		logRequestJSON(entry)
	default:
		logRequestText(entry)
	}

	// Send to monitoring system if configured
	sendToMonitoring(entry)
}

// logRequestText outputs the request log entry in human-readable format
func logRequestText(entry RequestLogEntry) {
	// Format log message
	logMsg := fmt.Sprintf("[%s] %s %s - %d - %v - %s - %s",
		entry.Timestamp.Format("2006-01-02 15:04:05"),
		entry.Method,
		entry.Path,
		entry.StatusCode,
		entry.Duration,
		entry.ClientIP,
		entry.RequestID,
	)

	if entry.UserID != "" {
		logMsg += fmt.Sprintf(" - User: %s (%s)", entry.UserID, entry.AuthType)
	}

	if entry.Error != "" {
		logMsg += fmt.Sprintf(" - Error: %s", entry.Error)
	}

	// Print to stdout
	fmt.Println(logMsg)
}

// logRequestJSON outputs the request log entry in JSON format
func logRequestJSON(entry RequestLogEntry) {
	// Convert to JSON
	jsonData, err := json.Marshal(entry)
	if err != nil {
		// Fallback to text logging if JSON marshaling fails
		log.Printf("Failed to marshal log entry to JSON: %v", err)
		logRequestText(entry)
		return
	}

	// Print JSON to stdout
	fmt.Println(string(jsonData))
}

// sendToMonitoring sends log entry to monitoring system
func sendToMonitoring(entry RequestLogEntry) {
	// TODO: Implement monitoring system integration
	// This could be:
	// - Send to GCP Cloud Logging
	// - Send to AWS CloudWatch
	// - Send to DataDog
	// - Send to custom monitoring service
	// - Store in time-series database

	monitoringEnabled := os.Getenv("MONITORING_ENABLED")
	if monitoringEnabled != "true" {
		return
	}

	monitoringService := os.Getenv("MONITORING_SERVICE")
	switch monitoringService {
	case "gcp":
		sendToGCPMonitoring(entry)
	case "aws":
		sendToAWSMonitoring(entry)
	case "datadog":
		sendToDataDogMonitoring(entry)
	default:
		// Default: just log that monitoring is enabled but service not configured
		if monitoringEnabled == "true" {
			log.Printf("Monitoring enabled but service '%s' not supported", monitoringService)
		}
	}
}

// sendToGCPMonitoring sends log entry to Google Cloud Logging
func sendToGCPMonitoring(entry RequestLogEntry) {
	// TODO: Implement GCP Cloud Logging integration
	// This would use the Google Cloud Logging client library
	log.Printf("GCP Monitoring: Would send log entry for request %s", entry.RequestID)
}

// sendToAWSMonitoring sends log entry to AWS CloudWatch
func sendToAWSMonitoring(entry RequestLogEntry) {
	// TODO: Implement AWS CloudWatch integration
	// This would use the AWS SDK for Go
	log.Printf("AWS Monitoring: Would send log entry for request %s", entry.RequestID)
}

// sendToDataDogMonitoring sends log entry to DataDog
func sendToDataDogMonitoring(entry RequestLogEntry) {
	// TODO: Implement DataDog integration
	// This would use the DataDog Go client
	log.Printf("DataDog Monitoring: Would send log entry for request %s", entry.RequestID)
}

// MetricsMiddleware collects request metrics for monitoring
func MetricsMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		start := time.Now()

		// Process request
		c.Next()

		duration := time.Since(start)
		statusCode := c.Writer.Status()
		method := c.Request.Method
		path := c.FullPath()
		requestID := c.GetString("request_id")

		// Create metrics data
		metrics := RequestMetrics{
			RequestID:    requestID,
			Method:       method,
			Path:         path,
			StatusCode:   statusCode,
			Duration:     duration,
			ClientIP:     c.ClientIP(),
			UserAgent:    c.Request.UserAgent(),
			RequestSize:  c.Request.ContentLength,
			ResponseSize: c.Writer.Size(),
			Timestamp:    start,
			IsError:      statusCode >= 400,
		}

		// Get user info if available
		if userID, exists := c.Get("user_id"); exists {
			metrics.UserID = userID.(string)
		}
		if authType, exists := c.Get("auth_type"); exists {
			metrics.AuthType = authType.(string)
		}

		// Send metrics to monitoring system
		sendMetricsToMonitoring(metrics)

		// Log metrics in structured format
		logMetrics(metrics)
	}
}

// DebugMiddleware provides debug information in development
func DebugMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		if gin.Mode() != gin.DebugMode {
			c.Next()
			return
		}

		fmt.Printf("DEBUG: Request Headers:\n")
		for name, values := range c.Request.Header {
			for _, value := range values {
				fmt.Printf("  %s: %s\n", name, value)
			}
		}

		fmt.Printf("DEBUG: Query Parameters:\n")
		for key, values := range c.Request.URL.Query() {
			for _, value := range values {
				fmt.Printf("  %s: %s\n", key, value)
			}
		}

		c.Next()

		fmt.Printf("DEBUG: Response Headers:\n")
		for name, values := range c.Writer.Header() {
			for _, value := range values {
				fmt.Printf("  %s: %s\n", name, value)
			}
		}
	}
}

// RequestMetrics represents metrics data for a request
type RequestMetrics struct {
	RequestID    string        `json:"request_id"`
	Method       string        `json:"method"`
	Path         string        `json:"path"`
	StatusCode   int           `json:"status_code"`
	Duration     time.Duration `json:"duration"`
	ClientIP     string        `json:"client_ip"`
	UserAgent    string        `json:"user_agent"`
	RequestSize  int64         `json:"request_size"`
	ResponseSize int           `json:"response_size"`
	UserID       string        `json:"user_id,omitempty"`
	AuthType     string        `json:"auth_type,omitempty"`
	Timestamp    time.Time     `json:"timestamp"`
	IsError      bool          `json:"is_error"`
}

// sendMetricsToMonitoring sends metrics to monitoring system
func sendMetricsToMonitoring(metrics RequestMetrics) {
	// TODO: Implement monitoring system integration
	// This could be:
	// - Send to Prometheus
	// - Send to GCP Cloud Monitoring
	// - Send to AWS CloudWatch Metrics
	// - Send to DataDog metrics

	monitoringEnabled := os.Getenv("METRICS_ENABLED")
	if monitoringEnabled != "true" {
		return
	}

	metricsService := os.Getenv("METRICS_SERVICE")
	switch metricsService {
	case "prometheus":
		sendToPrometheus(metrics)
	case "gcp":
		sendToGCPMetrics(metrics)
	case "aws":
		sendToAWSMetrics(metrics)
	default:
		// Default: just log that metrics are enabled but service not configured
		if monitoringEnabled == "true" {
			log.Printf("Metrics enabled but service '%s' not supported", metricsService)
		}
	}
}

// sendToPrometheus sends metrics to Prometheus
func sendToPrometheus(metrics RequestMetrics) {
	// TODO: Implement Prometheus metrics integration
	// This would use the Prometheus client library for Go
	log.Printf("Prometheus: Would send metrics for request %s", metrics.RequestID)
}

// sendToGCPMetrics sends metrics to Google Cloud Monitoring
func sendToGCPMetrics(metrics RequestMetrics) {
	// TODO: Implement GCP Cloud Monitoring integration
	log.Printf("GCP Metrics: Would send metrics for request %s", metrics.RequestID)
}

// sendToAWSMetrics sends metrics to AWS CloudWatch
func sendToAWSMetrics(metrics RequestMetrics) {
	// TODO: Implement AWS CloudWatch Metrics integration
	log.Printf("AWS Metrics: Would send metrics for request %s", metrics.RequestID)
}

// logMetrics logs metrics in structured format
func logMetrics(metrics RequestMetrics) {
	// Determine log format
	logFormat := os.Getenv("LOG_FORMAT")
	if logFormat == "" {
		logFormat = "text"
	}

	switch logFormat {
	case "json":
		logMetricsJSON(metrics)
	default:
		logMetricsText(metrics)
	}
}

// logMetricsText logs metrics in human-readable format
func logMetricsText(metrics RequestMetrics) {
	status := "OK"
	if metrics.IsError {
		status = "ERROR"
	}

	logMsg := fmt.Sprintf("METRIC: %s %s - %d - %v - %s - %s",
		metrics.Method,
		metrics.Path,
		metrics.StatusCode,
		metrics.Duration,
		status,
		metrics.RequestID,
	)

	if metrics.UserID != "" {
		logMsg += fmt.Sprintf(" - User: %s", metrics.UserID)
	}

	fmt.Println(logMsg)
}

// logMetricsJSON logs metrics in JSON format
func logMetricsJSON(metrics RequestMetrics) {
	jsonData, err := json.Marshal(metrics)
	if err != nil {
		log.Printf("Failed to marshal metrics to JSON: %v", err)
		logMetricsText(metrics)
		return
	}

	fmt.Println(string(jsonData))
}
