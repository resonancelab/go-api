package middleware

import (
	"context"
	"fmt"
	"net/http"
	"runtime/debug"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/resonancelab/psizero/shared/types"
)

// ErrorHandlerConfig configures error handling behavior
type ErrorHandlerConfig struct {
	EnableStackTrace     bool          `json:"enable_stack_trace"`
	MaxRequestTimeout    time.Duration `json:"max_request_timeout"`
	EnableCircuitBreaker bool          `json:"enable_circuit_breaker"`
	ErrorThreshold       int           `json:"error_threshold"`
	TimeWindow           time.Duration `json:"time_window"`
}

// CircuitBreakerState tracks circuit breaker status
type CircuitBreakerState struct {
	failures    int
	lastFailure time.Time
	isOpen      bool
}

var (
	defaultConfig = &ErrorHandlerConfig{
		EnableStackTrace:     false,
		MaxRequestTimeout:    30 * time.Second,
		EnableCircuitBreaker: true,
		ErrorThreshold:       10,
		TimeWindow:           1 * time.Minute,
	}
	circuitBreakers = make(map[string]*CircuitBreakerState)
)

// ErrorHandlerMiddleware provides comprehensive error handling
func ErrorHandlerMiddleware(config *ErrorHandlerConfig) gin.HandlerFunc {
	if config == nil {
		config = defaultConfig
	}

	return func(c *gin.Context) {
		// Set request timeout
		ctx, cancel := context.WithTimeout(c.Request.Context(), config.MaxRequestTimeout)
		defer cancel()
		c.Request = c.Request.WithContext(ctx)

		// Circuit breaker check
		if config.EnableCircuitBreaker {
			endpoint := c.Request.Method + ":" + c.FullPath()
			if isCircuitBreakerOpen(endpoint, config) {
				c.JSON(http.StatusServiceUnavailable, types.NewAPIError(
					"CIRCUIT_BREAKER_OPEN",
					"Service temporarily unavailable",
					"Circuit breaker is open due to high error rate",
					c.GetString("request_id"),
				))
				c.Abort()
				return
			}
		}

		// Recovery middleware
		defer func() {
			if err := recover(); err != nil {
				handlePanic(c, err, config)
			}
		}()

		// Execute request
		c.Next()

		// Handle errors after request processing
		if len(c.Errors) > 0 {
			handleRequestErrors(c, config)
		}

		// Update circuit breaker state
		if config.EnableCircuitBreaker {
			endpoint := c.Request.Method + ":" + c.FullPath()
			updateCircuitBreaker(endpoint, c.Writer.Status() >= 500, config)
		}
	}
}

// handlePanic handles panics and converts them to proper error responses
func handlePanic(c *gin.Context, err interface{}, config *ErrorHandlerConfig) {
	requestID := c.GetString("request_id")
	if requestID == "" {
		requestID = "unknown"
	}

	// Log the panic
	stack := debug.Stack()
	fmt.Printf("PANIC recovered: %v\nStack trace:\n%s\n", err, stack)

	// Prepare error details
	details := fmt.Sprintf("Internal server error: %v", err)
	if config.EnableStackTrace {
		details = fmt.Sprintf("Panic: %v\nStack: %s", err, stack)
	}

	// Send error response
	if !c.Writer.Written() {
		c.JSON(http.StatusInternalServerError, types.NewAPIError(
			"INTERNAL_PANIC",
			"Internal server error",
			details,
			requestID,
		))
	}
}

// handleRequestErrors processes accumulated request errors
func handleRequestErrors(c *gin.Context, config *ErrorHandlerConfig) {
	requestID := c.GetString("request_id")
	if requestID == "" {
		requestID = "unknown"
	}

	// Get the last error (most recent)
	lastError := c.Errors.Last()
	if lastError == nil {
		return
	}

	// Don't override if response already written
	if c.Writer.Written() {
		return
	}

	// Determine error type and status code
	statusCode := http.StatusInternalServerError
	errorCode := "INTERNAL_ERROR"
	message := "Internal server error"

	switch lastError.Type {
	case gin.ErrorTypeBind:
		statusCode = http.StatusBadRequest
		errorCode = "VALIDATION_ERROR"
		message = "Request validation failed"
	case gin.ErrorTypePublic:
		statusCode = http.StatusBadRequest
		errorCode = "REQUEST_ERROR"
		message = "Bad request"
	case gin.ErrorTypeRender:
		statusCode = http.StatusInternalServerError
		errorCode = "RENDER_ERROR"
		message = "Response rendering failed"
	}

	// Include stack trace if enabled
	details := lastError.Error()
	if config.EnableStackTrace && lastError.Meta != nil {
		if stackTrace, ok := lastError.Meta.(string); ok {
			details = fmt.Sprintf("%s\nStack: %s", details, stackTrace)
		}
	}

	c.JSON(statusCode, types.NewAPIError(
		errorCode,
		message,
		details,
		requestID,
	))
}

// isCircuitBreakerOpen checks if circuit breaker is open for an endpoint
func isCircuitBreakerOpen(endpoint string, config *ErrorHandlerConfig) bool {
	state, exists := circuitBreakers[endpoint]
	if !exists {
		circuitBreakers[endpoint] = &CircuitBreakerState{}
		return false
	}

	// Reset if time window has passed
	if time.Since(state.lastFailure) > config.TimeWindow {
		state.failures = 0
		state.isOpen = false
	}

	return state.isOpen
}

// updateCircuitBreaker updates circuit breaker state based on response
func updateCircuitBreaker(endpoint string, isError bool, config *ErrorHandlerConfig) {
	state, exists := circuitBreakers[endpoint]
	if !exists {
		state = &CircuitBreakerState{}
		circuitBreakers[endpoint] = state
	}

	if isError {
		state.failures++
		state.lastFailure = time.Now()

		// Open circuit breaker if threshold exceeded
		if state.failures >= config.ErrorThreshold {
			state.isOpen = true
		}
	} else {
		// Reset on successful request
		state.failures = 0
		state.isOpen = false
	}
}

// TimeoutMiddleware adds request timeout handling
func TimeoutMiddleware(timeout time.Duration) gin.HandlerFunc {
	return func(c *gin.Context) {
		ctx, cancel := context.WithTimeout(c.Request.Context(), timeout)
		defer cancel()

		c.Request = c.Request.WithContext(ctx)

		// Channel to signal completion
		done := make(chan bool, 1)

		go func() {
			c.Next()
			done <- true
		}()

		select {
		case <-done:
			// Request completed normally
			return
		case <-ctx.Done():
			// Request timed out
			requestID := c.GetString("request_id")
			c.JSON(http.StatusRequestTimeout, types.NewAPIError(
				"REQUEST_TIMEOUT",
				"Request timeout",
				fmt.Sprintf("Request exceeded timeout of %v", timeout),
				requestID,
			))
			c.Abort()
		}
	}
}

// ValidationMiddleware provides enhanced input validation
func ValidationMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		// Validate content type for POST/PUT requests
		if c.Request.Method == "POST" || c.Request.Method == "PUT" {
			contentType := c.GetHeader("Content-Type")
			if contentType != "application/json" && contentType != "" {
				c.JSON(http.StatusUnsupportedMediaType, types.NewAPIError(
					"UNSUPPORTED_MEDIA_TYPE",
					"Unsupported media type",
					"Only application/json is supported",
					c.GetString("request_id"),
				))
				c.Abort()
				return
			}
		}

		// Validate content length
		if c.Request.ContentLength > 10*1024*1024 { // 10MB limit
			c.JSON(http.StatusRequestEntityTooLarge, types.NewAPIError(
				"REQUEST_TOO_LARGE",
				"Request entity too large",
				"Request body exceeds 10MB limit",
				c.GetString("request_id"),
			))
			c.Abort()
			return
		}

		c.Next()
	}
}

// ResourceCleanupMiddleware ensures proper resource cleanup
func ResourceCleanupMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		// Store cleanup functions
		cleanupFuncs := make([]func(), 0)
		c.Set("cleanup_funcs", &cleanupFuncs)

		defer func() {
			// Execute all cleanup functions
			if funcs, exists := c.Get("cleanup_funcs"); exists {
				if cleanupList, ok := funcs.(*[]func()); ok {
					for _, cleanup := range *cleanupList {
						func() {
							defer func() {
								if r := recover(); r != nil {
									fmt.Printf("Cleanup function panicked: %v\n", r)
								}
							}()
							cleanup()
						}()
					}
				}
			}
		}()

		c.Next()
	}
}

// AddCleanupFunction adds a cleanup function to be executed after request
func AddCleanupFunction(c *gin.Context, cleanup func()) {
	if funcs, exists := c.Get("cleanup_funcs"); exists {
		if cleanupList, ok := funcs.(*[]func()); ok {
			*cleanupList = append(*cleanupList, cleanup)
		}
	}
}
