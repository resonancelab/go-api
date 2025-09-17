package router

import (
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/resonancelab/psizero/engines/srs"
	"github.com/resonancelab/psizero/gateway/services"
	"github.com/resonancelab/psizero/shared/types"
)

// SRS API types
type SRSRequest struct {
	Problem string                 `json:"problem" binding:"required" example:"3sat"`
	Spec    map[string]interface{} `json:"spec" binding:"required"`
	Config  *SRSEngineConfig       `json:"config,omitempty"`
}

type SRSEngineConfig struct {
	ParticleCount     int     `json:"particle_count,omitempty" example:"50"`
	MaxIterations     int     `json:"max_iterations,omitempty" example:"5000"`
	PlateauThreshold  float64 `json:"plateau_threshold,omitempty" example:"1e-6"`
	EntropyLambda     float64 `json:"entropy_lambda,omitempty" example:"0.02"`
	ResonanceStrength float64 `json:"resonance_strength,omitempty" example:"0.8"`
	TimeoutSeconds    int     `json:"timeout_seconds,omitempty" example:"300"`
}

type SRSResponse struct {
	Feasible    bool                   `json:"feasible"`
	Certificate *Certificate           `json:"certificate,omitempty"`
	Metrics     *types.Metrics         `json:"metrics"`
	Telemetry   []types.TelemetryPoint `json:"telemetry,omitempty"`
	Timing      *TimingInfo            `json:"timing"`
}

type Certificate struct {
	Assignment []int `json:"assignment"`
}

type TimingInfo struct {
	StartTime  time.Time `json:"start_time"`
	EndTime    time.Time `json:"end_time"`
	Duration   float64   `json:"duration_ms"`
	Iterations int       `json:"iterations"`
}

// SetupSRSRoutes configures SRS service routes with dependency injection
func SetupSRSRoutes(rg *gin.RouterGroup, container *services.ServiceContainer) {
	srsGroup := rg.Group("/srs")
	{
		srsGroup.POST("/solve", func(c *gin.Context) {
			solveProblem(c, container.GetSRSEngine())
		})
		srsGroup.GET("/problems", func(c *gin.Context) {
			listSupportedProblems(c, container.GetSRSEngine())
		})
		srsGroup.GET("/status", func(c *gin.Context) {
			getSRSStatus(c, container.GetSRSEngine())
		})
	}
}

// solveProblem handles SRS problem solving requests
// @Summary Solve NP-complete problems using Symbolic Resonance
// @Description Submit a problem specification for solving using advanced symbolic resonance algorithms
// @Tags SRS
// @Accept json
// @Produce json
// @Param request body SRSRequest true "Problem specification"
// @Success 200 {object} types.APIResponse{data=SRSResponse}
// @Failure 400 {object} types.APIResponse
// @Failure 401 {object} types.APIResponse
// @Failure 500 {object} types.APIResponse
// @Security ApiKeyAuth
// @Security BearerAuth
// @Router /v1/srs/solve [post]
func solveProblem(c *gin.Context, srsEngine *srs.SRSEngine) {
	requestID := c.GetString("request_id")
	var req SRSRequest

	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, types.NewAPIError(
			"SRS_001",
			"Invalid request format",
			err.Error(),
			requestID,
		))
		return
	}

	// Convert API config to SRS engine config
	engineConfig := convertToSRSConfig(req.Config)

	// Solve the problem using the actual SRS engine
	startTime := time.Now()

	solution, telemetry, err := srsEngine.SolveProblem(req.Problem, req.Spec, engineConfig)
	if err != nil {
		c.JSON(http.StatusInternalServerError, types.NewAPIError(
			"SRS_002",
			"Problem solving failed",
			err.Error(),
			requestID,
		))
		return
	}

	endTime := time.Now()
	duration := float64(endTime.Sub(startTime).Nanoseconds()) / 1e6

	// Convert solution to API response format
	response := SRSResponse{
		Feasible: solution.Feasible,
		Metrics: &types.Metrics{
			Entropy:           solution.Entropy,
			PlateauDetected:   solution.Satisfied == solution.Total,
			Dominance:         float64(solution.Satisfied) / float64(solution.Total),
			ResonanceStrength: 1.0 - solution.Energy,
			ConvergenceTime:   solution.ComputeTime * 1000, // Convert to ms
			Iterations:        solution.FoundAt,
		},
		Telemetry: telemetry,
		Timing: &TimingInfo{
			StartTime:  startTime,
			EndTime:    endTime,
			Duration:   duration,
			Iterations: solution.FoundAt,
		},
	}

	// Add certificate if solution found
	if solution.Feasible && len(solution.Assignment) > 0 {
		response.Certificate = &Certificate{
			Assignment: solution.Assignment,
		}
	}

	c.JSON(http.StatusOK, types.NewAPIResponse(response, requestID))
}

// convertToSRSConfig converts API config to SRS engine config
func convertToSRSConfig(apiConfig *SRSEngineConfig) *srs.SRSConfig {
	if apiConfig == nil {
		return srs.DefaultSRSConfig()
	}

	config := srs.DefaultSRSConfig()

	if apiConfig.ParticleCount > 0 {
		config.ParticleCount = apiConfig.ParticleCount
	}
	if apiConfig.MaxIterations > 0 {
		config.MaxIterations = apiConfig.MaxIterations
	}
	if apiConfig.PlateauThreshold > 0 {
		config.PlateauThreshold = apiConfig.PlateauThreshold
	}
	if apiConfig.EntropyLambda > 0 {
		config.EntropyLambda = apiConfig.EntropyLambda
	}
	if apiConfig.ResonanceStrength > 0 {
		config.ResonanceStrength = apiConfig.ResonanceStrength
	}
	if apiConfig.TimeoutSeconds > 0 {
		config.TimeoutSeconds = apiConfig.TimeoutSeconds
	}

	return config
}

// listSupportedProblems returns list of supported problem types
// @Summary List supported problem types
// @Description Get a list of all problem types supported by the SRS service
// @Tags SRS
// @Produce json
// @Success 200 {object} types.APIResponse{data=[]string}
// @Security ApiKeyAuth
// @Security BearerAuth
// @Router /v1/srs/problems [get]
func listSupportedProblems(c *gin.Context, srsEngine *srs.SRSEngine) {
	requestID := c.GetString("request_id")

	// These are the actually supported problem types in the SRS engine
	problems := []string{
		"3sat",
		"ksat",
		"subsetsum",
	}

	c.JSON(http.StatusOK, types.NewAPIResponse(problems, requestID))
}

// getSRSStatus returns service status
// @Summary Get SRS service status
// @Description Check the health and status of the SRS service
// @Tags SRS
// @Produce json
// @Success 200 {object} types.APIResponse
// @Security ApiKeyAuth
// @Security BearerAuth
// @Router /v1/srs/status [get]
func getSRSStatus(c *gin.Context, srsEngine *srs.SRSEngine) {
	requestID := c.GetString("request_id")

	// Get actual status from the SRS engine
	var engineStatus map[string]interface{}
	if srsEngine != nil {
		engineStatus = srsEngine.GetCurrentState()
	} else {
		engineStatus = map[string]interface{}{
			"status": "not_initialized",
		}
	}

	status := map[string]interface{}{
		"service":       "srs",
		"status":        "operational",
		"version":       "1.0.0",
		"engine_status": engineStatus,
	}

	c.JSON(http.StatusOK, types.NewAPIResponse(status, requestID))
}
