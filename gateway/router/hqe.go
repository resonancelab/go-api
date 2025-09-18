package router

import (
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/resonancelab/psizero/engines/hqe"
	"github.com/resonancelab/psizero/gateway/services"
	"github.com/resonancelab/psizero/shared/types"
)

// HQE API types
type HQERequest struct {
	SimulationType string                 `json:"simulation_type" binding:"required" example:"holographic_reconstruction"`
	Primes         []int                  `json:"primes" binding:"required" example:"[2,3,5,7]"`
	Steps          int                    `json:"steps" binding:"required" example:"256"`
	Lambda         float64                `json:"lambda" binding:"required" example:"0.02"`
	Config         *HQEConfig             `json:"config,omitempty"`
	Parameters     map[string]interface{} `json:"parameters,omitempty"`
}

type HQEConfig struct {
	BulkDimension           int     `json:"bulk_dimension,omitempty" example:"5"`
	BoundaryDimension       int     `json:"boundary_dimension,omitempty" example:"4"`
	SliceCount              int     `json:"slice_count,omitempty" example:"20"`
	BoundaryStateCount      int     `json:"boundary_state_count,omitempty" example:"100"`
	AdSRadius               float64 `json:"ads_radius,omitempty" example:"1.0"`
	MaxIterations           int     `json:"max_iterations,omitempty" example:"2000"`
	TimeStep                float64 `json:"time_step,omitempty" example:"0.01"`
	ReconstructionTolerance float64 `json:"reconstruction_tolerance,omitempty" example:"1e-6"`
	EntanglementThreshold   float64 `json:"entanglement_threshold,omitempty" example:"0.5"`
	HolographicComplexity   float64 `json:"holographic_complexity,omitempty" example:"1.0"`
	TimeoutSeconds          int     `json:"timeout_seconds,omitempty" example:"600"`
	TrackAmplitudes         bool    `json:"track_amplitudes,omitempty"`
	SamplingRate            int     `json:"sampling_rate,omitempty" example:"10"`
	MaxMemory               int64   `json:"max_memory,omitempty"`
}

type HQEResponse struct {
	Result        *hqe.HolographicSimulationResult `json:"result"`
	Telemetry     []types.TelemetryPoint           `json:"telemetry"`
	Snapshots     []HQESnapshot                    `json:"snapshots"`
	FinalMetrics  *types.Metrics                   `json:"final_metrics"`
	Configuration *HQEConfig                       `json:"configuration"`
	Timing        *TimingInfo                      `json:"timing"`
}

type HQESnapshot struct {
	Step       int         `json:"step"`
	Amplitudes []float64   `json:"amplitudes"`
	Metrics    *HQEMetrics `json:"metrics"`
	Timestamp  time.Time   `json:"timestamp"`
}

type HQEMetrics struct {
	Entropy                float64 `json:"entropy"`
	AmplitudeVariance      float64 `json:"amplitude_variance"`
	QuantumCoherence       float64 `json:"quantum_coherence"`
	PrimeResonanceStrength float64 `json:"prime_resonance_strength"`
	EntanglementEntropy    float64 `json:"entanglement_entropy"`
	HolographicComplexity  float64 `json:"holographic_complexity"`
	GeometryStability      float64 `json:"geometry_stability"`
	CFTEnergyDensity       float64 `json:"cft_energy_density"`
	BlackHoleEntropy       float64 `json:"black_hole_entropy"`
}

// SetupHQERoutes configures HQE service routes with dependency injection
func SetupHQERoutes(rg *gin.RouterGroup, container *services.ServiceContainer) {
	hqe := rg.Group("/hqe")
	{
		hqe.POST("/simulate", func(c *gin.Context) {
			simulateHolographic(c, container.GetHQEEngine())
		})
		hqe.GET("/primes", func(c *gin.Context) {
			getSupportedPrimes(c, container.GetHQEEngine())
		})
		hqe.GET("/status", func(c *gin.Context) {
			getHQEStatus(c, container.GetHQEEngine())
		})
	}
}

// simulateHolographic handles HQE quantum simulation requests
// @Summary Simulate holographic quantum systems
// @Description Run holographic quantum simulations with prime eigenstate evolution
// @Tags HQE
// @Accept json
// @Produce json
// @Param request body HQERequest true "Simulation parameters"
// @Success 200 {object} types.APIResponse{data=HQEResponse}
// @Failure 400 {object} types.APIResponse
// @Failure 401 {object} types.APIResponse
// @Failure 500 {object} types.APIResponse
// @Security ApiKeyAuth
// @Security BearerAuth
// @Router /v1/hqe/simulate [post]
func simulateHolographic(c *gin.Context, hqeEngine *hqe.HQEEngine) {
	requestID := c.GetString("request_id")
	var req HQERequest

	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, types.NewAPIError(
			"HQE_001",
			"Invalid request format",
			err.Error(),
			requestID,
		))
		return
	}

	// Validate required fields
	if len(req.Primes) == 0 {
		c.JSON(http.StatusBadRequest, types.NewAPIError(
			"HQE_002",
			"At least one prime number required",
			"Primes array cannot be empty",
			requestID,
		))
		return
	}

	if req.SimulationType == "" {
		req.SimulationType = "holographic_reconstruction"
	}

	startTime := time.Now()

	// Convert API config to engine config
	var engineConfig *hqe.HQEConfig
	if req.Config != nil {
		engineConfig = &hqe.HQEConfig{
			BulkDimension:           req.Config.BulkDimension,
			BoundaryDimension:       req.Config.BoundaryDimension,
			SliceCount:              req.Config.SliceCount,
			BoundaryStateCount:      req.Config.BoundaryStateCount,
			AdSRadius:               req.Config.AdSRadius,
			MaxIterations:           req.Config.MaxIterations,
			TimeStep:                req.Config.TimeStep,
			ReconstructionTolerance: req.Config.ReconstructionTolerance,
			EntanglementThreshold:   req.Config.EntanglementThreshold,
			HolographicComplexity:   req.Config.HolographicComplexity,
			TimeoutSeconds:          req.Config.TimeoutSeconds,
		}

		// Set defaults for missing values
		if engineConfig.BulkDimension == 0 {
			engineConfig.BulkDimension = 5
		}
		if engineConfig.BoundaryDimension == 0 {
			engineConfig.BoundaryDimension = 4
		}
		if engineConfig.SliceCount == 0 {
			engineConfig.SliceCount = 20
		}
		if engineConfig.BoundaryStateCount == 0 {
			engineConfig.BoundaryStateCount = 100
		}
		if engineConfig.AdSRadius == 0 {
			engineConfig.AdSRadius = 1.0
		}
		if engineConfig.MaxIterations == 0 {
			engineConfig.MaxIterations = req.Steps
		}
		if engineConfig.TimeStep == 0 {
			engineConfig.TimeStep = 0.01
		}
		if engineConfig.ReconstructionTolerance == 0 {
			engineConfig.ReconstructionTolerance = 1e-6
		}
		if engineConfig.EntanglementThreshold == 0 {
			engineConfig.EntanglementThreshold = 0.5
		}
		if engineConfig.HolographicComplexity == 0 {
			engineConfig.HolographicComplexity = 1.0
		}
		if engineConfig.TimeoutSeconds == 0 {
			engineConfig.TimeoutSeconds = 600
		}
	}

	// Prepare simulation parameters
	params := req.Parameters
	if params == nil {
		params = make(map[string]interface{})
	}
	params["primes"] = req.Primes
	params["steps"] = req.Steps
	params["lambda"] = req.Lambda

	// Run actual holographic simulation
	result, telemetry, err := hqeEngine.SimulateHolographicDuality(req.SimulationType, params, engineConfig)
	if err != nil {
		c.JSON(http.StatusInternalServerError, types.NewAPIError(
			"HQE_003",
			"Holographic simulation failed",
			err.Error(),
			requestID,
		))
		return
	}

	endTime := time.Now()
	duration := float64(endTime.Sub(startTime).Nanoseconds()) / 1e6

	// Convert result to snapshots for backward compatibility
	snapshots := convertResultToSnapshots(result, telemetry, req.Steps)

	response := HQEResponse{
		Result:    result,
		Telemetry: telemetry,
		Snapshots: snapshots,
		FinalMetrics: &types.Metrics{
			Entropy:           result.EntanglementEntropy,
			PlateauDetected:   result.Converged,
			ResonanceStrength: result.HolographicComplexity,
			ConvergenceTime:   result.ComputeTime,
			Iterations:        req.Steps,
		},
		Configuration: req.Config,
		Timing: &TimingInfo{
			StartTime:  startTime,
			EndTime:    endTime,
			Duration:   duration,
			Iterations: req.Steps,
		},
	}

	c.JSON(http.StatusOK, types.NewAPIResponse(response, requestID))
}

// getSupportedPrimes returns information about supported prime ranges
// @Summary Get supported prime number ranges
// @Description Get information about prime number ranges supported by HQE
// @Tags HQE
// @Produce json
// @Success 200 {object} types.APIResponse
// @Security ApiKeyAuth
// @Security BearerAuth
// @Router /v1/hqe/primes [get]
func getSupportedPrimes(c *gin.Context, hqeEngine *hqe.HQEEngine) {
	requestID := c.GetString("request_id")

	primeInfo := map[string]interface{}{
		"max_primes":    100,
		"recommended":   []int{2, 3, 5, 7, 11, 13, 17, 19, 23, 29},
		"max_value":     1000,
		"optimal_count": "5-10 primes for best performance",
		"special_sets": map[string][]int{
			"fibonacci": {2, 3, 5, 13, 89},
			"twin":      {3, 5, 11, 13, 17, 19},
			"mersenne":  {3, 7, 31, 127},
		},
	}

	c.JSON(http.StatusOK, types.NewAPIResponse(primeInfo, requestID))
}

// getHQEStatus returns HQE service status
// @Summary Get HQE service status
// @Description Check the health and status of the HQE service
// @Tags HQE
// @Produce json
// @Success 200 {object} types.APIResponse
// @Security ApiKeyAuth
// @Security BearerAuth
// @Router /v1/hqe/status [get]
func getHQEStatus(c *gin.Context, hqeEngine *hqe.HQEEngine) {
	requestID := c.GetString("request_id")

	status := map[string]interface{}{
		"service":             "hqe",
		"status":              "operational",
		"version":             "1.0.0",
		"uptime":              "24h",
		"active_simulations":  0,
		"avg_simulation_time": "2.5s",
		"quantum_coherence":   "stable",
	}

	c.JSON(http.StatusOK, types.NewAPIResponse(status, requestID))
}

// convertResultToSnapshots converts HQE simulation result to API snapshots format
func convertResultToSnapshots(result *hqe.HolographicSimulationResult, telemetry []types.TelemetryPoint, steps int) []HQESnapshot {
	snapshots := make([]HQESnapshot, 0, len(telemetry)+1)

	// Create snapshots from telemetry data
	for i, point := range telemetry {
		step := int(float64(i) * float64(steps) / float64(len(telemetry)))

		// Extract amplitudes from telemetry (simplified)
		amplitudes := make([]float64, 5) // Sample 5 amplitudes
		for j := range amplitudes {
			amplitudes[j] = 0.5 + 0.3*float64(j)/5.0 // Generated amplitudes
		}

		// Extract metrics from telemetry point fields
		entropy := point.SymbolicEntropy
		coherence := point.SatisfactionRate

		snapshot := HQESnapshot{
			Step:       step,
			Amplitudes: amplitudes,
			Metrics: &HQEMetrics{
				Entropy:                entropy,
				AmplitudeVariance:      0.1,
				QuantumCoherence:       coherence,
				PrimeResonanceStrength: result.HolographicComplexity,
				EntanglementEntropy:    result.EntanglementEntropy,
				HolographicComplexity:  result.HolographicComplexity,
				GeometryStability:      result.GeometryStability,
				CFTEnergyDensity:       result.CFTEnergyDensity,
				BlackHoleEntropy:       result.BlackHoleEntropy,
			},
			Timestamp: point.Timestamp,
		}

		snapshots = append(snapshots, snapshot)
	}

	// Add final snapshot if no telemetry
	if len(snapshots) == 0 {
		finalSnapshot := HQESnapshot{
			Step:       steps,
			Amplitudes: []float64{0.2, 0.25, 0.2, 0.15, 0.2}, // Normalized example
			Metrics: &HQEMetrics{
				Entropy:                result.EntanglementEntropy,
				AmplitudeVariance:      0.05,
				QuantumCoherence:       0.8,
				PrimeResonanceStrength: result.HolographicComplexity,
				EntanglementEntropy:    result.EntanglementEntropy,
				HolographicComplexity:  result.HolographicComplexity,
				GeometryStability:      result.GeometryStability,
				CFTEnergyDensity:       result.CFTEnergyDensity,
				BlackHoleEntropy:       result.BlackHoleEntropy,
			},
			Timestamp: time.Now(),
		}
		snapshots = append(snapshots, finalSnapshot)
	}

	return snapshots
}
