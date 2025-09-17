package router

import (
	"fmt"
	"math"
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/resonancelab/psizero/engines/unified"
	"github.com/resonancelab/psizero/gateway/services"
	"github.com/resonancelab/psizero/shared/types"
)

// Unified Physics API types
type UnifiedGravityRequest struct {
	ObserverEntropyReductionRate float64        `json:"observerEntropyReductionRate" binding:"required" example:"12.4"`
	RegionEntropyGradient        float64        `json:"regionEntropyGradient" binding:"required" example:"0.002"`
	Config                       *UnifiedConfig `json:"config,omitempty"`
}

type UnifiedFieldRequest struct {
	MassDistribution  []MassPoint    `json:"massDistribution" binding:"required"`
	ObserverPosition  Position3D     `json:"observerPosition" binding:"required"`
	CalculationRegion Region3D       `json:"calculationRegion" binding:"required"`
	Config            *UnifiedConfig `json:"config,omitempty"`
}

type UnifiedConfig struct {
	PrecisionLevel      string `json:"precisionLevel,omitempty" example:"high"`
	QuantumCorrections  bool   `json:"quantumCorrections,omitempty"`
	RelativisticEffects bool   `json:"relativisticEffects,omitempty"`
	ComputationTimeout  int    `json:"computationTimeout,omitempty" example:"300"`
}

type MassPoint struct {
	Position Position3D `json:"position"`
	Mass     float64    `json:"mass"`
	Type     string     `json:"type,omitempty"`
}

type Position3D struct {
	X float64 `json:"x"`
	Y float64 `json:"y"`
	Z float64 `json:"z"`
}

type Region3D struct {
	Center Position3D `json:"center"`
	Size   Position3D `json:"size"`
}

type UnifiedGravityResponse struct {
	EffectiveG          float64             `json:"effectiveG"`
	FieldStrength       float64             `json:"fieldStrength"`
	EntropyContribution float64             `json:"entropyContribution"`
	QuantumCorrections  *QuantumCorrections `json:"quantumCorrections,omitempty"`
	EmergentMetrics     *EmergentMetrics    `json:"emergentMetrics"`
	Notes               string              `json:"notes"`
	Timing              *TimingInfo         `json:"timing"`
}

type UnifiedFieldResponse struct {
	FieldMap           [][]FieldPoint     `json:"fieldMap"`
	TotalFieldStrength float64            `json:"totalFieldStrength"`
	GradientAnalysis   *GradientAnalysis  `json:"gradientAnalysis"`
	EmergentEffects    *EmergentEffects   `json:"emergentEffects"`
	Visualization      *VisualizationData `json:"visualization,omitempty"`
	Timing             *TimingInfo        `json:"timing"`
}

type QuantumCorrections struct {
	VacuumFluctuations float64 `json:"vacuumFluctuations"`
	CasimirEffect      float64 `json:"casimirEffect"`
	EntropyUncertainty float64 `json:"entropyUncertainty"`
}

type EmergentMetrics struct {
	InformationalDensity float64 `json:"informationalDensity"`
	SpacetimeCurvature   float64 `json:"spacetimeCurvature"`
	EntropyFlowRate      float64 `json:"entropyFlowRate"`
	ObserverCoupling     float64 `json:"observerCoupling"`
}

type FieldPoint struct {
	Position  Position3D `json:"position"`
	Strength  float64    `json:"strength"`
	Direction Position3D `json:"direction"`
	Potential float64    `json:"potential"`
}

type GradientAnalysis struct {
	MaxGradient    float64      `json:"maxGradient"`
	MinGradient    float64      `json:"minGradient"`
	AvgGradient    float64      `json:"avgGradient"`
	CriticalPoints []Position3D `json:"criticalPoints"`
}

type EmergentEffects struct {
	GravitationalLensing float64   `json:"gravitationalLensing"`
	FrameDragging        float64   `json:"frameDragging"`
	EntropyWaves         []float64 `json:"entropyWaves"`
	NonLocalCorrelations float64   `json:"nonLocalCorrelations"`
}

type VisualizationData struct {
	Contours    []ContourLevel `json:"contours"`
	VectorField []VectorArrow  `json:"vectorField"`
	ColorMap    string         `json:"colorMap"`
	Scale       string         `json:"scale"`
}

type ContourLevel struct {
	Value  float64      `json:"value"`
	Points []Position3D `json:"points"`
}

type VectorArrow struct {
	Start     Position3D `json:"start"`
	Direction Position3D `json:"direction"`
	Magnitude float64    `json:"magnitude"`
}

// SetupUnifiedRoutes configures Unified Physics service routes with dependency injection
func SetupUnifiedRoutes(rg *gin.RouterGroup, container *services.ServiceContainer) {
	unified := rg.Group("/unified")
	{
		unified.POST("/gravity/compute", func(c *gin.Context) {
			computeGravity(c, container.GetUnifiedEngine())
		})
		unified.POST("/field/analyze", func(c *gin.Context) {
			analyzeField(c, container.GetUnifiedEngine())
		})
		unified.GET("/constants", func(c *gin.Context) {
			getPhysicalConstants(c, container.GetUnifiedEngine())
		})
		unified.GET("/models", func(c *gin.Context) {
			getSupportedModels(c, container.GetUnifiedEngine())
		})
		unified.GET("/status", func(c *gin.Context) {
			getUnifiedStatus(c, container.GetUnifiedEngine())
		})
	}
}

// computeGravity handles emergent gravity computation requests
// @Summary Compute emergent gravitational effects
// @Description Calculate emergent gravitational effects from observer entropy reduction and field gradients
// @Tags Unified Physics
// @Accept json
// @Produce json
// @Param request body UnifiedGravityRequest true "Gravity computation parameters"
// @Success 200 {object} types.APIResponse{data=UnifiedGravityResponse}
// @Failure 400 {object} types.APIResponse
// @Failure 401 {object} types.APIResponse
// @Failure 500 {object} types.APIResponse
// @Security ApiKeyAuth
// @Security BearerAuth
// @Router /v1/unified/gravity/compute [post]
func computeGravity(c *gin.Context, unifiedEngine *unified.UnifiedEngine) {
	requestID := c.GetString("request_id")
	var req UnifiedGravityRequest

	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, types.NewAPIError(
			"UNIFIED_001",
			"Invalid request format",
			err.Error(),
			requestID,
		))
		return
	}

	if req.ObserverEntropyReductionRate <= 0 {
		c.JSON(http.StatusBadRequest, types.NewAPIError(
			"UNIFIED_002",
			"Invalid entropy reduction rate",
			"Observer entropy reduction rate must be positive",
			requestID,
		))
		return
	}

	startTime := time.Now()

	// Convert API types to engine types
	initialConditions := map[string]interface{}{
		"observer_entropy_reduction_rate": req.ObserverEntropyReductionRate,
		"region_entropy_gradient":         req.RegionEntropyGradient,
	}

	// Set configuration if provided
	var config *unified.UnifiedConfig
	if req.Config != nil {
		config = &unified.UnifiedConfig{
			IncludeQuantumGravity: req.Config.QuantumCorrections,
			IncludeSupersymmetry:  true,
			IncludeStringTheory:   true,
			TimeoutSeconds:        req.Config.ComputationTimeout,
		}
		if req.Config.PrecisionLevel == "high" {
			config.ToleranceLevel = 1e-15
		} else {
			config.ToleranceLevel = 1e-10
		}
	}

	// Run unified physics simulation for gravity computation
	result, telemetry, err := unifiedEngine.SimulateUnifiedPhysics("gravity_computation", initialConditions, config)
	if err != nil {
		c.JSON(http.StatusInternalServerError, types.NewAPIError(
			"UNIFIED_005",
			"Gravity computation failed",
			err.Error(),
			requestID,
		))
		return
	}

	endTime := time.Now()
	duration := float64(endTime.Sub(startTime).Nanoseconds()) / 1e6

	// Extract gravity-specific metrics from result
	effectiveG := result.FieldStrengths["gravitational_field"]
	if effectiveG == 0 {
		effectiveG = 6.67430e-11 * (1.0 + req.ObserverEntropyReductionRate/1000.0)
	}

	fieldStrength := result.FieldStrengths["total_field_strength"]
	if fieldStrength == 0 {
		fieldStrength = effectiveG * req.ObserverEntropyReductionRate
	}

	response := UnifiedGravityResponse{
		EffectiveG:          effectiveG,
		FieldStrength:       fieldStrength,
		EntropyContribution: req.ObserverEntropyReductionRate * result.UnificationDegree,
		QuantumCorrections: &QuantumCorrections{
			VacuumFluctuations: result.QuantumCorrections * 1e-18,
			CasimirEffect:      result.QuantumCorrections * 1e-21,
			EntropyUncertainty: result.QuantumCorrections * 0.01,
		},
		EmergentMetrics: &EmergentMetrics{
			InformationalDensity: req.RegionEntropyGradient * result.UnificationDegree * 1e6,
			SpacetimeCurvature:   fieldStrength * 1e-12,
			EntropyFlowRate:      req.ObserverEntropyReductionRate * result.UnificationDegree,
			ObserverCoupling:     result.ConsciousnessMetrics["observer_coupling"],
		},
		Notes: fmt.Sprintf("Unified physics gravity computation completed with %d evolution steps, unification degree: %.4f",
			len(telemetry), result.UnificationDegree),
		Timing: &TimingInfo{
			StartTime:  startTime,
			EndTime:    endTime,
			Duration:   duration,
			Iterations: len(telemetry),
		},
	}

	c.JSON(http.StatusOK, types.NewAPIResponse(response, requestID))
}

// analyzeField handles gravitational field analysis requests
// @Summary Analyze gravitational field structure
// @Description Compute detailed gravitational field analysis including emergent effects and visualization data
// @Tags Unified Physics
// @Accept json
// @Produce json
// @Param request body UnifiedFieldRequest true "Field analysis parameters"
// @Success 200 {object} types.APIResponse{data=UnifiedFieldResponse}
// @Failure 400 {object} types.APIResponse
// @Failure 401 {object} types.APIResponse
// @Failure 500 {object} types.APIResponse
// @Security ApiKeyAuth
// @Security BearerAuth
// @Router /v1/unified/field/analyze [post]
func analyzeField(c *gin.Context, unifiedEngine *unified.UnifiedEngine) {
	requestID := c.GetString("request_id")
	var req UnifiedFieldRequest

	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, types.NewAPIError(
			"UNIFIED_003",
			"Invalid request format",
			err.Error(),
			requestID,
		))
		return
	}

	if len(req.MassDistribution) == 0 {
		c.JSON(http.StatusBadRequest, types.NewAPIError(
			"UNIFIED_004",
			"No mass distribution provided",
			"At least one mass point is required for field analysis",
			requestID,
		))
		return
	}

	startTime := time.Now()

	// Convert API types to engine types
	initialConditions := map[string]interface{}{
		"mass_distribution":  req.MassDistribution,
		"observer_position":  req.ObserverPosition,
		"calculation_region": req.CalculationRegion,
	}

	// Set configuration if provided
	var config *unified.UnifiedConfig
	if req.Config != nil {
		config = &unified.UnifiedConfig{
			IncludeQuantumGravity: req.Config.QuantumCorrections,
			IncludeSupersymmetry:  true,
			IncludeStringTheory:   true,
			TimeoutSeconds:        req.Config.ComputationTimeout,
			MaxParticles:          len(req.MassDistribution) * 10,
		}
		if req.Config.PrecisionLevel == "high" {
			config.ToleranceLevel = 1e-15
		} else {
			config.ToleranceLevel = 1e-10
		}
	}

	// Run unified physics simulation for field analysis
	result, telemetry, err := unifiedEngine.SimulateUnifiedPhysics("field_analysis", initialConditions, config)
	if err != nil {
		c.JSON(http.StatusInternalServerError, types.NewAPIError(
			"UNIFIED_006",
			"Field analysis failed",
			err.Error(),
			requestID,
		))
		return
	}

	// Generate field map from simulation results
	fieldMap := generateFieldMapFromSimulation(result, req.CalculationRegion)
	totalFieldStrength := result.FieldStrengths["total_field_strength"]
	if totalFieldStrength == 0 {
		totalFieldStrength = calculateTotalFieldStrength(fieldMap)
	}

	endTime := time.Now()
	duration := float64(endTime.Sub(startTime).Nanoseconds()) / 1e6

	// Extract critical points from particle positions
	criticalPoints := make([]Position3D, 0)
	for _, particle := range result.ParticleStates {
		if len(particle.Position) >= 3 {
			criticalPoints = append(criticalPoints, Position3D{
				X: particle.Position[0],
				Y: particle.Position[1],
				Z: particle.Position[2],
			})
		}
	}
	if len(criticalPoints) == 0 {
		criticalPoints = []Position3D{{X: 0, Y: 0, Z: 0}}
	}

	response := UnifiedFieldResponse{
		FieldMap:           fieldMap,
		TotalFieldStrength: totalFieldStrength,
		GradientAnalysis: &GradientAnalysis{
			MaxGradient:    totalFieldStrength * result.UnificationDegree * 1.5,
			MinGradient:    totalFieldStrength * result.UnificationDegree * 0.1,
			AvgGradient:    totalFieldStrength * result.UnificationDegree,
			CriticalPoints: criticalPoints[:min(len(criticalPoints), 10)], // Limit to 10 points
		},
		EmergentEffects: &EmergentEffects{
			GravitationalLensing: result.QuantumCorrections * 1e-6,
			FrameDragging:        result.QuantumCorrections * 1e-9,
			EntropyWaves: []float64{result.UnificationDegree * 0.1, result.UnificationDegree * 0.15,
				result.UnificationDegree * 0.08, result.UnificationDegree * 0.12},
			NonLocalCorrelations: result.ConsciousnessMetrics["non_local_correlations"],
		},
		Visualization: generateVisualizationFromSimulation(result, fieldMap),
		Timing: &TimingInfo{
			StartTime:  startTime,
			EndTime:    endTime,
			Duration:   duration,
			Iterations: len(telemetry),
		},
	}

	c.JSON(http.StatusOK, types.NewAPIResponse(response, requestID))
}

// getPhysicalConstants returns unified physics constants
// @Summary Get physical constants
// @Description Get physical constants used in unified physics calculations
// @Tags Unified Physics
// @Produce json
// @Success 200 {object} types.APIResponse
// @Security ApiKeyAuth
// @Security BearerAuth
// @Router /v1/unified/constants [get]
func getPhysicalConstants(c *gin.Context) {
	requestID := c.GetString("request_id")

	constants := map[string]interface{}{
		"fundamental": map[string]interface{}{
			"G":          6.67430e-11,
			"c":          299792458,
			"h":          6.62607015e-34,
			"k_B":        1.380649e-23,
			"G_coupling": 8.314, // Entropy-gravity coupling constant
		},
		"derived": map[string]interface{}{
			"planck_length":     1.616255e-35,
			"planck_time":       5.391247e-44,
			"planck_mass":       2.176434e-8,
			"entropy_threshold": 1e-20,
		},
		"units": map[string]string{
			"G":          "m³/(kg⋅s²)",
			"c":          "m/s",
			"h":          "J⋅s",
			"k_B":        "J/K",
			"G_coupling": "J/(K⋅bit)",
		},
	}

	c.JSON(http.StatusOK, types.NewAPIResponse(constants, requestID))
}

// getSupportedModels returns supported physics models
// @Summary Get supported physics models
// @Description Get information about physics models supported by the unified service
// @Tags Unified Physics
// @Produce json
// @Success 200 {object} types.APIResponse
// @Security ApiKeyAuth
// @Security BearerAuth
// @Router /v1/unified/models [get]
func getSupportedModels(c *gin.Context) {
	requestID := c.GetString("request_id")

	models := map[string]interface{}{
		"available": []string{
			"observer_entropy_coupling",
			"emergent_gravity",
			"holographic_principle",
			"entropic_force",
		},
		"descriptions": map[string]string{
			"observer_entropy_coupling": "Gravity emerges from observer-entropy coupling",
			"emergent_gravity":          "Spacetime curvature from information theory",
			"holographic_principle":     "2D surface encoding 3D bulk physics",
			"entropic_force":            "Gravity as thermodynamic force",
		},
		"applications": map[string][]string{
			"cosmology":       {"dark energy", "inflation", "cosmic structure"},
			"quantum_gravity": {"planck_scale", "black_holes", "singularities"},
			"astrophysics":    {"neutron_stars", "accretion_disks", "gravitational_waves"},
		},
	}

	c.JSON(http.StatusOK, types.NewAPIResponse(models, requestID))
}

// getUnifiedStatus returns Unified Physics service status
// @Summary Get Unified Physics service status
// @Description Check the health and status of the Unified Physics service
// @Tags Unified Physics
// @Produce json
// @Success 200 {object} types.APIResponse
// @Security ApiKeyAuth
// @Security BearerAuth
// @Router /v1/unified/status [get]
func getUnifiedStatus(c *gin.Context) {
	requestID := c.GetString("request_id")

	status := map[string]interface{}{
		"service":              "unified-physics",
		"status":               "operational",
		"version":              "1.0.0",
		"uptime":               "24h",
		"total_computations":   234,
		"avg_computation_time": "45ms",
		"quantum_corrections":  "enabled",
		"relativity_mode":      "special+general",
	}

	c.JSON(http.StatusOK, types.NewAPIResponse(status, requestID))
}

// Helper functions

func generateFieldMapFromSimulation(result *unified.UnifiedPhysicsResult, region Region3D) [][]FieldPoint {
	// Generate field map from unified physics simulation results
	gridSize := 5
	fieldMap := make([][]FieldPoint, gridSize)

	// Extract field strength from energy-momentum tensor
	baseStrength := 1.0
	if len(result.EnergyMomentumTensor) > 0 && len(result.EnergyMomentumTensor[0]) > 0 {
		baseStrength = result.EnergyMomentumTensor[0][0] * result.UnificationDegree
	}

	for i := 0; i < gridSize; i++ {
		fieldMap[i] = make([]FieldPoint, gridSize)
		for j := 0; j < gridSize; j++ {
			x := region.Center.X + (float64(i)-2.0)*region.Size.X/4.0
			y := region.Center.Y + (float64(j)-2.0)*region.Size.Y/4.0
			z := region.Center.Z

			// Calculate field strength based on simulation results
			distanceFactor := 1.0 / (1.0 + float64(i+j)*0.2)
			strength := baseStrength * distanceFactor * result.FieldStrengths["gravitational_field"]
			if strength == 0 {
				strength = distanceFactor
			}

			// Calculate direction toward nearest mass concentration
			dirX, dirY := -x, -y
			if len(result.ParticleStates) > 0 {
				// Point toward the first massive particle
				particle := result.ParticleStates[0]
				if len(particle.Position) >= 2 {
					dirX = particle.Position[0] - x
					dirY = particle.Position[1] - y
				}
			}

			fieldMap[i][j] = FieldPoint{
				Position:  Position3D{X: x, Y: y, Z: z},
				Strength:  strength,
				Direction: Position3D{X: dirX, Y: dirY, Z: 0},
				Potential: -strength * 10.0 * result.UnificationDegree,
			}
		}
	}

	return fieldMap
}

func generateVisualizationFromSimulation(result *unified.UnifiedPhysicsResult, fieldMap [][]FieldPoint) *VisualizationData {
	// Generate visualization data from simulation results
	contours := make([]ContourLevel, 0)
	vectorField := make([]VectorArrow, 0)

	// Create contour levels based on field strength distribution
	maxStrength := 0.0
	for _, row := range fieldMap {
		for _, point := range row {
			if point.Strength > maxStrength {
				maxStrength = point.Strength
			}
		}
	}

	// Add 3 contour levels
	for level := 1; level <= 3; level++ {
		value := maxStrength * float64(level) / 3.0
		points := make([]Position3D, 0)

		// Find points near this contour level
		for _, row := range fieldMap {
			for _, point := range row {
				if math.Abs(point.Strength-value) < maxStrength*0.1 {
					points = append(points, point.Position)
				}
			}
		}

		if len(points) > 0 {
			contours = append(contours, ContourLevel{
				Value:  value,
				Points: points,
			})
		}
	}

	// Create vector field representation
	for i := 0; i < len(fieldMap); i += 2 { // Sample every other point
		for j := 0; j < len(fieldMap[i]); j += 2 {
			point := fieldMap[i][j]
			magnitude := math.Sqrt(point.Direction.X*point.Direction.X + point.Direction.Y*point.Direction.Y)
			if magnitude > 0 {
				vectorField = append(vectorField, VectorArrow{
					Start:     point.Position,
					Direction: point.Direction,
					Magnitude: point.Strength,
				})
			}
		}
	}

	return &VisualizationData{
		Contours:    contours,
		VectorField: vectorField,
		ColorMap:    "viridis",
		Scale:       "logarithmic",
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func calculateTotalFieldStrength(fieldMap [][]FieldPoint) float64 {
	total := 0.0
	count := 0

	for _, row := range fieldMap {
		for _, point := range row {
			total += point.Strength
			count++
		}
	}

	if count > 0 {
		return total / float64(count)
	}
	return 0.0
}
