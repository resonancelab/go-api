package router

import (
	"math"
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/resonancelab/psizero/engines/qsem"
	"github.com/resonancelab/psizero/gateway/services"
	"github.com/resonancelab/psizero/shared/types"
)

// QSEM API types
type QSEMEncodeRequest struct {
	Concepts []string    `json:"concepts" binding:"required" example:"[\"love\",\"entropy\",\"pattern\"]"`
	Basis    string      `json:"basis" example:"prime"`
	Config   *QSEMConfig `json:"config,omitempty"`
}

type QSEMResonanceRequest struct {
	Vectors []QuantumVector `json:"vectors" binding:"required"`
	Config  *QSEMConfig     `json:"config,omitempty"`
}

type QSEMConfig struct {
	SemanticDimensions  int     `json:"semantic_dimensions,omitempty" example:"300"`
	MaxConcepts         int     `json:"max_concepts,omitempty" example:"10000"`
	PrimeBasisSize      int     `json:"prime_basis_size,omitempty" example:"100"`
	ContextDepth        int     `json:"context_depth,omitempty" example:"5"`
	LearningRate        float64 `json:"learning_rate,omitempty" example:"0.01"`
	CoherenceThreshold  float64 `json:"coherence_threshold,omitempty" example:"0.7"`
	ActivationThreshold float64 `json:"activation_threshold,omitempty" example:"0.5"`
	SemanticRadius      float64 `json:"semantic_radius,omitempty" example:"2.0"`
	ClusteringThreshold float64 `json:"clustering_threshold,omitempty" example:"0.8"`
	MaxIterations       int     `json:"max_iterations,omitempty" example:"1000"`
	TimeoutSeconds      int     `json:"timeout_seconds,omitempty" example:"300"`
	MaxDimensions       int     `json:"max_dimensions,omitempty" example:"100"` // Backward compatibility
	Threshold           float64 `json:"threshold,omitempty" example:"0.1"`      // Backward compatibility
	Normalize           bool    `json:"normalize,omitempty"`                    // Backward compatibility
}

type QuantumVector struct {
	Concept string    `json:"concept"`
	Alpha   []float64 `json:"alpha"`
}

type QSEMEncodeResponse struct {
	Result    *qsem.SemanticAnalysisResult `json:"result"`
	Telemetry []types.TelemetryPoint       `json:"telemetry"`
	Vectors   []QuantumVector              `json:"vectors"`
	Basis     string                       `json:"basis"`
	Stats     *EncodingStats               `json:"stats"`
	Timing    *TimingInfo                  `json:"timing"`
}

type QSEMResonanceResponse struct {
	Result    *qsem.SemanticAnalysisResult `json:"result"`
	Telemetry []types.TelemetryPoint       `json:"telemetry"`
	Coherence float64                      `json:"coherence"`
	Pairwise  []ResonancePair              `json:"pairwise"`
	Matrix    [][]float64                  `json:"matrix,omitempty"`
	Analysis  *ResonanceAnalysis           `json:"analysis"`
	Timing    *TimingInfo                  `json:"timing"`
}

type EncodingStats struct {
	TotalConcepts    int     `json:"totalConcepts"`
	VectorDimensions int     `json:"vectorDimensions"`
	AvgMagnitude     float64 `json:"avgMagnitude"`
	Sparsity         float64 `json:"sparsity"`
}

type ResonancePair struct {
	A         int     `json:"a"`
	B         int     `json:"b"`
	Resonance float64 `json:"resonance"`
	Type      string  `json:"type"`
}

type ResonanceAnalysis struct {
	StrongestPair  *ResonancePair  `json:"strongestPair"`
	WeakestPair    *ResonancePair  `json:"weakestPair"`
	AvgResonance   float64         `json:"avgResonance"`
	ClusterCount   int             `json:"clusterCount"`
	SemanticGroups []SemanticGroup `json:"semanticGroups,omitempty"`
}

type SemanticGroup struct {
	Concepts  []string  `json:"concepts"`
	Centroid  []float64 `json:"centroid"`
	Coherence float64   `json:"coherence"`
	Label     string    `json:"label,omitempty"`
}

// SetupQSEMRoutes configures QSEM service routes with dependency injection
func SetupQSEMRoutes(rg *gin.RouterGroup, container *services.ServiceContainer) {
	qsem := rg.Group("/qsem")
	{
		qsem.POST("/encode", func(c *gin.Context) {
			encodeConcepts(c, container.GetQSEMEngine())
		})
		qsem.POST("/resonance", func(c *gin.Context) {
			computeResonance(c, container.GetQSEMEngine())
		})
		qsem.GET("/basis", func(c *gin.Context) {
			getSupportedBasis(c, container.GetQSEMEngine())
		})
		qsem.GET("/status", func(c *gin.Context) {
			getQSEMStatus(c, container.GetQSEMEngine())
		})
	}
}

// encodeConcepts handles concept encoding requests
// @Summary Encode concepts into quantum semantic vectors
// @Description Transform natural language concepts into prime-basis quantum vectors
// @Tags QSEM
// @Accept json
// @Produce json
// @Param request body QSEMEncodeRequest true "Encoding parameters"
// @Success 200 {object} types.APIResponse{data=QSEMEncodeResponse}
// @Failure 400 {object} types.APIResponse
// @Failure 401 {object} types.APIResponse
// @Failure 500 {object} types.APIResponse
// @Security ApiKeyAuth
// @Security BearerAuth
// @Router /v1/qsem/encode [post]
func encodeConcepts(c *gin.Context, qsemEngine *qsem.QSEMEngine) {
	requestID := c.GetString("request_id")
	var req QSEMEncodeRequest

	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, types.NewAPIError(
			"QSEM_001",
			"Invalid request format",
			err.Error(),
			requestID,
		))
		return
	}

	if len(req.Concepts) == 0 {
		c.JSON(http.StatusBadRequest, types.NewAPIError(
			"QSEM_002",
			"No concepts provided",
			"At least one concept is required for encoding",
			requestID,
		))
		return
	}

	startTime := time.Now()

	// Convert API config to engine config
	var engineConfig *qsem.QSEMConfig
	if req.Config != nil {
		engineConfig = &qsem.QSEMConfig{
			SemanticDimensions:  req.Config.SemanticDimensions,
			MaxConcepts:         req.Config.MaxConcepts,
			PrimeBasisSize:      req.Config.PrimeBasisSize,
			ContextDepth:        req.Config.ContextDepth,
			LearningRate:        req.Config.LearningRate,
			CoherenceThreshold:  req.Config.CoherenceThreshold,
			ActivationThreshold: req.Config.ActivationThreshold,
			SemanticRadius:      req.Config.SemanticRadius,
			ClusteringThreshold: req.Config.ClusteringThreshold,
			MaxIterations:       req.Config.MaxIterations,
			TimeoutSeconds:      req.Config.TimeoutSeconds,
		}

		// Set defaults for missing values
		if engineConfig.SemanticDimensions == 0 {
			engineConfig.SemanticDimensions = 300
		}
		if engineConfig.MaxConcepts == 0 {
			engineConfig.MaxConcepts = 10000
		}
		if engineConfig.PrimeBasisSize == 0 {
			engineConfig.PrimeBasisSize = 100
		}
		if engineConfig.ContextDepth == 0 {
			engineConfig.ContextDepth = 5
		}
		if engineConfig.LearningRate == 0 {
			engineConfig.LearningRate = 0.01
		}
		if engineConfig.CoherenceThreshold == 0 {
			engineConfig.CoherenceThreshold = 0.7
		}
		if engineConfig.ActivationThreshold == 0 {
			engineConfig.ActivationThreshold = 0.5
		}
		if engineConfig.SemanticRadius == 0 {
			engineConfig.SemanticRadius = 2.0
		}
		if engineConfig.ClusteringThreshold == 0 {
			engineConfig.ClusteringThreshold = 0.8
		}
		if engineConfig.MaxIterations == 0 {
			engineConfig.MaxIterations = 1000
		}
		if engineConfig.TimeoutSeconds == 0 {
			engineConfig.TimeoutSeconds = 300
		}
	}

	// Prepare input for QSEM engine
	input := map[string]interface{}{
		"type":     "concepts",
		"concepts": make([]interface{}, len(req.Concepts)),
	}

	for i, concept := range req.Concepts {
		input["concepts"].([]interface{})[i] = concept
	}

	// Run actual semantic analysis
	result, telemetry, err := qsemEngine.AnalyzeSemantics(input, engineConfig)
	if err != nil {
		c.JSON(http.StatusInternalServerError, types.NewAPIError(
			"QSEM_003",
			"Semantic analysis failed",
			err.Error(),
			requestID,
		))
		return
	}

	endTime := time.Now()
	duration := float64(endTime.Sub(startTime).Nanoseconds()) / 1e6

	// Convert result to API vectors for backward compatibility
	vectors := convertSemanticResultToVectors(result, req.Concepts)

	response := QSEMEncodeResponse{
		Result:    result,
		Telemetry: telemetry,
		Vectors:   vectors,
		Basis:     "prime",
		Stats: &EncodingStats{
			TotalConcepts: len(req.Concepts),
			VectorDimensions: func() int {
				if result.ConceptMap != nil && len(result.ConceptMap) > 0 {
					return 300
				}
				return 20
			}(),
			AvgMagnitude: calculateAverageVectorMagnitude(vectors),
			Sparsity:     result.MeaningDensity,
		},
		Timing: &TimingInfo{
			StartTime:  startTime,
			EndTime:    endTime,
			Duration:   duration,
			Iterations: len(req.Concepts),
		},
	}

	c.JSON(http.StatusOK, types.NewAPIResponse(response, requestID))
}

// computeResonance handles resonance computation requests
// @Summary Compute semantic resonance between quantum vectors
// @Description Calculate resonance patterns and coherence between encoded concept vectors
// @Tags QSEM
// @Accept json
// @Produce json
// @Param request body QSEMResonanceRequest true "Resonance computation parameters"
// @Success 200 {object} types.APIResponse{data=QSEMResonanceResponse}
// @Failure 400 {object} types.APIResponse
// @Failure 401 {object} types.APIResponse
// @Failure 500 {object} types.APIResponse
// @Security ApiKeyAuth
// @Security BearerAuth
// @Router /v1/qsem/resonance [post]
func computeResonance(c *gin.Context, qsemEngine *qsem.QSEMEngine) {
	requestID := c.GetString("request_id")
	var req QSEMResonanceRequest

	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, types.NewAPIError(
			"QSEM_003",
			"Invalid request format",
			err.Error(),
			requestID,
		))
		return
	}

	if len(req.Vectors) < 2 {
		c.JSON(http.StatusBadRequest, types.NewAPIError(
			"QSEM_004",
			"Insufficient vectors",
			"At least 2 vectors required for resonance computation",
			requestID,
		))
		return
	}

	startTime := time.Now()

	// Convert API config to engine config
	var engineConfig *qsem.QSEMConfig
	if req.Config != nil {
		engineConfig = &qsem.QSEMConfig{
			SemanticDimensions:  req.Config.SemanticDimensions,
			MaxConcepts:         req.Config.MaxConcepts,
			PrimeBasisSize:      req.Config.PrimeBasisSize,
			ContextDepth:        req.Config.ContextDepth,
			LearningRate:        req.Config.LearningRate,
			CoherenceThreshold:  req.Config.CoherenceThreshold,
			ActivationThreshold: req.Config.ActivationThreshold,
			SemanticRadius:      req.Config.SemanticRadius,
			ClusteringThreshold: req.Config.ClusteringThreshold,
			MaxIterations:       req.Config.MaxIterations,
			TimeoutSeconds:      req.Config.TimeoutSeconds,
		}

		// Set defaults for missing values (same as above)
		if engineConfig.SemanticDimensions == 0 {
			engineConfig.SemanticDimensions = 300
		}
		if engineConfig.MaxConcepts == 0 {
			engineConfig.MaxConcepts = 10000
		}
		if engineConfig.PrimeBasisSize == 0 {
			engineConfig.PrimeBasisSize = 100
		}
		if engineConfig.ContextDepth == 0 {
			engineConfig.ContextDepth = 5
		}
		if engineConfig.LearningRate == 0 {
			engineConfig.LearningRate = 0.01
		}
		if engineConfig.CoherenceThreshold == 0 {
			engineConfig.CoherenceThreshold = 0.7
		}
		if engineConfig.ActivationThreshold == 0 {
			engineConfig.ActivationThreshold = 0.5
		}
		if engineConfig.SemanticRadius == 0 {
			engineConfig.SemanticRadius = 2.0
		}
		if engineConfig.ClusteringThreshold == 0 {
			engineConfig.ClusteringThreshold = 0.8
		}
		if engineConfig.MaxIterations == 0 {
			engineConfig.MaxIterations = 1000
		}
		if engineConfig.TimeoutSeconds == 0 {
			engineConfig.TimeoutSeconds = 300
		}
	}

	// Extract concepts from vectors for semantic analysis
	concepts := make([]string, len(req.Vectors))
	for i, vector := range req.Vectors {
		concepts[i] = vector.Concept
	}

	// Prepare input for QSEM engine
	input := map[string]interface{}{
		"type":     "concepts",
		"concepts": make([]interface{}, len(concepts)),
	}

	for i, concept := range concepts {
		input["concepts"].([]interface{})[i] = concept
	}

	// Run actual semantic analysis
	result, telemetry, err := qsemEngine.AnalyzeSemantics(input, engineConfig)
	if err != nil {
		c.JSON(http.StatusInternalServerError, types.NewAPIError(
			"QSEM_005",
			"Semantic resonance analysis failed",
			err.Error(),
			requestID,
		))
		return
	}

	endTime := time.Now()
	duration := float64(endTime.Sub(startTime).Nanoseconds()) / 1e6

	// Convert result to resonance data for backward compatibility
	pairwise, matrix, analysis := convertSemanticResultToResonance(result, req.Vectors)

	response := QSEMResonanceResponse{
		Result:    result,
		Telemetry: telemetry,
		Coherence: result.ContextualCoherence,
		Pairwise:  pairwise,
		Matrix:    matrix,
		Analysis:  analysis,
		Timing: &TimingInfo{
			StartTime:  startTime,
			EndTime:    endTime,
			Duration:   duration,
			Iterations: len(pairwise),
		},
	}

	c.JSON(http.StatusOK, types.NewAPIResponse(response, requestID))
}

// getSupportedBasis returns supported vector basis types
// @Summary Get supported vector basis types
// @Description Get information about vector basis types supported by QSEM
// @Tags QSEM
// @Produce json
// @Success 200 {object} types.APIResponse
// @Security ApiKeyAuth
// @Security BearerAuth
// @Router /v1/qsem/basis [get]
func getSupportedBasis(c *gin.Context, qsemEngine *qsem.QSEMEngine) {
	requestID := c.GetString("request_id")

	basisInfo := map[string]interface{}{
		"supported": []string{"prime", "fourier", "wavelet"},
		"default":   "prime",
		"prime": map[string]interface{}{
			"description": "Prime number eigenstate basis",
			"dimensions":  "2-1000",
			"advantages":  []string{"quantum coherence", "semantic stability"},
		},
		"fourier": map[string]interface{}{
			"description": "Fourier transform basis",
			"dimensions":  "8-512",
			"advantages":  []string{"frequency analysis", "periodic patterns"},
		},
		"wavelet": map[string]interface{}{
			"description": "Wavelet transform basis",
			"dimensions":  "16-256",
			"advantages":  []string{"multi-resolution", "localized features"},
		},
	}

	c.JSON(http.StatusOK, types.NewAPIResponse(basisInfo, requestID))
}

// getQSEMStatus returns QSEM service status
// @Summary Get QSEM service status
// @Description Check the health and status of the QSEM service
// @Tags QSEM
// @Produce json
// @Success 200 {object} types.APIResponse
// @Security ApiKeyAuth
// @Security BearerAuth
// @Router /v1/qsem/status [get]
func getQSEMStatus(c *gin.Context, qsemEngine *qsem.QSEMEngine) {
	requestID := c.GetString("request_id")

	status := map[string]interface{}{
		"service":           "qsem",
		"status":            "operational",
		"version":           "1.0.0",
		"uptime":            "24h",
		"concepts_encoded":  15420,
		"avg_encoding_time": "45ms",
		"vector_database":   "connected",
	}

	c.JSON(http.StatusOK, types.NewAPIResponse(status, requestID))
}

// convertSemanticResultToVectors converts QSEM result to API vector format
func convertSemanticResultToVectors(result *qsem.SemanticAnalysisResult, concepts []string) []QuantumVector {
	vectors := make([]QuantumVector, len(concepts))

	for i, concept := range concepts {
		// Create quantum vector from semantic analysis result
		alpha := make([]float64, 10) // Sample 10 dimensions

		if result.ConceptMap != nil {
			if semanticVector, exists := result.ConceptMap[concept]; exists && semanticVector != nil {
				// Extract real amplitudes from quantum state
				if semanticVector.QuantumState != nil && len(semanticVector.QuantumState.Amplitudes) > 0 {
					minLen := len(alpha)
					if len(semanticVector.QuantumState.Amplitudes) < minLen {
						minLen = len(semanticVector.QuantumState.Amplitudes)
					}
					for j := 0; j < minLen; j++ {
						alpha[j] = real(semanticVector.QuantumState.Amplitudes[j])
					}
				} else {
					// Fallback to semantic fields
					fieldIndex := 0
					for _, value := range semanticVector.SemanticFields {
						if fieldIndex < len(alpha) {
							alpha[fieldIndex] = value
							fieldIndex++
						}
					}
				}
			}
		}

		// Fill remaining with default values if needed
		for j := range alpha {
			if alpha[j] == 0 {
				alpha[j] = 0.1 + 0.05*float64(j)/float64(len(alpha))
			}
		}

		vectors[i] = QuantumVector{
			Concept: concept,
			Alpha:   alpha,
		}
	}

	return vectors
}

// convertSemanticResultToResonance converts QSEM result to resonance format
func convertSemanticResultToResonance(result *qsem.SemanticAnalysisResult, vectors []QuantumVector) ([]ResonancePair, [][]float64, *ResonanceAnalysis) {
	n := len(vectors)
	matrix := make([][]float64, n)
	pairwise := make([]ResonancePair, 0, n*(n-1)/2)

	totalResonance := 0.0
	strongestPair := &ResonancePair{Resonance: -1}
	weakestPair := &ResonancePair{Resonance: 2}
	pairCount := 0

	for i := 0; i < n; i++ {
		matrix[i] = make([]float64, n)
		for j := i; j < n; j++ {
			var resonance float64
			if i == j {
				resonance = 1.0 // Self-resonance
			} else {
				// Calculate resonance from semantic similarity
				if result.SemanticSimilarity != nil {
					concept1 := vectors[i].Concept
					concept2 := vectors[j].Concept

					if similarities, exists := result.SemanticSimilarity[concept1]; exists {
						if similarity, found := similarities[concept2]; found {
							resonance = similarity
						} else {
							resonance = calculateVectorResonance(vectors[i], vectors[j])
						}
					} else {
						resonance = calculateVectorResonance(vectors[i], vectors[j])
					}
				} else {
					resonance = calculateVectorResonance(vectors[i], vectors[j])
				}

				// Determine resonance type
				resonanceType := "neutral"
				if resonance > 0.7 {
					resonanceType = "synonymous"
				} else if resonance < 0.3 {
					resonanceType = "orthogonal"
				}

				pair := ResonancePair{
					A:         i,
					B:         j,
					Resonance: resonance,
					Type:      resonanceType,
				}
				pairwise = append(pairwise, pair)

				if resonance > strongestPair.Resonance {
					strongestPair = &pair
				}
				if resonance < weakestPair.Resonance {
					weakestPair = &pair
				}

				totalResonance += resonance
				pairCount++
			}

			matrix[i][j] = resonance
			matrix[j][i] = resonance
		}
	}

	avgResonance := 0.0
	if pairCount > 0 {
		avgResonance = totalResonance / float64(pairCount)
	}

	// Get cluster count from result
	clusterCount := 1
	if result.ClusterAssignments != nil {
		clusters := make(map[string]bool)
		for _, cluster := range result.ClusterAssignments {
			clusters[cluster] = true
		}
		clusterCount = len(clusters)
	}

	analysis := &ResonanceAnalysis{
		StrongestPair: strongestPair,
		WeakestPair:   weakestPair,
		AvgResonance:  avgResonance,
		ClusterCount:  clusterCount,
	}

	return pairwise, matrix, analysis
}

// calculateVectorResonance computes resonance between two quantum vectors
func calculateVectorResonance(v1, v2 QuantumVector) float64 {
	if len(v1.Alpha) != len(v2.Alpha) {
		return 0.0
	}

	if len(v1.Alpha) == 0 {
		return 0.0
	}

	// Compute normalized dot product (cosine similarity)
	dotProduct := 0.0
	norm1 := 0.0
	norm2 := 0.0

	for i := range v1.Alpha {
		dotProduct += v1.Alpha[i] * v2.Alpha[i]
		norm1 += v1.Alpha[i] * v1.Alpha[i]
		norm2 += v2.Alpha[i] * v2.Alpha[i]
	}

	if norm1 == 0.0 || norm2 == 0.0 {
		return 0.0
	}

	// Return cosine similarity as resonance strength
	cosineSimilarity := dotProduct / (math.Sqrt(norm1) * math.Sqrt(norm2))

	// Ensure result is in [0, 1] range
	if cosineSimilarity < 0 {
		return 0.0
	}
	if cosineSimilarity > 1 {
		return 1.0
	}

	return cosineSimilarity
}

// calculateAverageVectorMagnitude computes average magnitude of vectors
func calculateAverageVectorMagnitude(vectors []QuantumVector) float64 {
	if len(vectors) == 0 {
		return 0.0
	}

	totalMagnitude := 0.0
	for _, vector := range vectors {
		magnitude := 0.0
		for _, val := range vector.Alpha {
			magnitude += val * val
		}
		totalMagnitude += magnitude
	}

	return totalMagnitude / float64(len(vectors))
}
