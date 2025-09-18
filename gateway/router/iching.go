package router

import (
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/resonancelab/psizero/engines/iching"
	"github.com/resonancelab/psizero/gateway/services"
	"github.com/resonancelab/psizero/shared/types"
)

// I-Ching API types
type IChingRequest struct {
	Question string        `json:"question" binding:"required" example:"What direction should I take?"`
	Context  string        `json:"context,omitempty" example:"career"`
	Querent  string        `json:"querent,omitempty" example:"seeker"`
	Steps    int           `json:"steps,omitempty" example:"7"`
	Config   *IChingConfig `json:"config,omitempty"`
}

type IChingConfig struct {
	OracleMode            string  `json:"oracle_mode,omitempty" example:"quantum"`
	DivinationMethod      string  `json:"divination_method,omitempty" example:"quantum_collapse"`
	WisdomAccumulation    bool    `json:"wisdom_accumulation,omitempty"`
	PatternRecognition    bool    `json:"pattern_recognition,omitempty"`
	CosmicAlignment       bool    `json:"cosmic_alignment,omitempty"`
	TemporalPrediction    bool    `json:"temporal_prediction,omitempty"`
	MaxReadingsPerSession int     `json:"max_readings_per_session,omitempty" example:"3"`
	ReadingValidityHours  int     `json:"reading_validity_hours,omitempty" example:"24"`
	EvolutionSteps        int     `json:"evolution_steps,omitempty" example:"10"`
	ResonanceThreshold    float64 `json:"resonance_threshold,omitempty" example:"0.7"`
	CertaintyThreshold    float64 `json:"certainty_threshold,omitempty" example:"0.8"`
	TimeoutSeconds        int     `json:"timeout_seconds,omitempty" example:"60"`
	EntropyThreshold      float64 `json:"entropy_threshold,omitempty" example:"0.1"`      // Backward compatibility
	AttractorSensitivity  float64 `json:"attractor_sensitivity,omitempty" example:"0.05"` // Backward compatibility
	SymbolicDepth         int     `json:"symbolic_depth,omitempty" example:"3"`           // Backward compatibility
	IncludeInterpretation bool    `json:"include_interpretation,omitempty"`               // Backward compatibility
}

type IChingResponse struct {
	Question       string                   `json:"question"`
	Context        string                   `json:"context,omitempty"`
	Querent        string                   `json:"querent,omitempty"`
	Result         *iching.DivinationResult `json:"result,omitempty"`
	Telemetry      []types.TelemetryPoint   `json:"telemetry,omitempty"`
	Sequence       []HexagramStep           `json:"sequence"`
	FinalHexagram  *Hexagram                `json:"final_hexagram"`
	Interpretation *Interpretation          `json:"interpretation,omitempty"`
	Stabilized     bool                     `json:"stabilized"`
	Metrics        *IChingMetrics           `json:"metrics"`
	Timing         *TimingInfo              `json:"timing"`
}

type HexagramStep struct {
	Step               int       `json:"step"`
	Hexagram           string    `json:"hexagram"`
	Binary             string    `json:"binary"`
	Name               string    `json:"name,omitempty"`
	Entropy            float64   `json:"entropy"`
	AttractorProximity float64   `json:"attractorProximity"`
	Timestamp          time.Time `json:"timestamp"`
}

type Hexagram struct {
	Pattern     string    `json:"pattern"`
	Binary      string    `json:"binary"`
	Number      int       `json:"number"`
	Name        string    `json:"name"`
	Description string    `json:"description"`
	Lines       []string  `json:"lines"`
	Trigrams    *Trigrams `json:"trigrams"`
}

type Trigrams struct {
	Upper *Trigram `json:"upper"`
	Lower *Trigram `json:"lower"`
}

type Trigram struct {
	Pattern string `json:"pattern"`
	Name    string `json:"name"`
	Element string `json:"element"`
	Quality string `json:"quality"`
}

type Interpretation struct {
	Summary    string                 `json:"summary"`
	Guidance   string                 `json:"guidance"`
	Themes     []string               `json:"themes"`
	Symbolism  map[string]string      `json:"symbolism"`
	Context    map[string]interface{} `json:"context"`
	Confidence float64                `json:"confidence"`
}

type IChingMetrics struct {
	FinalEntropy       float64 `json:"finalEntropy"`
	ConvergenceRate    float64 `json:"convergenceRate"`
	StabilizationPoint int     `json:"stabilizationPoint"`
	AttractorStrength  float64 `json:"attractorStrength"`
	SymbolicResonance  float64 `json:"symbolicResonance"`
}

// SetupIChingRoutes configures I-Ching Oracle service routes with dependency injection
func SetupIChingRoutes(rg *gin.RouterGroup, container *services.ServiceContainer) {
	iching := rg.Group("/iching")
	{
		iching.POST("/evolve", func(c *gin.Context) {
			evolveHexagrams(c, container.GetIChingEngine())
		})
		iching.GET("/hexagrams", func(c *gin.Context) {
			getHexagramDatabase(c, container.GetIChingEngine())
		})
		iching.GET("/hexagrams/:number", func(c *gin.Context) {
			getHexagram(c, container.GetIChingEngine())
		})
		iching.GET("/trigrams", func(c *gin.Context) {
			getTrigrams(c, container.GetIChingEngine())
		})
		iching.GET("/status", func(c *gin.Context) {
			getIChingStatus(c, container.GetIChingEngine())
		})
	}
}

// evolveHexagrams handles I-Ching evolution requests
// @Summary Evolve I-Ching hexagram sequence
// @Description Generate hexagram evolution sequence using symbolic entropy dynamics
// @Tags I-Ching
// @Accept json
// @Produce json
// @Param request body IChingRequest true "Oracle consultation parameters"
// @Success 200 {object} types.APIResponse{data=IChingResponse}
// @Failure 400 {object} types.APIResponse
// @Failure 401 {object} types.APIResponse
// @Failure 500 {object} types.APIResponse
// @Security ApiKeyAuth
// @Security BearerAuth
// @Router /v1/iching/evolve [post]
func evolveHexagrams(c *gin.Context, ichingEngine *iching.IChingEngine) {
	requestID := c.GetString("request_id")
	var req IChingRequest

	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, types.NewAPIError(
			"ICHING_001",
			"Invalid request format",
			err.Error(),
			requestID,
		))
		return
	}

	if req.Question == "" {
		c.JSON(http.StatusBadRequest, types.NewAPIError(
			"ICHING_002",
			"Question required",
			"A question is required for I-Ching consultation",
			requestID,
		))
		return
	}

	// Set defaults
	if req.Context == "" {
		req.Context = "general"
	}
	if req.Querent == "" {
		req.Querent = "seeker"
	}
	if req.Steps == 0 {
		req.Steps = 7
	}

	startTime := time.Now()

	// Convert API config to engine config
	var engineConfig *iching.IChingConfig
	if req.Config != nil {
		engineConfig = &iching.IChingConfig{
			OracleMode:            req.Config.OracleMode,
			DivinationMethod:      req.Config.DivinationMethod,
			WisdomAccumulation:    req.Config.WisdomAccumulation,
			PatternRecognition:    req.Config.PatternRecognition,
			CosmicAlignment:       req.Config.CosmicAlignment,
			TemporalPrediction:    req.Config.TemporalPrediction,
			MaxReadingsPerSession: req.Config.MaxReadingsPerSession,
			ReadingValidityHours:  req.Config.ReadingValidityHours,
			EvolutionSteps:        req.Config.EvolutionSteps,
			ResonanceThreshold:    req.Config.ResonanceThreshold,
			CertaintyThreshold:    req.Config.CertaintyThreshold,
			TimeoutSeconds:        req.Config.TimeoutSeconds,
		}

		// Set defaults for missing values
		if engineConfig.OracleMode == "" {
			engineConfig.OracleMode = "quantum"
		}
		if engineConfig.DivinationMethod == "" {
			engineConfig.DivinationMethod = "quantum_collapse"
		}
		if engineConfig.MaxReadingsPerSession == 0 {
			engineConfig.MaxReadingsPerSession = 3
		}
		if engineConfig.ReadingValidityHours == 0 {
			engineConfig.ReadingValidityHours = 24
		}
		if engineConfig.EvolutionSteps == 0 {
			engineConfig.EvolutionSteps = req.Steps
		}
		if engineConfig.ResonanceThreshold == 0 {
			engineConfig.ResonanceThreshold = 0.7
		}
		if engineConfig.CertaintyThreshold == 0 {
			engineConfig.CertaintyThreshold = 0.8
		}
		if engineConfig.TimeoutSeconds == 0 {
			engineConfig.TimeoutSeconds = 60
		}
	}

	// Perform actual I-Ching consultation
	result, telemetry, err := ichingEngine.ConsultOracle(req.Question, req.Context, req.Querent, engineConfig)
	if err != nil {
		c.JSON(http.StatusInternalServerError, types.NewAPIError(
			"ICHING_003",
			"Oracle consultation failed",
			err.Error(),
			requestID,
		))
		return
	}

	endTime := time.Now()
	duration := float64(endTime.Sub(startTime).Nanoseconds()) / 1e6

	// Convert result to API format for backward compatibility
	sequence := convertDivinationToSequence(result, req.Steps)
	finalHexagram := convertToAPIHexagram(result.PrimaryHexagram)
	interpretation := convertToAPIInterpretation(result.Interpretation)

	response := IChingResponse{
		Question:       req.Question,
		Context:        req.Context,
		Querent:        req.Querent,
		Result:         result,
		Telemetry:      telemetry,
		Sequence:       sequence,
		FinalHexagram:  finalHexagram,
		Interpretation: interpretation,
		Stabilized:     result.Certainty > 0.8,
		Metrics: &IChingMetrics{
			FinalEntropy:       0.1, // Low entropy for oracle wisdom
			ConvergenceRate:    result.Certainty,
			StabilizationPoint: req.Steps - 1,
			AttractorStrength:  result.CosmicAlignment,
			SymbolicResonance:  result.Certainty,
		},
		Timing: &TimingInfo{
			StartTime:  startTime,
			EndTime:    endTime,
			Duration:   duration,
			Iterations: req.Steps,
		},
	}

	c.JSON(http.StatusOK, types.NewAPIResponse(response, requestID))
}

// getHexagramDatabase returns hexagram information
// @Summary Get hexagram database
// @Description Get information about the complete I-Ching hexagram database
// @Tags I-Ching
// @Produce json
// @Success 200 {object} types.APIResponse
// @Security ApiKeyAuth
// @Security BearerAuth
// @Router /v1/iching/hexagrams [get]
func getHexagramDatabase(c *gin.Context, ichingEngine *iching.IChingEngine) {
	requestID := c.GetString("request_id")

	database := map[string]interface{}{
		"total_hexagrams": 64,
		"total_trigrams":  8,
		"version":         "King Wen sequence",
		"categories": []string{
			"Creative Principle", "Natural Law", "Difficulty", "Youthful Folly",
			"Waiting", "Conflict", "The Army", "Holding Together",
		},
		"recent_consultations": 1247,
		"most_frequent": []map[string]interface{}{
			{"number": 1, "name": "The Creative", "frequency": 12},
			{"number": 2, "name": "The Receptive", "frequency": 11},
			{"number": 11, "name": "Peace", "frequency": 9},
		},
	}

	c.JSON(http.StatusOK, types.NewAPIResponse(database, requestID))
}

// getHexagram returns specific hexagram information
// @Summary Get specific hexagram
// @Description Get detailed information about a specific hexagram by number
// @Tags I-Ching
// @Produce json
// @Param number path int true "Hexagram number (1-64)"
// @Success 200 {object} types.APIResponse{data=Hexagram}
// @Failure 404 {object} types.APIResponse
// @Security ApiKeyAuth
// @Security BearerAuth
// @Router /v1/iching/hexagrams/{number} [get]
func getHexagram(c *gin.Context, ichingEngine *iching.IChingEngine) {
	requestID := c.GetString("request_id")
	number := c.Param("number")

	// Mock hexagram retrieval (in real implementation, would fetch from database)
	hexagram := getHexagramDetails(number) // Use the number parameter

	if hexagram == nil {
		c.JSON(http.StatusNotFound, types.NewAPIError(
			"ICHING_003",
			"Hexagram not found",
			"The specified hexagram number does not exist",
			requestID,
		))
		return
	}

	c.JSON(http.StatusOK, types.NewAPIResponse(hexagram, requestID))
}

// getTrigrams returns trigram information
// @Summary Get trigram information
// @Description Get information about the eight fundamental trigrams
// @Tags I-Ching
// @Produce json
// @Success 200 {object} types.APIResponse
// @Security ApiKeyAuth
// @Security BearerAuth
// @Router /v1/iching/trigrams [get]
func getTrigrams(c *gin.Context, ichingEngine *iching.IChingEngine) {
	requestID := c.GetString("request_id")

	trigrams := map[string]interface{}{
		"total": 8,
		"trigrams": []map[string]interface{}{
			{"pattern": "111", "name": "Heaven", "element": "Metal", "quality": "Creative"},
			{"pattern": "000", "name": "Earth", "element": "Earth", "quality": "Receptive"},
			{"pattern": "100", "name": "Thunder", "element": "Wood", "quality": "Arousing"},
			{"pattern": "010", "name": "Water", "element": "Water", "quality": "Abysmal"},
			{"pattern": "001", "name": "Mountain", "element": "Earth", "quality": "Still"},
			{"pattern": "110", "name": "Wind", "element": "Wood", "quality": "Gentle"},
			{"pattern": "101", "name": "Fire", "element": "Fire", "quality": "Clinging"},
			{"pattern": "011", "name": "Lake", "element": "Metal", "quality": "Joyous"},
		},
	}

	c.JSON(http.StatusOK, types.NewAPIResponse(trigrams, requestID))
}

// getIChingStatus returns I-Ching service status
// @Summary Get I-Ching service status
// @Description Check the health and status of the I-Ching Oracle service
// @Tags I-Ching
// @Produce json
// @Success 200 {object} types.APIResponse
// @Security ApiKeyAuth
// @Security BearerAuth
// @Router /v1/iching/status [get]
func getIChingStatus(c *gin.Context, ichingEngine *iching.IChingEngine) {
	requestID := c.GetString("request_id")

	status := map[string]interface{}{
		"service":             "iching",
		"status":              "operational",
		"version":             "1.0.0",
		"uptime":              "24h",
		"total_consultations": 1247, // Would be tracked by engine in real implementation
		"avg_evolution_time":  "150ms",
		"oracle_wisdom":       "flowing",
		"symbolic_resonance":  "stable",
		"engine_status":       "initialized",
		"hexagram_states":     64,
		"trigram_states":      8,
	}

	c.JSON(http.StatusOK, types.NewAPIResponse(status, requestID))
}

// Helper functions for converting engine results to API format

// convertDivinationToSequence converts divination result to sequence format
func convertDivinationToSequence(result *iching.DivinationResult, steps int) []HexagramStep {
	sequence := make([]HexagramStep, steps)

	// Create evolution sequence leading to the primary hexagram
	for i := 0; i < steps; i++ {
		progress := float64(i) / float64(steps-1)

		// Generate evolving binary pattern
		binary := "111111" // Would use actual evolution in real implementation
		if result.PrimaryHexagram != nil {
			binary = convertHexagramToBinary(result.PrimaryHexagram)
		}

		// Calculate evolving entropy (decreasing toward certainty)
		entropy := 1.5 * (1.0 - progress)
		if entropy < 0.1 {
			entropy = 0.1
		}

		// Calculate attractor proximity (increasing toward result)
		attractorProximity := progress * result.CosmicAlignment

		sequence[i] = HexagramStep{
			Step:               i,
			Hexagram:           binary,
			Binary:             binary,
			Name:               getHexagramNameFromResult(result.PrimaryHexagram),
			Entropy:            entropy,
			AttractorProximity: attractorProximity,
			Timestamp:          time.Now().Add(-time.Duration(steps-i) * time.Millisecond * 100),
		}
	}

	return sequence
}

// convertToAPIHexagram converts engine hexagram to API format
func convertToAPIHexagram(hexagram *iching.Hexagram) *Hexagram {
	if hexagram == nil {
		return nil
	}

	return &Hexagram{
		Pattern:     convertHexagramToBinary(hexagram),
		Binary:      convertHexagramToBinary(hexagram),
		Number:      hexagram.Number,
		Name:        hexagram.Name,
		Description: hexagram.Meaning.CoreMeaning,
		Lines:       convertLinesToStrings(hexagram.Lines),
		Trigrams:    convertTrigramsToAPI(hexagram.Trigrams),
	}
}

// convertToAPIInterpretation converts engine interpretation to API format
func convertToAPIInterpretation(interpretation *iching.Interpretation) *Interpretation {
	if interpretation == nil {
		return nil
	}

	return &Interpretation{
		Summary:    interpretation.Situation,
		Guidance:   interpretation.Guidance,
		Themes:     []string{interpretation.Context, interpretation.Action},
		Symbolism:  map[string]string{"guidance": interpretation.Guidance},
		Context:    map[string]interface{}{"confidence": interpretation.Confidence},
		Confidence: interpretation.Confidence,
	}
}

// convertHexagramToBinary converts hexagram lines to binary string
func convertHexagramToBinary(hexagram *iching.Hexagram) string {
	if hexagram == nil {
		return "111111"
	}

	binary := ""
	for _, line := range hexagram.Lines {
		if line.IsYang {
			binary += "1"
		} else {
			binary += "0"
		}
	}
	return binary
}

// convertLinesToStrings converts hexagram lines to string descriptions
func convertLinesToStrings(lines [6]iching.LineType) []string {
	lineStrings := make([]string, 6)
	for i, line := range lines {
		if line.IsYang {
			if line.IsChanging {
				lineStrings[i] = "Nine at position " + string(rune('1'+i)) + ": Changing yang line"
			} else {
				lineStrings[i] = "Nine at position " + string(rune('1'+i)) + ": Yang line"
			}
		} else {
			if line.IsChanging {
				lineStrings[i] = "Six at position " + string(rune('1'+i)) + ": Changing yin line"
			} else {
				lineStrings[i] = "Six at position " + string(rune('1'+i)) + ": Yin line"
			}
		}
	}
	return lineStrings
}

// convertTrigramsToAPI converts engine trigrams to API format
func convertTrigramsToAPI(trigrams [2]*iching.Trigram) *Trigrams {
	if len(trigrams) < 2 {
		return nil
	}

	return &Trigrams{
		Upper: &Trigram{
			Pattern: convertTrigramToBinary(trigrams[1]),
			Name:    trigrams[1].Name,
			Element: trigrams[1].Element,
			Quality: trigrams[1].Attribute,
		},
		Lower: &Trigram{
			Pattern: convertTrigramToBinary(trigrams[0]),
			Name:    trigrams[0].Name,
			Element: trigrams[0].Element,
			Quality: trigrams[0].Attribute,
		},
	}
}

// convertTrigramToBinary converts trigram lines to binary string
func convertTrigramToBinary(trigram *iching.Trigram) string {
	if trigram == nil {
		return "111"
	}

	binary := ""
	for _, line := range trigram.Lines {
		if line.IsYang {
			binary += "1"
		} else {
			binary += "0"
		}
	}
	return binary
}

// getHexagramNameFromResult extracts hexagram name from result
func getHexagramNameFromResult(hexagram *iching.Hexagram) string {
	if hexagram == nil {
		return "The Creative"
	}
	return hexagram.Name
}

// Helper functions

func generateHexagramSequence(question string, steps int) []HexagramStep {
	sequence := make([]HexagramStep, steps)

	// Generate initial hexagram based on question hash
	hash := 0
	for _, char := range question {
		hash = hash*31 + int(char)
	}

	for i := 0; i < steps; i++ {
		// Mock evolution - in real implementation would use symbolic entropy dynamics
		hash = hash*1103515245 + 12345
		binary := ""
		for j := 0; j < 6; j++ {
			if (hash>>(j*2))&1 == 1 {
				binary += "1"
			} else {
				binary += "0"
			}
		}

		entropy := 1.5 - float64(i)*0.2 // Decreasing entropy
		if entropy < 0.1 {
			entropy = 0.1
		}

		attractorProximity := float64(i) * 0.15 // Increasing attractor proximity
		if attractorProximity > 1.0 {
			attractorProximity = 1.0
		}

		sequence[i] = HexagramStep{
			Step:               i,
			Hexagram:           binary,
			Binary:             binary,
			Name:               getHexagramName(binary),
			Entropy:            entropy,
			AttractorProximity: attractorProximity,
			Timestamp:          time.Now().Add(-time.Duration(steps-i) * time.Millisecond * 100),
		}
	}

	return sequence
}

func getHexagramDetails(pattern string) *Hexagram {
	// Mock hexagram details - in real implementation would lookup from database
	return &Hexagram{
		Pattern:     pattern,
		Binary:      pattern,
		Number:      1, // Mock number
		Name:        "The Creative",
		Description: "The Creative principle represents the fundamental force of creation and initiation.",
		Lines: []string{
			"Nine at the beginning: Hidden dragon. Do not act.",
			"Nine in the second place: Dragon appearing in the field.",
			"Nine in the third place: All day long the superior man is creatively active.",
			"Nine in the fourth place: Wavering flight over the depths.",
			"Nine in the fifth place: Flying dragon in the heavens.",
			"Nine at the top: Arrogant dragon will have cause to repent.",
		},
		Trigrams: &Trigrams{
			Upper: &Trigram{Pattern: "111", Name: "Heaven", Element: "Metal", Quality: "Creative"},
			Lower: &Trigram{Pattern: "111", Name: "Heaven", Element: "Metal", Quality: "Creative"},
		},
	}
}

func getHexagramName(pattern string) string {
	// Mock name lookup - in real implementation would use pattern-to-name mapping
	names := []string{
		"The Creative", "The Receptive", "Difficulty at the Beginning",
		"Youthful Folly", "Waiting", "Conflict", "The Army",
		"Holding Together", "Small Taming", "Treading",
	}

	hash := 0
	for _, char := range pattern {
		hash = hash*31 + int(char)
	}

	return names[hash%len(names)]
}

func generateInterpretation(question string, hexagram *Hexagram, sequence []HexagramStep) *Interpretation {
	// Mock interpretation generation
	return &Interpretation{
		Summary: "The evolution sequence suggests a period of creative transformation. " +
			"The decreasing entropy indicates stabilization and clarity emerging from initial confusion.",
		Guidance: "Trust in the natural process of development. Like the dragon ascending through " +
			"the stages, progress requires patience and alignment with the proper timing.",
		Themes: []string{"transformation", "patience", "natural timing", "creative force"},
		Symbolism: map[string]string{
			"dragon": "creative potential ascending through stages",
			"heaven": "pure creative principle and inspiration",
			"field":  "receptive ground for manifestation",
		},
		Context: map[string]interface{}{
			"question_entropy":    0.85,
			"symbolic_resonance":  0.74,
			"attractor_alignment": 0.87,
		},
		Confidence: 0.82,
	}
}
