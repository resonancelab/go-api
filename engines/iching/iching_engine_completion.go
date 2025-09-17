
package iching

import (
	"fmt"
	"math"
	"strings"
	"time"

	"github.com/resonancelab/psizero/shared/types"
)

// calculateCosmicAlignment computes cosmic alignment for the reading
func (ic *IChingEngine) calculateCosmicAlignment(hexagram *Hexagram) float64 {
	// Base alignment from oracle field
	baseAlignment := ic.oracleField.CosmicResonance
	
	// Celestial influences
	lunarAlignment := math.Abs(ic.oracleField.CelestialInfluence.MoonPhase - 0.5) * 2.0 // Peak at new/full moon
	solarAlignment := ic.oracleField.CelestialInfluence.SolarInfluence
	planetaryAlignment := ic.oracleField.CelestialInfluence.PlanetaryAlignment
	seasonalAlignment := ic.oracleField.CelestialInfluence.SeasonalEnergy
	
	celestialFactor := 0.25*lunarAlignment + 0.25*solarAlignment + 0.25*planetaryAlignment + 0.25*seasonalAlignment
	
	// Hexagram-specific alignment
	hexagramState := ic.hexagramStates[hexagram.Number]
	hexagramAlignment := hexagramState.Resonance + hexagramState.Coherence
	
	// Yin-Yang balance factor
	yangCount := 0
	for _, line := range hexagram.Lines {
		if line.IsYang {
			yangCount++
		}
	}
	yinYangBalance := 1.0 - math.Abs(float64(yangCount)/6.0-0.5)*2.0 // Peak at 3 yang, 3 yin
	
	// Combined alignment
	totalAlignment := 0.4*baseAlignment + 0.3*celestialFactor + 0.2*hexagramAlignment + 0.1*yinYangBalance
	return math.Min(totalAlignment, 1.0)
}

// generateInterpretation creates interpretation for the hexagram reading
func (ic *IChingEngine) generateInterpretation(primary, future *Hexagram, context, question string) *Interpretation {
	hexagramState := ic.hexagramStates[primary.Number]
	
	// Context-specific guidance
	var situation, guidance, outcome, timing, action, caution string
	
	switch strings.ToLower(context) {
	case "love", "relationship":
		situation = ic.generateLoveInterpretation(hexagramState)
		guidance = "Listen to your heart while maintaining balance"
		timing = "Emotional timing is crucial - wait for the right moment"
	case "career", "work", "business":
		situation = ic.generateCareerInterpretation(hexagramState)
		guidance = "Professional growth requires both patience and decisive action"
		timing = "Professional timing aligns with seasonal energies"
	case "health", "healing":
		situation = ic.generateHealthInterpretation(hexagramState)
		guidance = "Harmony between mind and body is essential"
		timing = "Healing follows natural cycles"
	default:
		situation = hexagramState.Meaning
		guidance = hexagramState.Judgment
		timing = "The time is according to the natural order"
	}
	
	// Generate outcome based on primary and future hexagrams
	if future != nil {
		futureState := ic.hexagramStates[future.Number]
		outcome = fmt.Sprintf("Transformation from %s to %s indicates %s", 
			hexagramState.Name, futureState.Name, ic.generateTransformationOutcome(hexagramState, futureState))
	} else {
		outcome = "The situation remains stable with " + hexagramState.Name
	}
	
	// Action and caution based on hexagram properties
	action = ic.generateActionAdvice(hexagramState, context)
	caution = ic.generateCautionAdvice(hexagramState)
	
	// Calculate confidence based on cosmic alignment and question clarity
	confidence := ic.calculateInterpretationConfidence(primary, future, question)
	
	return &Interpretation{
		Context:    context,
		Situation:  situation,
		Guidance:   guidance,
		Outcome:    outcome,
		Timing:     timing,
		Action:     action,
		Caution:    caution,
		Confidence: confidence,
	}
}

// generateLoveInterpretation creates love-specific interpretation
func (ic *IChingEngine) generateLoveInterpretation(hexagramState *HexagramState) string {
	loveKeywords := map[string]string{
		"Heaven":     "Divine love and spiritual connection await",
		"Earth":      "Nurturing and stable relationships flourish",
		"Thunder":    "Passionate but volatile emotional energy",
		"Mountain":   "Patient, enduring love that grows slowly",
		"Water":      "Deep emotional currents and intuitive connection",
		"Fire":       "Intense attraction and dynamic relationship energy",
		"Wind":       "Gentle, adaptable love that flows naturally",
		"Lake":      "Joyful, expressive emotional exchange",
	}
	
	for keyword, meaning := range loveKeywords {
		if strings.Contains(hexagramState.Name, keyword) || contains(hexagramState.Keywords, keyword) {
			return meaning
		}
	}
	
	return "Love requires balance between giving and receiving"
}

// generateCareerInterpretation creates career-specific interpretation
func (ic *IChingEngine) generateCareerInterpretation(hexagramState *HexagramState) string {
	careerKeywords := map[string]string{
		"Progress":    "Professional advancement is indicated",
		"Retreat":     "Strategic withdrawal or career change needed",
		"Leadership":  "Time to take charge and guide others",
		"Learning":    "Skill development and education are favored",
		"Prosperity":  "Financial success through diligent effort",
		"Challenge":   "Professional obstacles require creative solutions",
		"Innovation":  "New approaches and technologies bring success",
		"Cooperation": "Teamwork and partnerships are essential",
	}
	
	for keyword, meaning := range careerKeywords {
		if contains(hexagramState.Keywords, keyword) {
			return meaning
		}
	}
	
	return "Professional success comes through authentic effort and wise timing"
}

// generateHealthInterpretation creates health-specific interpretation
func (ic *IChingEngine) generateHealthInterpretation(hexagramState *HexagramState) string {
	healthKeywords := map[string]string{
		"Balance":    "Physical and emotional equilibrium supports healing",
		"Energy":     "Vital force is strong - focus on circulation",
		"Rest":       "Recovery requires adequate rest and reflection",
		"Movement":   "Physical activity and exercise are beneficial",
		"Nourishment": "Proper nutrition and self-care are essential",
		"Cleansing":  "Detoxification and purification support health",
		"Harmony":    "Mind-body connection facilitates healing",
		"Strength":   "Physical vitality and resilience are growing",
	}
	
	for keyword, meaning := range healthKeywords {
		if contains(hexagramState.Keywords, keyword) {
			return meaning
		}
	}
	
	return "Health improves through natural harmony and mindful living"
}

// contains checks if a slice contains a string (case-insensitive)
func contains(slice []string, item string) bool {
	for _, s := range slice {
		if strings.EqualFold(s, item) {
			return true
		}
	}
	return false
}

// generateTransformationOutcome describes the transformation between hexagrams
func (ic *IChingEngine) generateTransformationOutcome(from, to *HexagramState) string {
	// Analyze the change in energy patterns
	fromYang := 0
	toYang := 0
	
	for _, isYang := range from.YinYangPattern {
		if isYang {
			fromYang++
		}
	}
	
	for _, isYang := range to.YinYangPattern {
		if isYang {
			toYang++
		}
	}
	
	energyChange := toYang - fromYang
	
	if energyChange > 0 {
		return "increasing yang energy and active manifestation"
	} else if energyChange < 0 {
		return "increasing yin energy and receptive contemplation"
	} else {
		return "balanced transformation maintaining energetic equilibrium"
	}
}

// generateActionAdvice creates action recommendations
func (ic *IChingEngine) generateActionAdvice(hexagramState *HexagramState, context string) string {
	// Base advice on hexagram keywords and context
	actions := []string{
		"Take deliberate and mindful action",
		"Wait for the right timing before proceeding",
		"Seek harmony in all endeavors",
		"Embrace change with wisdom and flexibility",
		"Build strong foundations before expanding",
		"Trust your inner wisdom and intuition",
		"Collaborate with others for mutual benefit",
		"Maintain balance between effort and ease",
	}
	
	// Select action based on hexagram number (deterministic but varied)
	actionIndex := hexagramState.HexagramNumber % len(actions)
	return actions[actionIndex]
}

// generateCautionAdvice creates caution recommendations
func (ic *IChingEngine) generateCautionAdvice(hexagramState *HexagramState) string {
	cautions := []string{
		"Avoid hasty decisions without proper consideration",
		"Beware of extremes - seek the middle path",
		"Don't force outcomes against natural timing",
		"Guard against pride and overconfidence",
		"Remain flexible when circumstances change",
		"Don't neglect important relationships",
		"Avoid actions driven purely by emotion",
		"Be mindful of unintended consequences",
	}
	
	// Select caution based on hexagram characteristics
	cautionIndex := (hexagramState.HexagramNumber + 3) % len(cautions)
	return cautions[cautionIndex]
}

// calculateInterpretationConfidence calculates confidence in the interpretation
func (ic *IChingEngine) calculateInterpretationConfidence(primary, future *Hexagram, question string) float64 {
	// Base confidence from oracle field
	baseConfidence := ic.oracleField.DivineConnection
	
	// Question clarity factor
	questionWords := strings.Fields(question)
	questionClarity := math.Min(float64(len(questionWords))/10.0, 1.0) // More words = clearer question
	
	// Cosmic alignment factor
	cosmicFactor := ic.calculateCosmicAlignment(primary)
	
	// Hexagram stability (changing lines reduce confidence slightly)
	stabilityFactor := 1.0
	changingCount := 0
	for _, line := range primary.Lines {
		if line.IsChanging {
			changingCount++
		}
	}
	if changingCount > 0 {
		stabilityFactor = 1.0 - float64(changingCount)*0.1
	}
	
	// Future hexagram presence (adds confidence)
	futureFactor := 0.8
	if future != nil {
		futureFactor = 1.0
	}
	
	confidence := 0.3*baseConfidence + 0.2*questionClarity + 0.3*cosmicFactor + 0.1*stabilityFactor + 0.1*futureFactor
	return math.Min(confidence, 1.0)
}

// generateWisdomMessage creates a wisdom message for the reading
func (ic *IChingEngine) generateWisdomMessage(primary, future *Hexagram, interpretation *Interpretation) string {
	hexagramState := ic.hexagramStates[primary.Number]
	
	wisdomTemplates := []string{
		"The ancient wisdom teaches: %s. In your situation, %s.",
		"As the I-Ching reveals: %s. This guidance suggests %s.",
		"The oracle speaks: %s. For you, this means %s.",
		"Traditional wisdom says: %s. Applied to your question: %s.",
		"The cosmic pattern shows: %s. Your path forward involves %s.",
	}
	
	// Select template based on hexagram
	templateIndex := hexagramState.HexagramNumber % len(wisdomTemplates)
	template := wisdomTemplates[templateIndex]
	
	// Ancient wisdom from hexagram
	ancientWisdom := hexagramState.Image
	if ancientWisdom == "" {
		ancientWisdom = hexagramState.Judgment
	}
	
	// Personal application
	personalApplication := interpretation.Guidance
	
	return fmt.Sprintf(template, ancientWisdom, personalApplication)
}

// calculateDivinationCertainty calculates overall certainty of the reading
func (ic *IChingEngine) calculateDivinationCertainty(hexagram *Hexagram, cosmicAlignment float64) float64 {
	// Factors affecting certainty
	hexagramResonance := ic.hexagramStates[hexagram.Number].Resonance
	oracleWisdom := ic.oracleField.WisdomLevel
	fieldCoherence := ic.oracleField.QuantumState.Coherence
	
	// Line clarity (how definitive the lines are)
	lineClarity := 0.0
	for _, line := range hexagram.Lines {
		lineClarity += line.Resonance
	}
	lineClarity /= 6.0
	
	certainty := 0.25*cosmicAlignment + 0.25*hexagramResonance + 0.2*oracleWisdom + 0.15*fieldCoherence + 0.15*lineClarity
	return math.Min(certainty, 1.0)
}

// updateWisdomMatrix updates the wisdom accumulation system
func (ic *IChingEngine) updateWisdomMatrix(divination *Divination) {
	ic.wisdomMatrix.TotalReadings++
	
	// Update wisdom level based on reading complexity
	wisdomGain := 0.01 // Base wisdom gain per reading
	if len(divination.ChangingLines) > 0 {
		wisdomGain += 0.005 * float64(len(divination.ChangingLines))
	}
	if divination.FutureHexagram != nil {
		wisdomGain += 0.01
	}
	
	ic.wisdomMatrix.WisdomLevel += wisdomGain
	ic.wisdomMatrix.WisdomLevel = math.Min(ic.wisdomMatrix.WisdomLevel, 1.0)
	
	// Update oracle field wisdom
	ic.oracleField.WisdomLevel = ic.wisdomMatrix.WisdomLevel
	
	// Record patterns
	if ic.config.PatternRecognition {
		pattern := fmt.Sprintf("%d", divination.PrimaryHexagram.Number)
		if divination.FutureHexagram != nil {
			pattern += fmt.Sprintf("->%d", divination.FutureHexagram.Number)
		}
		
		if wisdomPattern, exists := ic.wisdomMatrix.PatternRecognition[pattern]; exists {
			wisdomPattern.Frequency++
		} else {
			ic.wisdomMatrix.PatternRecognition[pattern] = &WisdomPattern{
				Pattern:   []int{divination.PrimaryHexagram.Number},
				Frequency: 1,
				Accuracy:  0.5,
				Insights:  make([]string, 0),
			}
		}
	}
	
	// Generate insights
	if ic.wisdomMatrix.TotalReadings%10 == 0 { // Every 10 readings
		insight := ic.generateInsight()
		ic.wisdomMatrix.Insights = append(ic.wisdomMatrix.Insights, insight)
	}
}

// generateInsight creates a new insight based on accumulated wisdom
func (ic *IChingEngine) generateInsight() *Insight {
	insightTypes := []string{"pattern", "correlation", "prediction"}
	insightType := insightTypes[ic.wisdomMatrix.TotalReadings%len(insightTypes)]
	
	var content string
	var confidence float64
	
	switch insightType {
	case "pattern":
		content = "Recurring patterns suggest cycles of transformation in human experience"
		confidence = ic.wisdomMatrix.WisdomLevel * 0.8
	case "correlation":
		content = "Cosmic timing and personal readiness often align in meaningful ways"
		confidence = ic.wisdomMatrix.WisdomLevel * 0.7
	case "prediction":
		content = "Future outcomes depend on present choices and natural timing"
		confidence = ic.wisdomMatrix.WisdomLevel * 0.6
	}
	
	return &Insight{
		ID:         fmt.Sprintf("insight_%d", time.Now().UnixNano()),
		Type:       insightType,
		Content:    content,
		Confidence: confidence,
		Evidence:   []string{"accumulated reading patterns", "cosmic correlations"},
		Timestamp:  time.Now(),
	}
}

// recordDivinationTelemetry records telemetry for the divination
func (ic *IChingEngine) recordDivinationTelemetry(divination *Divination) {
	yangCount := 0
	changingCount := 0
	
	for _, line := range divination.PrimaryHexagram.Lines {
		if line.IsYang {
			yangCount++
		}
		if line.IsChanging {
			changingCount++
		}
	}
	
	yinYangBalance := float64(yangCount) / 6.0
	changeIntensity := float64(changingCount) / 6.0
	
	point := types.TelemetryPoint{
		Step:              ic.currentReading,
		SymbolicEntropy:   1.0 - divination.Certainty, // Higher entropy = lower certainty
		LyapunovMetric:    divination.CosmicAlignment,
		SatisfactionRate:  divination.Interpretation.Confidence,
		ResonanceStrength: ic.oracleField.CosmicResonance,
		Dominance:         yinYangBalance,
		Timestamp:         time.Now(),
	}
	
	ic.telemetryPoints = append(ic.telemetryPoints, point)
	
	// Update evolution metrics
	ic.evolutionMetrics["wisdom_level"] = ic.wisdomMatrix.WisdomLevel
	ic.evolutionMetrics["cosmic_alignment"] = divination.CosmicAlignment
	ic.evolutionMetrics["reading_certainty"] = divination.Certainty
	ic.evolutionMetrics["yin_yang_balance"] = yinYangBalance
	ic.evolutionMetrics["change_intensity"] = changeIntensity
	
	ic.currentReading++
}

// getHexagramData returns the traditional hexagram data
func (ic *IChingEngine) getHexagramData() map[int]struct {
	name         string
	chineseName  string
	yinYangPattern []bool
	meaning      string
	judgment     string
	image        string
	keywords     []string
	element      string
	season       string
	direction    string
	planet       string
	relatedHexagrams []int
} {
	// Simplified hexagram data (in a full implementation, this would be much more comprehensive)
	return map[int]struct {
		name         string
		chineseName  string
		yinYangPattern []bool
		meaning      string
		judgment     string
		image        string
		keywords     []string
		element      string
		season       string
		direction    string
		planet       string
		relatedHexagrams []int
	}{
		1: {
			name:         "The Creative",
			chineseName:  "Qián",
			yinYangPattern: []bool{true, true, true, true, true, true}, // All yang
			meaning:      "Creative force, leadership, initiative",
			judgment:     "The Creative works sublime success, furthering through perseverance",
			image:        "Heaven moves vigorously; the superior person makes himself strong and untiring",
			keywords:     []string{"leadership", "creativity", "initiative", "strength"},
			element:      "Metal",
			season:       "Late autumn",
			direction:    "Northwest",
			planet:       "Jupiter",
			relatedHexagrams: []int{2, 43, 44},
		},
		2: {
			name:         "The Receptive",
			chineseName:  "Kūn",
			yinYangPattern: []bool{false, false, false, false, false, false}, // All yin
			meaning:      "Receptive force, nurturing, devotion",
			judgment:     "The Receptive brings sublime success, furthering through the perseverance of a mare",
			image:        "Earth's condition is receptive devotion; the superior person carries the outer world",
			keywords:     []string{"receptivity", "nurturing", "devotion", "support"},
			element:      "Earth",
			season:       "Late summer",
			direction:    "Southwest",
			planet:       "Saturn",
			relatedHexagrams: []int{1, 23, 24},
		},
		// ... (In a full implementation, all 64 hexagrams would be defined)
		// For brevity, including just a few examples
		3: {
			name:         "Difficulty at the Beginning",
			chineseName:  "Zhūn",
			yinYangPattern: []bool{true, false, false, false, true, false},
			meaning:      "Initial difficulties, birth pains, new beginnings",
			judgment:     "Difficulty at the Beginning works supreme success, furthering through perseverance",
			image:        "Thunder and rain fill the air; the superior person brings order out of confusion",
			keywords:     []string{"beginning", "difficulty", "perseverance", "growth"},
			element:      "Water",
			season:       "Early spring",
			direction:    "North",
			planet:       "Mercury",
			relatedHexagrams: []int{4, 5, 60},
		},
	}
}

// getTrigramData returns the eight trigram data
func (ic *IChingEngine) getTrigramData() map[int]struct {
	name         string
	chineseName  string
	yinYangPattern []bool
	element      string
	attribute    string
	family       string
	bodyPart     string
	animal       string
	direction    string
} {
	return map[int]struct {
		name         string
		chineseName  string
		yinYangPattern []bool
		element      string
		attribute    string
		family       string
		bodyPart     string
		animal       string
		direction    string
	}{
		0: { // ☷ Earth/Kun
			name:         "Earth",
			chineseName:  "Kūn",
			yinYangPattern: []bool{false, false, false},
			element:      "Earth",
			attribute:    "Devoted, yielding",
			family:       "Mother",
			bodyPart:     "Belly",
			animal:       "Mare",
			direction:    "Southwest",
		},
		1: { // ☶ Mountain/Gen
			name:         "Mountain",
			chineseName:  "Gèn",
			yinYangPattern: []bool{true, false, false},
			element:      "Earth",
			attribute:    "Keeping still",
			family:       "Youngest son",
			bodyPart:     "Hand",
			animal:       "Dog",
			direction:    "Northeast",
		},
		2: { // ☵ Water/Kan
			name:         "Water",
			chineseName:  "Kǎn",
			yinYangPattern: []bool{false, true, false},
			element:      "Water",
			attribute:    "Abysmal",
			family:       "Middle son",
			bodyPart:     "Ear",
			animal:       "Pig",
			direction:    "North",
		},
		3: { // ☴ Wind/Xun
			name:         "Wind",
			chineseName:  "Xùn",
			yinYangPattern: []bool{true, true, false},
			element:      "Wood",
			attribute:    "Gentle",
			family:       "Eldest daughter",
			bodyPart:     "Thigh",
			animal:       "Rooster",
			direction:    "Southeast",
		},
		4: { // ☳ Thunder/Zhen
			name:         "Thunder",
			chineseName:  "Zhèn",
			yinYangPattern: []bool{false, false, true},
			element:      "Wood",
			attribute:    "Arousing",
			family:       "Eldest son",
			bodyPart:     "Foot",
			animal:       "Dragon",
			direction:    "East",
		},
		5: { // ☲ Fire/Li
			name:         "Fire",
			chineseName:  "Lí",
			yinYangPattern: []bool{true, false, true},
			element:      "Fire",
			attribute:    "Clinging",
			family:       "Middle daughter",
			bodyPart:     "Eye",
			animal:       "Pheasant",
			direction:    "South",
		},
		6: { // ☱ Lake/Dui
			name:         "Lake",
			chineseName:  "Duì",
			yinYangPattern: []bool{false, true, true},
			element:      "Metal",
			attribute:    "Joyous",
			family:       "Youngest daughter",
			bodyPart:     "Mouth",
			animal:       "Sheep",
			direction:    "West",
		},
		7: { // ☰ Heaven/Qian
			name:         "Heaven",
			chineseName:  "Qián",
			yinYangPattern: []bool{true, true, true},
			element:      "Metal",
			attribute:    "Creative",
			family:       "Father",
			bodyPart:     "Head",
			animal:       "Horse",
			direction:    "Northwest",
		},
	}
}

// Utility and management methods

// GetTelemetry returns current telemetry data
func (ic *IChingEngine) GetTelemetry() []types.TelemetryPoint {
	ic.mu.RLock()
	defer ic.mu.RUnlock()
	
	telemetry := make([]types.TelemetryPoint, len(ic.telemetryPoints))
	copy(telemetry, ic.telemetryPoints)
	return telemetry
}

// GetCurrentState returns the current state of the I-Ching engine
func (ic *IChingEngine) GetCurrentState() map[string]interface{} {
	ic.mu.RLock()
	defer ic.mu.RUnlock()
	
	state := map[string]interface{}{
		"current_reading":   ic.currentReading,
		"elapsed_time":      time.Since(ic.startTime).Seconds(),
		"total_divinations": len(ic.divinations),
		"hexagram_states":   len(ic.hexagramStates),
		"trigram_states":    len(ic.trigramStates),
	}
	
	if ic.oracleField != nil {
		state["oracle_field"] = map[string]interface{}{
			"cosmic_resonance":   ic.oracleField.CosmicResonance,
			"temporal_flow":      ic.oracleField.TemporalFlow,
			"wisdom_level":       ic.oracleField.WisdomLevel,
			"divine_connection":  ic.oracleField.DivineConnection,
			"yin_yang_balance":   ic.oracleField.Balance.Balance,
			"elemental_balance":  ic.oracleField.FiveElements.Balance,
			"moon_phase":         ic.oracleField.CelestialInfluence.MoonPhase,
			"seasonal_energy":    ic.oracleField.CelestialInfluence.SeasonalEnergy,
		}
	}
	
	if ic.wisdomMatrix != nil {
		state["wisdom_matrix"] = map[string]interface{}{
			"total_readings":       ic.wisdomMatrix.TotalReadings,
			"wisdom_level":         ic.wisdomMatrix.WisdomLevel,
			"prediction_accuracy":  ic.wisdomMatrix.PredictionAccuracy,
			"pattern_count":        len(ic.wisdomMatrix.PatternRecognition),
			"insight_count":        len(ic.wisdomMatrix.Insights),
		}
	}
	
	if ic.currentHexagram != nil {
		state["current_hexagram"] = map[string]interface{}{
			"number":       ic.currentHexagram.Number,
			"name":         ic.currentHexagram.Name,
			"chinese_name": ic.currentHexagram.ChineseName,
			"changing_lines": ic.currentHexagram.ChangingLines,
		}
	}
	
	state["evolution_metrics"] = ic.evolutionMetrics
	
	return state
}

// Reset resets the I-Ching engine to initial state
func (ic *IChingEngine) Reset() {
	ic.mu.Lock()
	defer ic.mu.Unlock()
	
	ic.hexagramStates = make(map[int]*HexagramState)
	ic.trigramStates = make(map[int]*TrigramState)
	ic.oracleField = nil
	ic.divinations = make([]*Divination, 0)
	ic.hexagramEvolution = nil
	ic.wisdomMatrix = nil
	ic.currentReading = 0
	ic.currentHexagram = nil
	ic.telemetryPoints = make([]types.TelemetryPoint, 0)
	ic.evolutionMetrics = make(map[string]float64)
}

// SetConfig updates the I-Ching configuration
func (ic *IChingEngine) SetConfig(config *IChingConfig) {
	ic.mu.Lock()
	defer ic.mu.Unlock()
	
	if config != nil {
		ic.config = config
	}
}

// GetConfig returns the current I-Ching configuration
func (ic *IChingEngine) GetConfig() *IChingConfig {
	ic.mu.RLock()
	defer ic.mu.RUnlock()
	
	configCopy := *ic.config
	return &configCopy
}

// GetHexagram returns a hexagram by number
func (ic *IChingEngine) GetHexagram(number int) (*HexagramState, error) {
	ic.mu.RLock()
	defer ic.mu.RUnlock()
	
	if hexagram, exists := ic.hexagramStates[number]; exists {
		return hexagram, nil
	}
	
	return nil, fmt.Errorf("hexagram %d not found", number)
}

// GetDivinations returns all divinations
func (ic *IChingEngine) GetDivinations() []*Divination {
	ic.mu.RLock()
	defer ic.mu.RUnlock()
	
	divinations := make([]*Divination, len(ic.divinations))
	copy(divinations, ic.divinations)
	return divinations
}

// GetWisdomInsights returns accumulated wisdom insights
func (ic *IChingEngine) GetWisdomInsights() []*Insight {
	ic.mu.RLock()
	defer ic.mu.RUnlock()
	
	if ic.wisdomMatrix != nil {
		insights := make([]*Insight, len(ic.wisdomMatrix.Insights))
		copy(insights, ic.wisdomMatrix.Insights)
		