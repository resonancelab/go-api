package iching

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

// initializeOracleField sets up the unified I-Ching quantum field
func (ic *IChingEngine) initializeOracleField() error {
	// Create quantum state for oracle field
	amplitudes := ic.generateOracleFieldAmplitudes()
	quantumState, err := ic.resonanceEngine.CreateQuantumState(amplitudes)
	if err != nil {
		return fmt.Errorf("failed to create oracle field state: %w", err)
	}

	// Initialize Yin-Yang balance
	balance := &YinYangBalance{
		YinStrength:  0.5,
		YangStrength: 0.5,
		Balance:      0.0, // Perfect balance
		Harmony:      0.8,
		Flow:         0.6,
	}

	// Initialize Five Elements
	elements := &ElementalMatrix{
		Wood:            0.2,
		Fire:            0.2,
		Earth:           0.2,
		Metal:           0.2,
		Water:           0.2,
		GenerativeFlow:  0.7,
		DestructiveFlow: 0.1,
		Balance:         0.8,
	}

	// Initialize celestial influences
	celestial := &CelestialState{
		MoonPhase:          ic.calculateCurrentMoonPhase(),
		SolarInfluence:     0.6,
		PlanetaryAlignment: ic.calculatePlanetaryAlignment(),
		SeasonalEnergy:     ic.calculateSeasonalEnergy(),
		CosmicResonance:    0.7,
	}

	ic.oracleField = &OracleField{
		ID:                 "cosmic_oracle_field",
		QuantumState:       quantumState,
		CosmicResonance:    0.7,
		TemporalFlow:       0.6,
		CausalConnections:  make(map[string]float64),
		WisdomLevel:        0.3, // Will grow with use
		DivineConnection:   0.8,
		Balance:            balance,
		FiveElements:       elements,
		CelestialInfluence: celestial,
	}

	return nil
}

// generateOracleFieldAmplitudes creates quantum amplitudes for oracle field
func (ic *IChingEngine) generateOracleFieldAmplitudes() []complex128 {
	dimension := ic.resonanceEngine.GetDimension()
	amplitudes := make([]complex128, dimension)

	// Create oracle field using cosmic harmony principles
	phi := (1.0 + math.Sqrt(5.0)) / 2.0 // Golden ratio for cosmic harmony

	normFactor := 0.0

	for i := 0; i < dimension; i++ {
		// Each amplitude corresponds to a hexagram (1-64) or cosmic principle
		hexagramIndex := i % 64

		// Oracle field amplitude based on hexagram significance and cosmic harmony
		amplitude := math.Exp(-float64(i)/(phi*64.0)) * (1.0 + 0.05*rand.Float64())

		// Phase based on hexagram number and cosmic cycles
		phase := 2.0 * math.Pi * float64(hexagramIndex) / 64.0
		phase += phi * float64(i) / float64(dimension) // Add golden ratio modulation

		amplitudes[i] = complex(amplitude*math.Cos(phase), amplitude*math.Sin(phase))
		normFactor += amplitude * amplitude
	}

	// Normalize
	if normFactor > 0 {
		normFactor = math.Sqrt(normFactor)
		for i := range amplitudes {
			amplitudes[i] /= complex(normFactor, 0)
		}
	}

	return amplitudes
}

// calculateCurrentMoonPhase calculates current moon phase (0-1)
func (ic *IChingEngine) calculateCurrentMoonPhase() float64 {
	// Simplified lunar cycle calculation
	now := time.Now()
	// Lunar month is approximately 29.53 days
	lunarCycle := 29.53 * 24 * time.Hour

	// Reference new moon (approximate)
	referenceNewMoon := time.Date(2024, 1, 11, 0, 0, 0, 0, time.UTC)

	elapsed := now.Sub(referenceNewMoon)
	cycles := elapsed.Hours() / lunarCycle.Hours()
	phase := cycles - math.Floor(cycles) // Fractional part

	return phase
}

// calculatePlanetaryAlignment calculates planetary influence
func (ic *IChingEngine) calculatePlanetaryAlignment() float64 {
	// Simplified planetary alignment calculation
	// In a real implementation, this would use actual astronomical data
	now := time.Now()
	dayOfYear := now.YearDay()

	// Create a pseudo-planetary cycle
	alignment := 0.5 + 0.3*math.Sin(2.0*math.Pi*float64(dayOfYear)/365.0)
	alignment += 0.2 * math.Sin(2.0*math.Pi*float64(dayOfYear)/687.0) // Mars cycle approximation

	return math.Max(0.0, math.Min(1.0, alignment))
}

// calculateSeasonalEnergy calculates seasonal energy influence
func (ic *IChingEngine) calculateSeasonalEnergy() float64 {
	now := time.Now()
	dayOfYear := float64(now.YearDay())

	// Seasonal energy peaks at solstices and equinoxes
	springEquinox := 80.0   // Approximately March 21
	summerSolstice := 172.0 // Approximately June 21
	autumnEquinox := 266.0  // Approximately September 23
	winterSolstice := 355.0 // Approximately December 21

	// Calculate distance to nearest seasonal milestone
	distances := []float64{
		math.Abs(dayOfYear - springEquinox),
		math.Abs(dayOfYear - summerSolstice),
		math.Abs(dayOfYear - autumnEquinox),
		math.Abs(dayOfYear - winterSolstice),
	}

	minDistance := distances[0]
	for _, d := range distances[1:] {
		if d < minDistance {
			minDistance = d
		}
	}

	// Energy is higher closer to seasonal transitions
	energy := 1.0 - (minDistance / 45.0) // Peak energy within 45 days of transition
	return math.Max(0.3, math.Min(1.0, energy))
}

// initializeHexagramStates creates quantum states for all 64 hexagrams
func (ic *IChingEngine) initializeHexagramStates() error {
	// Initialize trigram states first
	if err := ic.initializeTrigramStates(); err != nil {
		return fmt.Errorf("failed to initialize trigram states: %w", err)
	}

	// Create all 64 hexagram states
	hexagramData := ic.getHexagramData()

	for i := 1; i <= 64; i++ {
		data := hexagramData[i]

		// Create quantum state for hexagram
		amplitudes := ic.generateHexagramAmplitudes(i, data.yinYangPattern)
		quantumState, err := ic.resonanceEngine.CreateQuantumState(amplitudes)
		if err != nil {
			return fmt.Errorf("failed to create hexagram %d state: %w", i, err)
		}

		// Get trigrams
		lowerTrigramNum := ic.getLowerTrigramNumber(data.yinYangPattern)
		upperTrigramNum := ic.getUpperTrigramNumber(data.yinYangPattern)

		hexagramState := &HexagramState{
			HexagramNumber:   i,
			Name:             data.name,
			ChineseName:      data.chineseName,
			QuantumState:     quantumState,
			YinYangPattern:   data.yinYangPattern,
			UpperTrigram:     ic.trigramStates[upperTrigramNum],
			LowerTrigram:     ic.trigramStates[lowerTrigramNum],
			Meaning:          data.meaning,
			Judgment:         data.judgment,
			Image:            data.image,
			Keywords:         data.keywords,
			Element:          data.element,
			Season:           data.season,
			Direction:        data.direction,
			Planet:           data.planet,
			Resonance:        0.0,
			Coherence:        quantumState.Coherence,
			Activation:       0.0,
			ChangingLines:    make([]int, 0),
			RelatedHexagrams: data.relatedHexagrams,
		}

		ic.hexagramStates[i] = hexagramState
	}

	return nil
}

// initializeTrigramStates creates quantum states for all 8 trigrams
func (ic *IChingEngine) initializeTrigramStates() error {
	trigramData := ic.getTrigramData()

	for i := 0; i < 8; i++ {
		data := trigramData[i]

		// Create quantum state for trigram
		amplitudes := ic.generateTrigramAmplitudes(i, data.yinYangPattern)
		quantumState, err := ic.resonanceEngine.CreateQuantumState(amplitudes)
		if err != nil {
			return fmt.Errorf("failed to create trigram %d state: %w", i, err)
		}

		trigramState := &TrigramState{
			TrigramNumber:  i,
			Name:           data.name,
			ChineseName:    data.chineseName,
			QuantumState:   quantumState,
			YinYangPattern: data.yinYangPattern,
			Element:        data.element,
			Attribute:      data.attribute,
			Family:         data.family,
			BodyPart:       data.bodyPart,
			Animal:         data.animal,
			Direction:      data.direction,
			Resonance:      0.0,
		}

		ic.trigramStates[i] = trigramState
	}

	return nil
}

// generateHexagramAmplitudes creates quantum amplitudes for a hexagram
func (ic *IChingEngine) generateHexagramAmplitudes(hexagramNum int, yinYangPattern []bool) []complex128 {
	dimension := ic.resonanceEngine.GetDimension()
	amplitudes := make([]complex128, dimension)

	normFactor := 0.0

	// Calculate hexagram characteristics
	yangCount := 0
	for _, isYang := range yinYangPattern {
		if isYang {
			yangCount++
		}
	}
	yinYangRatio := float64(yangCount) / float64(len(yinYangPattern))

	for i := 0; i < dimension; i++ {
		prime := float64(ic.resonanceEngine.GetPrimes().GetNthPrime(i))

		// Hexagram amplitude based on yin-yang pattern and number significance
		amplitude := 1.0

		// Modulate by hexagram position in the 64
		positionFactor := math.Exp(-float64(i)/float64(dimension)) * (1.0 + 0.1*yinYangRatio)

		// Add pattern-specific modulation
		patternPhase := 0.0
		for j, isYang := range yinYangPattern {
			if isYang {
				patternPhase += math.Pi / float64(j+1)
			} else {
				patternPhase -= math.Pi / float64(j+1)
			}
		}

		amplitude = positionFactor * (1.0 + 0.2*rand.Float64())

		// Phase based on hexagram number, prime, and pattern
		phase := 2.0*math.Pi*float64(hexagramNum)/64.0 + prime/100.0 + patternPhase

		amplitudes[i] = complex(amplitude*math.Cos(phase), amplitude*math.Sin(phase))
		normFactor += amplitude * amplitude
	}

	// Normalize
	if normFactor > 0 {
		normFactor = math.Sqrt(normFactor)
		for i := range amplitudes {
			amplitudes[i] /= complex(normFactor, 0)
		}
	}

	return amplitudes
}

// generateTrigramAmplitudes creates quantum amplitudes for a trigram
func (ic *IChingEngine) generateTrigramAmplitudes(trigramNum int, yinYangPattern []bool) []complex128 {
	dimension := ic.resonanceEngine.GetDimension()
	amplitudes := make([]complex128, dimension)

	normFactor := 0.0

	for i := 0; i < dimension; i++ {
		prime := float64(ic.resonanceEngine.GetPrimes().GetNthPrime(i))

		// Trigram amplitude
		amplitude := math.Exp(-float64(i)/float64(dimension/8)) * (1.0 + 0.15*rand.Float64())

		// Phase based on trigram characteristics
		phase := 2.0*math.Pi*float64(trigramNum)/8.0 + prime/50.0

		amplitudes[i] = complex(amplitude*math.Cos(phase), amplitude*math.Sin(phase))
		normFactor += amplitude * amplitude
	}

	// Normalize
	if normFactor > 0 {
		normFactor = math.Sqrt(normFactor)
		for i := range amplitudes {
			amplitudes[i] /= complex(normFactor, 0)
		}
	}

	return amplitudes
}

// getLowerTrigramNumber extracts lower trigram number from hexagram pattern
func (ic *IChingEngine) getLowerTrigramNumber(pattern []bool) int {
	// Lower trigram is lines 0, 1, 2 (bottom three)
	num := 0
	for i := 0; i < 3 && i < len(pattern); i++ {
		if pattern[i] {
			num += 1 << uint(i)
		}
	}
	return num
}

// getUpperTrigramNumber extracts upper trigram number from hexagram pattern
func (ic *IChingEngine) getUpperTrigramNumber(pattern []bool) int {
	// Upper trigram is lines 3, 4, 5 (top three)
	num := 0
	for i := 3; i < 6 && i < len(pattern); i++ {
		if pattern[i] {
			num += 1 << uint(i-3)
		}
	}
	return num
}

// initializeWisdomMatrix sets up the wisdom accumulation system
func (ic *IChingEngine) initializeWisdomMatrix() error {
	ic.wisdomMatrix = &WisdomMatrix{
		TotalReadings:      0,
		PatternRecognition: make(map[string]*WisdomPattern),
		PredictionAccuracy: 0.5, // Start neutral
		WisdomLevel:        0.1, // Start low
		Insights:           make([]*Insight, 0),
		Correlations:       make(map[string]map[string]float64),
		KnowledgeGraph: &KnowledgeGraph{
			Nodes:        make(map[string]*KnowledgeNode),
			Edges:        make([]*KnowledgeEdge, 0),
			Centrality:   make(map[string]float64),
			Connectivity: 0.0,
		},
	}

	// Initialize hexagram nodes in knowledge graph
	for i := 1; i <= 64; i++ {
		nodeID := fmt.Sprintf("hexagram_%d", i)
		node := &KnowledgeNode{
			ID:          nodeID,
			Type:        "hexagram",
			Properties:  make(map[string]interface{}),
			Connections: make([]string, 0),
			Weight:      1.0,
		}

		if hexState, exists := ic.hexagramStates[i]; exists {
			node.Properties["name"] = hexState.Name
			node.Properties["element"] = hexState.Element
			node.Properties["keywords"] = hexState.Keywords
		}

		ic.wisdomMatrix.KnowledgeGraph.Nodes[nodeID] = node
	}

	return nil
}

// performDivination performs the actual I-Ching reading
func (ic *IChingEngine) performDivination(question, context, querent string) (*DivinationResult, error) {
	timeout := time.Duration(ic.config.TimeoutSeconds) * time.Second

	// Generate primary hexagram
	primaryHexagram, err := ic.generateHexagram(question, context)
	if err != nil {
		return nil, fmt.Errorf("failed to generate primary hexagram: %w", err)
	}

	ic.currentHexagram = primaryHexagram

	// Determine changing lines
	changingLines := ic.determineChangingLines(primaryHexagram)

	// Generate future hexagram if there are changing lines
	var futureHexagram *Hexagram
	if len(changingLines) > 0 {
		futureHexagram, err = ic.generateFutureHexagram(primaryHexagram, changingLines)
		if err != nil {
			return nil, fmt.Errorf("failed to generate future hexagram: %w", err)
		}
	}

	// Calculate cosmic alignment
	cosmicAlignment := ic.calculateCosmicAlignment(primaryHexagram)

	// Generate interpretation
	interpretation := ic.generateInterpretation(primaryHexagram, futureHexagram, context, question)

	// Generate wisdom message
	wisdomMessage := ic.generateWisdomMessage(primaryHexagram, futureHexagram, interpretation)

	// Calculate certainty
	certainty := ic.calculateDivinationCertainty(primaryHexagram, cosmicAlignment)

	// Create divination record
	divination := &Divination{
		ID:              fmt.Sprintf("div_%d", time.Now().UnixNano()),
		Question:        question,
		Querent:         querent,
		Context:         context,
		Method:          ic.config.DivinationMethod,
		PrimaryHexagram: primaryHexagram,
		FutureHexagram:  futureHexagram,
		ChangingLines:   changingLines,
		Interpretation:  interpretation,
		WisdomMessage:   wisdomMessage,
		CosmicAlignment: cosmicAlignment,
		Certainty:       certainty,
		Timestamp:       time.Now(),
		ExpirationTime:  time.Now().Add(time.Duration(ic.config.ReadingValidityHours) * time.Hour),
	}

	ic.divinations = append(ic.divinations, divination)

	// Update wisdom matrix
	ic.updateWisdomMatrix(divination)

	// Record telemetry
	ic.recordDivinationTelemetry(divination)

	// Generate result
	result := &DivinationResult{
		SessionID:        divination.ID,
		Question:         question,
		PrimaryHexagram:  primaryHexagram,
		FutureHexagram:   futureHexagram,
		ChangingLines:    changingLines,
		Interpretation:   interpretation,
		WisdomMessage:    wisdomMessage,
		CosmicAlignment:  cosmicAlignment,
		Certainty:        certainty,
		PredictedOutcome: interpretation.Outcome,
		Guidance:         interpretation.Guidance,
		Timing:           interpretation.Timing,
		ProcessingTime:   time.Since(ic.startTime).Seconds(),
		Success:          certainty > ic.config.CertaintyThreshold,
	}

	return result, nil
}

// generateHexagram generates a hexagram through quantum collapse
func (ic *IChingEngine) generateHexagram(question, context string) (*Hexagram, error) {
	// Hash question and context for deterministic randomness
	questionHash := ic.hashString(question + context)

	// Generate six lines through quantum measurement
	lines := [6]LineType{}

	for i := 0; i < 6; i++ {
		line := ic.generateLine(questionHash, i)
		lines[i] = line
	}

	// Determine hexagram number from line pattern
	hexagramNumber := ic.calculateHexagramNumber(lines)

	// Get hexagram state
	hexagramState, exists := ic.hexagramStates[hexagramNumber]
	if !exists {
		return nil, fmt.Errorf("hexagram %d not found", hexagramNumber)
	}

	// Create trigrams
	lowerTrigram := ic.createTrigramFromLines(lines[0:3])
	upperTrigram := ic.createTrigramFromLines(lines[3:6])

	// Create hexagram meaning
	meaning := &HexagramMeaning{
		CoreMeaning: hexagramState.Meaning,
		Judgment:    hexagramState.Judgment,
		Image:       hexagramState.Image,
		Commentary:  ic.generateCommentary(hexagramState, question),
		Keywords:    hexagramState.Keywords,
		Advice:      ic.generateAdvice(hexagramState, context),
		Warning:     ic.generateWarning(hexagramState),
		Opportunity: ic.generateOpportunity(hexagramState),
		Challenge:   ic.generateChallenge(hexagramState),
	}

	hexagram := &Hexagram{
		Number:         hexagramNumber,
		Lines:          lines,
		Name:           hexagramState.Name,
		ChineseName:    hexagramState.ChineseName,
		Trigrams:       [2]*Trigram{lowerTrigram, upperTrigram},
		Meaning:        meaning,
		Interpretation: nil, // Will be set later
		ChangingLines:  make([]int, 0),
		FutureHexagram: nil,
		Timestamp:      time.Now(),
	}

	return hexagram, nil
}

// generateLine generates a single yin/yang line with changing potential
func (ic *IChingEngine) generateLine(questionHash uint64, lineIndex int) LineType {
	// Use quantum state collapse simulation
	seed := questionHash + uint64(lineIndex)*1000 + uint64(time.Now().Nanosecond())
	rand.Seed(int64(seed))

	// Quantum probability for yin/yang
	probability := rand.Float64()

	// Cosmic influences
	cosmicInfluence := ic.oracleField.CosmicResonance
	lunarInfluence := ic.oracleField.CelestialInfluence.MoonPhase

	// Adjust probability based on cosmic factors
	adjustedProbability := probability + cosmicInfluence*0.1 + lunarInfluence*0.05

	isYang := adjustedProbability > 0.5

	// Determine if line is changing (rarer event)
	changeProbability := rand.Float64()
	isChanging := changeProbability < 0.125 // 1/8 probability for changing line

	lineType := "yin"
	if isYang && !isChanging {
		lineType = "yang"
	} else if !isYang && isChanging {
		lineType = "changing_yin"
	} else if isYang && isChanging {
		lineType = "changing_yang"
	}

	return LineType{
		Type:        lineType,
		IsYang:      isYang,
		IsChanging:  isChanging,
		Probability: adjustedProbability,
		Resonance:   ic.calculateLineResonance(lineIndex, isYang),
	}
}

// hashString creates a hash from a string for deterministic randomness
func (ic *IChingEngine) hashString(s string) uint64 {
	hash := uint64(5381)
	for _, char := range s {
		hash = ((hash << 5) + hash) + uint64(char)
	}
	return hash
}

// calculateHexagramNumber determines hexagram number from six lines
func (ic *IChingEngine) calculateHexagramNumber(lines [6]LineType) int {
	// Convert lines to binary number (yang=1, yin=0)
	binary := 0
	for i, line := range lines {
		if line.IsYang {
			binary += 1 << uint(i)
		}
	}

	// Map binary to traditional hexagram numbering (King Wen sequence)
	// This is a simplified mapping - real implementation would use the traditional sequence
	hexagramNumber := (binary % 64) + 1

	return hexagramNumber
}

// createTrigramFromLines creates a trigram from three lines
func (ic *IChingEngine) createTrigramFromLines(lines []LineType) *Trigram {
	if len(lines) != 3 {
		return nil
	}

	// Convert to trigram pattern
	trigramLines := [3]LineType{lines[0], lines[1], lines[2]}

	// Calculate trigram number
	trigramNumber := 0
	for i, line := range trigramLines {
		if line.IsYang {
			trigramNumber += 1 << uint(i)
		}
	}

	// Get trigram data
	trigramData := ic.getTrigramData()[trigramNumber]

	trigram := &Trigram{
		Number:      trigramNumber,
		Lines:       trigramLines,
		Name:        trigramData.name,
		ChineseName: trigramData.chineseName,
		Element:     trigramData.element,
		Attribute:   trigramData.attribute,
		Direction:   trigramData.direction,
	}

	return trigram
}

// calculateLineResonance calculates resonance for a line
func (ic *IChingEngine) calculateLineResonance(lineIndex int, isYang bool) float64 {
	// Line resonance based on position and type
	positionWeight := []float64{0.1, 0.15, 0.2, 0.25, 0.3, 0.35}[lineIndex] // Higher lines have more influence

	typeResonance := 0.5
	if isYang {
		typeResonance = 0.6
	} else {
		typeResonance = 0.4
	}

	return positionWeight * typeResonance * ic.oracleField.CosmicResonance
}

// determineChangingLines identifies which lines are changing
func (ic *IChingEngine) determineChangingLines(hexagram *Hexagram) []int {
	changingLines := make([]int, 0)

	for i, line := range hexagram.Lines {
		if line.IsChanging {
			changingLines = append(changingLines, i+1) // 1-indexed
		}
	}

	return changingLines
}

// generateFutureHexagram creates the hexagram after changes
func (ic *IChingEngine) generateFutureHexagram(primary *Hexagram, changingLines []int) (*Hexagram, error) {
	// Create new lines with changes applied
	newLines := primary.Lines

	for _, lineNum := range changingLines {
		lineIndex := lineNum - 1 // Convert to 0-indexed
		if lineIndex >= 0 && lineIndex < 6 {
			// Flip the line
			line := newLines[lineIndex]
			line.IsYang = !line.IsYang
			line.IsChanging = false // No longer changing

			if line.IsYang {
				line.Type = "yang"
			} else {
				line.Type = "yin"
			}

			newLines[lineIndex] = line
		}
	}

	// Calculate new hexagram number
	newHexagramNumber := ic.calculateHexagramNumber(newLines)

	// Get new hexagram state
	hexagramState, exists := ic.hexagramStates[newHexagramNumber]
	if !exists {
		return nil, fmt.Errorf("future hexagram %d not found", newHexagramNumber)
	}

	// Create trigrams
	lowerTrigram := ic.createTrigramFromLines(newLines[0:3])
	upperTrigram := ic.createTrigramFromLines(newLines[3:6])

	// Create meaning
	meaning := &HexagramMeaning{
		CoreMeaning: hexagramState.Meaning,
		Judgment:    hexagramState.Judgment,
		Image:       hexagramState.Image,
		Commentary:  "Future state after transformation",
		Keywords:    hexagramState.Keywords,
		Advice:      "Embrace the coming change",
		Warning:     "Change brings both opportunity and challenge",
		Opportunity: "New possibilities will emerge",
		Challenge:   "Adaptation will be required",
	}

	futureHexagram := &Hexagram{
		Number:         newHexagramNumber,
		Lines:          newLines,
		Name:           hexagramState.Name,
		ChineseName:    hexagramState.ChineseName,
		Trigrams:       [2]*Trigram{lowerTrigram, upperTrigram},
		Meaning:        meaning,
		Interpretation: nil,
		ChangingLines:  make([]int, 0),
		FutureHexagram: nil,
		Timestamp:      time.Now(),
	}

	return futureHexagram, nil
}
