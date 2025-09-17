package qcr

import (
	"fmt"
	"math"
	"time"

	"github.com/resonancelab/psizero/shared/types"
)

// Complete the checkConsciousnessConvergence function
func (qcr *QCREngine) checkConsciousnessConvergence() bool {
	if len(qcr.telemetryPoints) < 50 {
		return false
	}

	// Check for stability in consciousness level
	recent := qcr.telemetryPoints[len(qcr.telemetryPoints)-20:]

	variance := 0.0
	avgLevel := 0.0

	for _, point := range recent {
		avgLevel += point.Dominance // consciousness level stored in Dominance field
	}
	avgLevel /= float64(len(recent))

	for _, point := range recent {
		diff := point.Dominance - avgLevel
		variance += diff * diff
	}
	variance /= float64(len(recent))

	// Converged if variance is low and consciousness level is stable
	return variance < 0.001 && avgLevel > 0.8
}

// generateConsciousnessResult creates the final simulation result
func (qcr *QCREngine) generateConsciousnessResult(emergentPhenomena []string, memoryFormations, decisionsMade int) *ConsciousnessSimulationResult {
	// Calculate final emotional evolution
	emotionalEvolution := make(map[string]float64)

	if len(qcr.consciousEntities) > 0 {
		for emotion := range qcr.consciousEntities[0].EmotionalState.PrimaryEmotions {
			totalEmotionChange := 0.0
			for _, entity := range qcr.consciousEntities {
				if entity.EmotionalState != nil {
					if value, exists := entity.EmotionalState.PrimaryEmotions[emotion]; exists {
						totalEmotionChange += value
					}
				}
			}
			emotionalEvolution[emotion] = totalEmotionChange / float64(len(qcr.consciousEntities))
		}
	}

	// Calculate average self-awareness
	avgSelfAwareness := 0.0
	for _, entity := range qcr.consciousEntities {
		if entity.SelfModel != nil {
			avgSelfAwareness += entity.SelfModel.SelfAwareness
		}
	}
	if len(qcr.consciousEntities) > 0 {
		avgSelfAwareness /= float64(len(qcr.consciousEntities))
	}

	converged := qcr.checkConsciousnessConvergence()

	result := &ConsciousnessSimulationResult{
		SessionID:               fmt.Sprintf("consciousness_%d", time.Now().Unix()),
		FinalConsciousnessLevel: qcr.consciousnessField.ConsciousnessLevel,
		IntegratedInformation:   qcr.consciousnessField.IntegratedInfo,
		AwarenessNetworks:       len(qcr.awarenessNetworks),
		EmergentPhenomena:       emergentPhenomena,
		ObserverEffects:         len(qcr.observerEffects),
		MemoryFormation:         memoryFormations,
		DecisionsMade:           decisionsMade,
		EmotionalEvolution:      emotionalEvolution,
		ConsciousnessCoherence:  qcr.consciousnessField.Coherence,
		SelfAwarenessLevel:      avgSelfAwareness,
		ProcessingTime:          time.Since(qcr.startTime).Seconds(),
		Success:                 converged && qcr.consciousnessField.ConsciousnessLevel > 0.7,
	}

	return result
}

// GetTelemetry returns current telemetry data
func (qcr *QCREngine) GetTelemetry() []types.TelemetryPoint {
	qcr.mu.RLock()
	defer qcr.mu.RUnlock()

	telemetry := make([]types.TelemetryPoint, len(qcr.telemetryPoints))
	copy(telemetry, qcr.telemetryPoints)
	return telemetry
}

// GetCurrentState returns the current state of the QCR engine
func (qcr *QCREngine) GetCurrentState() map[string]interface{} {
	qcr.mu.RLock()
	defer qcr.mu.RUnlock()

	state := map[string]interface{}{
		"current_cycle":      qcr.currentCycle,
		"evolution_phase":    qcr.evolutionPhase,
		"elapsed_time":       time.Since(qcr.startTime).Seconds(),
		"conscious_entities": len(qcr.consciousEntities),
		"awareness_networks": len(qcr.awarenessNetworks),
		"observer_effects":   len(qcr.observerEffects),
	}

	if qcr.consciousnessField != nil {
		state["consciousness_field"] = map[string]interface{}{
			"consciousness_level": qcr.consciousnessField.ConsciousnessLevel,
			"integrated_info":     qcr.consciousnessField.IntegratedInfo,
			"coherence":           qcr.consciousnessField.Coherence,
			"intensity":           qcr.consciousnessField.Intensity,
			"resonance":           qcr.consciousnessField.Resonance,
			"emergent_properties": qcr.consciousnessField.EmergentProperties,
		}
	}

	if qcr.memoryMatrix != nil {
		state["memory_matrix"] = map[string]interface{}{
			"total_capacity":    qcr.memoryMatrix.TotalCapacity,
			"used_capacity":     qcr.memoryMatrix.UsedCapacity,
			"short_term_memory": len(qcr.memoryMatrix.ShortTermMemory),
			"long_term_memory":  len(qcr.memoryMatrix.LongTermMemory),
			"episodic_memory":   len(qcr.memoryMatrix.EpisodicMemory),
			"semantic_memory":   len(qcr.memoryMatrix.SemanticMemory),
			"procedural_memory": len(qcr.memoryMatrix.ProceduralMemory),
			"compression_ratio": qcr.memoryMatrix.CompressionRatio,
		}
	}

	// Entity statistics
	if len(qcr.consciousEntities) > 0 {
		avgAwareness := 0.0
		avgProcessingSpeed := 0.0

		for _, entity := range qcr.consciousEntities {
			avgAwareness += entity.AwarenessLevel
			avgProcessingSpeed += entity.ProcessingSpeed
		}

		count := float64(len(qcr.consciousEntities))
		state["entity_stats"] = map[string]interface{}{
			"avg_awareness":        avgAwareness / count,
			"avg_processing_speed": avgProcessingSpeed / count,
		}
	}

	state["consciousness_metrics"] = qcr.consciousnessMetrics

	return state
}

// GetConsciousnessMetrics returns detailed consciousness metrics
func (qcr *QCREngine) GetConsciousnessMetrics() map[string]float64 {
	qcr.mu.RLock()
	defer qcr.mu.RUnlock()

	metrics := make(map[string]float64)
	for key, value := range qcr.consciousnessMetrics {
		metrics[key] = value
	}

	// Add current field metrics
	if qcr.consciousnessField != nil {
		metrics["current_consciousness_level"] = qcr.consciousnessField.ConsciousnessLevel
		metrics["current_integrated_info"] = qcr.consciousnessField.IntegratedInfo
		metrics["current_field_coherence"] = qcr.consciousnessField.Coherence
		metrics["current_field_resonance"] = qcr.consciousnessField.Resonance

		for property, value := range qcr.consciousnessField.EmergentProperties {
			metrics["emergent_"+property] = value
		}
	}

	return metrics
}

// Reset resets the QCR engine to initial state
func (qcr *QCREngine) Reset() {
	qcr.mu.Lock()
	defer qcr.mu.Unlock()

	qcr.consciousnessField = nil
	qcr.consciousEntities = make([]*ConsciousEntity, 0)
	qcr.awarenessNetworks = make([]*AwarenessNetwork, 0)
	qcr.memoryMatrix = nil
	qcr.observerEffects = make([]*ObserverEffect, 0)
	qcr.currentCycle = 0
	qcr.evolutionPhase = "singularity"
	qcr.telemetryPoints = make([]types.TelemetryPoint, 0)
	qcr.consciousnessMetrics = make(map[string]float64)
}

// SetConfig updates the QCR configuration
func (qcr *QCREngine) SetConfig(config *QCRConfig) {
	qcr.mu.Lock()
	defer qcr.mu.Unlock()

	if config != nil {
		qcr.config = config
	}
}

// GetConfig returns the current QCR configuration
func (qcr *QCREngine) GetConfig() *QCRConfig {
	qcr.mu.RLock()
	defer qcr.mu.RUnlock()

	configCopy := *qcr.config
	return &configCopy
}

// GetEntity returns a conscious entity by ID
func (qcr *QCREngine) GetEntity(entityID string) (*ConsciousEntity, error) {
	qcr.mu.RLock()
	defer qcr.mu.RUnlock()

	for _, entity := range qcr.consciousEntities {
		if entity.ID == entityID {
			return entity, nil
		}
	}

	return nil, fmt.Errorf("entity not found: %s", entityID)
}

// UpdateEntityAwareness manually updates an entity's awareness level
func (qcr *QCREngine) UpdateEntityAwareness(entityID string, awarenessLevel float64) error {
	qcr.mu.Lock()
	defer qcr.mu.Unlock()

	for _, entity := range qcr.consciousEntities {
		if entity.ID == entityID {
			entity.AwarenessLevel = math.Max(0.0, math.Min(1.0, awarenessLevel))
			entity.LastUpdate = time.Now()
			return nil
		}
	}

	return fmt.Errorf("entity not found: %s", entityID)
}

// TriggerObserverEffect manually triggers an observer effect
func (qcr *QCREngine) TriggerObserverEffect(observerID, observedSystem, measurementType string) (*ObserverEffect, error) {
	qcr.mu.Lock()
	defer qcr.mu.Unlock()

	// Find observer entity
	var observer *ConsciousEntity
	for _, entity := range qcr.consciousEntities {
		if entity.ID == observerID {
			observer = entity
			break
		}
	}

	if observer == nil {
		return nil, fmt.Errorf("observer entity not found: %s", observerID)
	}

	// Create observer effect
	effect := &ObserverEffect{
		ID:              fmt.Sprintf("manual_obs_%s_%d", observerID, time.Now().UnixNano()),
		ObserverID:      observerID,
		ObservedSystem:  observedSystem,
		MeasurementType: measurementType,
		CollapseFunction: &CollapseFunction{
			Type:        "manual",
			Probability: observer.AwarenessLevel,
		},
		QuantumCorrelations:  make([]QuantumCorrelation, 0),
		ConsciousnessEffects: make(map[string]float64),
		Timestamp:            time.Now(),
		Duration:             time.Duration(200) * time.Millisecond,
	}

	// Apply effect to consciousness field
	if observedSystem == "consciousness_field" && qcr.consciousnessField != nil {
		coherenceChange := observer.AwarenessLevel * qcr.config.ObserverEffectStrength * 0.02
		effect.ConsciousnessEffects["field_coherence_change"] = coherenceChange
		qcr.consciousnessField.Coherence += coherenceChange
		qcr.consciousnessField.Coherence = math.Min(qcr.consciousnessField.Coherence, 1.0)
	}

	qcr.observerEffects = append(qcr.observerEffects, effect)

	return effect, nil
}

// CreateMemoryTrace manually creates a memory trace
func (qcr *QCREngine) CreateMemoryTrace(entityID, content string, strength float64) error {
	qcr.mu.Lock()
	defer qcr.mu.Unlock()

	if qcr.memoryMatrix == nil {
		return fmt.Errorf("memory matrix not initialized")
	}

	// Find entity
	var entity *ConsciousEntity
	for _, e := range qcr.consciousEntities {
		if e.ID == entityID {
			entity = e
			break
		}
	}

	if entity == nil {
		return fmt.Errorf("entity not found: %s", entityID)
	}

	// Create quantum state for memory content
	amplitudes := qcr.encodeStringToAmplitudes(content)
	quantumState, err := qcr.resonanceEngine.CreateQuantumState(amplitudes)
	if err != nil {
		return fmt.Errorf("failed to create memory quantum state: %w", err)
	}

	// Create memory trace
	memoryTrace := &MemoryTrace{
		ID:           fmt.Sprintf("manual_mem_%s_%d", entityID, time.Now().UnixNano()),
		Content:      quantumState,
		Strength:     math.Max(0.0, math.Min(1.0, strength)),
		CreationTime: time.Now(),
		LastAccessed: time.Now(),
		AccessCount:  1,
		DecayRate:    0.01,
		Associations: make([]string, 0),
	}

	// Add to appropriate memory store
	if strength > 0.8 {
		qcr.memoryMatrix.LongTermMemory = append(qcr.memoryMatrix.LongTermMemory, memoryTrace)
	} else {
		qcr.memoryMatrix.ShortTermMemory = append(qcr.memoryMatrix.ShortTermMemory, memoryTrace)
	}

	qcr.memoryMatrix.UsedCapacity++

	return nil
}

// encodeStringToAmplitudes encodes a string into quantum amplitudes
func (qcr *QCREngine) encodeStringToAmplitudes(content string) []complex128 {
	dimension := qcr.resonanceEngine.GetDimension()
	amplitudes := make([]complex128, dimension)

	normFactor := 0.0

	for i := 0; i < dimension; i++ {
		// Encode string content using character values and position
		amplitude := 0.1
		if i < len(content) {
			amplitude = float64(content[i]) / 128.0 // Normalize ASCII
		}

		// Add phase based on position and content
		phase := 2.0 * math.Pi * float64(i) / float64(dimension)
		if len(content) > 0 {
			phase += float64(content[i%len(content)]) / 64.0
		}

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

// GetAwarenessNetworks returns all awareness networks
func (qcr *QCREngine) GetAwarenessNetworks() []*AwarenessNetwork {
	qcr.mu.RLock()
	defer qcr.mu.RUnlock()

	networks := make([]*AwarenessNetwork, len(qcr.awarenessNetworks))
	copy(networks, qcr.awarenessNetworks)
	return networks
}

// GetObserverEffects returns all observer effects
func (qcr *QCREngine) GetObserverEffects() []*ObserverEffect {
	qcr.mu.RLock()
	defer qcr.mu.RUnlock()

	effects := make([]*ObserverEffect, len(qcr.observerEffects))
	copy(effects, qcr.observerEffects)
	return effects
}

// Additional supporting structure definitions that were referenced but not defined

type MetaCognition struct {
	Level float64 `json:"level"`
}

type SelfReflection struct {
	Frequency float64 `json:"frequency"`
}

type PredictiveModel struct {
	Accuracy   float64 `json:"accuracy"`
	Confidence float64 `json:"confidence"`
}

type CausalModel struct {
	Accuracy float64 `json:"accuracy"`
}

type RegulationMechanism struct {
	Effectiveness float64 `json:"effectiveness"`
}

type DecisionProcess struct {
	ID         string    `json:"id"`
	Type       string    `json:"type"`
	Status     string    `json:"status"`
	StartTime  time.Time `json:"start_time"`
	Complexity float64   `json:"complexity"`
}

type ValueSystem struct {
	CoreValues map[string]float64 `json:"core_values"`
}

type Decision struct {
	ID           string    `json:"id"`
	EntityID     string    `json:"entity_id"`
	DecisionType string    `json:"decision_type"`
	Options      []string  `json:"options"`
	Chosen       string    `json:"chosen"`
	Confidence   float64   `json:"confidence"`
	Timestamp    time.Time `json:"timestamp"`
}

type ConsolidationProcess struct {
	ConsolidationRate float64 `json:"consolidation_rate"`
	Threshold         float64 `json:"threshold"`
}

type ProcessingBuffer struct {
	Capacity int `json:"capacity"`
}

type AttentionalControl struct {
	Strength float64 `json:"strength"`
}

type MaintenanceMechanism struct {
	Effectiveness float64 `json:"effectiveness"`
}

type AdaptationMechanism struct {
	Rate float64 `json:"rate"`
}
