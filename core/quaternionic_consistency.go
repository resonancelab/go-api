package core

import (
	"fmt"
	"math"
	"math/cmplx"
	"sync"
	"time"
)

// QuaternionicConsistencyChecker validates mathematical consistency of quaternionic states
type QuaternionicConsistencyChecker struct {
	// Reference states for consistency validation
	ReferenceStates map[string]*QuaternionicState `json:"reference_states"`

	// Consistency parameters
	PhaseTolerance     float64       `json:"phase_tolerance"`
	AmplitudeTolerance float64       `json:"amplitude_tolerance"`
	CoherenceThreshold float64       `json:"coherence_threshold"`
	ValidationWindow   time.Duration `json:"validation_window"`
	MaxDriftRate       float64       `json:"max_drift_rate"`

	// Validation statistics
	TotalValidations        int64     `json:"total_validations"`
	ConsistentValidations   int64     `json:"consistent_validations"`
	InconsistentValidations int64     `json:"inconsistent_validations"`
	LastValidationTime      time.Time `json:"last_validation_time"`

	// Thread safety
	mu sync.RWMutex `json:"-"`
}

// ConsistencyCheckResult represents the result of a consistency check
type ConsistencyCheckResult struct {
	IsConsistent   bool                   `json:"is_consistent"`
	NodeID         string                 `json:"node_id"`
	StateID        string                 `json:"state_id"`
	Timestamp      time.Time              `json:"timestamp"`
	PhaseDrift     float64                `json:"phase_drift"`
	AmplitudeDrift float64                `json:"amplitude_drift"`
	CoherenceDrift float64                `json:"coherence_drift"`
	DriftRate      float64                `json:"drift_rate"`
	Errors         []string               `json:"errors,omitempty"`
	Metrics        map[string]interface{} `json:"metrics"`
}

// NewQuaternionicConsistencyChecker creates a new consistency checker
func NewQuaternionicConsistencyChecker(phaseTolerance, amplitudeTolerance, coherenceThreshold, maxDriftRate float64, validationWindow time.Duration) *QuaternionicConsistencyChecker {
	return &QuaternionicConsistencyChecker{
		ReferenceStates:         make(map[string]*QuaternionicState),
		PhaseTolerance:          phaseTolerance,
		AmplitudeTolerance:      amplitudeTolerance,
		CoherenceThreshold:      coherenceThreshold,
		ValidationWindow:        validationWindow,
		MaxDriftRate:            maxDriftRate,
		TotalValidations:        0,
		ConsistentValidations:   0,
		InconsistentValidations: 0,
		LastValidationTime:      time.Now(),
	}
}

// RegisterReferenceState registers a reference state for consistency validation
func (qcc *QuaternionicConsistencyChecker) RegisterReferenceState(stateID string, state *QuaternionicState) error {
	qcc.mu.Lock()
	defer qcc.mu.Unlock()

	if state == nil {
		return fmt.Errorf("reference state cannot be nil")
	}

	if err := state.Validate(); err != nil {
		return fmt.Errorf("invalid reference state: %w", err)
	}

	// Clone the state to avoid external modifications
	qcc.ReferenceStates[stateID] = state.Clone()
	return nil
}

// CheckConsistency validates a quaternionic state against its reference
func (qcc *QuaternionicConsistencyChecker) CheckConsistency(stateID string, currentState *QuaternionicState) *ConsistencyCheckResult {
	qcc.mu.Lock()
	defer qcc.mu.Unlock()

	qcc.TotalValidations++
	qcc.LastValidationTime = time.Now()

	result := &ConsistencyCheckResult{
		IsConsistent: true,
		StateID:      stateID,
		Timestamp:    time.Now(),
		Errors:       make([]string, 0),
		Metrics:      make(map[string]interface{}),
	}

	referenceState, exists := qcc.ReferenceStates[stateID]
	if !exists {
		result.IsConsistent = false
		result.Errors = append(result.Errors, fmt.Sprintf("no reference state found for ID: %s", stateID))
		return result
	}

	if currentState == nil {
		result.IsConsistent = false
		result.Errors = append(result.Errors, "current state is nil")
		return result
	}

	// Validate current state
	if err := currentState.Validate(); err != nil {
		result.IsConsistent = false
		result.Errors = append(result.Errors, fmt.Sprintf("current state validation failed: %v", err))
		return result
	}

	// Check phase consistency
	phaseDrift := qcc.calculatePhaseDrift(referenceState, currentState)
	result.PhaseDrift = phaseDrift

	if math.Abs(phaseDrift) > qcc.PhaseTolerance {
		result.IsConsistent = false
		result.Errors = append(result.Errors, fmt.Sprintf("phase drift %.6f exceeds tolerance %.6f", phaseDrift, qcc.PhaseTolerance))
	}

	// Check amplitude consistency
	amplitudeDrift := qcc.calculateAmplitudeDrift(referenceState, currentState)
	result.AmplitudeDrift = amplitudeDrift

	if math.Abs(amplitudeDrift) > qcc.AmplitudeTolerance {
		result.IsConsistent = false
		result.Errors = append(result.Errors, fmt.Sprintf("amplitude drift %.6f exceeds tolerance %.6f", amplitudeDrift, qcc.AmplitudeTolerance))
	}

	// Check coherence consistency
	coherenceDrift := currentState.Coherence - referenceState.Coherence
	result.CoherenceDrift = coherenceDrift

	if math.Abs(coherenceDrift) > (1.0 - qcc.CoherenceThreshold) {
		result.IsConsistent = false
		result.Errors = append(result.Errors, fmt.Sprintf("coherence drift %.6f exceeds threshold %.6f", coherenceDrift, qcc.CoherenceThreshold))
	}

	// Check drift rate
	timeDiff := currentState.Time - referenceState.Time
	if timeDiff > 0 {
		totalDrift := math.Sqrt(phaseDrift*phaseDrift + amplitudeDrift*amplitudeDrift)
		driftRate := totalDrift / timeDiff
		result.DriftRate = driftRate

		if driftRate > qcc.MaxDriftRate {
			result.IsConsistent = false
			result.Errors = append(result.Errors, fmt.Sprintf("drift rate %.6f exceeds maximum %.6f", driftRate, qcc.MaxDriftRate))
		}
	}

	// Calculate additional metrics
	result.Metrics = qcc.calculateConsistencyMetrics(referenceState, currentState)

	// Update statistics
	if result.IsConsistent {
		qcc.ConsistentValidations++
	} else {
		qcc.InconsistentValidations++
	}

	return result
}

// calculatePhaseDrift calculates the phase difference between states
func (qcc *QuaternionicConsistencyChecker) calculatePhaseDrift(reference, current *QuaternionicState) float64 {
	phaseDiff := current.Phase - reference.Phase

	// Handle phase wrapping
	for phaseDiff > math.Pi {
		phaseDiff -= 2 * math.Pi
	}
	for phaseDiff < -math.Pi {
		phaseDiff += 2 * math.Pi
	}

	return phaseDiff
}

// calculateAmplitudeDrift calculates the amplitude difference between states
func (qcc *QuaternionicConsistencyChecker) calculateAmplitudeDrift(reference, current *QuaternionicState) float64 {
	refAmplitude := reference.ComputeQuaternionicAmplitude()
	currAmplitude := current.ComputeQuaternionicAmplitude()

	// Calculate relative amplitude difference
	refMag := cmplx.Abs(refAmplitude)
	currMag := cmplx.Abs(currAmplitude)

	if refMag == 0 {
		if currMag == 0 {
			return 0.0
		}
		return currMag
	}

	return (currMag - refMag) / refMag
}

// calculateConsistencyMetrics calculates additional consistency metrics
func (qcc *QuaternionicConsistencyChecker) calculateConsistencyMetrics(reference, current *QuaternionicState) map[string]interface{} {
	metrics := make(map[string]interface{})

	// Position consistency
	if len(reference.Position) == len(current.Position) {
		positionDrift := 0.0
		for i := range reference.Position {
			diff := current.Position[i] - reference.Position[i]
			positionDrift += diff * diff
		}
		metrics["position_drift"] = math.Sqrt(positionDrift)
	}

	// Time evolution consistency
	timeDiff := current.Time - reference.Time
	metrics["time_evolution"] = timeDiff

	// Normalization factor consistency
	normDrift := current.NormalizationFactor - reference.NormalizationFactor
	metrics["normalization_drift"] = normDrift

	// Gaussian coordinates consistency
	if len(reference.GaussianCoords) == len(current.GaussianCoords) {
		gaussianDrift := 0.0
		for i := range reference.GaussianCoords {
			diff := current.GaussianCoords[i] - reference.GaussianCoords[i]
			gaussianDrift += diff * diff
		}
		metrics["gaussian_drift"] = math.Sqrt(gaussianDrift)
	}

	// Eisenstein coordinates consistency
	if len(reference.EisensteinCoords) == len(current.EisensteinCoords) {
		eisensteinDrift := 0.0
		for i := range reference.EisensteinCoords {
			diff := current.EisensteinCoords[i] - reference.EisensteinCoords[i]
			eisensteinDrift += diff * diff
		}
		metrics["eisenstein_drift"] = math.Sqrt(eisensteinDrift)
	}

	// Timestamp consistency
	timestampDrift := current.Timestamp.Sub(reference.Timestamp).Seconds()
	metrics["timestamp_drift"] = timestampDrift

	return metrics
}

// ValidateStateEvolution validates the evolution of a quaternionic state
func (qcc *QuaternionicConsistencyChecker) ValidateStateEvolution(stateID string, stateHistory []*QuaternionicState) *EvolutionValidationResult {
	qcc.mu.RLock()
	defer qcc.mu.RUnlock()

	result := &EvolutionValidationResult{
		StateID:          stateID,
		IsEvolutionValid: true,
		Timestamp:        time.Now(),
		Errors:           make([]string, 0),
		Metrics:          make(map[string]interface{}),
	}

	if len(stateHistory) < 2 {
		result.IsEvolutionValid = false
		result.Errors = append(result.Errors, "insufficient state history for evolution validation")
		return result
	}

	// Validate chronological order
	for i := 1; i < len(stateHistory); i++ {
		if stateHistory[i].Time < stateHistory[i-1].Time {
			result.IsEvolutionValid = false
			result.Errors = append(result.Errors, fmt.Sprintf("non-chronological time evolution at index %d", i))
		}
		if stateHistory[i].Timestamp.Before(stateHistory[i-1].Timestamp) {
			result.IsEvolutionValid = false
			result.Errors = append(result.Errors, fmt.Sprintf("non-chronological timestamp at index %d", i))
		}
	}

	// Validate phase continuity
	phaseVelocities := make([]float64, len(stateHistory)-1)
	for i := 0; i < len(stateHistory)-1; i++ {
		timeDiff := stateHistory[i+1].Time - stateHistory[i].Time
		if timeDiff > 0 {
			phaseDiff := qcc.calculatePhaseDrift(stateHistory[i], stateHistory[i+1])
			phaseVelocities[i] = math.Abs(phaseDiff) / timeDiff
		}
	}

	// Check for phase discontinuities
	maxPhaseVelocity := 0.0
	avgPhaseVelocity := 0.0
	validVelocities := 0

	for _, velocity := range phaseVelocities {
		if velocity > 0 {
			if velocity > maxPhaseVelocity {
				maxPhaseVelocity = velocity
			}
			avgPhaseVelocity += velocity
			validVelocities++
		}
	}

	if validVelocities > 0 {
		avgPhaseVelocity /= float64(validVelocities)
	}

	// Validate against maximum allowed phase velocity (should be related to frequency)
	maxAllowedVelocity := 4 * math.Pi // Roughly 2 full rotations per second
	if maxPhaseVelocity > maxAllowedVelocity {
		result.IsEvolutionValid = false
		result.Errors = append(result.Errors, fmt.Sprintf("phase velocity %.3f exceeds maximum allowed %.3f", maxPhaseVelocity, maxAllowedVelocity))
	}

	// Store metrics
	result.Metrics["max_phase_velocity"] = maxPhaseVelocity
	result.Metrics["avg_phase_velocity"] = avgPhaseVelocity
	result.Metrics["evolution_duration"] = stateHistory[len(stateHistory)-1].Time - stateHistory[0].Time
	result.Metrics["state_count"] = len(stateHistory)

	return result
}

// GetConsistencyStatistics returns consistency validation statistics
func (qcc *QuaternionicConsistencyChecker) GetConsistencyStatistics() map[string]interface{} {
	qcc.mu.RLock()
	defer qcc.mu.RUnlock()

	consistencyRate := 0.0
	if qcc.TotalValidations > 0 {
		consistencyRate = float64(qcc.ConsistentValidations) / float64(qcc.TotalValidations)
	}

	return map[string]interface{}{
		"total_validations":        qcc.TotalValidations,
		"consistent_validations":   qcc.ConsistentValidations,
		"inconsistent_validations": qcc.InconsistentValidations,
		"consistency_rate":         consistencyRate,
		"last_validation_time":     qcc.LastValidationTime,
		"reference_states_count":   len(qcc.ReferenceStates),
		"phase_tolerance":          qcc.PhaseTolerance,
		"amplitude_tolerance":      qcc.AmplitudeTolerance,
		"coherence_threshold":      qcc.CoherenceThreshold,
		"max_drift_rate":           qcc.MaxDriftRate,
		"validation_window":        qcc.ValidationWindow,
	}
}

// RemoveStaleReferences removes reference states older than the validation window
func (qcc *QuaternionicConsistencyChecker) RemoveStaleReferences() int {
	qcc.mu.Lock()
	defer qcc.mu.Unlock()

	removed := 0
	cutoffTime := time.Now().Add(-qcc.ValidationWindow)

	for stateID, state := range qcc.ReferenceStates {
		if state.Timestamp.Before(cutoffTime) {
			delete(qcc.ReferenceStates, stateID)
			removed++
		}
	}

	return removed
}

// ValidateMathematicalCorrectness performs comprehensive mathematical validation
func (qcc *QuaternionicConsistencyChecker) ValidateMathematicalCorrectness(state *QuaternionicState) *MathematicalValidationResult {
	result := &MathematicalValidationResult{
		IsMathematicallyValid: true,
		Timestamp:             time.Now(),
		Errors:                make([]string, 0),
		Properties:            make(map[string]interface{}),
	}

	if state == nil {
		result.IsMathematicallyValid = false
		result.Errors = append(result.Errors, "state is nil")
		return result
	}

	// Validate normalization
	amplitude := state.ComputeQuaternionicAmplitude()
	magnitude := cmplx.Abs(amplitude)
	if math.Abs(magnitude-1.0) > 1e-10 {
		result.IsMathematicallyValid = false
		result.Errors = append(result.Errors, fmt.Sprintf("state not properly normalized: magnitude = %.10f", magnitude))
	}

	// Validate phase range
	if state.Phase < -2*math.Pi || state.Phase > 2*math.Pi {
		result.IsMathematicallyValid = false
		result.Errors = append(result.Errors, fmt.Sprintf("phase out of reasonable range: %.3f", state.Phase))
	}

	// Validate coherence bounds
	if state.Coherence < -1.0 || state.Coherence > 1.0 {
		result.IsMathematicallyValid = false
		result.Errors = append(result.Errors, fmt.Sprintf("coherence out of bounds: %.3f", state.Coherence))
	}

	// Validate coordinate consistency
	if len(state.Position) != len(state.GaussianCoords) || len(state.Position) != len(state.EisensteinCoords) {
		result.IsMathematicallyValid = false
		result.Errors = append(result.Errors, "coordinate dimensions inconsistent")
	}

	// Store mathematical properties
	result.Properties["magnitude"] = magnitude
	result.Properties["phase_range"] = fmt.Sprintf("[%.3f, %.3f]", -2*math.Pi, 2*math.Pi)
	result.Properties["coherence_range"] = "[-1.0, 1.0]"
	result.Properties["coordinate_dimensions"] = len(state.Position)
	result.Properties["normalization_factor"] = state.NormalizationFactor

	return result
}

// EvolutionValidationResult represents the result of evolution validation
type EvolutionValidationResult struct {
	StateID          string                 `json:"state_id"`
	IsEvolutionValid bool                   `json:"is_evolution_valid"`
	Timestamp        time.Time              `json:"timestamp"`
	Errors           []string               `json:"errors,omitempty"`
	Metrics          map[string]interface{} `json:"metrics"`
}

// MathematicalValidationResult represents mathematical validation results
type MathematicalValidationResult struct {
	IsMathematicallyValid bool                   `json:"is_mathematically_valid"`
	Timestamp             time.Time              `json:"timestamp"`
	Errors                []string               `json:"errors,omitempty"`
	Properties            map[string]interface{} `json:"properties"`
}
