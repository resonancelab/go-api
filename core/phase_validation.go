package core

import (
	"fmt"
	"math"
	"sync"
	"time"
)

// PhaseAlignmentValidator validates phase alignment across distributed nodes
type PhaseAlignmentValidator struct {
	// Node phase measurements
	NodePhases map[string]*NodePhaseMeasurement `json:"node_phases"`

	// Validation parameters
	PhaseTolerance     float64       `json:"phase_tolerance"`     // Maximum allowed phase difference
	CoherenceThreshold float64       `json:"coherence_threshold"` // Minimum required coherence
	ValidationWindow   time.Duration `json:"validation_window"`   // Time window for validation
	MaxLatency         time.Duration `json:"max_latency"`         // Maximum allowed network latency

	// Validation statistics
	TotalValidations      int64     `json:"total_validations"`
	SuccessfulValidations int64     `json:"successful_validations"`
	FailedValidations     int64     `json:"failed_validations"`
	LastValidationTime    time.Time `json:"last_validation_time"`

	// Thread safety
	mu sync.RWMutex `json:"-"`
}

// NodePhaseMeasurement represents phase measurements for a single node
type NodePhaseMeasurement struct {
	NodeID         string        `json:"node_id"`
	Phase          float64       `json:"phase"`
	Coherence      float64       `json:"coherence"`
	Timestamp      time.Time     `json:"timestamp"`
	Latency        time.Duration `json:"latency"`
	PrimeBasis     []int         `json:"prime_basis"`
	AmplitudeData  []complex128  `json:"amplitude_data"`
	ValidationHash string        `json:"validation_hash"`
}

// ValidationResult represents the result of a phase alignment validation
type ValidationResult struct {
	IsValid         bool                   `json:"is_valid"`
	GlobalCoherence float64                `json:"global_coherence"`
	PhaseVariance   float64                `json:"phase_variance"`
	NodeCount       int                    `json:"node_count"`
	Timestamp       time.Time              `json:"timestamp"`
	Errors          []string               `json:"errors,omitempty"`
	NodeResults     map[string]interface{} `json:"node_results"`
}

// NewPhaseAlignmentValidator creates a new phase alignment validator
func NewPhaseAlignmentValidator(phaseTolerance, coherenceThreshold float64, validationWindow, maxLatency time.Duration) *PhaseAlignmentValidator {
	return &PhaseAlignmentValidator{
		NodePhases:            make(map[string]*NodePhaseMeasurement),
		PhaseTolerance:        phaseTolerance,
		CoherenceThreshold:    coherenceThreshold,
		ValidationWindow:      validationWindow,
		MaxLatency:            maxLatency,
		TotalValidations:      0,
		SuccessfulValidations: 0,
		FailedValidations:     0,
		LastValidationTime:    time.Now(),
	}
}

// RecordNodePhase records a phase measurement from a node
func (pav *PhaseAlignmentValidator) RecordNodePhase(nodeID string, phase, coherence float64, primeBasis []int, amplitudeData []complex128, latency time.Duration) error {
	pav.mu.Lock()
	defer pav.mu.Unlock()

	// Validate input
	if nodeID == "" {
		return fmt.Errorf("node ID cannot be empty")
	}

	if len(primeBasis) == 0 {
		return fmt.Errorf("prime basis cannot be empty")
	}

	if len(amplitudeData) != len(primeBasis) {
		return fmt.Errorf("amplitude data length %d doesn't match prime basis length %d", len(amplitudeData), len(primeBasis))
	}

	// Create measurement
	measurement := &NodePhaseMeasurement{
		NodeID:        nodeID,
		Phase:         phase,
		Coherence:     coherence,
		Timestamp:     time.Now(),
		Latency:       latency,
		PrimeBasis:    make([]int, len(primeBasis)),
		AmplitudeData: make([]complex128, len(amplitudeData)),
	}

	// Copy data
	copy(measurement.PrimeBasis, primeBasis)
	copy(measurement.AmplitudeData, amplitudeData)

	// Generate validation hash
	measurement.ValidationHash = pav.generateValidationHash(measurement)

	pav.NodePhases[nodeID] = measurement
	return nil
}

// ValidatePhaseAlignment performs comprehensive phase alignment validation
func (pav *PhaseAlignmentValidator) ValidatePhaseAlignment() *ValidationResult {
	pav.mu.Lock()
	defer pav.mu.Unlock()

	pav.TotalValidations++
	pav.LastValidationTime = time.Now()

	result := &ValidationResult{
		IsValid:     true,
		Timestamp:   time.Now(),
		Errors:      make([]string, 0),
		NodeResults: make(map[string]interface{}),
	}

	// Check minimum node count
	if len(pav.NodePhases) < 2 {
		result.IsValid = false
		result.Errors = append(result.Errors, "insufficient nodes for validation (minimum 2 required)")
		return result
	}

	// Validate individual node measurements
	validNodes := make([]*NodePhaseMeasurement, 0)
	for nodeID, measurement := range pav.NodePhases {
		nodeResult := pav.validateNodeMeasurement(measurement)
		result.NodeResults[nodeID] = nodeResult

		if nodeResult["is_valid"].(bool) {
			validNodes = append(validNodes, measurement)
		} else {
			result.IsValid = false
			result.Errors = append(result.Errors, fmt.Sprintf("node %s validation failed: %v", nodeID, nodeResult["errors"]))
		}
	}

	result.NodeCount = len(validNodes)

	if len(validNodes) < 2 {
		result.IsValid = false
		result.Errors = append(result.Errors, "insufficient valid nodes for coherence calculation")
		return result
	}

	// Calculate global coherence
	globalCoherence := pav.calculateGlobalCoherence(validNodes)
	result.GlobalCoherence = globalCoherence

	// Calculate phase variance
	phaseVariance := pav.calculatePhaseVariance(validNodes)
	result.PhaseVariance = phaseVariance

	// Validate coherence threshold
	if globalCoherence < pav.CoherenceThreshold {
		result.IsValid = false
		result.Errors = append(result.Errors, fmt.Sprintf("global coherence %.3f below threshold %.3f", globalCoherence, pav.CoherenceThreshold))
	}

	// Validate phase variance
	maxAllowedVariance := pav.PhaseTolerance * pav.PhaseTolerance
	if phaseVariance > maxAllowedVariance {
		result.IsValid = false
		result.Errors = append(result.Errors, fmt.Sprintf("phase variance %.6f exceeds maximum allowed %.6f", phaseVariance, maxAllowedVariance))
	}

	// Update statistics
	if result.IsValid {
		pav.SuccessfulValidations++
	} else {
		pav.FailedValidations++
	}

	return result
}

// validateNodeMeasurement validates a single node's phase measurement
func (pav *PhaseAlignmentValidator) validateNodeMeasurement(measurement *NodePhaseMeasurement) map[string]interface{} {
	result := map[string]interface{}{
		"is_valid": true,
		"errors":   make([]string, 0),
	}

	// Check timestamp freshness
	age := time.Since(measurement.Timestamp)
	if age > pav.ValidationWindow {
		result["is_valid"] = false
		result["errors"] = append(result["errors"].([]string), fmt.Sprintf("measurement too old: %v", age))
	}

	// Check latency
	if measurement.Latency > pav.MaxLatency {
		result["is_valid"] = false
		result["errors"] = append(result["errors"].([]string), fmt.Sprintf("latency too high: %v", measurement.Latency))
	}

	// Validate phase range
	if measurement.Phase < -math.Pi || measurement.Phase > math.Pi {
		result["is_valid"] = false
		result["errors"] = append(result["errors"].([]string), fmt.Sprintf("phase out of range: %.3f", measurement.Phase))
	}

	// Validate coherence range
	if measurement.Coherence < 0 || measurement.Coherence > 1 {
		result["is_valid"] = false
		result["errors"] = append(result["errors"].([]string), fmt.Sprintf("coherence out of range: %.3f", measurement.Coherence))
	}

	// Validate amplitude normalization
	totalAmplitude := 0.0
	for _, amp := range measurement.AmplitudeData {
		totalAmplitude += real(amp * complex(real(amp), -imag(amp))) // |amp|²
	}

	if math.Abs(totalAmplitude-1.0) > 1e-6 {
		result["is_valid"] = false
		result["errors"] = append(result["errors"].([]string), fmt.Sprintf("amplitudes not normalized: sum=%.6f", totalAmplitude))
	}

	// Validate prime basis consistency
	for i, prime := range measurement.PrimeBasis {
		if prime <= 1 {
			result["is_valid"] = false
			result["errors"] = append(result["errors"].([]string), fmt.Sprintf("invalid prime at index %d: %d", i, prime))
		}
	}

	return result
}

// calculateGlobalCoherence calculates global coherence across all valid nodes
func (pav *PhaseAlignmentValidator) calculateGlobalCoherence(nodes []*NodePhaseMeasurement) float64 {
	if len(nodes) < 2 {
		return 0.0
	}

	totalCoherence := 0.0
	pairCount := 0

	// Calculate pairwise coherence
	for i := 0; i < len(nodes); i++ {
		for j := i + 1; j < len(nodes); j++ {
			// Calculate phase difference
			phaseDiff := math.Abs(nodes[i].Phase - nodes[j].Phase)

			// Handle phase wrapping
			if phaseDiff > math.Pi {
				phaseDiff = 2*math.Pi - phaseDiff
			}

			// Coherence contribution
			pairCoherence := math.Cos(phaseDiff)

			// Weight by individual node coherences
			weightedCoherence := pairCoherence * math.Sqrt(nodes[i].Coherence*nodes[j].Coherence)

			totalCoherence += weightedCoherence
			pairCount++
		}
	}

	if pairCount == 0 {
		return 0.0
	}

	return totalCoherence / float64(pairCount)
}

// calculatePhaseVariance calculates the variance in phase measurements
func (pav *PhaseAlignmentValidator) calculatePhaseVariance(nodes []*NodePhaseMeasurement) float64 {
	if len(nodes) < 2 {
		return 0.0
	}

	// Calculate mean phase
	totalPhase := 0.0
	for _, node := range nodes {
		totalPhase += node.Phase
	}
	meanPhase := totalPhase / float64(len(nodes))

	// Calculate variance
	variance := 0.0
	for _, node := range nodes {
		phaseDiff := node.Phase - meanPhase

		// Handle phase wrapping for variance calculation
		if phaseDiff > math.Pi {
			phaseDiff -= 2 * math.Pi
		} else if phaseDiff < -math.Pi {
			phaseDiff += 2 * math.Pi
		}

		variance += phaseDiff * phaseDiff
	}

	return variance / float64(len(nodes))
}

// GetValidationStatistics returns validation statistics
func (pav *PhaseAlignmentValidator) GetValidationStatistics() map[string]interface{} {
	pav.mu.RLock()
	defer pav.mu.RUnlock()

	successRate := 0.0
	if pav.TotalValidations > 0 {
		successRate = float64(pav.SuccessfulValidations) / float64(pav.TotalValidations)
	}

	return map[string]interface{}{
		"total_validations":      pav.TotalValidations,
		"successful_validations": pav.SuccessfulValidations,
		"failed_validations":     pav.FailedValidations,
		"success_rate":           successRate,
		"last_validation_time":   pav.LastValidationTime,
		"active_nodes":           len(pav.NodePhases),
		"phase_tolerance":        pav.PhaseTolerance,
		"coherence_threshold":    pav.CoherenceThreshold,
		"validation_window":      pav.ValidationWindow,
		"max_latency":            pav.MaxLatency,
	}
}

// GetNodePhase retrieves the latest phase measurement for a node
func (pav *PhaseAlignmentValidator) GetNodePhase(nodeID string) (*NodePhaseMeasurement, bool) {
	pav.mu.RLock()
	defer pav.mu.RUnlock()

	measurement, exists := pav.NodePhases[nodeID]
	return measurement, exists
}

// RemoveStaleMeasurements removes measurements older than the validation window
func (pav *PhaseAlignmentValidator) RemoveStaleMeasurements() int {
	pav.mu.Lock()
	defer pav.mu.Unlock()

	removed := 0
	cutoffTime := time.Now().Add(-pav.ValidationWindow)

	for nodeID, measurement := range pav.NodePhases {
		if measurement.Timestamp.Before(cutoffTime) {
			delete(pav.NodePhases, nodeID)
			removed++
		}
	}

	return removed
}

// generateValidationHash generates a hash for measurement validation
func (pav *PhaseAlignmentValidator) generateValidationHash(measurement *NodePhaseMeasurement) string {
	// Simple hash based on measurement data
	// In production, this would use cryptographic hashing
	data := fmt.Sprintf("%s:%.6f:%.6f:%d",
		measurement.NodeID,
		measurement.Phase,
		measurement.Coherence,
		measurement.Timestamp.UnixNano())

	// Simple rolling hash for demonstration
	hash := uint64(0)
	for _, char := range data {
		hash = hash*31 + uint64(char)
	}

	return fmt.Sprintf("%016x", hash)
}

// ValidateNetworkLatency validates network latency for phase alignment
func (pav *PhaseAlignmentValidator) ValidateNetworkLatency() *LatencyValidationResult {
	pav.mu.RLock()
	defer pav.mu.RUnlock()

	result := &LatencyValidationResult{
		Timestamp:     time.Now(),
		NodeLatencies: make(map[string]time.Duration),
		IsValid:       true,
		Errors:        make([]string, 0),
	}

	minLatency := time.Duration(math.MaxInt64)
	maxLatency := time.Duration(0)
	totalLatency := time.Duration(0)

	for nodeID, measurement := range pav.NodePhases {
		latency := measurement.Latency
		result.NodeLatencies[nodeID] = latency

		if latency > pav.MaxLatency {
			result.IsValid = false
			result.Errors = append(result.Errors, fmt.Sprintf("node %s latency %v exceeds maximum %v", nodeID, latency, pav.MaxLatency))
		}

		if latency < minLatency {
			minLatency = latency
		}
		if latency > maxLatency {
			maxLatency = latency
		}
		totalLatency += latency
	}

	if len(pav.NodePhases) > 0 {
		result.AverageLatency = totalLatency / time.Duration(len(pav.NodePhases))
		result.MinLatency = minLatency
		result.MaxLatency = maxLatency
		result.LatencyVariance = pav.calculateLatencyVariance()
	}

	return result
}

// calculateLatencyVariance calculates variance in network latencies
func (pav *PhaseAlignmentValidator) calculateLatencyVariance() float64 {
	if len(pav.NodePhases) < 2 {
		return 0.0
	}

	// Calculate mean latency
	totalLatency := 0.0
	for _, measurement := range pav.NodePhases {
		totalLatency += float64(measurement.Latency.Nanoseconds())
	}
	meanLatency := totalLatency / float64(len(pav.NodePhases))

	// Calculate variance
	variance := 0.0
	for _, measurement := range pav.NodePhases {
		latencyDiff := float64(measurement.Latency.Nanoseconds()) - meanLatency
		variance += latencyDiff * latencyDiff
	}

	return variance / float64(len(pav.NodePhases))
}

// LatencyValidationResult represents network latency validation results
type LatencyValidationResult struct {
	Timestamp       time.Time                `json:"timestamp"`
	NodeLatencies   map[string]time.Duration `json:"node_latencies"`
	AverageLatency  time.Duration            `json:"average_latency"`
	MinLatency      time.Duration            `json:"min_latency"`
	MaxLatency      time.Duration            `json:"max_latency"`
	LatencyVariance float64                  `json:"latency_variance"`
	IsValid         bool                     `json:"is_valid"`
	Errors          []string                 `json:"errors,omitempty"`
}
