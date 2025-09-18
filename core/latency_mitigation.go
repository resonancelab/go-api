package core

import (
	"fmt"
	"math"
	"sync"
	"time"
)

// LatencyMitigator handles network latency compensation for phase alignment
type LatencyMitigator struct {
	// Node latency measurements
	NodeLatencies map[string]*LatencyMeasurement `json:"node_latencies"`

	// Latency prediction parameters
	PredictionWindow   time.Duration `json:"prediction_window"`   // Time window for latency prediction
	MaxLatencyDrift    time.Duration `json:"max_latency_drift"`   // Maximum allowed latency drift
	LatencyThreshold   time.Duration `json:"latency_threshold"`   // Threshold for latency compensation
	CompensationFactor float64       `json:"compensation_factor"` // Compensation strength factor

	// Statistical tracking
	TotalMeasurements     int64     `json:"total_measurements"`
	SuccessfulPredictions int64     `json:"successful_predictions"`
	FailedPredictions     int64     `json:"failed_predictions"`
	LastUpdateTime        time.Time `json:"last_update_time"`

	// Thread safety
	mu sync.RWMutex `json:"-"`
}

// LatencyMeasurement represents latency measurements for a single node
type LatencyMeasurement struct {
	NodeID           string               `json:"node_id"`
	Measurements     []LatencySample      `json:"measurements"`
	PredictedLatency time.Duration        `json:"predicted_latency"`
	LatencyVariance  float64              `json:"latency_variance"`
	TrendSlope       float64              `json:"trend_slope"` // Latency trend (ms/s)
	LastUpdate       time.Time            `json:"last_update"`
	Compensation     *LatencyCompensation `json:"compensation"`
}

// LatencySample represents a single latency measurement
type LatencySample struct {
	Timestamp time.Time     `json:"timestamp"`
	Latency   time.Duration `json:"latency"`
	RoundTrip bool          `json:"round_trip"` // True for round-trip, false for one-way
}

// LatencyCompensation contains compensation parameters for a node
type LatencyCompensation struct {
	PhaseOffset    float64       `json:"phase_offset"` // Phase offset to apply (radians)
	TimeOffset     time.Duration `json:"time_offset"`  // Time offset to apply
	Confidence     float64       `json:"confidence"`   // Confidence in compensation (0-1)
	LastCalculated time.Time     `json:"last_calculated"`
	ValidUntil     time.Time     `json:"valid_until"`
}

// CompensationResult represents the result of latency compensation
type CompensationResult struct {
	NodeID             string        `json:"node_id"`
	OriginalLatency    time.Duration `json:"original_latency"`
	CompensatedLatency time.Duration `json:"compensated_latency"`
	PhaseAdjustment    float64       `json:"phase_adjustment"` // Radians
	TimeAdjustment     time.Duration `json:"time_adjustment"`
	Confidence         float64       `json:"confidence"`
	Timestamp          time.Time     `json:"timestamp"`
	Error              string        `json:"error,omitempty"`
}

// LatencyStatistics contains comprehensive latency statistics
type LatencyStatistics struct {
	AverageLatency     time.Duration               `json:"average_latency"`
	MinLatency         time.Duration               `json:"min_latency"`
	MaxLatency         time.Duration               `json:"max_latency"`
	LatencyVariance    float64                     `json:"latency_variance"`
	PredictionAccuracy float64                     `json:"prediction_accuracy"`
	NodeStatistics     map[string]NodeLatencyStats `json:"node_statistics"`
	Timestamp          time.Time                   `json:"timestamp"`
}

// NodeLatencyStats contains latency statistics for a single node
type NodeLatencyStats struct {
	AverageLatency    time.Duration `json:"average_latency"`
	LatencyVariance   float64       `json:"latency_variance"`
	PredictionError   time.Duration `json:"prediction_error"`
	CompensationCount int64         `json:"compensation_count"`
	SuccessRate       float64       `json:"success_rate"`
}

// NewLatencyMitigator creates a new latency mitigator
func NewLatencyMitigator(predictionWindow, maxLatencyDrift, latencyThreshold time.Duration, compensationFactor float64) *LatencyMitigator {
	return &LatencyMitigator{
		NodeLatencies:         make(map[string]*LatencyMeasurement),
		PredictionWindow:      predictionWindow,
		MaxLatencyDrift:       maxLatencyDrift,
		LatencyThreshold:      latencyThreshold,
		CompensationFactor:    compensationFactor,
		TotalMeasurements:     0,
		SuccessfulPredictions: 0,
		FailedPredictions:     0,
		LastUpdateTime:        time.Now(),
	}
}

// RecordLatencyMeasurement records a latency measurement for a node
func (lm *LatencyMitigator) RecordLatencyMeasurement(nodeID string, latency time.Duration, roundTrip bool) error {
	lm.mu.Lock()
	defer lm.mu.Unlock()

	lm.TotalMeasurements++

	// Get or create measurement record
	measurement, exists := lm.NodeLatencies[nodeID]
	if !exists {
		measurement = &LatencyMeasurement{
			NodeID:       nodeID,
			Measurements: make([]LatencySample, 0),
		}
		lm.NodeLatencies[nodeID] = measurement
	}

	// Add new measurement
	sample := LatencySample{
		Timestamp: time.Now(),
		Latency:   latency,
		RoundTrip: roundTrip,
	}
	measurement.Measurements = append(measurement.Measurements, sample)
	measurement.LastUpdate = time.Now()

	// Clean old measurements
	lm.cleanOldMeasurements(measurement)

	// Update predictions and compensation
	if err := lm.updateLatencyPrediction(measurement); err != nil {
		return fmt.Errorf("failed to update latency prediction: %w", err)
	}

	return nil
}

// cleanOldMeasurements removes measurements older than the prediction window
func (lm *LatencyMitigator) cleanOldMeasurements(measurement *LatencyMeasurement) {
	cutoffTime := time.Now().Add(-lm.PredictionWindow)
	validMeasurements := make([]LatencySample, 0)

	for _, sample := range measurement.Measurements {
		if sample.Timestamp.After(cutoffTime) {
			validMeasurements = append(validMeasurements, sample)
		}
	}

	measurement.Measurements = validMeasurements
}

// updateLatencyPrediction updates latency prediction and compensation for a node
func (lm *LatencyMitigator) updateLatencyPrediction(measurement *LatencyMeasurement) error {
	if len(measurement.Measurements) < 2 {
		return nil // Need at least 2 measurements for prediction
	}

	// Calculate latency statistics
	latencies := make([]time.Duration, len(measurement.Measurements))
	timestamps := make([]time.Time, len(measurement.Measurements))

	for i, sample := range measurement.Measurements {
		latencies[i] = sample.Latency
		timestamps[i] = sample.Timestamp
	}

	// Calculate average latency
	totalLatency := time.Duration(0)
	for _, latency := range latencies {
		totalLatency += latency
	}
	averageLatency := totalLatency / time.Duration(len(latencies))
	measurement.PredictedLatency = averageLatency

	// Calculate variance
	variance := 0.0
	for _, latency := range latencies {
		diff := float64(latency - averageLatency)
		variance += diff * diff
	}
	measurement.LatencyVariance = variance / float64(len(latencies))

	// Calculate trend (linear regression on latency over time)
	if len(measurement.Measurements) >= 3 {
		measurement.TrendSlope = lm.calculateLatencyTrend(timestamps, latencies)
	}

	// Calculate compensation
	compensation := lm.calculateLatencyCompensation(measurement)
	measurement.Compensation = compensation

	return nil
}

// calculateLatencyTrend calculates the trend in latency over time using linear regression
func (lm *LatencyMitigator) calculateLatencyTrend(timestamps []time.Time, latencies []time.Duration) float64 {
	if len(timestamps) != len(latencies) || len(timestamps) < 2 {
		return 0.0
	}

	n := float64(len(timestamps))
	startTime := timestamps[0]

	// Calculate sums
	sumX := 0.0
	sumY := 0.0
	sumXY := 0.0
	sumX2 := 0.0

	for i, timestamp := range timestamps {
		x := timestamp.Sub(startTime).Seconds()        // Time in seconds
		y := float64(latencies[i].Nanoseconds()) / 1e6 // Latency in milliseconds

		sumX += x
		sumY += y
		sumXY += x * y
		sumX2 += x * x
	}

	// Calculate slope
	denominator := n*sumX2 - sumX*sumX
	if math.Abs(denominator) < 1e-10 {
		return 0.0
	}

	slope := (n*sumXY - sumX*sumY) / denominator
	return slope // ms/s
}

// calculateLatencyCompensation calculates compensation parameters for a node
func (lm *LatencyMitigator) calculateLatencyCompensation(measurement *LatencyMeasurement) *LatencyCompensation {
	compensation := &LatencyCompensation{
		LastCalculated: time.Now(),
		ValidUntil:     time.Now().Add(lm.PredictionWindow),
	}

	// Calculate confidence based on measurement count and variance
	measurementCount := len(measurement.Measurements)
	if measurementCount < 3 {
		compensation.Confidence = 0.5 // Low confidence with few measurements
	} else {
		// Higher confidence with more measurements and lower variance
		countFactor := math.Min(float64(measurementCount)/10.0, 1.0)
		varianceFactor := 1.0 / (1.0 + measurement.LatencyVariance/1000000.0) // Normalize variance
		compensation.Confidence = countFactor * varianceFactor
	}

	// Calculate time offset based on predicted latency
	if measurement.PredictedLatency > lm.LatencyThreshold {
		// Apply compensation for high latency
		compensationFactor := lm.CompensationFactor * compensation.Confidence
		timeOffset := time.Duration(float64(measurement.PredictedLatency) * compensationFactor)
		compensation.TimeOffset = timeOffset
	}

	// Calculate phase offset (assuming 1ms = 2π * f * dt, with f ≈ 1Hz for simplicity)
	// This is a simplified model - in practice, this would be based on the actual frequency
	phaseOffset := 2.0 * math.Pi * float64(measurement.PredictedLatency.Nanoseconds()) / 1e9
	compensation.PhaseOffset = phaseOffset * lm.CompensationFactor * compensation.Confidence

	return compensation
}

// CompensateLatency applies latency compensation to a phase measurement
func (lm *LatencyMitigator) CompensateLatency(nodeID string, measuredPhase float64, measurementTime time.Time) *CompensationResult {
	lm.mu.RLock()
	defer lm.mu.RUnlock()

	result := &CompensationResult{
		NodeID:          nodeID,
		Timestamp:       time.Now(),
		PhaseAdjustment: 0.0,
		TimeAdjustment:  0,
		Confidence:      0.0,
	}

	measurement, exists := lm.NodeLatencies[nodeID]
	if !exists {
		result.Error = fmt.Sprintf("no latency measurements found for node %s", nodeID)
		return result
	}

	if measurement.Compensation == nil {
		result.Error = fmt.Sprintf("no compensation data available for node %s", nodeID)
		return result
	}

	// Check if compensation is still valid
	if time.Now().After(measurement.Compensation.ValidUntil) {
		result.Error = fmt.Sprintf("compensation data expired for node %s", nodeID)
		return result
	}

	// Apply compensation
	result.OriginalLatency = measurement.PredictedLatency
	result.CompensatedLatency = measurement.PredictedLatency - measurement.Compensation.TimeOffset
	result.PhaseAdjustment = measurement.Compensation.PhaseOffset
	result.TimeAdjustment = measurement.Compensation.TimeOffset
	result.Confidence = measurement.Compensation.Confidence

	return result
}

// GetLatencyStatistics returns comprehensive latency statistics
func (lm *LatencyMitigator) GetLatencyStatistics() *LatencyStatistics {
	lm.mu.RLock()
	defer lm.mu.RUnlock()

	stats := &LatencyStatistics{
		NodeStatistics: make(map[string]NodeLatencyStats),
		Timestamp:      time.Now(),
	}

	if len(lm.NodeLatencies) == 0 {
		return stats
	}

	// Calculate global statistics
	allLatencies := make([]time.Duration, 0)
	minLatency := time.Duration(math.MaxInt64)
	maxLatency := time.Duration(0)
	totalLatency := time.Duration(0)

	for _, measurement := range lm.NodeLatencies {
		if len(measurement.Measurements) == 0 {
			continue
		}

		// Use the most recent measurement for global stats
		recentLatency := measurement.Measurements[len(measurement.Measurements)-1].Latency
		allLatencies = append(allLatencies, recentLatency)
		totalLatency += recentLatency

		if recentLatency < minLatency {
			minLatency = recentLatency
		}
		if recentLatency > maxLatency {
			maxLatency = recentLatency
		}

		// Calculate node-specific statistics
		nodeStats := lm.calculateNodeLatencyStats(measurement)
		stats.NodeStatistics[measurement.NodeID] = nodeStats
	}

	if len(allLatencies) > 0 {
		stats.AverageLatency = totalLatency / time.Duration(len(allLatencies))
		stats.MinLatency = minLatency
		stats.MaxLatency = maxLatency

		// Calculate variance
		variance := 0.0
		for _, latency := range allLatencies {
			diff := float64(latency - stats.AverageLatency)
			variance += diff * diff
		}
		stats.LatencyVariance = variance / float64(len(allLatencies))
	}

	// Calculate prediction accuracy
	if lm.TotalMeasurements > 0 {
		stats.PredictionAccuracy = float64(lm.SuccessfulPredictions) / float64(lm.TotalMeasurements)
	}

	return stats
}

// calculateNodeLatencyStats calculates statistics for a single node
func (lm *LatencyMitigator) calculateNodeLatencyStats(measurement *LatencyMeasurement) NodeLatencyStats {
	stats := NodeLatencyStats{}

	if len(measurement.Measurements) == 0 {
		return stats
	}

	// Calculate average latency
	totalLatency := time.Duration(0)
	for _, sample := range measurement.Measurements {
		totalLatency += sample.Latency
	}
	stats.AverageLatency = totalLatency / time.Duration(len(measurement.Measurements))

	// Calculate variance
	variance := 0.0
	for _, sample := range measurement.Measurements {
		diff := float64(sample.Latency - stats.AverageLatency)
		variance += diff * diff
	}
	stats.LatencyVariance = variance / float64(len(measurement.Measurements))

	// Prediction error (simplified - would need actual vs predicted comparison)
	stats.PredictionError = time.Duration(math.Sqrt(stats.LatencyVariance) * 1e6) // Convert to nanoseconds

	// Compensation statistics
	if measurement.Compensation != nil {
		stats.CompensationCount = 1 // Simplified
		stats.SuccessRate = measurement.Compensation.Confidence
	}

	return stats
}

// PredictLatency predicts latency for a node at a future time
func (lm *LatencyMitigator) PredictLatency(nodeID string, predictionTime time.Time) (time.Duration, float64, error) {
	lm.mu.RLock()
	defer lm.mu.RUnlock()

	measurement, exists := lm.NodeLatencies[nodeID]
	if !exists {
		return 0, 0.0, fmt.Errorf("no latency measurements found for node %s", nodeID)
	}

	if len(measurement.Measurements) < 2 {
		return measurement.PredictedLatency, 0.5, nil // Low confidence
	}

	// Simple linear prediction based on trend
	timeDiff := predictionTime.Sub(measurement.LastUpdate).Seconds()
	predictedLatencyNs := float64(measurement.PredictedLatency.Nanoseconds()) +
		measurement.TrendSlope*timeDiff*1e6 // Convert slope from ms/s to ns/s

	if predictedLatencyNs < 0 {
		predictedLatencyNs = 0
	}

	predictedLatency := time.Duration(predictedLatencyNs)

	// Calculate confidence based on trend stability and measurement count
	measurementCount := len(measurement.Measurements)
	countConfidence := math.Min(float64(measurementCount)/10.0, 1.0)
	trendConfidence := 1.0 / (1.0 + math.Abs(measurement.TrendSlope)/100.0) // Normalize trend
	confidence := countConfidence * trendConfidence

	return predictedLatency, confidence, nil
}

// RemoveStaleMeasurements removes measurements older than the prediction window
func (lm *LatencyMitigator) RemoveStaleMeasurements() int {
	lm.mu.Lock()
	defer lm.mu.Unlock()

	removed := 0
	cutoffTime := time.Now().Add(-lm.PredictionWindow)

	for nodeID, measurement := range lm.NodeLatencies {
		validMeasurements := make([]LatencySample, 0)

		for _, sample := range measurement.Measurements {
			if sample.Timestamp.After(cutoffTime) {
				validMeasurements = append(validMeasurements, sample)
			}
		}

		if len(validMeasurements) == 0 {
			delete(lm.NodeLatencies, nodeID)
			removed++
		} else {
			measurement.Measurements = validMeasurements
		}
	}

	return removed
}

// GetNodeLatency retrieves latency information for a specific node
func (lm *LatencyMitigator) GetNodeLatency(nodeID string) (*LatencyMeasurement, bool) {
	lm.mu.RLock()
	defer lm.mu.RUnlock()

	measurement, exists := lm.NodeLatencies[nodeID]
	return measurement, exists
}

// Reset resets the latency mitigator state
func (lm *LatencyMitigator) Reset() {
	lm.mu.Lock()
	defer lm.mu.Unlock()

	lm.NodeLatencies = make(map[string]*LatencyMeasurement)
	lm.TotalMeasurements = 0
	lm.SuccessfulPredictions = 0
	lm.FailedPredictions = 0
	lm.LastUpdateTime = time.Now()
}
