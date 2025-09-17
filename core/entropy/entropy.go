package entropy

import (
	"math"
	"sort"
	"time"

	"github.com/resonancelab/psizero/core/hilbert"
)

// EntropyEvolution manages entropy dynamics according to S(t) = S₀e^(-λt)
type EntropyEvolution struct {
	initialEntropy float64
	lambda         float64
	startTime      time.Time
	history        []EntropyPoint
	maxHistory     int
}

// EntropyPoint represents entropy measurement at a specific time
type EntropyPoint struct {
	Time     time.Time `json:"time"`
	Entropy  float64   `json:"entropy"`
	Gradient float64   `json:"gradient"`
	Step     int       `json:"step"`
}

// PlateauDetector analyzes entropy evolution for convergence plateaus
type PlateauDetector struct {
	windowSize        int
	tolerance         float64
	minPlateau        int
	gradientThreshold float64
}

// ObservationCapacity computes OC = -ΔS_internal/Δt
type ObservationCapacity struct {
	internalEntropy []float64
	timePoints      []time.Time
	windowSize      int
}

// ConvergenceMetrics contains convergence analysis results
type ConvergenceMetrics struct {
	PlateauDetected  bool    `json:"plateau_detected"`
	PlateauStartStep int     `json:"plateau_start_step,omitempty"`
	PlateauDuration  int     `json:"plateau_duration,omitempty"`
	FinalEntropy     float64 `json:"final_entropy"`
	ConvergenceRate  float64 `json:"convergence_rate"`
	StabilityScore   float64 `json:"stability_score"`
	ObserverCapacity float64 `json:"observer_capacity"`
}

// CollapseAnalysis contains quantum state collapse probability analysis
type CollapseAnalysis struct {
	CollapseProbability  float64   `json:"collapse_probability"`
	ExpectedCollapseTime float64   `json:"expected_collapse_time"`
	EntropyIntegral      float64   `json:"entropy_integral"`
	CriticalPoints       []float64 `json:"critical_points,omitempty"`
}

// NewEntropyEvolution creates a new entropy evolution tracker
func NewEntropyEvolution(initialEntropy, lambda float64, maxHistory int) *EntropyEvolution {
	if maxHistory <= 0 {
		maxHistory = 10000 // Default history size
	}

	return &EntropyEvolution{
		initialEntropy: initialEntropy,
		lambda:         lambda,
		startTime:      time.Now(),
		history:        make([]EntropyPoint, 0, maxHistory),
		maxHistory:     maxHistory,
	}
}

// ComputeEntropy calculates entropy at time t using S(t) = S₀e^(-λt)
func (ee *EntropyEvolution) ComputeEntropy(t float64) float64 {
	if t < 0 {
		t = 0
	}
	return ee.initialEntropy * math.Exp(-ee.lambda*t)
}

// ComputeEntropyAtStep calculates entropy for a given iteration step
func (ee *EntropyEvolution) ComputeEntropyAtStep(step int, timePerStep float64) float64 {
	t := float64(step) * timePerStep
	return ee.ComputeEntropy(t)
}

// AddEntropyPoint records a new entropy measurement
func (ee *EntropyEvolution) AddEntropyPoint(entropy float64, step int) {
	now := time.Now()

	// Calculate gradient if we have previous points
	gradient := 0.0
	if len(ee.history) > 0 {
		lastPoint := ee.history[len(ee.history)-1]
		dt := now.Sub(lastPoint.Time).Seconds()
		if dt > 0 {
			gradient = (entropy - lastPoint.Entropy) / dt
		}
	}

	point := EntropyPoint{
		Time:     now,
		Entropy:  entropy,
		Gradient: gradient,
		Step:     step,
	}

	ee.history = append(ee.history, point)

	// Trim history if too long
	if len(ee.history) > ee.maxHistory {
		ee.history = ee.history[len(ee.history)-ee.maxHistory:]
	}
}

// CollapseProbability calculates P_collapse = 1 - e^(-∫S(t)dt)
func (ee *EntropyEvolution) CollapseProbability(timeEnd float64) float64 {
	if timeEnd <= 0 || ee.lambda == 0 {
		return 0.0
	}

	// Analytical solution: ∫₀ᵗ S₀e^(-λτ)dτ = S₀/λ * (1 - e^(-λt))
	integral := (ee.initialEntropy / ee.lambda) * (1.0 - math.Exp(-ee.lambda*timeEnd))

	// P_collapse = 1 - e^(-integral)
	probability := 1.0 - math.Exp(-integral)

	// Clamp to [0, 1]
	if probability < 0 {
		return 0.0
	}
	if probability > 1 {
		return 1.0
	}

	return probability
}

// ExpectedCollapseTime calculates when collapse probability reaches threshold
func (ee *EntropyEvolution) ExpectedCollapseTime(threshold float64) float64 {
	if threshold <= 0 || threshold >= 1 || ee.lambda == 0 {
		return math.Inf(1)
	}

	// Solve 1 - e^(-S₀/λ * (1 - e^(-λt))) = threshold
	// This requires numerical solution for general case

	// For small thresholds, approximate solution
	if threshold < 0.1 {
		return -math.Log(1.0-threshold*ee.lambda/ee.initialEntropy) / ee.lambda
	}

	// Binary search for more accurate solution
	return ee.binarySearchCollapseTime(threshold, 0.0, 100.0/ee.lambda)
}

// binarySearchCollapseTime finds collapse time using binary search
func (ee *EntropyEvolution) binarySearchCollapseTime(threshold, tMin, tMax float64) float64 {
	tolerance := 1e-6
	maxIterations := 100

	for i := 0; i < maxIterations; i++ {
		tMid := (tMin + tMax) / 2.0
		prob := ee.CollapseProbability(tMid)

		if math.Abs(prob-threshold) < tolerance {
			return tMid
		}

		if prob < threshold {
			tMin = tMid
		} else {
			tMax = tMid
		}
	}

	return (tMin + tMax) / 2.0
}

// GetCurrentEntropy returns the most recent entropy measurement
func (ee *EntropyEvolution) GetCurrentEntropy() (float64, bool) {
	if len(ee.history) == 0 {
		return 0.0, false
	}

	return ee.history[len(ee.history)-1].Entropy, true
}

// GetEntropyHistory returns a copy of the entropy history
func (ee *EntropyEvolution) GetEntropyHistory() []EntropyPoint {
	history := make([]EntropyPoint, len(ee.history))
	copy(history, ee.history)
	return history
}

// Reset resets the entropy evolution tracker
func (ee *EntropyEvolution) Reset(initialEntropy, lambda float64) {
	ee.initialEntropy = initialEntropy
	ee.lambda = lambda
	ee.startTime = time.Now()
	ee.history = ee.history[:0] // Clear but keep capacity
}

// NewPlateauDetector creates a new plateau detection system
func NewPlateauDetector(windowSize int, tolerance float64, minPlateau int, gradientThreshold float64) *PlateauDetector {
	return &PlateauDetector{
		windowSize:        windowSize,
		tolerance:         tolerance,
		minPlateau:        minPlateau,
		gradientThreshold: gradientThreshold,
	}
}

// DetectPlateau analyzes entropy history for convergence plateaus
func (pd *PlateauDetector) DetectPlateau(history []EntropyPoint) (bool, int, int) {
	if len(history) < pd.windowSize+pd.minPlateau {
		return false, -1, 0
	}

	// Look for plateau in the most recent data
	recentHistory := history
	if len(history) > pd.windowSize*3 {
		recentHistory = history[len(history)-pd.windowSize*3:]
	}

	return pd.findPlateau(recentHistory)
}

// findPlateau searches for plateau patterns in entropy data
func (pd *PlateauDetector) findPlateau(history []EntropyPoint) (bool, int, int) {
	if len(history) < pd.windowSize+pd.minPlateau {
		return false, -1, 0
	}

	// Calculate rolling variance and gradient
	for start := 0; start <= len(history)-pd.windowSize-pd.minPlateau; start++ {
		plateauLength := pd.analyzeWindow(history[start:])

		if plateauLength >= pd.minPlateau {
			return true, start, plateauLength
		}
	}

	return false, -1, 0
}

// analyzeWindow analyzes a window for plateau characteristics
func (pd *PlateauDetector) analyzeWindow(window []EntropyPoint) int {
	if len(window) < pd.windowSize {
		return 0
	}

	// Calculate variance in the initial window
	variance := pd.calculateVariance(window[:pd.windowSize])
	if variance > pd.tolerance {
		return 0
	}

	// Extend the plateau as far as possible
	plateauLength := pd.windowSize
	baseEntropy := pd.calculateMean(window[:pd.windowSize])

	for i := pd.windowSize; i < len(window); i++ {
		// Check if this point maintains the plateau
		if math.Abs(window[i].Entropy-baseEntropy) > pd.tolerance {
			break
		}

		// Check gradient constraint
		if math.Abs(window[i].Gradient) > pd.gradientThreshold {
			break
		}

		plateauLength++
	}

	return plateauLength
}

// calculateVariance computes variance of entropy values in a window
func (pd *PlateauDetector) calculateVariance(window []EntropyPoint) float64 {
	if len(window) == 0 {
		return 0.0
	}

	mean := pd.calculateMean(window)
	variance := 0.0

	for _, point := range window {
		diff := point.Entropy - mean
		variance += diff * diff
	}

	return variance / float64(len(window))
}

// calculateMean computes mean entropy in a window
func (pd *PlateauDetector) calculateMean(window []EntropyPoint) float64 {
	if len(window) == 0 {
		return 0.0
	}

	sum := 0.0
	for _, point := range window {
		sum += point.Entropy
	}

	return sum / float64(len(window))
}

// NewObservationCapacity creates a new observation capacity calculator
func NewObservationCapacity(windowSize int) *ObservationCapacity {
	return &ObservationCapacity{
		internalEntropy: make([]float64, 0, windowSize*2),
		timePoints:      make([]time.Time, 0, windowSize*2),
		windowSize:      windowSize,
	}
}

// AddMeasurement adds a new internal entropy measurement
func (oc *ObservationCapacity) AddMeasurement(entropy float64, timestamp time.Time) {
	oc.internalEntropy = append(oc.internalEntropy, entropy)
	oc.timePoints = append(oc.timePoints, timestamp)

	// Trim if too long
	if len(oc.internalEntropy) > oc.windowSize*2 {
		trimSize := oc.windowSize
		oc.internalEntropy = oc.internalEntropy[trimSize:]
		oc.timePoints = oc.timePoints[trimSize:]
	}
}

// ComputeObservationCapacity calculates OC = -ΔS_internal/Δt
func (oc *ObservationCapacity) ComputeObservationCapacity() float64 {
	if len(oc.internalEntropy) < 2 {
		return 0.0
	}

	// Use recent measurements for calculation
	recentSize := oc.windowSize
	if len(oc.internalEntropy) < recentSize {
		recentSize = len(oc.internalEntropy)
	}

	startIdx := len(oc.internalEntropy) - recentSize

	// Calculate entropy change over time
	deltaS := oc.internalEntropy[len(oc.internalEntropy)-1] - oc.internalEntropy[startIdx]
	deltaT := oc.timePoints[len(oc.timePoints)-1].Sub(oc.timePoints[startIdx]).Seconds()

	if deltaT == 0 {
		return 0.0
	}

	// OC = -ΔS_internal/Δt
	return -deltaS / deltaT
}

// AnalyzeConvergence performs comprehensive convergence analysis
func AnalyzeConvergence(entropyEvolution *EntropyEvolution, plateauDetector *PlateauDetector,
	observationCapacity *ObservationCapacity, state *hilbert.QuantumState) *ConvergenceMetrics {

	history := entropyEvolution.GetEntropyHistory()

	// Detect plateau
	plateauDetected, plateauStart, plateauDuration := plateauDetector.DetectPlateau(history)

	// Get final entropy
	finalEntropy := 0.0
	if len(history) > 0 {
		finalEntropy = history[len(history)-1].Entropy
	}

	// Calculate convergence rate
	convergenceRate := calculateConvergenceRate(history)

	// Calculate stability score
	stabilityScore := calculateStabilityScore(history, plateauDetector.windowSize)

	// Get observation capacity
	observerCapacity := observationCapacity.ComputeObservationCapacity()

	return &ConvergenceMetrics{
		PlateauDetected:  plateauDetected,
		PlateauStartStep: plateauStart,
		PlateauDuration:  plateauDuration,
		FinalEntropy:     finalEntropy,
		ConvergenceRate:  convergenceRate,
		StabilityScore:   stabilityScore,
		ObserverCapacity: observerCapacity,
	}
}

// calculateConvergenceRate estimates the rate of entropy convergence
func calculateConvergenceRate(history []EntropyPoint) float64 {
	if len(history) < 10 {
		return 0.0
	}

	// Use linear regression on log(entropy) vs time for exponential fit
	var sumX, sumY, sumXY, sumX2 float64
	n := len(history)
	startTime := history[0].Time

	for _, point := range history {
		if point.Entropy <= 0 {
			continue
		}

		x := point.Time.Sub(startTime).Seconds()
		y := math.Log(point.Entropy)

		sumX += x
		sumY += y
		sumXY += x * y
		sumX2 += x * x
	}

	// Calculate slope (convergence rate)
	denominator := float64(n)*sumX2 - sumX*sumX
	if math.Abs(denominator) < 1e-15 {
		return 0.0
	}

	slope := (float64(n)*sumXY - sumX*sumY) / denominator
	return -slope // Return positive rate for convergence
}

// calculateStabilityScore measures the stability of recent entropy values
func calculateStabilityScore(history []EntropyPoint, windowSize int) float64 {
	if len(history) < windowSize {
		return 0.0
	}

	// Analyze recent window
	recentWindow := history[len(history)-windowSize:]

	// Calculate coefficient of variation (std dev / mean)
	mean := 0.0
	for _, point := range recentWindow {
		mean += point.Entropy
	}
	mean /= float64(len(recentWindow))

	if mean == 0 {
		return 0.0
	}

	variance := 0.0
	for _, point := range recentWindow {
		diff := point.Entropy - mean
		variance += diff * diff
	}
	variance /= float64(len(recentWindow))

	stdDev := math.Sqrt(variance)
	coeffVar := stdDev / mean

	// Stability score: higher is more stable (inverse of coefficient of variation)
	return 1.0 / (1.0 + coeffVar)
}

// PerformCollapseAnalysis analyzes quantum state collapse dynamics
func PerformCollapseAnalysis(entropyEvolution *EntropyEvolution, timeHorizon float64) *CollapseAnalysis {
	// Calculate collapse probability over time horizon
	collapseProbability := entropyEvolution.CollapseProbability(timeHorizon)

	// Calculate expected collapse time (50% probability)
	expectedCollapseTime := entropyEvolution.ExpectedCollapseTime(0.5)

	// Calculate entropy integral
	entropyIntegral := (entropyEvolution.initialEntropy / entropyEvolution.lambda) *
		(1.0 - math.Exp(-entropyEvolution.lambda*timeHorizon))

	// Find critical points (where collapse probability reaches certain thresholds)
	thresholds := []float64{0.1, 0.25, 0.5, 0.75, 0.9}
	criticalPoints := make([]float64, len(thresholds))

	for i, threshold := range thresholds {
		criticalPoints[i] = entropyEvolution.ExpectedCollapseTime(threshold)
	}

	return &CollapseAnalysis{
		CollapseProbability:  collapseProbability,
		ExpectedCollapseTime: expectedCollapseTime,
		EntropyIntegral:      entropyIntegral,
		CriticalPoints:       criticalPoints,
	}
}

// PredictEntropyEvolution predicts future entropy values
func (ee *EntropyEvolution) PredictEntropyEvolution(steps int, timePerStep float64) []float64 {
	predictions := make([]float64, steps)
	currentTime := time.Since(ee.startTime).Seconds()

	for i := 0; i < steps; i++ {
		futureTime := currentTime + float64(i+1)*timePerStep
		predictions[i] = ee.ComputeEntropy(futureTime)
	}

	return predictions
}

// GetEntropyStatistics returns statistical summary of entropy evolution
func (ee *EntropyEvolution) GetEntropyStatistics() map[string]float64 {
	if len(ee.history) == 0 {
		return map[string]float64{}
	}

	values := make([]float64, len(ee.history))
	for i, point := range ee.history {
		values[i] = point.Entropy
	}

	sort.Float64s(values)

	stats := map[string]float64{
		"count":  float64(len(values)),
		"min":    values[0],
		"max":    values[len(values)-1],
		"median": values[len(values)/2],
		"q25":    values[len(values)/4],
		"q75":    values[3*len(values)/4],
	}

	// Calculate mean and standard deviation
	sum := 0.0
	for _, v := range values {
		sum += v
	}
	mean := sum / float64(len(values))
	stats["mean"] = mean

	variance := 0.0
	for _, v := range values {
		diff := v - mean
		variance += diff * diff
	}
	variance /= float64(len(values))
	stats["std"] = math.Sqrt(variance)

	return stats
}
