package core

import (
	"fmt"
	"math"
	"sort"
	"sync"
	"time"
)

// TelemetryCollector manages comprehensive metrics collection for the Reson.net system
type TelemetryCollector struct {
	// Core metrics storage
	CoherenceMetrics   []CoherenceMetric   `json:"coherence_metrics"`
	ResonanceMetrics   []ResonanceMetric   `json:"resonance_metrics"`
	PhaseMetrics       []PhaseMetric       `json:"phase_metrics"`
	PerformanceMetrics []PerformanceMetric `json:"performance_metrics"`
	NetworkMetrics     []NetworkMetric     `json:"network_metrics"`
	EconomicMetrics    []EconomicMetric    `json:"economic_metrics"`

	// Aggregation settings
	AggregationWindow  time.Duration `json:"aggregation_window"`
	MaxMetricsHistory  int           `json:"max_metrics_history"`
	CollectionInterval time.Duration `json:"collection_interval"`

	// Statistical tracking
	TotalCollections   int64                           `json:"total_collections"`
	LastCollectionTime time.Time                       `json:"last_collection_time"`
	MetricsByNode      map[string]TelemetryNodeMetrics `json:"metrics_by_node"`

	// Alert thresholds
	CoherenceThreshold  float64       `json:"coherence_threshold"`
	PhaseDriftThreshold float64       `json:"phase_drift_threshold"`
	LatencyThreshold    time.Duration `json:"latency_threshold"`

	// Thread safety
	mu sync.RWMutex `json:"-"`
}

// CoherenceMetric represents coherence measurements
type CoherenceMetric struct {
	NodeID          string    `json:"node_id"`
	GlobalCoherence float64   `json:"global_coherence"`
	LocalCoherence  float64   `json:"local_coherence"`
	PhaseVariance   float64   `json:"phase_variance"`
	Connectivity    float64   `json:"connectivity"`
	Timestamp       time.Time `json:"timestamp"`
}

// ResonanceMetric represents resonance measurements
type ResonanceMetric struct {
	NodeID         string    `json:"node_id"`
	PrimeResonance float64   `json:"prime_resonance"`
	EnergyLevel    float64   `json:"energy_level"`
	Frequency      float64   `json:"frequency"`
	Amplitude      float64   `json:"amplitude"`
	QualityFactor  float64   `json:"quality_factor"`
	Timestamp      time.Time `json:"timestamp"`
}

// PhaseMetric represents phase alignment measurements
type PhaseMetric struct {
	NodeID        string    `json:"node_id"`
	PhaseValue    float64   `json:"phase_value"`
	PhaseDrift    float64   `json:"phase_drift"`
	PhaseVelocity float64   `json:"phase_velocity"`
	SyncQuality   float64   `json:"sync_quality"`
	Compensation  float64   `json:"compensation"`
	Timestamp     time.Time `json:"timestamp"`
}

// PerformanceMetric represents system performance measurements
type PerformanceMetric struct {
	NodeID           string        `json:"node_id"`
	OperationsPerSec float64       `json:"operations_per_sec"`
	MemoryUsage      float64       `json:"memory_usage"`
	CPUUsage         float64       `json:"cpu_usage"`
	Latency          time.Duration `json:"latency"`
	Throughput       float64       `json:"throughput"`
	ErrorRate        float64       `json:"error_rate"`
	Timestamp        time.Time     `json:"timestamp"`
}

// NetworkMetric represents network performance measurements
type NetworkMetric struct {
	NodeID          string        `json:"node_id"`
	NetworkLatency  time.Duration `json:"network_latency"`
	PacketLoss      float64       `json:"packet_loss"`
	Bandwidth       float64       `json:"bandwidth"`
	ConnectionCount int           `json:"connection_count"`
	SyncLatency     time.Duration `json:"sync_latency"`
	Timestamp       time.Time     `json:"timestamp"`
}

// EconomicMetric represents economic system measurements
type EconomicMetric struct {
	NodeID              string    `json:"node_id"`
	TokenBalance        float64   `json:"token_balance"`
	TransactionRate     float64   `json:"transaction_rate"`
	ResourceUtilization float64   `json:"resource_utilization"`
	RewardRate          float64   `json:"reward_rate"`
	MarketPrice         float64   `json:"market_price"`
	Timestamp           time.Time `json:"timestamp"`
}

// TelemetryNodeMetrics contains aggregated metrics for a single node
type TelemetryNodeMetrics struct {
	NodeID             string    `json:"node_id"`
	AverageCoherence   float64   `json:"average_coherence"`
	AverageResonance   float64   `json:"average_resonance"`
	PhaseStability     float64   `json:"phase_stability"`
	PerformanceScore   float64   `json:"performance_score"`
	NetworkHealth      float64   `json:"network_health"`
	EconomicEfficiency float64   `json:"economic_efficiency"`
	LastUpdate         time.Time `json:"last_update"`
	MetricsCount       int64     `json:"metrics_count"`
}

// TelemetryReport contains comprehensive system telemetry
type TelemetryReport struct {
	GlobalMetrics      GlobalMetrics                   `json:"global_metrics"`
	NodeMetrics        map[string]TelemetryNodeMetrics `json:"node_metrics"`
	TrendAnalysis      TrendAnalysis                   `json:"trend_analysis"`
	AlertStatus        AlertStatus                     `json:"alert_status"`
	PerformanceSummary PerformanceSummary              `json:"performance_summary"`
	Timestamp          time.Time                       `json:"timestamp"`
}

// GlobalMetrics contains system-wide aggregated metrics
type GlobalMetrics struct {
	AverageCoherence     float64 `json:"average_coherence"`
	GlobalResonance      float64 `json:"global_resonance"`
	PhaseSynchronization float64 `json:"phase_synchronization"`
	SystemPerformance    float64 `json:"system_performance"`
	NetworkHealth        float64 `json:"network_health"`
	EconomicActivity     float64 `json:"economic_activity"`
	ActiveNodes          int     `json:"active_nodes"`
}

// TrendAnalysis contains trend analysis for key metrics
type TrendAnalysis struct {
	CoherenceTrend   TrendData `json:"coherence_trend"`
	PerformanceTrend TrendData `json:"performance_trend"`
	NetworkTrend     TrendData `json:"network_trend"`
	EconomicTrend    TrendData `json:"economic_trend"`
}

// TrendData contains trend information for a metric
type TrendData struct {
	Slope          float64       `json:"slope"`
	Direction      string        `json:"direction"` // "increasing", "decreasing", "stable"
	Confidence     float64       `json:"confidence"`
	PredictedValue float64       `json:"predicted_value"`
	TimeHorizon    time.Duration `json:"time_horizon"`
}

// AlertStatus contains current alert status
type AlertStatus struct {
	ActiveAlerts  []Alert `json:"active_alerts"`
	CriticalCount int     `json:"critical_count"`
	WarningCount  int     `json:"warning_count"`
	InfoCount     int     `json:"info_count"`
}

// Alert represents a system alert
type Alert struct {
	ID        string    `json:"id"`
	Type      string    `json:"type"`     // "critical", "warning", "info"
	Category  string    `json:"category"` // "coherence", "performance", "network", "economic"
	Message   string    `json:"message"`
	NodeID    string    `json:"node_id,omitempty"`
	Value     float64   `json:"value"`
	Threshold float64   `json:"threshold"`
	Timestamp time.Time `json:"timestamp"`
}

// PerformanceSummary contains performance analysis
type PerformanceSummary struct {
	TopPerformingNodes      []string `json:"top_performing_nodes"`
	UnderperformingNodes    []string `json:"underperforming_nodes"`
	SystemEfficiency        float64  `json:"system_efficiency"`
	BottleneckAnalysis      string   `json:"bottleneck_analysis"`
	OptimizationSuggestions []string `json:"optimization_suggestions"`
}

// NewTelemetryCollector creates a new telemetry collector
func NewTelemetryCollector(aggregationWindow time.Duration, maxHistory int, collectionInterval time.Duration) *TelemetryCollector {
	return &TelemetryCollector{
		CoherenceMetrics:    make([]CoherenceMetric, 0),
		ResonanceMetrics:    make([]ResonanceMetric, 0),
		PhaseMetrics:        make([]PhaseMetric, 0),
		PerformanceMetrics:  make([]PerformanceMetric, 0),
		NetworkMetrics:      make([]NetworkMetric, 0),
		EconomicMetrics:     make([]EconomicMetric, 0),
		AggregationWindow:   aggregationWindow,
		MaxMetricsHistory:   maxHistory,
		CollectionInterval:  collectionInterval,
		TotalCollections:    0,
		LastCollectionTime:  time.Now(),
		MetricsByNode:       make(map[string]TelemetryNodeMetrics),
		CoherenceThreshold:  0.8,
		PhaseDriftThreshold: 0.1,
		LatencyThreshold:    100 * time.Millisecond,
	}
}

// RecordCoherenceMetric records a coherence measurement
func (tc *TelemetryCollector) RecordCoherenceMetric(nodeID string, globalCoherence, localCoherence, phaseVariance, connectivity float64) {
	tc.mu.Lock()
	defer tc.mu.Unlock()

	metric := CoherenceMetric{
		NodeID:          nodeID,
		GlobalCoherence: globalCoherence,
		LocalCoherence:  localCoherence,
		PhaseVariance:   phaseVariance,
		Connectivity:    connectivity,
		Timestamp:       time.Now(),
	}

	tc.CoherenceMetrics = append(tc.CoherenceMetrics, metric)
	tc.cleanOldMetrics()
	tc.updateNodeMetrics(nodeID)
}

// RecordResonanceMetric records a resonance measurement
func (tc *TelemetryCollector) RecordResonanceMetric(nodeID string, primeResonance, energyLevel, frequency, amplitude, qualityFactor float64) {
	tc.mu.Lock()
	defer tc.mu.Unlock()

	metric := ResonanceMetric{
		NodeID:         nodeID,
		PrimeResonance: primeResonance,
		EnergyLevel:    energyLevel,
		Frequency:      frequency,
		Amplitude:      amplitude,
		QualityFactor:  qualityFactor,
		Timestamp:      time.Now(),
	}

	tc.ResonanceMetrics = append(tc.ResonanceMetrics, metric)
	tc.cleanOldMetrics()
	tc.updateNodeMetrics(nodeID)
}

// RecordPhaseMetric records a phase measurement
func (tc *TelemetryCollector) RecordPhaseMetric(nodeID string, phaseValue, phaseDrift, phaseVelocity, syncQuality, compensation float64) {
	tc.mu.Lock()
	defer tc.mu.Unlock()

	metric := PhaseMetric{
		NodeID:        nodeID,
		PhaseValue:    phaseValue,
		PhaseDrift:    phaseDrift,
		PhaseVelocity: phaseVelocity,
		SyncQuality:   syncQuality,
		Compensation:  compensation,
		Timestamp:     time.Now(),
	}

	tc.PhaseMetrics = append(tc.PhaseMetrics, metric)
	tc.cleanOldMetrics()
	tc.updateNodeMetrics(nodeID)
}

// RecordPerformanceMetric records a performance measurement
func (tc *TelemetryCollector) RecordPerformanceMetric(nodeID string, opsPerSec, memoryUsage, cpuUsage float64, latency time.Duration, throughput, errorRate float64) {
	tc.mu.Lock()
	defer tc.mu.Unlock()

	metric := PerformanceMetric{
		NodeID:           nodeID,
		OperationsPerSec: opsPerSec,
		MemoryUsage:      memoryUsage,
		CPUUsage:         cpuUsage,
		Latency:          latency,
		Throughput:       throughput,
		ErrorRate:        errorRate,
		Timestamp:        time.Now(),
	}

	tc.PerformanceMetrics = append(tc.PerformanceMetrics, metric)
	tc.cleanOldMetrics()
	tc.updateNodeMetrics(nodeID)
}

// RecordNetworkMetric records a network measurement
func (tc *TelemetryCollector) RecordNetworkMetric(nodeID string, networkLatency time.Duration, packetLoss, bandwidth float64, connectionCount int, syncLatency time.Duration) {
	tc.mu.Lock()
	defer tc.mu.Unlock()

	metric := NetworkMetric{
		NodeID:          nodeID,
		NetworkLatency:  networkLatency,
		PacketLoss:      packetLoss,
		Bandwidth:       bandwidth,
		ConnectionCount: connectionCount,
		SyncLatency:     syncLatency,
		Timestamp:       time.Now(),
	}

	tc.NetworkMetrics = append(tc.NetworkMetrics, metric)
	tc.cleanOldMetrics()
	tc.updateNodeMetrics(nodeID)
}

// RecordEconomicMetric records an economic measurement
func (tc *TelemetryCollector) RecordEconomicMetric(nodeID string, tokenBalance, transactionRate, resourceUtilization, rewardRate, marketPrice float64) {
	tc.mu.Lock()
	defer tc.mu.Unlock()

	metric := EconomicMetric{
		NodeID:              nodeID,
		TokenBalance:        tokenBalance,
		TransactionRate:     transactionRate,
		ResourceUtilization: resourceUtilization,
		RewardRate:          rewardRate,
		MarketPrice:         marketPrice,
		Timestamp:           time.Now(),
	}

	tc.EconomicMetrics = append(tc.EconomicMetrics, metric)
	tc.cleanOldMetrics()
	tc.updateNodeMetrics(nodeID)
}

// cleanOldMetrics removes metrics older than the aggregation window
func (tc *TelemetryCollector) cleanOldMetrics() {
	cutoffTime := time.Now().Add(-tc.AggregationWindow)

	// Clean coherence metrics
	validCoherence := make([]CoherenceMetric, 0)
	for _, metric := range tc.CoherenceMetrics {
		if metric.Timestamp.After(cutoffTime) {
			validCoherence = append(validCoherence, metric)
		}
	}
	tc.CoherenceMetrics = validCoherence

	// Clean other metrics similarly
	validResonance := make([]ResonanceMetric, 0)
	for _, metric := range tc.ResonanceMetrics {
		if metric.Timestamp.After(cutoffTime) {
			validResonance = append(validResonance, metric)
		}
	}
	tc.ResonanceMetrics = validResonance

	// Continue for other metric types...
	validPhase := make([]PhaseMetric, 0)
	for _, metric := range tc.PhaseMetrics {
		if metric.Timestamp.After(cutoffTime) {
			validPhase = append(validPhase, metric)
		}
	}
	tc.PhaseMetrics = validPhase

	validPerformance := make([]PerformanceMetric, 0)
	for _, metric := range tc.PerformanceMetrics {
		if metric.Timestamp.After(cutoffTime) {
			validPerformance = append(validPerformance, metric)
		}
	}
	tc.PerformanceMetrics = validPerformance

	validNetwork := make([]NetworkMetric, 0)
	for _, metric := range tc.NetworkMetrics {
		if metric.Timestamp.After(cutoffTime) {
			validNetwork = append(validNetwork, metric)
		}
	}
	tc.NetworkMetrics = validNetwork

	validEconomic := make([]EconomicMetric, 0)
	for _, metric := range tc.EconomicMetrics {
		if metric.Timestamp.After(cutoffTime) {
			validEconomic = append(validEconomic, metric)
		}
	}
	tc.EconomicMetrics = validEconomic
}

// updateNodeMetrics updates aggregated metrics for a node
func (tc *TelemetryCollector) updateNodeMetrics(nodeID string) {
	nodeMetrics := tc.MetricsByNode[nodeID]
	if nodeMetrics.NodeID == "" {
		nodeMetrics = TelemetryNodeMetrics{
			NodeID: nodeID,
		}
	}

	// Calculate averages from recent metrics
	nodeMetrics.AverageCoherence = tc.calculateNodeAverageCoherence(nodeID)
	nodeMetrics.AverageResonance = tc.calculateNodeAverageResonance(nodeID)
	nodeMetrics.PhaseStability = tc.calculateNodePhaseStability(nodeID)
	nodeMetrics.PerformanceScore = tc.calculateNodePerformanceScore(nodeID)
	nodeMetrics.NetworkHealth = tc.calculateNodeNetworkHealth(nodeID)
	nodeMetrics.EconomicEfficiency = tc.calculateNodeEconomicEfficiency(nodeID)
	nodeMetrics.LastUpdate = time.Now()
	nodeMetrics.MetricsCount++

	tc.MetricsByNode[nodeID] = nodeMetrics
}

// calculateNodeAverageCoherence calculates average coherence for a node
func (tc *TelemetryCollector) calculateNodeAverageCoherence(nodeID string) float64 {
	coherenceValues := make([]float64, 0)

	for _, metric := range tc.CoherenceMetrics {
		if metric.NodeID == nodeID {
			coherenceValues = append(coherenceValues, metric.GlobalCoherence)
		}
	}

	if len(coherenceValues) == 0 {
		return 0.0
	}

	sum := 0.0
	for _, value := range coherenceValues {
		sum += value
	}

	return sum / float64(len(coherenceValues))
}

// calculateNodeAverageResonance calculates average resonance for a node
func (tc *TelemetryCollector) calculateNodeAverageResonance(nodeID string) float64 {
	resonanceValues := make([]float64, 0)

	for _, metric := range tc.ResonanceMetrics {
		if metric.NodeID == nodeID {
			resonanceValues = append(resonanceValues, metric.PrimeResonance)
		}
	}

	if len(resonanceValues) == 0 {
		return 0.0
	}

	sum := 0.0
	for _, value := range resonanceValues {
		sum += value
	}

	return sum / float64(len(resonanceValues))
}

// calculateNodePhaseStability calculates phase stability for a node
func (tc *TelemetryCollector) calculateNodePhaseStability(nodeID string) float64 {
	phaseValues := make([]float64, 0)

	for _, metric := range tc.PhaseMetrics {
		if metric.NodeID == nodeID {
			phaseValues = append(phaseValues, metric.PhaseDrift)
		}
	}

	if len(phaseValues) < 2 {
		return 1.0 // Perfect stability with insufficient data
	}

	// Calculate variance of phase drift
	sum := 0.0
	for _, value := range phaseValues {
		sum += value
	}
	mean := sum / float64(len(phaseValues))

	variance := 0.0
	for _, value := range phaseValues {
		diff := value - mean
		variance += diff * diff
	}
	variance /= float64(len(phaseValues))

	// Convert variance to stability score (lower variance = higher stability)
	stability := 1.0 / (1.0 + variance)
	return stability
}

// calculateNodePerformanceScore calculates performance score for a node
func (tc *TelemetryCollector) calculateNodePerformanceScore(nodeID string) float64 {
	performanceValues := make([]float64, 0)

	for _, metric := range tc.PerformanceMetrics {
		if metric.NodeID == nodeID {
			// Combine multiple performance metrics into a single score
			score := metric.OperationsPerSec*0.4 +
				(1.0-metric.MemoryUsage)*0.2 +
				(1.0-metric.CPUUsage)*0.2 +
				(1.0-metric.ErrorRate)*0.2
			performanceValues = append(performanceValues, score)
		}
	}

	if len(performanceValues) == 0 {
		return 0.0
	}

	sum := 0.0
	for _, value := range performanceValues {
		sum += value
	}

	return sum / float64(len(performanceValues))
}

// calculateNodeNetworkHealth calculates network health for a node
func (tc *TelemetryCollector) calculateNodeNetworkHealth(nodeID string) float64 {
	networkValues := make([]float64, 0)

	for _, metric := range tc.NetworkMetrics {
		if metric.NodeID == nodeID {
			// Combine network metrics into health score
			latencyScore := 1.0 / (1.0 + float64(metric.NetworkLatency.Nanoseconds())/1e9) // Normalize latency
			lossScore := 1.0 - metric.PacketLoss
			connectionScore := float64(metric.ConnectionCount) / 10.0 // Normalize connections
			if connectionScore > 1.0 {
				connectionScore = 1.0
			}

			health := latencyScore*0.4 + lossScore*0.3 + connectionScore*0.3
			networkValues = append(networkValues, health)
		}
	}

	if len(networkValues) == 0 {
		return 0.0
	}

	sum := 0.0
	for _, value := range networkValues {
		sum += value
	}

	return sum / float64(len(networkValues))
}

// calculateNodeEconomicEfficiency calculates economic efficiency for a node
func (tc *TelemetryCollector) calculateNodeEconomicEfficiency(nodeID string) float64 {
	economicValues := make([]float64, 0)

	for _, metric := range tc.EconomicMetrics {
		if metric.NodeID == nodeID {
			// Combine economic metrics into efficiency score
			utilizationScore := metric.ResourceUtilization
			rewardScore := metric.RewardRate / 100.0 // Normalize reward rate
			if rewardScore > 1.0 {
				rewardScore = 1.0
			}

			efficiency := utilizationScore*0.6 + rewardScore*0.4
			economicValues = append(economicValues, efficiency)
		}
	}

	if len(economicValues) == 0 {
		return 0.0
	}

	sum := 0.0
	for _, value := range economicValues {
		sum += value
	}

	return sum / float64(len(economicValues))
}

// GenerateTelemetryReport generates a comprehensive telemetry report
func (tc *TelemetryCollector) GenerateTelemetryReport() *TelemetryReport {
	tc.mu.RLock()
	defer tc.mu.RUnlock()

	report := &TelemetryReport{
		NodeMetrics:        make(map[string]TelemetryNodeMetrics),
		TrendAnalysis:      tc.generateTrendAnalysis(),
		AlertStatus:        tc.generateAlertStatus(),
		PerformanceSummary: tc.generatePerformanceSummary(),
		Timestamp:          time.Now(),
	}

	// Calculate global metrics
	report.GlobalMetrics = tc.calculateGlobalMetrics()

	// Copy node metrics
	for nodeID, metrics := range tc.MetricsByNode {
		report.NodeMetrics[nodeID] = metrics
	}

	return report
}

// calculateGlobalMetrics calculates system-wide aggregated metrics
func (tc *TelemetryCollector) calculateGlobalMetrics() GlobalMetrics {
	metrics := GlobalMetrics{}

	// Calculate average coherence
	if len(tc.CoherenceMetrics) > 0 {
		sum := 0.0
		for _, metric := range tc.CoherenceMetrics {
			sum += metric.GlobalCoherence
		}
		metrics.AverageCoherence = sum / float64(len(tc.CoherenceMetrics))
	}

	// Calculate global resonance
	if len(tc.ResonanceMetrics) > 0 {
		sum := 0.0
		for _, metric := range tc.ResonanceMetrics {
			sum += metric.PrimeResonance
		}
		metrics.GlobalResonance = sum / float64(len(tc.ResonanceMetrics))
	}

	// Calculate phase synchronization
	if len(tc.PhaseMetrics) > 0 {
		sum := 0.0
		for _, metric := range tc.PhaseMetrics {
			sum += metric.SyncQuality
		}
		metrics.PhaseSynchronization = sum / float64(len(tc.PhaseMetrics))
	}

	// Calculate system performance
	if len(tc.PerformanceMetrics) > 0 {
		sum := 0.0
		for _, metric := range tc.PerformanceMetrics {
			sum += metric.OperationsPerSec
		}
		metrics.SystemPerformance = sum / float64(len(tc.PerformanceMetrics))
	}

	// Calculate network health
	if len(tc.NetworkMetrics) > 0 {
		sum := 0.0
		for _, metric := range tc.NetworkMetrics {
			health := 1.0 / (1.0 + float64(metric.NetworkLatency.Nanoseconds())/1e9)
			sum += health
		}
		metrics.NetworkHealth = sum / float64(len(tc.NetworkMetrics))
	}

	// Calculate economic activity
	if len(tc.EconomicMetrics) > 0 {
		sum := 0.0
		for _, metric := range tc.EconomicMetrics {
			sum += metric.TransactionRate
		}
		metrics.EconomicActivity = sum / float64(len(tc.EconomicMetrics))
	}

	// Count active nodes
	activeNodes := make(map[string]bool)
	for _, metric := range tc.CoherenceMetrics {
		activeNodes[metric.NodeID] = true
	}
	for _, metric := range tc.ResonanceMetrics {
		activeNodes[metric.NodeID] = true
	}
	for _, metric := range tc.PhaseMetrics {
		activeNodes[metric.NodeID] = true
	}
	for _, metric := range tc.PerformanceMetrics {
		activeNodes[metric.NodeID] = true
	}
	for _, metric := range tc.NetworkMetrics {
		activeNodes[metric.NodeID] = true
	}
	for _, metric := range tc.EconomicMetrics {
		activeNodes[metric.NodeID] = true
	}

	metrics.ActiveNodes = len(activeNodes)

	return metrics
}

// generateTrendAnalysis generates trend analysis for key metrics
func (tc *TelemetryCollector) generateTrendAnalysis() TrendAnalysis {
	return TrendAnalysis{
		CoherenceTrend: tc.calculateTrend(tc.CoherenceMetrics, func(m interface{}) (float64, time.Time) {
			metric := m.(CoherenceMetric)
			return metric.GlobalCoherence, metric.Timestamp
		}),
		PerformanceTrend: tc.calculateTrend(tc.PerformanceMetrics, func(m interface{}) (float64, time.Time) {
			metric := m.(PerformanceMetric)
			return metric.OperationsPerSec, metric.Timestamp
		}),
		NetworkTrend: tc.calculateTrend(tc.NetworkMetrics, func(m interface{}) (float64, time.Time) {
			metric := m.(NetworkMetric)
			return float64(metric.NetworkLatency.Nanoseconds()), metric.Timestamp
		}),
		EconomicTrend: tc.calculateTrend(tc.EconomicMetrics, func(m interface{}) (float64, time.Time) {
			metric := m.(EconomicMetric)
			return metric.TransactionRate, metric.Timestamp
		}),
	}
}

// calculateTrend calculates trend for a series of metrics
func (tc *TelemetryCollector) calculateTrend(metrics interface{}, extractor func(interface{}) (float64, time.Time)) TrendData {
	// This is a simplified implementation - in practice, you'd use proper statistical analysis
	return TrendData{
		Slope:          0.0,
		Direction:      "stable",
		Confidence:     0.5,
		PredictedValue: 0.0,
		TimeHorizon:    time.Hour,
	}
}

// generateAlertStatus generates current alert status
func (tc *TelemetryCollector) generateAlertStatus() AlertStatus {
	alerts := make([]Alert, 0)

	// Check coherence alerts
	for _, metric := range tc.CoherenceMetrics {
		if metric.GlobalCoherence < tc.CoherenceThreshold {
			alerts = append(alerts, Alert{
				ID:        fmt.Sprintf("coherence-%s-%d", metric.NodeID, time.Now().Unix()),
				Type:      "warning",
				Category:  "coherence",
				Message:   fmt.Sprintf("Low coherence on node %s: %.3f", metric.NodeID, metric.GlobalCoherence),
				NodeID:    metric.NodeID,
				Value:     metric.GlobalCoherence,
				Threshold: tc.CoherenceThreshold,
				Timestamp: time.Now(),
			})
		}
	}

	// Check phase drift alerts
	for _, metric := range tc.PhaseMetrics {
		if math.Abs(metric.PhaseDrift) > tc.PhaseDriftThreshold {
			alerts = append(alerts, Alert{
				ID:        fmt.Sprintf("phase-%s-%d", metric.NodeID, time.Now().Unix()),
				Type:      "warning",
				Category:  "phase",
				Message:   fmt.Sprintf("High phase drift on node %s: %.3f", metric.NodeID, metric.PhaseDrift),
				NodeID:    metric.NodeID,
				Value:     metric.PhaseDrift,
				Threshold: tc.PhaseDriftThreshold,
				Timestamp: time.Now(),
			})
		}
	}

	// Check network latency alerts
	for _, metric := range tc.NetworkMetrics {
		if metric.NetworkLatency > tc.LatencyThreshold {
			alerts = append(alerts, Alert{
				ID:        fmt.Sprintf("network-%s-%d", metric.NodeID, time.Now().Unix()),
				Type:      "warning",
				Category:  "network",
				Message:   fmt.Sprintf("High latency on node %s: %v", metric.NodeID, metric.NetworkLatency),
				NodeID:    metric.NodeID,
				Value:     float64(metric.NetworkLatency.Nanoseconds()),
				Threshold: float64(tc.LatencyThreshold.Nanoseconds()),
				Timestamp: time.Now(),
			})
		}
	}

	// Count alerts by type
	criticalCount := 0
	warningCount := 0
	infoCount := 0

	for _, alert := range alerts {
		switch alert.Type {
		case "critical":
			criticalCount++
		case "warning":
			warningCount++
		case "info":
			infoCount++
		}
	}

	return AlertStatus{
		ActiveAlerts:  alerts,
		CriticalCount: criticalCount,
		WarningCount:  warningCount,
		InfoCount:     infoCount,
	}
}

// generatePerformanceSummary generates performance analysis summary
func (tc *TelemetryCollector) generatePerformanceSummary() PerformanceSummary {
	// Sort nodes by performance score
	type nodeScore struct {
		nodeID string
		score  float64
	}

	scores := make([]nodeScore, 0, len(tc.MetricsByNode))
	for nodeID, metrics := range tc.MetricsByNode {
		scores = append(scores, nodeScore{nodeID, metrics.PerformanceScore})
	}

	sort.Slice(scores, func(i, j int) bool {
		return scores[i].score > scores[j].score
	})

	// Get top and bottom performers
	topPerformers := make([]string, 0, 3)
	underPerformers := make([]string, 0, 3)

	for i, score := range scores {
		if i < 3 {
			topPerformers = append(topPerformers, score.nodeID)
		}
		if i >= len(scores)-3 {
			underPerformers = append(underPerformers, score.nodeID)
		}
	}

	// Calculate system efficiency
	totalEfficiency := 0.0
	for _, metrics := range tc.MetricsByNode {
		totalEfficiency += metrics.PerformanceScore
	}
	systemEfficiency := totalEfficiency / float64(len(tc.MetricsByNode))

	return PerformanceSummary{
		TopPerformingNodes:   topPerformers,
		UnderperformingNodes: underPerformers,
		SystemEfficiency:     systemEfficiency,
		BottleneckAnalysis:   "Analysis based on collected metrics",
		OptimizationSuggestions: []string{
			"Consider redistributing load from underperforming nodes",
			"Monitor network latency for optimization opportunities",
			"Review coherence thresholds for optimal synchronization",
		},
	}
}

// GetNodeMetrics retrieves metrics for a specific node
func (tc *TelemetryCollector) GetNodeMetrics(nodeID string) (TelemetryNodeMetrics, bool) {
	tc.mu.RLock()
	defer tc.mu.RUnlock()

	metrics, exists := tc.MetricsByNode[nodeID]
	return metrics, exists
}

// Reset resets the telemetry collector
func (tc *TelemetryCollector) Reset() {
	tc.mu.Lock()
	defer tc.mu.Unlock()

	tc.CoherenceMetrics = make([]CoherenceMetric, 0)
	tc.ResonanceMetrics = make([]ResonanceMetric, 0)
	tc.PhaseMetrics = make([]PhaseMetric, 0)
	tc.PerformanceMetrics = make([]PerformanceMetric, 0)
	tc.NetworkMetrics = make([]NetworkMetric, 0)
	tc.EconomicMetrics = make([]EconomicMetric, 0)
	tc.TotalCollections = 0
	tc.LastCollectionTime = time.Now()
	tc.MetricsByNode = make(map[string]TelemetryNodeMetrics)
}
