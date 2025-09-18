package core

import (
	"fmt"
	"math"
	"math/rand"
	"sync"
	"time"
)

// ChaosEngineering manages controlled chaos experiments for testing system resilience
type ChaosEngineering struct {
	// Experiment configuration
	Experiments       map[string]*ChaosExperiment `json:"experiments"`
	ActiveExperiments []string                    `json:"active_experiments"`

	// System components to test
	NetworkManager NetworkManager      `json:"-"` // Injected dependency
	NodeManager    NodeManager         `json:"-"` // Injected dependency
	Telemetry      *TelemetryCollector `json:"-"` // Injected dependency

	// Experiment results
	Results map[string]*ExperimentResult `json:"results"`

	// Safety controls
	SafetyEnabled bool    `json:"safety_enabled"`
	MaxDisruption float64 `json:"max_disruption"` // Maximum allowed disruption (0-1)
	AutoRecovery  bool    `json:"auto_recovery"`

	// Monitoring
	HealthChecks    []HealthCheck              `json:"health_checks"`
	RecoveryMetrics map[string]*RecoveryMetric `json:"recovery_metrics"`

	// Thread safety
	mu sync.RWMutex `json:"-"`
}

// ChaosExperiment represents a single chaos experiment
type ChaosExperiment struct {
	ID          string           `json:"id"`
	Name        string           `json:"name"`
	Description string           `json:"description"`
	Type        ChaosType        `json:"type"`
	Status      ExperimentStatus `json:"status"`

	// Timing
	StartTime time.Time     `json:"start_time"`
	EndTime   time.Time     `json:"end_time"`
	Duration  time.Duration `json:"duration"`

	// Parameters
	Parameters ChaosParameters `json:"parameters"`

	// Targets
	TargetNodes []string      `json:"target_nodes"`
	TargetLinks []NetworkLink `json:"target_links"`

	// Monitoring
	Metrics  []ChaosMetric  `json:"metrics"`
	Triggers []ChaosTrigger `json:"triggers"`

	// Safety
	SafetyRules []SafetyRule `json:"safety_rules"`
}

// ChaosType defines the type of chaos experiment
type ChaosType string

const (
	NetworkPartition   ChaosType = "network_partition"
	NetworkLatency     ChaosType = "network_latency"
	NetworkPacketLoss  ChaosType = "network_packet_loss"
	NodeFailure        ChaosType = "node_failure"
	NodeRestart        ChaosType = "node_restart"
	ResourceExhaustion ChaosType = "resource_exhaustion"
	TimeSkew           ChaosType = "time_skew"
	PhaseDrift         ChaosType = "phase_drift"
)

// ExperimentStatus represents the status of a chaos experiment
type ExperimentStatus string

const (
	ExperimentPending   ExperimentStatus = "pending"
	ExperimentRunning   ExperimentStatus = "running"
	ExperimentCompleted ExperimentStatus = "completed"
	ExperimentFailed    ExperimentStatus = "failed"
	ExperimentAborted   ExperimentStatus = "aborted"
)

// ChaosParameters contains parameters for chaos experiments
type ChaosParameters struct {
	// Network chaos parameters
	PartitionDuration time.Duration `json:"partition_duration"`
	LatencyIncrease   time.Duration `json:"latency_increase"`
	PacketLossRate    float64       `json:"packet_loss_rate"`

	// Node chaos parameters
	NodeFailureDuration time.Duration `json:"node_failure_duration"`
	ResourceLimit       float64       `json:"resource_limit"`

	// Phase chaos parameters
	PhaseDriftAmount float64       `json:"phase_drift_amount"`
	TimeSkewAmount   time.Duration `json:"time_skew_amount"`

	// General parameters
	Intensity   float64 `json:"intensity"`   // 0-1 scale
	Probability float64 `json:"probability"` // Probability of applying chaos
}

// NetworkLink represents a network connection between nodes
type NetworkLink struct {
	SourceNode string        `json:"source_node"`
	TargetNode string        `json:"target_node"`
	Latency    time.Duration `json:"latency"`
	PacketLoss float64       `json:"packet_loss"`
	Bandwidth  float64       `json:"bandwidth"`
}

// ChaosMetric represents a metric collected during chaos experiments
type ChaosMetric struct {
	Name         string      `json:"name"`
	Value        interface{} `json:"value"`
	Timestamp    time.Time   `json:"timestamp"`
	NodeID       string      `json:"node_id,omitempty"`
	ExperimentID string      `json:"experiment_id"`
}

// ChaosTrigger represents a trigger for chaos events
type ChaosTrigger struct {
	Condition     string        `json:"condition"`
	Action        string        `json:"action"`
	Threshold     float64       `json:"threshold"`
	Cooldown      time.Duration `json:"cooldown"`
	LastTriggered time.Time     `json:"last_triggered"`
}

// SafetyRule represents a safety rule for chaos experiments
type SafetyRule struct {
	Name      string `json:"name"`
	Condition string `json:"condition"`
	Action    string `json:"action"`
	Severity  string `json:"severity"` // "warning", "critical", "fatal"
	Enabled   bool   `json:"enabled"`
}

// ExperimentResult contains the results of a chaos experiment
type ExperimentResult struct {
	ExperimentID string        `json:"experiment_id"`
	Success      bool          `json:"success"`
	StartTime    time.Time     `json:"start_time"`
	EndTime      time.Time     `json:"end_time"`
	Duration     time.Duration `json:"duration"`

	// Impact metrics
	SystemImpact SystemImpact          `json:"system_impact"`
	NodeImpacts  map[string]NodeImpact `json:"node_impacts"`

	// Recovery metrics
	RecoveryTime    time.Duration `json:"recovery_time"`
	RecoverySuccess bool          `json:"recovery_success"`

	// Observations
	Observations    []string `json:"observations"`
	Recommendations []string `json:"recommendations"`

	// Raw data
	Metrics []ChaosMetric `json:"metrics"`
}

// SystemImpact represents the overall system impact of a chaos experiment
type SystemImpact struct {
	CoherenceDrop    float64       `json:"coherence_drop"`
	PerformanceDrop  float64       `json:"performance_drop"`
	AvailabilityDrop float64       `json:"availability_drop"`
	RecoveryTime     time.Duration `json:"recovery_time"`
	CriticalEvents   int           `json:"critical_events"`
}

// NodeImpact represents the impact on a specific node
type NodeImpact struct {
	NodeID            string        `json:"node_id"`
	CoherenceImpact   float64       `json:"coherence_impact"`
	PerformanceImpact float64       `json:"performance_impact"`
	ConnectivityLoss  time.Duration `json:"connectivity_loss"`
	ErrorCount        int           `json:"error_count"`
	RecoveryTime      time.Duration `json:"recovery_time"`
}

// HealthCheck represents a system health check
type HealthCheck struct {
	Name         string        `json:"name"`
	Description  string        `json:"description"`
	Interval     time.Duration `json:"interval"`
	LastCheck    time.Time     `json:"last_check"`
	Status       string        `json:"status"` // "healthy", "degraded", "critical"
	ErrorMessage string        `json:"error_message,omitempty"`
}

// RecoveryMetric tracks system recovery after chaos events
type RecoveryMetric struct {
	EventID      string        `json:"event_id"`
	EventType    string        `json:"event_type"`
	StartTime    time.Time     `json:"start_time"`
	RecoveryTime time.Duration `json:"recovery_time"`
	Success      bool          `json:"success"`
	Metrics      []ChaosMetric `json:"metrics"`
}

// NetworkManager interface for network operations (simplified)
type NetworkManager interface {
	PartitionNetwork(sourceNode, targetNode string, duration time.Duration) error
	AddLatency(sourceNode, targetNode string, latency time.Duration) error
	AddPacketLoss(sourceNode, targetNode string, lossRate float64) error
	RestoreNetwork(sourceNode, targetNode string) error
	GetNetworkLinks() []NetworkLink
}

// NodeManager interface for node operations (simplified)
type NodeManager interface {
	StopNode(nodeID string, duration time.Duration) error
	RestartNode(nodeID string) error
	GetActiveNodes() []string
	ExhaustResources(nodeID string, resourceType string, limit float64) error
}

// NewChaosEngineering creates a new chaos engineering system
func NewChaosEngineering(networkManager NetworkManager, nodeManager NodeManager, telemetry *TelemetryCollector) *ChaosEngineering {
	ce := &ChaosEngineering{
		Experiments:       make(map[string]*ChaosExperiment),
		ActiveExperiments: make([]string, 0),
		Results:           make(map[string]*ExperimentResult),
		NetworkManager:    networkManager,
		NodeManager:       nodeManager,
		Telemetry:         telemetry,
		SafetyEnabled:     true,
		MaxDisruption:     0.8, // Allow up to 80% disruption
		AutoRecovery:      true,
		HealthChecks:      make([]HealthCheck, 0),
		RecoveryMetrics:   make(map[string]*RecoveryMetric),
	}

	// Initialize default health checks
	ce.initializeHealthChecks()

	return ce
}

// initializeHealthChecks sets up default health checks
func (ce *ChaosEngineering) initializeHealthChecks() {
	ce.HealthChecks = []HealthCheck{
		{
			Name:        "system_coherence",
			Description: "Monitor overall system coherence",
			Interval:    10 * time.Second,
			Status:      "unknown",
		},
		{
			Name:        "node_connectivity",
			Description: "Monitor node connectivity",
			Interval:    5 * time.Second,
			Status:      "unknown",
		},
		{
			Name:        "performance_stability",
			Description: "Monitor performance stability",
			Interval:    15 * time.Second,
			Status:      "unknown",
		},
		{
			Name:        "consensus_health",
			Description: "Monitor consensus mechanism health",
			Interval:    20 * time.Second,
			Status:      "unknown",
		},
	}
}

// CreateExperiment creates a new chaos experiment
func (ce *ChaosEngineering) CreateExperiment(name, description string, expType ChaosType, duration time.Duration, parameters ChaosParameters) (*ChaosExperiment, error) {
	ce.mu.Lock()
	defer ce.mu.Unlock()

	experimentID := fmt.Sprintf("chaos-%s-%d", expType, time.Now().Unix())

	experiment := &ChaosExperiment{
		ID:          experimentID,
		Name:        name,
		Description: description,
		Type:        expType,
		Status:      ExperimentPending,
		Duration:    duration,
		Parameters:  parameters,
		Metrics:     make([]ChaosMetric, 0),
		Triggers:    make([]ChaosTrigger, 0),
		SafetyRules: ce.getDefaultSafetyRules(),
	}

	ce.Experiments[experimentID] = experiment

	return experiment, nil
}

// getDefaultSafetyRules returns default safety rules for experiments
func (ce *ChaosEngineering) getDefaultSafetyRules() []SafetyRule {
	return []SafetyRule{
		{
			Name:      "max_coherence_drop",
			Condition: "system_coherence < 0.5",
			Action:    "abort_experiment",
			Severity:  "critical",
			Enabled:   true,
		},
		{
			Name:      "node_isolation",
			Condition: "isolated_nodes > 2",
			Action:    "trigger_recovery",
			Severity:  "warning",
			Enabled:   true,
		},
		{
			Name:      "performance_critical",
			Condition: "system_performance < 0.3",
			Action:    "abort_experiment",
			Severity:  "critical",
			Enabled:   true,
		},
	}
}

// StartExperiment starts a chaos experiment
func (ce *ChaosEngineering) StartExperiment(experimentID string) error {
	ce.mu.Lock()
	experiment, exists := ce.Experiments[experimentID]
	ce.mu.Unlock()

	if !exists {
		return fmt.Errorf("experiment %s not found", experimentID)
	}

	// Check safety rules before starting
	if err := ce.checkSafetyRules(experiment); err != nil {
		return fmt.Errorf("safety check failed: %w", err)
	}

	experiment.Status = ExperimentRunning
	experiment.StartTime = time.Now()
	experiment.EndTime = experiment.StartTime.Add(experiment.Duration)

	ce.mu.Lock()
	ce.ActiveExperiments = append(ce.ActiveExperiments, experimentID)
	ce.mu.Unlock()

	// Start the experiment in a goroutine
	go ce.runExperiment(experiment)

	return nil
}

// runExperiment executes the chaos experiment
func (ce *ChaosEngineering) runExperiment(experiment *ChaosExperiment) {
	defer ce.completeExperiment(experiment)

	// Apply chaos based on experiment type
	switch experiment.Type {
	case NetworkPartition:
		ce.applyNetworkPartition(experiment)
	case NetworkLatency:
		ce.applyNetworkLatency(experiment)
	case NetworkPacketLoss:
		ce.applyNetworkPacketLoss(experiment)
	case NodeFailure:
		ce.applyNodeFailure(experiment)
	case NodeRestart:
		ce.applyNodeRestart(experiment)
	case ResourceExhaustion:
		ce.applyResourceExhaustion(experiment)
	case TimeSkew:
		ce.applyTimeSkew(experiment)
	case PhaseDrift:
		ce.applyPhaseDrift(experiment)
	}

	// Monitor experiment during execution
	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			if time.Now().After(experiment.EndTime) {
				return
			}

			// Check safety rules during execution
			if err := ce.checkSafetyRules(experiment); err != nil {
				fmt.Printf("Safety rule violation in experiment %s: %v\n", experiment.ID, err)
				if ce.AutoRecovery {
					ce.triggerRecovery(experiment)
				}
				return
			}

			// Collect metrics
			ce.collectExperimentMetrics(experiment)

		}
	}
}

// applyNetworkPartition applies network partition chaos
func (ce *ChaosEngineering) applyNetworkPartition(experiment *ChaosExperiment) {
	if ce.NetworkManager == nil {
		return
	}

	// Select random nodes to partition
	activeNodes := ce.NodeManager.GetActiveNodes()
	if len(activeNodes) < 2 {
		return
	}

	// Create partitions between random pairs of nodes
	partitionCount := int(float64(len(activeNodes)) * experiment.Parameters.Intensity)
	if partitionCount < 1 {
		partitionCount = 1
	}

	for i := 0; i < partitionCount; i++ {
		sourceIdx := rand.Intn(len(activeNodes))
		targetIdx := rand.Intn(len(activeNodes))
		if sourceIdx == targetIdx {
			targetIdx = (targetIdx + 1) % len(activeNodes)
		}

		sourceNode := activeNodes[sourceIdx]
		targetNode := activeNodes[targetIdx]

		if rand.Float64() < experiment.Parameters.Probability {
			ce.NetworkManager.PartitionNetwork(sourceNode, targetNode, experiment.Parameters.PartitionDuration)

			link := NetworkLink{
				SourceNode: sourceNode,
				TargetNode: targetNode,
			}
			experiment.TargetLinks = append(experiment.TargetLinks, link)
		}
	}
}

// applyNetworkLatency applies network latency chaos
func (ce *ChaosEngineering) applyNetworkLatency(experiment *ChaosExperiment) {
	if ce.NetworkManager == nil {
		return
	}

	activeNodes := ce.NodeManager.GetActiveNodes()
	links := ce.NetworkManager.GetNetworkLinks()

	for _, link := range links {
		if rand.Float64() < experiment.Parameters.Probability {
			// Check if nodes are still active
			sourceActive := false
			targetActive := false
			for _, node := range activeNodes {
				if node == link.SourceNode {
					sourceActive = true
				}
				if node == link.TargetNode {
					targetActive = true
				}
			}

			if sourceActive && targetActive {
				latencyIncrease := time.Duration(float64(experiment.Parameters.LatencyIncrease) * experiment.Parameters.Intensity)
				ce.NetworkManager.AddLatency(link.SourceNode, link.TargetNode, latencyIncrease)
				experiment.TargetLinks = append(experiment.TargetLinks, link)
			}
		}
	}
}

// applyNetworkPacketLoss applies network packet loss chaos
func (ce *ChaosEngineering) applyNetworkPacketLoss(experiment *ChaosExperiment) {
	if ce.NetworkManager == nil {
		return
	}

	activeNodes := ce.NodeManager.GetActiveNodes()
	links := ce.NetworkManager.GetNetworkLinks()

	for _, link := range links {
		if rand.Float64() < experiment.Parameters.Probability {
			// Check if nodes are still active
			sourceActive := false
			targetActive := false
			for _, node := range activeNodes {
				if node == link.SourceNode {
					sourceActive = true
				}
				if node == link.TargetNode {
					targetActive = true
				}
			}

			if sourceActive && targetActive {
				lossRate := experiment.Parameters.PacketLossRate * experiment.Parameters.Intensity
				ce.NetworkManager.AddPacketLoss(link.SourceNode, link.TargetNode, lossRate)
				experiment.TargetLinks = append(experiment.TargetLinks, link)
			}
		}
	}
}

// applyNodeFailure applies node failure chaos
func (ce *ChaosEngineering) applyNodeFailure(experiment *ChaosExperiment) {
	if ce.NodeManager == nil {
		return
	}

	activeNodes := ce.NodeManager.GetActiveNodes()
	failureCount := int(float64(len(activeNodes)) * experiment.Parameters.Intensity)
	if failureCount < 1 {
		failureCount = 1
	}
	if failureCount > len(activeNodes)-1 { // Keep at least one node running
		failureCount = len(activeNodes) - 1
	}

	// Randomly select nodes to fail
	selectedNodes := make([]string, 0, failureCount)
	availableNodes := make([]string, len(activeNodes))
	copy(availableNodes, activeNodes)

	for i := 0; i < failureCount && len(availableNodes) > 1; i++ {
		idx := rand.Intn(len(availableNodes))
		selectedNode := availableNodes[idx]

		if rand.Float64() < experiment.Parameters.Probability {
			ce.NodeManager.StopNode(selectedNode, experiment.Parameters.NodeFailureDuration)
			selectedNodes = append(selectedNodes, selectedNode)
			experiment.TargetNodes = append(experiment.TargetNodes, selectedNode)
		}

		// Remove from available nodes
		availableNodes = append(availableNodes[:idx], availableNodes[idx+1:]...)
	}
}

// applyNodeRestart applies node restart chaos
func (ce *ChaosEngineering) applyNodeRestart(experiment *ChaosExperiment) {
	if ce.NodeManager == nil {
		return
	}

	activeNodes := ce.NodeManager.GetActiveNodes()
	restartCount := int(float64(len(activeNodes)) * experiment.Parameters.Intensity)
	if restartCount < 1 {
		restartCount = 1
	}

	for i := 0; i < restartCount && i < len(activeNodes); i++ {
		if rand.Float64() < experiment.Parameters.Probability {
			nodeID := activeNodes[rand.Intn(len(activeNodes))]
			ce.NodeManager.RestartNode(nodeID)
			experiment.TargetNodes = append(experiment.TargetNodes, nodeID)
		}
	}
}

// applyResourceExhaustion applies resource exhaustion chaos
func (ce *ChaosEngineering) applyResourceExhaustion(experiment *ChaosExperiment) {
	if ce.NodeManager == nil {
		return
	}

	activeNodes := ce.NodeManager.GetActiveNodes()
	exhaustCount := int(float64(len(activeNodes)) * experiment.Parameters.Intensity)
	if exhaustCount < 1 {
		exhaustCount = 1
	}

	resourceTypes := []string{"cpu", "memory", "disk", "network"}

	for i := 0; i < exhaustCount && i < len(activeNodes); i++ {
		if rand.Float64() < experiment.Parameters.Probability {
			nodeID := activeNodes[rand.Intn(len(activeNodes))]
			resourceType := resourceTypes[rand.Intn(len(resourceTypes))]
			limit := experiment.Parameters.ResourceLimit * experiment.Parameters.Intensity

			ce.NodeManager.ExhaustResources(nodeID, resourceType, limit)
			experiment.TargetNodes = append(experiment.TargetNodes, nodeID)
		}
	}
}

// applyTimeSkew applies time skew chaos
func (ce *ChaosEngineering) applyTimeSkew(experiment *ChaosExperiment) {
	// This would require integration with the node's time synchronization system
	// For now, we'll just record the intent
	fmt.Printf("Applying time skew chaos: %v to %d nodes\n",
		experiment.Parameters.TimeSkewAmount, len(experiment.TargetNodes))
}

// applyPhaseDrift applies phase drift chaos
func (ce *ChaosEngineering) applyPhaseDrift(experiment *ChaosExperiment) {
	// This would require integration with the phase synchronization system
	// For now, we'll just record the intent
	fmt.Printf("Applying phase drift chaos: %.3f radians to %d nodes\n",
		experiment.Parameters.PhaseDriftAmount, len(experiment.TargetNodes))
}

// collectExperimentMetrics collects metrics during experiment execution
func (ce *ChaosEngineering) collectExperimentMetrics(experiment *ChaosExperiment) {
	// Collect system-wide metrics
	if ce.Telemetry != nil {
		report := ce.Telemetry.GenerateTelemetryReport()

		metric := ChaosMetric{
			Name:         "system_coherence",
			Value:        report.GlobalMetrics.AverageCoherence,
			Timestamp:    time.Now(),
			ExperimentID: experiment.ID,
		}
		experiment.Metrics = append(experiment.Metrics, metric)

		metric = ChaosMetric{
			Name:         "system_performance",
			Value:        report.GlobalMetrics.SystemPerformance,
			Timestamp:    time.Now(),
			ExperimentID: experiment.ID,
		}
		experiment.Metrics = append(experiment.Metrics, metric)
	}
}

// checkSafetyRules checks if any safety rules are violated
func (ce *ChaosEngineering) checkSafetyRules(experiment *ChaosExperiment) error {
	if !ce.SafetyEnabled {
		return nil
	}

	for _, rule := range experiment.SafetyRules {
		if !rule.Enabled {
			continue
		}

		violated := ce.evaluateSafetyRule(rule, experiment)
		if violated {
			return fmt.Errorf("safety rule '%s' violated: %s", rule.Name, rule.Condition)
		}
	}

	return nil
}

// evaluateSafetyRule evaluates a safety rule condition
func (ce *ChaosEngineering) evaluateSafetyRule(rule SafetyRule, experiment *ChaosExperiment) bool {
	// This is a simplified implementation - in practice, you'd have a more sophisticated
	// rule evaluation engine
	switch rule.Condition {
	case "system_coherence < 0.5":
		if ce.Telemetry != nil {
			report := ce.Telemetry.GenerateTelemetryReport()
			return report.GlobalMetrics.AverageCoherence < 0.5
		}
	case "system_performance < 0.3":
		if ce.Telemetry != nil {
			report := ce.Telemetry.GenerateTelemetryReport()
			return report.GlobalMetrics.SystemPerformance < 300000 // Simplified threshold
		}
	case "isolated_nodes > 2":
		// Simplified check - in practice, you'd check actual network topology
		return len(experiment.TargetNodes) > 2
	}

	return false
}

// triggerRecovery triggers recovery procedures
func (ce *ChaosEngineering) triggerRecovery(experiment *ChaosExperiment) {
	fmt.Printf("Triggering recovery for experiment %s\n", experiment.ID)

	// Restore network partitions
	for _, link := range experiment.TargetLinks {
		if ce.NetworkManager != nil {
			ce.NetworkManager.RestoreNetwork(link.SourceNode, link.TargetNode)
		}
	}

	// Restart failed nodes
	for _, nodeID := range experiment.TargetNodes {
		if ce.NodeManager != nil {
			ce.NodeManager.RestartNode(nodeID)
		}
	}

	experiment.Status = ExperimentAborted
}

// completeExperiment completes a chaos experiment
func (ce *ChaosEngineering) completeExperiment(experiment *ChaosExperiment) {
	experiment.Status = ExperimentCompleted
	experiment.EndTime = time.Now()

	// Remove from active experiments
	ce.mu.Lock()
	for i, id := range ce.ActiveExperiments {
		if id == experiment.ID {
			ce.ActiveExperiments = append(ce.ActiveExperiments[:i], ce.ActiveExperiments[i+1:]...)
			break
		}
	}
	ce.mu.Unlock()

	// Generate experiment result
	result := ce.generateExperimentResult(experiment)
	ce.mu.Lock()
	ce.Results[experiment.ID] = result
	ce.mu.Unlock()

	fmt.Printf("Chaos experiment %s completed: %v\n", experiment.ID, result.Success)
}

// generateExperimentResult generates a result for a completed experiment
func (ce *ChaosEngineering) generateExperimentResult(experiment *ChaosExperiment) *ExperimentResult {
	result := &ExperimentResult{
		ExperimentID:    experiment.ID,
		Success:         experiment.Status == ExperimentCompleted,
		StartTime:       experiment.StartTime,
		EndTime:         experiment.EndTime,
		Duration:        experiment.EndTime.Sub(experiment.StartTime),
		NodeImpacts:     make(map[string]NodeImpact),
		Observations:    make([]string, 0),
		Recommendations: make([]string, 0),
		Metrics:         experiment.Metrics,
	}

	// Calculate system impact
	result.SystemImpact = ce.calculateSystemImpact(experiment)

	// Calculate recovery metrics
	result.RecoveryTime = ce.calculateRecoveryTime(experiment)
	result.RecoverySuccess = result.RecoveryTime < 5*time.Minute // Simplified threshold

	// Generate observations and recommendations
	result.Observations = ce.generateObservations(experiment, result)
	result.Recommendations = ce.generateRecommendations(experiment, result)

	return result
}

// calculateSystemImpact calculates the overall system impact
func (ce *ChaosEngineering) calculateSystemImpact(experiment *ChaosExperiment) SystemImpact {
	impact := SystemImpact{}

	if len(experiment.Metrics) == 0 {
		return impact
	}

	// Calculate coherence drop
	initialCoherence := 0.0
	finalCoherence := 0.0
	coherenceCount := 0

	for _, metric := range experiment.Metrics {
		if metric.Name == "system_coherence" {
			if coherenceCount == 0 {
				initialCoherence = metric.Value.(float64)
			}
			finalCoherence = metric.Value.(float64)
			coherenceCount++
		}
	}

	if coherenceCount > 0 {
		impact.CoherenceDrop = math.Max(0, initialCoherence-finalCoherence)
	}

	// Simplified calculations for other metrics
	impact.PerformanceDrop = 0.1          // Placeholder
	impact.AvailabilityDrop = 0.05        // Placeholder
	impact.RecoveryTime = 2 * time.Minute // Placeholder
	impact.CriticalEvents = len(experiment.TargetNodes)

	return impact
}

// calculateRecoveryTime calculates recovery time (simplified)
func (ce *ChaosEngineering) calculateRecoveryTime(experiment *ChaosExperiment) time.Duration {
	// Simplified - in practice, you'd track actual recovery time
	return time.Duration(len(experiment.TargetNodes)) * 30 * time.Second
}

// generateObservations generates observations from experiment results
func (ce *ChaosEngineering) generateObservations(experiment *ChaosExperiment, result *ExperimentResult) []string {
	observations := make([]string, 0)

	observations = append(observations, fmt.Sprintf("Experiment %s completed in %v", experiment.ID, result.Duration))
	observations = append(observations, fmt.Sprintf("Affected %d nodes and %d network links", len(experiment.TargetNodes), len(experiment.TargetLinks)))
	observations = append(observations, fmt.Sprintf("System coherence dropped by %.3f", result.SystemImpact.CoherenceDrop))

	if result.RecoverySuccess {
		observations = append(observations, fmt.Sprintf("System recovered successfully in %v", result.RecoveryTime))
	} else {
		observations = append(observations, "System recovery was slower than expected")
	}

	return observations
}

// generateRecommendations generates recommendations based on experiment results
func (ce *ChaosEngineering) generateRecommendations(experiment *ChaosExperiment, result *ExperimentResult) []string {
	recommendations := make([]string, 0)

	if result.SystemImpact.CoherenceDrop > 0.3 {
		recommendations = append(recommendations, "Consider improving phase synchronization algorithms")
	}

	if result.RecoveryTime > 3*time.Minute {
		recommendations = append(recommendations, "Implement faster recovery mechanisms")
	}

	if len(experiment.TargetNodes) > 0 {
		recommendations = append(recommendations, "Add more redundancy to prevent single points of failure")
	}

	return recommendations
}

// GetExperimentStatus returns the status of a chaos experiment
func (ce *ChaosEngineering) GetExperimentStatus(experimentID string) (ExperimentStatus, error) {
	ce.mu.RLock()
	defer ce.mu.RUnlock()

	experiment, exists := ce.Experiments[experimentID]
	if !exists {
		return "", fmt.Errorf("experiment %s not found", experimentID)
	}

	return experiment.Status, nil
}

// GetExperimentResult returns the result of a completed experiment
func (ce *ChaosEngineering) GetExperimentResult(experimentID string) (*ExperimentResult, error) {
	ce.mu.RLock()
	defer ce.mu.RUnlock()

	result, exists := ce.Results[experimentID]
	if !exists {
		return nil, fmt.Errorf("result for experiment %s not found", experimentID)
	}

	return result, nil
}

// ListExperiments returns a list of all experiments
func (ce *ChaosEngineering) ListExperiments() []*ChaosExperiment {
	ce.mu.RLock()
	defer ce.mu.RUnlock()

	experiments := make([]*ChaosExperiment, 0, len(ce.Experiments))
	for _, experiment := range ce.Experiments {
		experiments = append(experiments, experiment)
	}

	return experiments
}

// StopExperiment stops a running chaos experiment
func (ce *ChaosEngineering) StopExperiment(experimentID string) error {
	ce.mu.Lock()
	defer ce.mu.Unlock()

	experiment, exists := ce.Experiments[experimentID]
	if !exists {
		return fmt.Errorf("experiment %s not found", experimentID)
	}

	if experiment.Status != ExperimentRunning {
		return fmt.Errorf("experiment %s is not running", experimentID)
	}

	experiment.Status = ExperimentAborted
	experiment.EndTime = time.Now()

	// Trigger recovery
	ce.triggerRecovery(experiment)

	return nil
}

// RunHealthChecks runs all configured health checks
func (ce *ChaosEngineering) RunHealthChecks() []HealthCheck {
	ce.mu.Lock()
	defer ce.mu.Unlock()

	for i := range ce.HealthChecks {
		check := &ce.HealthChecks[i]

		// Skip if not enough time has passed
		if time.Since(check.LastCheck) < check.Interval {
			continue
		}

		// Run the health check
		check.LastCheck = time.Now()
		check.Status, check.ErrorMessage = ce.runHealthCheck(check)
	}

	return ce.HealthChecks
}

// runHealthCheck runs a single health check
func (ce *ChaosEngineering) runHealthCheck(check *HealthCheck) (string, string) {
	switch check.Name {
	case "system_coherence":
		if ce.Telemetry != nil {
			report := ce.Telemetry.GenerateTelemetryReport()
			coherence := report.GlobalMetrics.AverageCoherence
			if coherence > 0.8 {
				return "healthy", ""
			} else if coherence > 0.6 {
				return "degraded", fmt.Sprintf("Low coherence: %.3f", coherence)
			} else {
				return "critical", fmt.Sprintf("Critical coherence: %.3f", coherence)
			}
		}
	case "node_connectivity":
		if ce.NodeManager != nil {
			activeNodes := ce.NodeManager.GetActiveNodes()
			if len(activeNodes) > 2 {
				return "healthy", ""
			} else {
				return "critical", fmt.Sprintf("Only %d nodes active", len(activeNodes))
			}
		}
	case "performance_stability":
		if ce.Telemetry != nil {
			report := ce.Telemetry.GenerateTelemetryReport()
			performance := report.GlobalMetrics.SystemPerformance
			if performance > 500000 {
				return "healthy", ""
			} else if performance > 200000 {
				return "degraded", fmt.Sprintf("Low performance: %.0f", performance)
			} else {
				return "critical", fmt.Sprintf("Critical performance: %.0f", performance)
			}
		}
	case "consensus_health":
		// Simplified consensus health check
		return "healthy", ""
	}

	return "unknown", "Health check not implemented"
}

// GetSystemResilienceScore calculates an overall system resilience score
func (ce *ChaosEngineering) GetSystemResilienceScore() float64 {
	ce.mu.RLock()
	defer ce.mu.RUnlock()

	if len(ce.Results) == 0 {
		return 1.0 // Perfect score if no experiments run
	}

	totalScore := 0.0
	experimentCount := 0

	for _, result := range ce.Results {
		if result.Success {
			score := 1.0

			// Reduce score based on impact
			score -= result.SystemImpact.CoherenceDrop * 0.3
			score -= result.SystemImpact.PerformanceDrop * 0.3
			score -= result.SystemImpact.AvailabilityDrop * 0.2

			// Reduce score if recovery was slow
			if result.RecoveryTime > 2*time.Minute {
				score -= 0.2
			}

			// Ensure score stays within bounds
			if score < 0 {
				score = 0
			}

			totalScore += score
			experimentCount++
		}
	}

	if experimentCount == 0 {
		return 1.0
	}

	return totalScore / float64(experimentCount)
}
