package core

import (
	"fmt"
	"math"
	"sync"
	"time"
)

// QuaternionicSynchronizer manages synchronization of quaternionic states across distributed nodes
type QuaternionicSynchronizer struct {
	// Node states indexed by node ID
	NodeStates map[string]*NodeQuaternionicState `json:"node_states"`

	// Synchronization parameters
	CouplingStrength     float64       `json:"coupling_strength"`     // K in Kuramoto model
	NaturalFrequency     float64       `json:"natural_frequency"`     // ω₀ base frequency
	SyncInterval         time.Duration `json:"sync_interval"`         // Synchronization interval
	MaxPhaseDrift        float64       `json:"max_phase_drift"`       // Maximum allowed phase drift
	ConvergenceThreshold float64       `json:"convergence_threshold"` // Convergence threshold

	// Synchronization statistics
	TotalSyncOperations int64     `json:"total_sync_operations"`
	SuccessfulSyncs     int64     `json:"successful_syncs"`
	FailedSyncs         int64     `json:"failed_syncs"`
	LastSyncTime        time.Time `json:"last_sync_time"`
	GlobalCoherence     float64   `json:"global_coherence"`

	// Thread safety
	mu sync.RWMutex `json:"-"`
}

// NodeQuaternionicState represents the quaternionic state of a single node
type NodeQuaternionicState struct {
	NodeID            string                   `json:"node_id"`
	QuaternionicState *QuaternionicState       `json:"quaternionic_state"`
	PrimeOscillators  map[int]*PrimeOscillator `json:"prime_oscillators"`
	LastUpdate        time.Time                `json:"last_update"`
	SyncPhase         float64                  `json:"sync_phase"`
	Coherence         float64                  `json:"coherence"`
	Connectivity      map[string]float64       `json:"connectivity"` // Connection strength to other nodes
}

// SyncResult represents the result of a synchronization operation
type SyncResult struct {
	NodeID          string    `json:"node_id"`
	Success         bool      `json:"success"`
	PhaseAdjustment float64   `json:"phase_adjustment"`
	CoherenceChange float64   `json:"coherence_change"`
	GlobalCoherence float64   `json:"global_coherence"`
	Timestamp       time.Time `json:"timestamp"`
	Error           string    `json:"error,omitempty"`
}

// SynchronizationMetrics contains comprehensive synchronization metrics
type SynchronizationMetrics struct {
	GlobalCoherence     float64                `json:"global_coherence"`
	PhaseVariance       float64                `json:"phase_variance"`
	AverageConnectivity float64                `json:"average_connectivity"`
	NodeCount           int                    `json:"node_count"`
	SyncRate            float64                `json:"sync_rate"`
	ConvergenceRate     float64                `json:"convergence_rate"`
	NodeMetrics         map[string]NodeMetrics `json:"node_metrics"`
	Timestamp           time.Time              `json:"timestamp"`
}

// NodeMetrics contains metrics for a single node
type NodeMetrics struct {
	LocalCoherence    float64 `json:"local_coherence"`
	PhaseDrift        float64 `json:"phase_drift"`
	ConnectivityCount int     `json:"connectivity_count"`
	SyncSuccessRate   float64 `json:"sync_success_rate"`
}

// NewQuaternionicSynchronizer creates a new quaternionic synchronizer
func NewQuaternionicSynchronizer(couplingStrength, naturalFrequency, maxPhaseDrift, convergenceThreshold float64, syncInterval time.Duration) *QuaternionicSynchronizer {
	return &QuaternionicSynchronizer{
		NodeStates:           make(map[string]*NodeQuaternionicState),
		CouplingStrength:     couplingStrength,
		NaturalFrequency:     naturalFrequency,
		SyncInterval:         syncInterval,
		MaxPhaseDrift:        maxPhaseDrift,
		ConvergenceThreshold: convergenceThreshold,
		TotalSyncOperations:  0,
		SuccessfulSyncs:      0,
		FailedSyncs:          0,
		LastSyncTime:         time.Now(),
		GlobalCoherence:      0.0,
	}
}

// RegisterNode registers a node for synchronization
func (qs *QuaternionicSynchronizer) RegisterNode(nodeID string, initialState *QuaternionicState) error {
	qs.mu.Lock()
	defer qs.mu.Unlock()

	if initialState == nil {
		return fmt.Errorf("initial state cannot be nil")
	}

	// Validate the initial state
	if err := initialState.Validate(); err != nil {
		return fmt.Errorf("invalid initial state: %w", err)
	}

	nodeState := &NodeQuaternionicState{
		NodeID:            nodeID,
		QuaternionicState: initialState.Clone(),
		PrimeOscillators:  make(map[int]*PrimeOscillator),
		LastUpdate:        time.Now(),
		SyncPhase:         initialState.Phase,
		Coherence:         initialState.Coherence,
		Connectivity:      make(map[string]float64),
	}

	qs.NodeStates[nodeID] = nodeState

	// Initialize connectivity to existing nodes
	qs.initializeNodeConnectivity(nodeID)

	return nil
}

// initializeNodeConnectivity sets up initial connectivity between nodes
func (qs *QuaternionicSynchronizer) initializeNodeConnectivity(newNodeID string) {
	newNode := qs.NodeStates[newNodeID]

	// Connect to all existing nodes with uniform strength
	connectionStrength := 1.0 / float64(len(qs.NodeStates))

	for nodeID := range qs.NodeStates {
		if nodeID != newNodeID {
			newNode.Connectivity[nodeID] = connectionStrength
			qs.NodeStates[nodeID].Connectivity[newNodeID] = connectionStrength
		}
	}
}

// Synchronize performs a full synchronization cycle across all nodes
func (qs *QuaternionicSynchronizer) Synchronize() *SynchronizationMetrics {
	qs.mu.Lock()
	defer qs.mu.Unlock()

	qs.TotalSyncOperations++
	qs.LastSyncTime = time.Now()

	if len(qs.NodeStates) < 2 {
		return qs.createEmptyMetrics()
	}

	// Perform synchronization using Kuramoto model
	syncResults := qs.performKuramotoSynchronization()

	// Update global coherence
	qs.GlobalCoherence = qs.calculateGlobalCoherence()

	// Update statistics
	for _, result := range syncResults {
		if result.Success {
			qs.SuccessfulSyncs++
		} else {
			qs.FailedSyncs++
		}
	}

	return qs.createSynchronizationMetrics(syncResults)
}

// performKuramotoSynchronization implements the Kuramoto synchronization algorithm
func (qs *QuaternionicSynchronizer) performKuramotoSynchronization() map[string]*SyncResult {
	results := make(map[string]*SyncResult)

	// Get all node IDs
	nodeIDs := make([]string, 0, len(qs.NodeStates))
	for nodeID := range qs.NodeStates {
		nodeIDs = append(nodeIDs, nodeID)
	}

	// Calculate phase adjustments for each node
	for _, nodeID := range nodeIDs {
		result := qs.synchronizeNode(nodeID, nodeIDs)
		results[nodeID] = result

		// Apply the phase adjustment if successful
		if result.Success {
			nodeState := qs.NodeStates[nodeID]
			nodeState.SyncPhase += result.PhaseAdjustment
			nodeState.QuaternionicState.Phase = nodeState.SyncPhase
			nodeState.LastUpdate = time.Now()
		}
	}

	return results
}

// synchronizeNode calculates the synchronization adjustment for a single node
func (qs *QuaternionicSynchronizer) synchronizeNode(nodeID string, allNodeIDs []string) *SyncResult {
	nodeState := qs.NodeStates[nodeID]
	phaseAdjustment := 0.0
	totalWeight := 0.0

	// Calculate coupling from all other nodes
	for _, otherNodeID := range allNodeIDs {
		if otherNodeID == nodeID {
			continue
		}

		otherNodeState := qs.NodeStates[otherNodeID]

		// Get connection strength
		connectionStrength := nodeState.Connectivity[otherNodeID]
		if connectionStrength == 0 {
			continue
		}

		// Calculate phase difference
		phaseDiff := otherNodeState.SyncPhase - nodeState.SyncPhase

		// Handle phase wrapping
		for phaseDiff > math.Pi {
			phaseDiff -= 2 * math.Pi
		}
		for phaseDiff < -math.Pi {
			phaseDiff += 2 * math.Pi
		}

		// Kuramoto coupling: K * sin(Δφ)
		coupling := qs.CouplingStrength * connectionStrength * math.Sin(phaseDiff)
		phaseAdjustment += coupling
		totalWeight += connectionStrength
	}

	// Normalize by total connection weight
	if totalWeight > 0 {
		phaseAdjustment /= totalWeight
	}

	// Check if adjustment is within acceptable bounds
	if math.Abs(phaseAdjustment) > qs.MaxPhaseDrift {
		return &SyncResult{
			NodeID:          nodeID,
			Success:         false,
			PhaseAdjustment: phaseAdjustment,
			Timestamp:       time.Now(),
			Error:           fmt.Sprintf("phase adjustment %.6f exceeds maximum drift %.6f", phaseAdjustment, qs.MaxPhaseDrift),
		}
	}

	// Calculate coherence change
	oldCoherence := nodeState.Coherence
	newCoherence := qs.calculateNodeCoherence(nodeID)
	coherenceChange := newCoherence - oldCoherence
	nodeState.Coherence = newCoherence

	return &SyncResult{
		NodeID:          nodeID,
		Success:         true,
		PhaseAdjustment: phaseAdjustment,
		CoherenceChange: coherenceChange,
		GlobalCoherence: qs.GlobalCoherence,
		Timestamp:       time.Now(),
	}
}

// calculateNodeCoherence calculates the coherence of a single node
func (qs *QuaternionicSynchronizer) calculateNodeCoherence(nodeID string) float64 {
	nodeState := qs.NodeStates[nodeID]
	totalCoherence := 0.0
	totalWeight := 0.0

	// Calculate coherence with all connected nodes
	for otherNodeID, weight := range nodeState.Connectivity {
		if weight == 0 {
			continue
		}

		otherNodeState := qs.NodeStates[otherNodeID]

		// Calculate phase difference
		phaseDiff := otherNodeState.SyncPhase - nodeState.SyncPhase

		// Handle phase wrapping
		for phaseDiff > math.Pi {
			phaseDiff -= 2 * math.Pi
		}
		for phaseDiff < -math.Pi {
			phaseDiff += 2 * math.Pi
		}

		// Coherence contribution
		coherence := math.Cos(phaseDiff)
		totalCoherence += weight * coherence
		totalWeight += weight
	}

	if totalWeight == 0 {
		return 1.0 // Perfect coherence with self
	}

	return totalCoherence / totalWeight
}

// calculateGlobalCoherence calculates the global coherence across all nodes
func (qs *QuaternionicSynchronizer) calculateGlobalCoherence() float64 {
	if len(qs.NodeStates) < 2 {
		return 1.0
	}

	totalCoherence := 0.0
	pairCount := 0

	// Calculate pairwise coherence
	nodeIDs := make([]string, 0, len(qs.NodeStates))
	for nodeID := range qs.NodeStates {
		nodeIDs = append(nodeIDs, nodeID)
	}

	for i := 0; i < len(nodeIDs); i++ {
		for j := i + 1; j < len(nodeIDs); j++ {
			nodeI := qs.NodeStates[nodeIDs[i]]
			nodeJ := qs.NodeStates[nodeIDs[j]]

			// Calculate phase difference
			phaseDiff := nodeI.SyncPhase - nodeJ.SyncPhase

			// Handle phase wrapping
			for phaseDiff > math.Pi {
				phaseDiff -= 2 * math.Pi
			}
			for phaseDiff < -math.Pi {
				phaseDiff += 2 * math.Pi
			}

			// Coherence contribution
			coherence := math.Cos(phaseDiff)
			totalCoherence += coherence
			pairCount++
		}
	}

	if pairCount == 0 {
		return 1.0
	}

	return totalCoherence / float64(pairCount)
}

// createSynchronizationMetrics creates comprehensive synchronization metrics
func (qs *QuaternionicSynchronizer) createSynchronizationMetrics(syncResults map[string]*SyncResult) *SynchronizationMetrics {
	metrics := &SynchronizationMetrics{
		GlobalCoherence: qs.GlobalCoherence,
		NodeCount:       len(qs.NodeStates),
		NodeMetrics:     make(map[string]NodeMetrics),
		Timestamp:       time.Now(),
	}

	// Calculate phase variance
	phases := make([]float64, 0, len(qs.NodeStates))
	for _, nodeState := range qs.NodeStates {
		phases = append(phases, nodeState.SyncPhase)
	}
	metrics.PhaseVariance = qs.calculateVariance(phases)

	// Calculate average connectivity
	totalConnectivity := 0.0
	connectionCount := 0
	for _, nodeState := range qs.NodeStates {
		for _, weight := range nodeState.Connectivity {
			totalConnectivity += weight
			connectionCount++
		}
	}
	if connectionCount > 0 {
		metrics.AverageConnectivity = totalConnectivity / float64(connectionCount)
	}

	// Calculate sync rate
	if qs.TotalSyncOperations > 0 {
		metrics.SyncRate = float64(qs.SuccessfulSyncs) / float64(qs.TotalSyncOperations)
	}

	// Create node metrics
	for nodeID, nodeState := range qs.NodeStates {
		syncResult := syncResults[nodeID]

		nodeMetrics := NodeMetrics{
			LocalCoherence:    nodeState.Coherence,
			ConnectivityCount: len(nodeState.Connectivity),
		}

		if syncResult != nil {
			nodeMetrics.PhaseDrift = syncResult.PhaseAdjustment
			if syncResult.Success {
				nodeMetrics.SyncSuccessRate = 1.0
			}
		}

		metrics.NodeMetrics[nodeID] = nodeMetrics
	}

	return metrics
}

// calculateVariance calculates the variance of a slice of float64 values
func (qs *QuaternionicSynchronizer) calculateVariance(values []float64) float64 {
	if len(values) < 2 {
		return 0.0
	}

	// Calculate mean
	sum := 0.0
	for _, value := range values {
		sum += value
	}
	mean := sum / float64(len(values))

	// Calculate variance
	variance := 0.0
	for _, value := range values {
		diff := value - mean
		variance += diff * diff
	}

	return variance / float64(len(values))
}

// createEmptyMetrics creates empty metrics for insufficient node count
func (qs *QuaternionicSynchronizer) createEmptyMetrics() *SynchronizationMetrics {
	return &SynchronizationMetrics{
		GlobalCoherence:     1.0,
		PhaseVariance:       0.0,
		AverageConnectivity: 0.0,
		NodeCount:           len(qs.NodeStates),
		SyncRate:            0.0,
		ConvergenceRate:     0.0,
		NodeMetrics:         make(map[string]NodeMetrics),
		Timestamp:           time.Now(),
	}
}

// GetNodeState retrieves the state of a specific node
func (qs *QuaternionicSynchronizer) GetNodeState(nodeID string) (*NodeQuaternionicState, bool) {
	qs.mu.RLock()
	defer qs.mu.RUnlock()

	state, exists := qs.NodeStates[nodeID]
	return state, exists
}

// UpdateNodeConnectivity updates the connectivity between two nodes
func (qs *QuaternionicSynchronizer) UpdateNodeConnectivity(nodeID1, nodeID2 string, strength float64) error {
	qs.mu.Lock()
	defer qs.mu.Unlock()

	node1, exists1 := qs.NodeStates[nodeID1]
	node2, exists2 := qs.NodeStates[nodeID2]

	if !exists1 {
		return fmt.Errorf("node %s not found", nodeID1)
	}
	if !exists2 {
		return fmt.Errorf("node %s not found", nodeID2)
	}

	if strength < 0 || strength > 1 {
		return fmt.Errorf("connectivity strength must be between 0 and 1, got %.3f", strength)
	}

	node1.Connectivity[nodeID2] = strength
	node2.Connectivity[nodeID1] = strength

	return nil
}

// GetSynchronizationStatistics returns comprehensive synchronization statistics
func (qs *QuaternionicSynchronizer) GetSynchronizationStatistics() map[string]interface{} {
	qs.mu.RLock()
	defer qs.mu.RUnlock()

	syncRate := 0.0
	if qs.TotalSyncOperations > 0 {
		syncRate = float64(qs.SuccessfulSyncs) / float64(qs.TotalSyncOperations)
	}

	return map[string]interface{}{
		"total_sync_operations": qs.TotalSyncOperations,
		"successful_syncs":      qs.SuccessfulSyncs,
		"failed_syncs":          qs.FailedSyncs,
		"sync_rate":             syncRate,
		"global_coherence":      qs.GlobalCoherence,
		"node_count":            len(qs.NodeStates),
		"coupling_strength":     qs.CouplingStrength,
		"natural_frequency":     qs.NaturalFrequency,
		"max_phase_drift":       qs.MaxPhaseDrift,
		"convergence_threshold": qs.ConvergenceThreshold,
		"sync_interval":         qs.SyncInterval,
		"last_sync_time":        qs.LastSyncTime,
	}
}

// IsConverged checks if the synchronization has converged
func (qs *QuaternionicSynchronizer) IsConverged() bool {
	qs.mu.RLock()
	defer qs.mu.RUnlock()

	return qs.GlobalCoherence >= qs.ConvergenceThreshold
}

// Reset resets the synchronizer state
func (qs *QuaternionicSynchronizer) Reset() {
	qs.mu.Lock()
	defer qs.mu.Unlock()

	qs.NodeStates = make(map[string]*NodeQuaternionicState)
	qs.TotalSyncOperations = 0
	qs.SuccessfulSyncs = 0
	qs.FailedSyncs = 0
	qs.LastSyncTime = time.Now()
	qs.GlobalCoherence = 0.0
}
