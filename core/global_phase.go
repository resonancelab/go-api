package core

import (
	"fmt"
	"math"
	"sync"
	"time"
)

// GlobalPhaseState manages the global phase state Φ⃗(t) = (Φ₁(t), Φ₂(t), ..., Φₙ(t)) from Reson.net paper
type GlobalPhaseState struct {
	// Oscillator states indexed by prime
	Oscillators map[int]*PrimeOscillator `json:"oscillators"`

	// Quaternionic states for distributed nodes
	QuaternionicStates map[string]*QuaternionicState `json:"quaternionic_states"`

	// Global coherence C(t) = Σᵢⱼwᵢⱼ·cos(Φᵢ(t) - Φⱼ(t))
	GlobalCoherence float64 `json:"global_coherence"`

	// Phase locking threshold from paper
	PhaseLockThreshold float64 `json:"phase_lock_threshold"`

	// Coupling weights between oscillators
	CouplingWeights [][]float64 `json:"coupling_weights"`

	// Synchronization metadata
	LastSyncTime time.Time     `json:"last_sync_time"`
	SyncInterval time.Duration `json:"sync_interval"`
	NodeCount    int           `json:"node_count"`

	// Thread safety
	mu sync.RWMutex `json:"-"`
}

// NewGlobalPhaseState creates a new global phase state manager
func NewGlobalPhaseState(phaseLockThreshold float64, syncInterval time.Duration) *GlobalPhaseState {
	return &GlobalPhaseState{
		Oscillators:        make(map[int]*PrimeOscillator),
		QuaternionicStates: make(map[string]*QuaternionicState),
		PhaseLockThreshold: phaseLockThreshold,
		CouplingWeights:    nil, // Will be initialized when oscillators are added
		LastSyncTime:       time.Now(),
		SyncInterval:       syncInterval,
		NodeCount:          0,
	}
}

// AddOscillator adds a prime oscillator to the global state
func (gps *GlobalPhaseState) AddOscillator(oscillator *PrimeOscillator) error {
	gps.mu.Lock()
	defer gps.mu.Unlock()

	if oscillator == nil {
		return fmt.Errorf("oscillator cannot be nil")
	}

	if err := oscillator.Validate(); err != nil {
		return fmt.Errorf("invalid oscillator: %w", err)
	}

	gps.Oscillators[oscillator.Prime] = oscillator

	// Reinitialize coupling weights matrix
	gps.initializeCouplingWeights()

	return nil
}

// AddQuaternionicState adds a quaternionic state for a distributed node
func (gps *GlobalPhaseState) AddQuaternionicState(nodeID string, state *QuaternionicState) error {
	gps.mu.Lock()
	defer gps.mu.Unlock()

	if state == nil {
		return fmt.Errorf("quaternionic state cannot be nil")
	}

	if err := state.Validate(); err != nil {
		return fmt.Errorf("invalid quaternionic state: %w", err)
	}

	gps.QuaternionicStates[nodeID] = state
	gps.NodeCount = len(gps.QuaternionicStates)

	return nil
}

// initializeCouplingWeights creates the coupling weights matrix
func (gps *GlobalPhaseState) initializeCouplingWeights() {
	numOscillators := len(gps.Oscillators)
	if numOscillators == 0 {
		return
	}

	gps.CouplingWeights = make([][]float64, numOscillators)
	for i := range gps.CouplingWeights {
		gps.CouplingWeights[i] = make([]float64, numOscillators)
		// Initialize with uniform coupling (can be customized)
		for j := range gps.CouplingWeights[i] {
			if i != j {
				gps.CouplingWeights[i][j] = 1.0 / float64(numOscillators-1)
			}
		}
	}
}

// UpdateGlobalCoherence computes C(t) = Σᵢⱼwᵢⱼ·cos(Φᵢ(t) - Φⱼ(t))
func (gps *GlobalPhaseState) UpdateGlobalCoherence() float64 {
	gps.mu.RLock()
	defer gps.mu.RUnlock()

	return gps.updateGlobalCoherenceInternal()
}

// updateGlobalCoherenceInternal computes coherence without acquiring locks (assumes lock is held)
func (gps *GlobalPhaseState) updateGlobalCoherenceInternal() float64 {
	if len(gps.Oscillators) <= 1 {
		gps.GlobalCoherence = 1.0
		return gps.GlobalCoherence
	}

	coherence := 0.0
	totalWeight := 0.0

	// Convert map to slice for indexed access
	oscList := make([]*PrimeOscillator, 0, len(gps.Oscillators))
	for _, osc := range gps.Oscillators {
		oscList = append(oscList, osc)
	}

	// Compute pairwise coherence
	for i := 0; i < len(oscList); i++ {
		for j := i + 1; j < len(oscList); j++ {
			weight := 1.0 // Default weight
			if gps.CouplingWeights != nil && i < len(gps.CouplingWeights) && j < len(gps.CouplingWeights[i]) {
				weight = gps.CouplingWeights[i][j]
			}

			// cos(Φᵢ(t) - Φⱼ(t))
			phaseDiff := oscList[i].Phase - oscList[j].Phase
			coherence += weight * math.Cos(phaseDiff)
			totalWeight += weight
		}
	}

	if totalWeight > 0 {
		gps.GlobalCoherence = coherence / totalWeight
	} else {
		gps.GlobalCoherence = 0.0
	}

	return gps.GlobalCoherence
}

// EvolvePhaseState evolves all oscillators by deltaTime
func (gps *GlobalPhaseState) EvolvePhaseState(deltaTime float64) {
	gps.mu.Lock()
	defer gps.mu.Unlock()

	// Evolve all oscillators
	for _, osc := range gps.Oscillators {
		osc.UpdatePhase(deltaTime)
	}

	// Evolve quaternionic states
	for _, qstate := range gps.QuaternionicStates {
		if qstate.PrimeOscillator != nil {
			qstate.UpdatePhase(deltaTime, qstate.PrimeOscillator.Frequency)
		}
	}

	// Update global coherence (internal method to avoid deadlock)
	gps.updateGlobalCoherenceInternal()
	gps.LastSyncTime = time.Now()
}

// SynchronizePhases performs phase synchronization across oscillators
func (gps *GlobalPhaseState) SynchronizePhases(couplingStrength float64) {
	gps.mu.Lock()
	defer gps.mu.Unlock()

	if len(gps.Oscillators) <= 1 {
		return
	}

	// Convert to slice for easier iteration
	oscList := make([]*PrimeOscillator, 0, len(gps.Oscillators))
	for _, osc := range gps.Oscillators {
		oscList = append(oscList, osc)
	}

	// Apply Kuramoto-like synchronization
	for i := 0; i < len(oscList); i++ {
		phaseVelocity := 0.0

		for j := 0; j < len(oscList); j++ {
			if i == j {
				continue
			}

			weight := 1.0
			if gps.CouplingWeights != nil && i < len(gps.CouplingWeights) && j < len(gps.CouplingWeights[i]) {
				weight = gps.CouplingWeights[i][j]
			}

			// Kuramoto coupling: Σⱼ K·wᵢⱼ·sin(Φⱼ - Φᵢ)
			phaseDiff := oscList[j].Phase - oscList[i].Phase
			phaseVelocity += couplingStrength * weight * math.Sin(phaseDiff)
		}

		// Update phase with coupling
		oscList[i].Phase += phaseVelocity * 0.01 // Small time step
	}
}

// CheckPhaseLocking determines if the system has achieved phase locking
func (gps *GlobalPhaseState) CheckPhaseLocking() bool {
	gps.mu.RLock()
	defer gps.mu.RUnlock()

	return math.Abs(gps.GlobalCoherence) >= gps.PhaseLockThreshold
}

// GetPhaseDistribution returns the current phase distribution
func (gps *GlobalPhaseState) GetPhaseDistribution() map[int]float64 {
	gps.mu.RLock()
	defer gps.mu.RUnlock()

	distribution := make(map[int]float64)
	for prime, osc := range gps.Oscillators {
		distribution[prime] = osc.Phase
	}

	return distribution
}

// GetQuaternionicState retrieves a quaternionic state by node ID
func (gps *GlobalPhaseState) GetQuaternionicState(nodeID string) (*QuaternionicState, bool) {
	gps.mu.RLock()
	defer gps.mu.RUnlock()

	state, exists := gps.QuaternionicStates[nodeID]
	return state, exists
}

// GetOscillator retrieves an oscillator by prime number
func (gps *GlobalPhaseState) GetOscillator(prime int) (*PrimeOscillator, bool) {
	gps.mu.RLock()
	defer gps.mu.RUnlock()

	osc, exists := gps.Oscillators[prime]
	return osc, exists
}

// ComputeResonanceMatrix computes pairwise resonance between all oscillators
func (gps *GlobalPhaseState) ComputeResonanceMatrix() [][]float64 {
	gps.mu.RLock()
	defer gps.mu.RUnlock()

	oscList := make([]*PrimeOscillator, 0, len(gps.Oscillators))
	for _, osc := range gps.Oscillators {
		oscList = append(oscList, osc)
	}

	matrix := make([][]float64, len(oscList))
	for i := range matrix {
		matrix[i] = make([]float64, len(oscList))
		for j := range matrix[i] {
			if i == j {
				matrix[i][j] = 1.0 // Self-resonance
			} else {
				matrix[i][j] = oscList[i].ComputeResonance(oscList[j])
			}
		}
	}

	return matrix
}

// ExportState exports the current global phase state for serialization
func (gps *GlobalPhaseState) ExportState() *GlobalPhaseStateSnapshot {
	gps.mu.RLock()
	defer gps.mu.RUnlock()

	snapshot := &GlobalPhaseStateSnapshot{
		Timestamp:          time.Now(),
		GlobalCoherence:    gps.GlobalCoherence,
		NodeCount:          gps.NodeCount,
		PhaseDistribution:  make(map[int]float64),
		QuaternionicStates: make(map[string]*QuaternionicState),
	}

	// Copy phase distribution
	for prime, osc := range gps.Oscillators {
		snapshot.PhaseDistribution[prime] = osc.Phase
	}

	// Copy quaternionic states
	for nodeID, state := range gps.QuaternionicStates {
		snapshot.QuaternionicStates[nodeID] = state.Clone()
	}

	return snapshot
}

// ImportState imports a global phase state snapshot
func (gps *GlobalPhaseState) ImportState(snapshot *GlobalPhaseStateSnapshot) error {
	gps.mu.Lock()
	defer gps.mu.Unlock()

	if snapshot == nil {
		return fmt.Errorf("snapshot cannot be nil")
	}

	gps.GlobalCoherence = snapshot.GlobalCoherence
	gps.NodeCount = snapshot.NodeCount
	gps.LastSyncTime = snapshot.Timestamp

	// Update oscillator phases
	for prime, phase := range snapshot.PhaseDistribution {
		if osc, exists := gps.Oscillators[prime]; exists {
			osc.Phase = phase
			osc.LastUpdate = snapshot.Timestamp
		}
	}

	// Update quaternionic states
	gps.QuaternionicStates = make(map[string]*QuaternionicState)
	for nodeID, state := range snapshot.QuaternionicStates {
		gps.QuaternionicStates[nodeID] = state.Clone()
	}

	return nil
}

// String returns a string representation of the global phase state
func (gps *GlobalPhaseState) String() string {
	gps.mu.RLock()
	defer gps.mu.RUnlock()

	return fmt.Sprintf("GlobalPhaseState{oscillators=%d, nodes=%d, C(t)=%.3f, locked=%t}",
		len(gps.Oscillators), gps.NodeCount, gps.GlobalCoherence, gps.CheckPhaseLocking())
}

// GlobalPhaseStateSnapshot represents a snapshot of the global phase state
type GlobalPhaseStateSnapshot struct {
	Timestamp          time.Time                     `json:"timestamp"`
	GlobalCoherence    float64                       `json:"global_coherence"`
	NodeCount          int                           `json:"node_count"`
	PhaseDistribution  map[int]float64               `json:"phase_distribution"`
	QuaternionicStates map[string]*QuaternionicState `json:"quaternionic_states"`
}
