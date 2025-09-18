package hilbert

import (
	"fmt"
	"math"
	"math/cmplx"
	"sync"
	"time"

	"github.com/resonancelab/psizero/core/primes"
)

// QuantumState represents a state vector in the prime-based Hilbert space
// |ψ⟩ = Σ_{p∈ℙ} α_p|p⟩ where Σ|α_p|² = 1
type QuantumState struct {
	Amplitudes []complex128 `json:"amplitudes"`
	PrimeBasis []int        `json:"prime_basis"`
	Entropy    float64      `json:"entropy"`
	Coherence  float64      `json:"coherence"`
	Energy     float64      `json:"energy"`
	LastUpdate time.Time    `json:"last_update"`
	Normalized bool         `json:"normalized"`
	mu         sync.RWMutex
}

// HilbertSpace manages quantum states in prime-based Hilbert space
type HilbertSpace struct {
	dimension   int
	primeBasis  []int
	primeEngine *primes.PrimeEngine
	states      map[string]*QuantumState
	mu          sync.RWMutex
}

// StateMetrics contains computed metrics for a quantum state
type StateMetrics struct {
	Entropy           float64 `json:"entropy"`
	Coherence         float64 `json:"coherence"`
	Energy            float64 `json:"energy"`
	Purity            float64 `json:"purity"`
	EntanglementDepth int     `json:"entanglement_depth"`
	Locality          float64 `json:"locality"`
}

// NewHilbertSpace creates a new prime-based Hilbert space
func NewHilbertSpace(dimension int, primeEngine *primes.PrimeEngine) (*HilbertSpace, error) {
	if dimension <= 0 {
		return nil, fmt.Errorf("dimension must be positive, got %d", dimension)
	}

	if primeEngine == nil {
		return nil, fmt.Errorf("prime engine cannot be nil")
	}

	// Get prime basis for the Hilbert space
	primeBasis := primeEngine.GetPrimeBasis(dimension)
	if len(primeBasis) < dimension {
		return nil, fmt.Errorf("insufficient primes for dimension %d", dimension)
	}

	return &HilbertSpace{
		dimension:   dimension,
		primeBasis:  primeBasis,
		primeEngine: primeEngine,
		states:      make(map[string]*QuantumState),
	}, nil
}

// CreateState creates a new quantum state with given amplitudes
func (hs *HilbertSpace) CreateState(amplitudes []complex128) (*QuantumState, error) {
	if len(amplitudes) != hs.dimension {
		return nil, fmt.Errorf("amplitudes length %d doesn't match dimension %d",
			len(amplitudes), hs.dimension)
	}

	// Copy amplitudes to avoid external modification
	ampCopy := make([]complex128, len(amplitudes))
	copy(ampCopy, amplitudes)

	// Copy prime basis
	basisCopy := make([]int, len(hs.primeBasis))
	copy(basisCopy, hs.primeBasis)

	state := &QuantumState{
		Amplitudes: ampCopy,
		PrimeBasis: basisCopy,
		LastUpdate: time.Now(),
		Normalized: false,
	}

	// Normalize the state
	if err := hs.NormalizeState(state); err != nil {
		return nil, fmt.Errorf("failed to normalize state: %w", err)
	}

	// Compute initial metrics
	hs.UpdateStateMetrics(state)

	return state, nil
}

// CreateBasisState creates a state with amplitude 1 for a specific prime basis element
func (hs *HilbertSpace) CreateBasisState(primeIndex int) (*QuantumState, error) {
	if primeIndex < 0 || primeIndex >= hs.dimension {
		return nil, fmt.Errorf("prime index %d out of range [0, %d)", primeIndex, hs.dimension)
	}

	amplitudes := make([]complex128, hs.dimension)
	amplitudes[primeIndex] = complex(1.0, 0.0)

	return hs.CreateState(amplitudes)
}

// CreateSuperposition creates a uniform superposition state
func (hs *HilbertSpace) CreateSuperposition() (*QuantumState, error) {
	amplitude := complex(1.0/math.Sqrt(float64(hs.dimension)), 0.0)
	amplitudes := make([]complex128, hs.dimension)

	for i := range amplitudes {
		amplitudes[i] = amplitude
	}

	return hs.CreateState(amplitudes)
}

// normalizeStateInternal normalizes the state without locking (assumes lock is held)
func (hs *HilbertSpace) normalizeStateInternal(state *QuantumState) error {
	// Calculate norm squared
	normSquared := 0.0
	for _, amp := range state.Amplitudes {
		normSquared += real(amp * cmplx.Conj(amp))
	}

	if normSquared == 0 {
		return fmt.Errorf("cannot normalize zero state")
	}

	// Normalize amplitudes
	norm := math.Sqrt(normSquared)
	for i := range state.Amplitudes {
		state.Amplitudes[i] /= complex(norm, 0)
	}

	state.Normalized = true
	state.LastUpdate = time.Now()

	return nil
}

// NormalizeState ensures the state satisfies Σ|α_p|² = 1
func (hs *HilbertSpace) NormalizeState(state *QuantumState) error {
	if state == nil {
		return fmt.Errorf("state cannot be nil")
	}

	state.mu.Lock()
	defer state.mu.Unlock()

	return hs.normalizeStateInternal(state)
}

// updateStateMetricsInternal updates metrics without locking (assumes lock is held)
func (hs *HilbertSpace) updateStateMetricsInternal(state *QuantumState) {
	state.Entropy = hs.computeVonNeumannEntropy(state)
	state.Coherence = hs.computeCoherence(state)
	state.Energy = hs.computeEnergy(state)
	state.LastUpdate = time.Now()
}

// UpdateStateMetrics computes and updates all metrics for the quantum state
func (hs *HilbertSpace) UpdateStateMetrics(state *QuantumState) {
	if state == nil {
		return
	}

	state.mu.Lock()
	defer state.mu.Unlock()

	hs.updateStateMetricsInternal(state)
}

// computeVonNeumannEntropy calculates the von Neumann entropy S = -Tr(ρ ln ρ)
// For pure states: S = -Σᵢ |αᵢ|² ln|αᵢ|²
func (hs *HilbertSpace) computeVonNeumannEntropy(state *QuantumState) float64 {
	entropy := 0.0

	for _, amp := range state.Amplitudes {
		probability := real(amp * cmplx.Conj(amp))
		if probability > 1e-15 { // Avoid log(0)
			entropy -= probability * math.Log(probability)
		}
	}

	return entropy
}

// computeCoherence measures quantum coherence of the state
// Based on l1-norm of off-diagonal elements in computational basis
func (hs *HilbertSpace) computeCoherence(state *QuantumState) float64 {
	coherence := 0.0

	// For a pure state, coherence is related to the interference terms
	// We compute it as the sum of cross-terms between different basis states
	for i := range state.Amplitudes {
		for j := i + 1; j < len(state.Amplitudes); j++ {
			// Coherence contribution from amplitude cross-products
			crossTerm := state.Amplitudes[i] * cmplx.Conj(state.Amplitudes[j])
			coherence += cmplx.Abs(crossTerm)
		}
	}

	return coherence
}

// computeEnergy calculates the energy expectation value ⟨ψ|H|ψ⟩
// Using prime-based Hamiltonian H = Σₚ p|p⟩⟨p|
func (hs *HilbertSpace) computeEnergy(state *QuantumState) float64 {
	energy := 0.0

	for i, amp := range state.Amplitudes {
		probability := real(amp * cmplx.Conj(amp))
		primeEnergy := float64(state.PrimeBasis[i])
		energy += probability * primeEnergy
	}

	return energy
}

// EvolveState applies time evolution to the quantum state
// |ψ(t+dt)⟩ = U(dt)|ψ(t)⟩ where U(dt) = exp(-iHdt/ℏ)
func (hs *HilbertSpace) EvolveState(state *QuantumState, dt float64, hamiltonian [][]complex128) error {
	if state == nil {
		return fmt.Errorf("state cannot be nil")
	}

	if len(hamiltonian) != hs.dimension || len(hamiltonian[0]) != hs.dimension {
		return fmt.Errorf("hamiltonian dimensions don't match Hilbert space")
	}

	state.mu.Lock()
	defer state.mu.Unlock()

	// For small dt, use first-order approximation: U ≈ I - iHdt
	newAmplitudes := make([]complex128, hs.dimension)

	for i := range newAmplitudes {
		newAmplitudes[i] = state.Amplitudes[i] // Identity term

		// Apply -iHdt term
		for j := range state.Amplitudes {
			evolutionTerm := complex(0, -dt) * hamiltonian[i][j] * state.Amplitudes[j]
			newAmplitudes[i] += evolutionTerm
		}
	}

	state.Amplitudes = newAmplitudes
	state.LastUpdate = time.Now()
	state.Normalized = false

	// Renormalize after evolution
	if err := hs.normalizeStateInternal(state); err != nil {
		return err
	}

	// Update metrics
	hs.updateStateMetricsInternal(state)

	return nil
}

// CreatePrimeHamiltonian generates the prime-based Hamiltonian matrix
// H = Σₚ p|p⟩⟨p| (diagonal matrix with primes on diagonal)
func (hs *HilbertSpace) CreatePrimeHamiltonian() [][]complex128 {
	hamiltonian := make([][]complex128, hs.dimension)

	for i := range hamiltonian {
		hamiltonian[i] = make([]complex128, hs.dimension)
		// Diagonal element is the prime value
		hamiltonian[i][i] = complex(float64(hs.primeBasis[i]), 0)
	}

	return hamiltonian
}

// CreateResonanceHamiltonian generates Hamiltonian with resonance coupling
// H = Σₚ p|p⟩⟨p| + λ Σₚ,q R(p,q)|p⟩⟨q|
func (hs *HilbertSpace) CreateResonanceHamiltonian(coupling float64) [][]complex128 {
	hamiltonian := hs.CreatePrimeHamiltonian()

	// Add resonance coupling terms
	for i := range hamiltonian {
		for j := range hamiltonian[i] {
			if i != j {
				resonance := hs.primeEngine.ComputePrimeResonance(
					hs.primeBasis[i],
					hs.primeBasis[j],
				)
				hamiltonian[i][j] += complex(coupling*resonance, 0)
			}
		}
	}

	return hamiltonian
}

// MeasureState performs a quantum measurement in the computational basis
// Returns the index of the measured basis state and collapses the state
func (hs *HilbertSpace) MeasureState(state *QuantumState) (int, error) {
	if state == nil {
		return -1, fmt.Errorf("state cannot be nil")
	}

	state.mu.Lock()
	defer state.mu.Unlock()

	// Calculate probabilities
	probabilities := make([]float64, len(state.Amplitudes))
	for i, amp := range state.Amplitudes {
		probabilities[i] = real(amp * cmplx.Conj(amp))
	}

	// Generate random number for measurement outcome
	// Note: In production, should use cryptographically secure random number
	r := math.Mod(float64(time.Now().UnixNano()), 1.0)

	// Find measurement outcome
	cumulative := 0.0
	for i, prob := range probabilities {
		cumulative += prob
		if r <= cumulative {
			// Collapse state to measured basis state
			for j := range state.Amplitudes {
				if j == i {
					state.Amplitudes[j] = complex(1.0, 0.0)
				} else {
					state.Amplitudes[j] = complex(0.0, 0.0)
				}
			}
			state.LastUpdate = time.Now()
			return i, nil
		}
	}

	// Fallback (shouldn't happen with proper normalization)
	return len(state.Amplitudes) - 1, nil
}

// ComputeInnerProduct calculates ⟨ψ₁|ψ₂⟩
func (hs *HilbertSpace) ComputeInnerProduct(state1, state2 *QuantumState) (complex128, error) {
	if state1 == nil || state2 == nil {
		return 0, fmt.Errorf("states cannot be nil")
	}

	if len(state1.Amplitudes) != len(state2.Amplitudes) {
		return 0, fmt.Errorf("states have different dimensions")
	}

	state1.mu.RLock()
	state2.mu.RLock()
	defer state1.mu.RUnlock()
	defer state2.mu.RUnlock()

	innerProduct := complex(0, 0)
	for i := range state1.Amplitudes {
		innerProduct += cmplx.Conj(state1.Amplitudes[i]) * state2.Amplitudes[i]
	}

	return innerProduct, nil
}

// ComputeFidelity calculates the fidelity between two quantum states
// F(ρ₁, ρ₂) = |⟨ψ₁|ψ₂⟩|² for pure states
func (hs *HilbertSpace) ComputeFidelity(state1, state2 *QuantumState) (float64, error) {
	innerProduct, err := hs.ComputeInnerProduct(state1, state2)
	if err != nil {
		return 0, fmt.Errorf("failed to compute inner product: %w", err)
	}

	fidelity := real(innerProduct * cmplx.Conj(innerProduct))
	return fidelity, nil
}

// StoreState stores a quantum state with given ID
func (hs *HilbertSpace) StoreState(id string, state *QuantumState) {
	hs.mu.Lock()
	defer hs.mu.Unlock()

	hs.states[id] = state
}

// RetrieveState retrieves a stored quantum state by ID
func (hs *HilbertSpace) RetrieveState(id string) (*QuantumState, error) {
	hs.mu.RLock()
	defer hs.mu.RUnlock()

	state, exists := hs.states[id]
	if !exists {
		return nil, fmt.Errorf("state with ID %s not found", id)
	}

	return state, nil
}

// GetDimension returns the dimension of the Hilbert space
func (hs *HilbertSpace) GetDimension() int {
	return hs.dimension
}

// GetPrimeBasis returns a copy of the prime basis
func (hs *HilbertSpace) GetPrimeBasis() []int {
	basis := make([]int, len(hs.primeBasis))
	copy(basis, hs.primeBasis)
	return basis
}

// GetStoredStatesCount returns the number of stored states
func (hs *HilbertSpace) GetStoredStatesCount() int {
	hs.mu.RLock()
	defer hs.mu.RUnlock()

	return len(hs.states)
}

// CloneState creates a deep copy of a quantum state
func (hs *HilbertSpace) CloneState(state *QuantumState) *QuantumState {
	if state == nil {
		return nil
	}

	state.mu.RLock()
	defer state.mu.RUnlock()

	amplitudes := make([]complex128, len(state.Amplitudes))
	copy(amplitudes, state.Amplitudes)

	basis := make([]int, len(state.PrimeBasis))
	copy(basis, state.PrimeBasis)

	return &QuantumState{
		Amplitudes: amplitudes,
		PrimeBasis: basis,
		Entropy:    state.Entropy,
		Coherence:  state.Coherence,
		Energy:     state.Energy,
		LastUpdate: time.Now(),
		Normalized: state.Normalized,
	}
}

// HolographicReconstruction represents the result of holographic reconstruction
type HolographicReconstruction struct {
	ReconstructedState  *QuantumState `json:"reconstructed_state"`
	Confidence          float64       `json:"confidence"`
	Coherence           float64       `json:"coherence"`
	ReconstructionError float64       `json:"reconstruction_error"`
	UsedFragments       int           `json:"used_fragments"`
	TotalFragments      int           `json:"total_fragments"`
}

// ReconstructFromFragments performs holographic reconstruction from partial state fragments
func (hs *HilbertSpace) ReconstructFromFragments(fragments []*QuantumState, referenceState *QuantumState) (*HolographicReconstruction, error) {
	if len(fragments) == 0 {
		return nil, fmt.Errorf("no fragments provided for reconstruction")
	}

	// Initialize reconstruction with the first fragment
	reconstructed := hs.CloneState(fragments[0])
	if reconstructed == nil {
		return nil, fmt.Errorf("failed to clone first fragment")
	}

	// Apply holographic reconstruction algorithm
	for i := 1; i < len(fragments); i++ {
		if err := hs.holographicMerge(reconstructed, fragments[i]); err != nil {
			return nil, fmt.Errorf("failed to merge fragment %d: %w", i, err)
		}
	}

	// Apply reference-guided enhancement if reference is provided
	if referenceState != nil {
		if err := hs.referenceGuidedEnhancement(reconstructed, referenceState); err != nil {
			return nil, fmt.Errorf("failed to apply reference enhancement: %w", err)
		}
	}

	// Renormalize the reconstructed state
	if err := hs.NormalizeState(reconstructed); err != nil {
		return nil, fmt.Errorf("failed to normalize reconstructed state: %w", err)
	}

	// Update metrics
	hs.UpdateStateMetrics(reconstructed)

	// Calculate reconstruction quality metrics
	confidence := hs.calculateReconstructionConfidence(reconstructed, fragments)
	coherence := reconstructed.Coherence
	reconstructionError := hs.calculateReconstructionError(reconstructed, fragments)

	result := &HolographicReconstruction{
		ReconstructedState:  reconstructed,
		Confidence:          confidence,
		Coherence:           coherence,
		ReconstructionError: reconstructionError,
		UsedFragments:       len(fragments),
		TotalFragments:      len(fragments),
	}

	return result, nil
}

// holographicMerge merges a fragment into the reconstructed state using holographic principles
func (hs *HilbertSpace) holographicMerge(reconstructed, fragment *QuantumState) error {
	if len(reconstructed.Amplitudes) != len(fragment.Amplitudes) {
		return fmt.Errorf("fragment dimension mismatch")
	}

	reconstructed.mu.Lock()
	fragment.mu.RLock()
	defer reconstructed.mu.Unlock()
	defer fragment.mu.RUnlock()

	// Calculate interference pattern between reconstructed state and fragment
	interference := make([]complex128, len(reconstructed.Amplitudes))

	for i := range reconstructed.Amplitudes {
		// Holographic interference: combine amplitudes with phase coherence
		reconstructedAmp := reconstructed.Amplitudes[i]
		fragmentAmp := fragment.Amplitudes[i]

		// Calculate cross-interference term
		crossTerm := reconstructedAmp * cmplx.Conj(fragmentAmp)

		// Apply holographic reconstruction formula
		// New amplitude = old + fragment + interference correction
		interference[i] = reconstructedAmp + fragmentAmp + 0.5*crossTerm
	}

	// Apply coherence weighting based on fragment quality
	coherenceWeight := hs.calculateFragmentCoherence(fragment)
	coherenceWeightComplex := complex(coherenceWeight, 0)
	invCoherenceWeightComplex := complex(1-coherenceWeight, 0)

	for i := range interference {
		// Weighted combination preserving phase information
		reconstructed.Amplitudes[i] = invCoherenceWeightComplex*reconstructed.Amplitudes[i] +
			coherenceWeightComplex*interference[i]
	}

	reconstructed.LastUpdate = time.Now()
	reconstructed.Normalized = false

	return nil
}

// referenceGuidedEnhancement enhances reconstruction using a reference state
func (hs *HilbertSpace) referenceGuidedEnhancement(reconstructed, reference *QuantumState) error {
	if len(reconstructed.Amplitudes) != len(reference.Amplitudes) {
		return fmt.Errorf("reference state dimension mismatch")
	}

	reconstructed.mu.Lock()
	reference.mu.RLock()
	defer reconstructed.mu.Unlock()
	defer reference.mu.RUnlock()

	// Calculate similarity between reconstructed and reference states
	similarity, err := hs.ComputeFidelity(reconstructed, reference)
	if err != nil {
		return fmt.Errorf("failed to compute similarity: %w", err)
	}

	// Apply enhancement based on similarity
	enhancementFactor := math.Min(similarity*2.0, 1.0) // Scale similarity to enhancement factor

	for i := range reconstructed.Amplitudes {
		// Guided enhancement: pull toward reference state
		difference := reference.Amplitudes[i] - reconstructed.Amplitudes[i]
		enhancementTerm := complex(enhancementFactor*0.3, 0) * difference
		reconstructed.Amplitudes[i] += enhancementTerm
	}

	reconstructed.LastUpdate = time.Now()
	reconstructed.Normalized = false

	return nil
}

// calculateFragmentCoherence calculates the coherence quality of a fragment
func (hs *HilbertSpace) calculateFragmentCoherence(fragment *QuantumState) float64 {
	if fragment == nil {
		return 0.0
	}

	// Use existing coherence calculation but normalize to [0,1]
	coherence := hs.computeCoherence(fragment)

	// Normalize coherence to [0,1] range
	maxCoherence := float64(len(fragment.Amplitudes)) * float64(len(fragment.Amplitudes)-1) / 2.0
	if maxCoherence > 0 {
		coherence = math.Min(coherence/maxCoherence, 1.0)
	}

	return coherence
}

// calculateReconstructionConfidence calculates confidence in the reconstruction
func (hs *HilbertSpace) calculateReconstructionConfidence(reconstructed *QuantumState, fragments []*QuantumState) float64 {
	if len(fragments) == 0 {
		return 0.0
	}

	totalConfidence := 0.0

	// Calculate average fidelity with all fragments
	for _, fragment := range fragments {
		fidelity, err := hs.ComputeFidelity(reconstructed, fragment)
		if err == nil {
			totalConfidence += fidelity
		}
	}

	confidence := totalConfidence / float64(len(fragments))

	// Factor in reconstruction coherence
	coherenceFactor := math.Min(reconstructed.Coherence*2.0, 1.0)

	return (confidence + coherenceFactor) / 2.0
}

// calculateReconstructionError calculates the reconstruction error
func (hs *HilbertSpace) calculateReconstructionError(reconstructed *QuantumState, fragments []*QuantumState) float64 {
	if len(fragments) == 0 {
		return 1.0
	}

	totalError := 0.0

	// Calculate average distance to fragments
	for _, fragment := range fragments {
		fidelity, err := hs.ComputeFidelity(reconstructed, fragment)
		if err == nil {
			// Error = 1 - fidelity
			totalError += (1.0 - fidelity)
		} else {
			totalError += 1.0 // Maximum error
		}
	}

	return totalError / float64(len(fragments))
}

// CreateHolographicMemory creates a holographic memory system for pattern storage
func (hs *HilbertSpace) CreateHolographicMemory(capacity int) *HolographicMemory {
	return &HolographicMemory{
		hilbertSpace: hs,
		capacity:     capacity,
		patterns:     make(map[string]*HolographicPattern),
		interference: make([][]complex128, capacity),
	}
}

// HolographicMemory represents a holographic memory system
type HolographicMemory struct {
	hilbertSpace *HilbertSpace
	capacity     int
	patterns     map[string]*HolographicPattern
	interference [][]complex128
	mu           sync.RWMutex
}

// HolographicPattern represents a stored holographic pattern
type HolographicPattern struct {
	ID          string        `json:"id"`
	State       *QuantumState `json:"-"`
	Strength    float64       `json:"strength"`
	AccessCount int           `json:"access_count"`
	LastAccess  time.Time     `json:"last_access"`
}

// StorePattern stores a pattern in holographic memory
func (hm *HolographicMemory) StorePattern(id string, state *QuantumState) error {
	hm.mu.Lock()
	defer hm.mu.Unlock()

	if len(hm.patterns) >= hm.capacity {
		return fmt.Errorf("memory capacity exceeded")
	}

	// Create interference pattern for storage
	pattern := &HolographicPattern{
		ID:          id,
		State:       hm.hilbertSpace.CloneState(state),
		Strength:    1.0,
		AccessCount: 0,
		LastAccess:  time.Now(),
	}

	hm.patterns[id] = pattern

	// Update interference matrix
	hm.updateInterferenceMatrix()

	return nil
}

// RetrievePattern retrieves a pattern from holographic memory
func (hm *HolographicMemory) RetrievePattern(id string) (*QuantumState, error) {
	hm.mu.RLock()
	pattern, exists := hm.patterns[id]
	hm.mu.RUnlock()

	if !exists {
		return nil, fmt.Errorf("pattern not found: %s", id)
	}

	// Update access statistics
	hm.mu.Lock()
	pattern.AccessCount++
	pattern.LastAccess = time.Now()
	hm.mu.Unlock()

	// Reconstruct pattern using holographic principles
	reconstructed, err := hm.reconstructPattern(pattern)
	if err != nil {
		return nil, fmt.Errorf("failed to reconstruct pattern: %w", err)
	}

	return reconstructed, nil
}

// reconstructPattern reconstructs a pattern using holographic reconstruction
func (hm *HolographicMemory) reconstructPattern(pattern *HolographicPattern) (*QuantumState, error) {
	reconstructed := hm.hilbertSpace.CloneState(pattern.State)
	if reconstructed == nil {
		return nil, fmt.Errorf("failed to clone pattern state")
	}

	// Apply holographic reconstruction from interference patterns
	// This is a simplified implementation - in practice would use more sophisticated algorithms

	// Apply strength-based enhancement
	strengthFactor := math.Min(pattern.Strength*1.5, 1.0)
	for i := range reconstructed.Amplitudes {
		reconstructed.Amplitudes[i] *= complex(strengthFactor, 0)
	}

	// Renormalize
	if err := hm.hilbertSpace.NormalizeState(reconstructed); err != nil {
		return nil, fmt.Errorf("failed to renormalize reconstructed state: %w", err)
	}

	return reconstructed, nil
}

// updateInterferenceMatrix updates the holographic interference matrix
func (hm *HolographicMemory) updateInterferenceMatrix() {
	dimension := hm.hilbertSpace.GetDimension()
	hm.interference = make([][]complex128, dimension)

	for i := range hm.interference {
		hm.interference[i] = make([]complex128, dimension)

		// Calculate interference from all stored patterns
		for _, pattern := range hm.patterns {
			if i < len(pattern.State.Amplitudes) {
				amplitude := pattern.State.Amplitudes[i]
				// Add to interference matrix
				hm.interference[i][i] += amplitude * complex(pattern.Strength, 0)
			}
		}
	}
}

// GetMemoryStats returns statistics about the holographic memory
func (hm *HolographicMemory) GetMemoryStats() map[string]interface{} {
	hm.mu.RLock()
	defer hm.mu.RUnlock()

	totalAccess := 0
	oldestAccess := time.Now()
	newestAccess := time.Time{}

	for _, pattern := range hm.patterns {
		totalAccess += pattern.AccessCount
		if pattern.LastAccess.Before(oldestAccess) {
			oldestAccess = pattern.LastAccess
		}
		if pattern.LastAccess.After(newestAccess) {
			newestAccess = pattern.LastAccess
		}
	}

	avgStrength := 0.0
	if len(hm.patterns) > 0 {
		for _, pattern := range hm.patterns {
			avgStrength += pattern.Strength
		}
		avgStrength /= float64(len(hm.patterns))
	}

	return map[string]interface{}{
		"capacity":        hm.capacity,
		"stored_patterns": len(hm.patterns),
		"utilization":     float64(len(hm.patterns)) / float64(hm.capacity),
		"total_access":    totalAccess,
		"avg_access":      float64(totalAccess) / math.Max(float64(len(hm.patterns)), 1),
		"avg_strength":    avgStrength,
		"oldest_access":   oldestAccess,
		"newest_access":   newestAccess,
	}
}
