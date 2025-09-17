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

// NormalizeState ensures the state satisfies Σ|α_p|² = 1
func (hs *HilbertSpace) NormalizeState(state *QuantumState) error {
	if state == nil {
		return fmt.Errorf("state cannot be nil")
	}

	state.mu.Lock()
	defer state.mu.Unlock()

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

// UpdateStateMetrics computes and updates all metrics for the quantum state
func (hs *HilbertSpace) UpdateStateMetrics(state *QuantumState) {
	if state == nil {
		return
	}

	state.mu.Lock()
	defer state.mu.Unlock()

	state.Entropy = hs.computeVonNeumannEntropy(state)
	state.Coherence = hs.computeCoherence(state)
	state.Energy = hs.computeEnergy(state)
	state.LastUpdate = time.Now()
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

	// Normalize by maximum possible coherence
	maxCoherence := float64(len(state.Amplitudes) * (len(state.Amplitudes) - 1) / 2)
	if maxCoherence > 0 {
		coherence /= maxCoherence
	}

	return math.Min(coherence, 1.0)
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
	return hs.NormalizeState(state)
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
