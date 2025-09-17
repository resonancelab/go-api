package core

import (
	"fmt"
	"time"

	"github.com/resonancelab/psizero/core/entropy"
	"github.com/resonancelab/psizero/core/hilbert"
	"github.com/resonancelab/psizero/core/operators"
	"github.com/resonancelab/psizero/core/primes"
)

// ResonanceEngine is the main engine implementing the Ψ0=1 formalism
type ResonanceEngine struct {
	primeEngine         *primes.PrimeEngine
	hilbertSpace        *hilbert.HilbertSpace
	entropyEvolution    *entropy.EntropyEvolution
	plateauDetector     *entropy.PlateauDetector
	observationCapacity *entropy.ObservationCapacity

	// Core operators
	primeOperator   *operators.PrimeOperator
	resonanceOps    map[int]*operators.ResonanceOperator
	eulerTransform  *operators.EulerTransform
	mobiusTransform *operators.MobiusTransform

	// Configuration
	dimension     int
	maxPrimeLimit int
	initialized   bool
}

// EngineConfig contains configuration for the resonance engine
type EngineConfig struct {
	Dimension        int     `json:"dimension"`
	MaxPrimeLimit    int     `json:"max_prime_limit"`
	InitialEntropy   float64 `json:"initial_entropy"`
	EntropyLambda    float64 `json:"entropy_lambda"`
	PlateauTolerance float64 `json:"plateau_tolerance"`
	PlateauWindow    int     `json:"plateau_window"`
	HistorySize      int     `json:"history_size"`
}

// DefaultEngineConfig returns a default configuration
func DefaultEngineConfig() *EngineConfig {
	return &EngineConfig{
		Dimension:        50,    // 50-dimensional prime space
		MaxPrimeLimit:    1000,  // Primes up to 1000
		InitialEntropy:   1.5,   // Initial entropy
		EntropyLambda:    0.02,  // Entropy decay rate
		PlateauTolerance: 1e-6,  // Plateau detection tolerance
		PlateauWindow:    10,    // Window size for plateau detection
		HistorySize:      10000, // Maximum history points
	}
}

// NewResonanceEngine creates a new resonance engine with the given configuration
func NewResonanceEngine(config *EngineConfig) (*ResonanceEngine, error) {
	if config == nil {
		config = DefaultEngineConfig()
	}

	// Validate configuration
	if err := validateConfig(config); err != nil {
		return nil, fmt.Errorf("invalid configuration: %w", err)
	}

	// Initialize prime engine
	primeEngine := primes.NewPrimeEngine(config.MaxPrimeLimit)

	// Initialize Hilbert space
	hilbertSpace, err := hilbert.NewHilbertSpace(config.Dimension, primeEngine)
	if err != nil {
		return nil, fmt.Errorf("failed to create Hilbert space: %w", err)
	}

	// Initialize entropy evolution
	entropyEvolution := entropy.NewEntropyEvolution(
		config.InitialEntropy,
		config.EntropyLambda,
		config.HistorySize,
	)

	// Initialize plateau detector
	plateauDetector := entropy.NewPlateauDetector(
		config.PlateauWindow,
		config.PlateauTolerance,
		config.PlateauWindow/2,     // Minimum plateau length
		config.PlateauTolerance*10, // Gradient threshold
	)

	// Initialize observation capacity tracker
	observationCapacity := entropy.NewObservationCapacity(config.PlateauWindow)

	// Get prime basis for operators
	primeBasis := hilbertSpace.GetPrimeBasis()

	// Initialize core operators
	primeOperator := operators.NewPrimeOperator(primeBasis)
	eulerTransform := operators.NewEulerTransform(primeBasis, primeEngine)
	mobiusTransform := operators.NewMobiusTransform(primeBasis, primeEngine)

	engine := &ResonanceEngine{
		primeEngine:         primeEngine,
		hilbertSpace:        hilbertSpace,
		entropyEvolution:    entropyEvolution,
		plateauDetector:     plateauDetector,
		observationCapacity: observationCapacity,
		primeOperator:       primeOperator,
		resonanceOps:        make(map[int]*operators.ResonanceOperator),
		eulerTransform:      eulerTransform,
		mobiusTransform:     mobiusTransform,
		dimension:           config.Dimension,
		maxPrimeLimit:       config.MaxPrimeLimit,
		initialized:         true,
	}

	return engine, nil
}

// validateConfig validates the engine configuration
func validateConfig(config *EngineConfig) error {
	if config.Dimension <= 0 {
		return fmt.Errorf("dimension must be positive, got %d", config.Dimension)
	}

	if config.MaxPrimeLimit <= 0 {
		return fmt.Errorf("max prime limit must be positive, got %d", config.MaxPrimeLimit)
	}

	if config.InitialEntropy <= 0 {
		return fmt.Errorf("initial entropy must be positive, got %f", config.InitialEntropy)
	}

	if config.EntropyLambda <= 0 {
		return fmt.Errorf("entropy lambda must be positive, got %f", config.EntropyLambda)
	}

	if config.PlateauTolerance <= 0 {
		return fmt.Errorf("plateau tolerance must be positive, got %f", config.PlateauTolerance)
	}

	if config.PlateauWindow <= 0 {
		return fmt.Errorf("plateau window must be positive, got %d", config.PlateauWindow)
	}

	if config.HistorySize <= 0 {
		return fmt.Errorf("history size must be positive, got %d", config.HistorySize)
	}

	return nil
}

// CreateQuantumState creates a new quantum state with given amplitudes
func (re *ResonanceEngine) CreateQuantumState(amplitudes []complex128) (*hilbert.QuantumState, error) {
	if !re.initialized {
		return nil, fmt.Errorf("engine not initialized")
	}

	return re.hilbertSpace.CreateState(amplitudes)
}

// CreateSuperpositionState creates a uniform superposition state
func (re *ResonanceEngine) CreateSuperpositionState() (*hilbert.QuantumState, error) {
	if !re.initialized {
		return nil, fmt.Errorf("engine not initialized")
	}

	return re.hilbertSpace.CreateSuperposition()
}

// CreateBasisState creates a basis state for a specific prime
func (re *ResonanceEngine) CreateBasisState(primeIndex int) (*hilbert.QuantumState, error) {
	if !re.initialized {
		return nil, fmt.Errorf("engine not initialized")
	}

	return re.hilbertSpace.CreateBasisState(primeIndex)
}

// GetResonanceOperator gets or creates a resonance operator for a given number
func (re *ResonanceEngine) GetResonanceOperator(number int, strength float64) *operators.ResonanceOperator {
	key := number
	if op, exists := re.resonanceOps[key]; exists {
		return op
	}

	primeBasis := re.hilbertSpace.GetPrimeBasis()
	op := operators.NewResonanceOperator(primeBasis, re.primeEngine, number, strength)
	re.resonanceOps[key] = op

	return op
}

// ApplyOperator applies an operator to a quantum state
func (re *ResonanceEngine) ApplyOperator(state *hilbert.QuantumState, operator operators.Operator) (*hilbert.QuantumState, error) {
	if !re.initialized {
		return nil, fmt.Errorf("engine not initialized")
	}

	newState, err := operator.Apply(state)
	if err != nil {
		return nil, fmt.Errorf("failed to apply operator %s: %w", operator.GetName(), err)
	}

	// Renormalize if needed
	if !newState.Normalized {
		if err := re.hilbertSpace.NormalizeState(newState); err != nil {
			return nil, fmt.Errorf("failed to normalize state after operator: %w", err)
		}
	}

	// Update state metrics
	re.hilbertSpace.UpdateStateMetrics(newState)

	return newState, nil
}

// EvolveState evolves a quantum state using the prime Hamiltonian
func (re *ResonanceEngine) EvolveState(state *hilbert.QuantumState, dt float64) (*hilbert.QuantumState, error) {
	if !re.initialized {
		return nil, fmt.Errorf("engine not initialized")
	}

	hamiltonian := re.hilbertSpace.CreatePrimeHamiltonian()
	err := re.hilbertSpace.EvolveState(state, dt, hamiltonian)
	if err != nil {
		return nil, fmt.Errorf("failed to evolve state: %w", err)
	}

	// Update metrics after evolution
	re.hilbertSpace.UpdateStateMetrics(state)

	return state, nil
}

// EvolveStateWithResonance evolves a state using resonance-coupled Hamiltonian
func (re *ResonanceEngine) EvolveStateWithResonance(state *hilbert.QuantumState, dt, coupling float64) (*hilbert.QuantumState, error) {
	if !re.initialized {
		return nil, fmt.Errorf("engine not initialized")
	}

	hamiltonian := re.hilbertSpace.CreateResonanceHamiltonian(coupling)
	err := re.hilbertSpace.EvolveState(state, dt, hamiltonian)
	if err != nil {
		return nil, fmt.Errorf("failed to evolve state with resonance: %w", err)
	}

	// Update metrics after evolution
	re.hilbertSpace.UpdateStateMetrics(state)

	return state, nil
}

// RecordEntropyPoint records an entropy measurement for convergence tracking
func (re *ResonanceEngine) RecordEntropyPoint(state *hilbert.QuantumState, step int) {
	if !re.initialized || state == nil {
		return
	}

	// Update state metrics to get current entropy
	re.hilbertSpace.UpdateStateMetrics(state)

	// Record entropy evolution
	re.entropyEvolution.AddEntropyPoint(state.Entropy, step)

	// Record for observation capacity
	re.observationCapacity.AddMeasurement(state.Entropy, time.Now())
}

// AnalyzeConvergence performs comprehensive convergence analysis
func (re *ResonanceEngine) AnalyzeConvergence(state *hilbert.QuantumState) *entropy.ConvergenceMetrics {
	if !re.initialized {
		return &entropy.ConvergenceMetrics{}
	}

	return entropy.AnalyzeConvergence(
		re.entropyEvolution,
		re.plateauDetector,
		re.observationCapacity,
		state,
	)
}

// PerformCollapseAnalysis analyzes quantum state collapse dynamics
func (re *ResonanceEngine) PerformCollapseAnalysis(timeHorizon float64) *entropy.CollapseAnalysis {
	if !re.initialized {
		return &entropy.CollapseAnalysis{}
	}

	return entropy.PerformCollapseAnalysis(re.entropyEvolution, timeHorizon)
}

// GetPrimeBasis returns the prime basis used by the engine
func (re *ResonanceEngine) GetPrimeBasis() []int {
	if !re.initialized {
		return []int{}
	}

	return re.hilbertSpace.GetPrimeBasis()
}

// GetDimension returns the dimension of the Hilbert space
func (re *ResonanceEngine) GetDimension() int {
	return re.dimension
}

// GetPrimeEngine returns the prime engine
func (re *ResonanceEngine) GetPrimeEngine() *primes.PrimeEngine {
	return re.primeEngine
}

// GetHilbertSpace returns the Hilbert space manager
func (re *ResonanceEngine) GetHilbertSpace() *hilbert.HilbertSpace {
	return re.hilbertSpace
}

// GetEntropyEvolution returns the entropy evolution tracker
func (re *ResonanceEngine) GetEntropyEvolution() *entropy.EntropyEvolution {
	return re.entropyEvolution
}

// Reset resets the engine state
func (re *ResonanceEngine) Reset(config *EngineConfig) error {
	if config == nil {
		config = DefaultEngineConfig()
	}

	if err := validateConfig(config); err != nil {
		return fmt.Errorf("invalid configuration: %w", err)
	}

	// Reset entropy evolution
	re.entropyEvolution.Reset(config.InitialEntropy, config.EntropyLambda)

	// Reset observation capacity
	re.observationCapacity = entropy.NewObservationCapacity(config.PlateauWindow)

	// Clear resonance operators cache
	re.resonanceOps = make(map[int]*operators.ResonanceOperator)

	return nil
}

// CreateFactorizationOperator creates a factorization operator for a given number
func (re *ResonanceEngine) CreateFactorizationOperator(targetNumber int) *operators.FactorizationOperator {
	if !re.initialized {
		return nil
	}

	primeBasis := re.hilbertSpace.GetPrimeBasis()
	return operators.NewFactorizationOperator(primeBasis, re.primeEngine, targetNumber)
}

// CreateResonanceLockingOperator creates a resonance locking evolution operator
func (re *ResonanceEngine) CreateResonanceLockingOperator(resonanceNumber int,
	lambda, targetResonance, dt float64) *operators.ResonanceLockingOperator {
	if !re.initialized {
		return nil
	}

	hamiltonian := re.hilbertSpace.CreatePrimeHamiltonian()
	resonanceOp := re.GetResonanceOperator(resonanceNumber, 1.0)

	return operators.NewResonanceLockingOperator(
		hamiltonian, resonanceOp, lambda, targetResonance, dt)
}

// ComputeStateMetrics computes comprehensive metrics for a quantum state
func (re *ResonanceEngine) ComputeStateMetrics(state *hilbert.QuantumState) map[string]float64 {
	if !re.initialized || state == nil {
		return map[string]float64{}
	}

	// Update state metrics
	re.hilbertSpace.UpdateStateMetrics(state)

	// Get observation capacity
	observerCapacity := re.observationCapacity.ComputeObservationCapacity()

	// Calculate additional metrics
	metrics := map[string]float64{
		"entropy":           state.Entropy,
		"coherence":         state.Coherence,
		"energy":            state.Energy,
		"observer_capacity": observerCapacity,
	}

	// Add prime-specific metrics
	primeBasis := state.PrimeBasis
	if len(primeBasis) > 0 && len(state.Amplitudes) == len(primeBasis) {
		// Calculate dominant prime
		maxProb := 0.0
		dominantPrime := 0
		for i, amp := range state.Amplitudes {
			prob := real(amp * complex(real(amp), -imag(amp))) // |amp|²
			if prob > maxProb {
				maxProb = prob
				dominantPrime = primeBasis[i]
			}
		}

		metrics["dominant_prime"] = float64(dominantPrime)
		metrics["max_probability"] = maxProb
		metrics["effective_dimension"] = calculateEffectiveDimension(state.Amplitudes)
	}

	return metrics
}

// calculateEffectiveDimension calculates the effective dimension of the quantum state
func calculateEffectiveDimension(amplitudes []complex128) float64 {
	// Calculate participation ratio: 1 / Σᵢ |αᵢ|⁴
	sum := 0.0
	for _, amp := range amplitudes {
		prob := real(amp * complex(real(amp), -imag(amp)))
		sum += prob * prob
	}

	if sum == 0 {
		return 0.0
	}

	return 1.0 / sum
}

// IsInitialized returns whether the engine is properly initialized
func (re *ResonanceEngine) IsInitialized() bool {
	return re.initialized
}

// GetEngineStatus returns current engine status and statistics
func (re *ResonanceEngine) GetEngineStatus() map[string]interface{} {
	status := map[string]interface{}{
		"initialized":     re.initialized,
		"dimension":       re.dimension,
		"max_prime_limit": re.maxPrimeLimit,
		"stored_states":   0,
		"resonance_ops":   len(re.resonanceOps),
	}

	if re.initialized {
		status["stored_states"] = re.hilbertSpace.GetStoredStatesCount()
		status["entropy_history_size"] = len(re.entropyEvolution.GetEntropyHistory())
		status["prime_basis"] = re.hilbertSpace.GetPrimeBasis()[:min(10, len(re.hilbertSpace.GetPrimeBasis()))] // First 10 primes
	}

	return status
}

// min returns the minimum of two integers
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
