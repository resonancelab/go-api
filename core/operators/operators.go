package operators

import (
	"fmt"
	"math"
	"math/cmplx"

	"github.com/resonancelab/psizero/core/hilbert"
	"github.com/resonancelab/psizero/core/primes"
)

// Operator interface defines quantum operators in the prime-based Hilbert space
type Operator interface {
	Apply(state *hilbert.QuantumState) (*hilbert.QuantumState, error)
	GetMatrix(dimension int) ([][]complex128, error)
	GetName() string
}

// PrimeOperator implements P̂|p⟩ = p|p⟩
type PrimeOperator struct {
	primeBasis []int
}

// NumberOperator implements N̂|n⟩ = n|n⟩ for composite numbers
type NumberOperator struct {
	primeBasis  []int
	primeEngine *primes.PrimeEngine
}

// FactorizationOperator implements F̂|n⟩ = Σᵢ√(aᵢ/A)|pᵢ⟩
type FactorizationOperator struct {
	primeBasis   []int
	primeEngine  *primes.PrimeEngine
	targetNumber int
}

// ResonanceOperator implements R(n)|p⟩ = e^(2πi log_p n)|p⟩
type ResonanceOperator struct {
	primeBasis      []int
	primeEngine     *primes.PrimeEngine
	resonanceNumber int
	strength        float64
}

// EulerTransform implements Ê|n⟩ = e^(2πi φ(n)/n)|n⟩
type EulerTransform struct {
	primeBasis  []int
	primeEngine *primes.PrimeEngine
}

// MobiusTransform implements M̂|n⟩ = μ(n)|n⟩
type MobiusTransform struct {
	primeBasis  []int
	primeEngine *primes.PrimeEngine
}

// VonMangoldtTransform implements Λ̂|n⟩ = Λ(n)|n⟩
type VonMangoldtTransform struct {
	primeBasis  []int
	primeEngine *primes.PrimeEngine
}

// ResonanceLockingOperator implements the evolution equation
// d/dt|Ψ_C⟩ = iĤ|Ψ_C⟩ - λ(R̂-r_stable)|Ψ_C⟩
type ResonanceLockingOperator struct {
	hamiltonian     [][]complex128
	resonanceOp     *ResonanceOperator
	lambda          float64
	targetResonance float64
	dt              float64
}

// NewPrimeOperator creates a new prime operator
func NewPrimeOperator(primeBasis []int) *PrimeOperator {
	return &PrimeOperator{
		primeBasis: primeBasis,
	}
}

// Apply applies the prime operator to a quantum state
func (po *PrimeOperator) Apply(state *hilbert.QuantumState) (*hilbert.QuantumState, error) {
	if state == nil {
		return nil, fmt.Errorf("state cannot be nil")
	}

	if len(state.Amplitudes) != len(po.primeBasis) {
		return nil, fmt.Errorf("state dimension doesn't match operator")
	}

	newAmplitudes := make([]complex128, len(state.Amplitudes))

	// Apply P̂|p⟩ = p|p⟩ (multiply each amplitude by its corresponding prime)
	for i, amp := range state.Amplitudes {
		primeValue := complex(float64(po.primeBasis[i]), 0)
		newAmplitudes[i] = amp * primeValue
	}

	return &hilbert.QuantumState{
		Amplitudes: newAmplitudes,
		PrimeBasis: state.PrimeBasis,
		Normalized: false, // Needs renormalization
	}, nil
}

// GetMatrix returns the matrix representation of the prime operator
func (po *PrimeOperator) GetMatrix(dimension int) ([][]complex128, error) {
	if dimension != len(po.primeBasis) {
		return nil, fmt.Errorf("dimension mismatch")
	}

	matrix := make([][]complex128, dimension)
	for i := range matrix {
		matrix[i] = make([]complex128, dimension)
		// Diagonal matrix with primes on diagonal
		matrix[i][i] = complex(float64(po.primeBasis[i]), 0)
	}

	return matrix, nil
}

// GetName returns the operator name
func (po *PrimeOperator) GetName() string {
	return "PrimeOperator"
}

// NewResonanceOperator creates a new resonance operator
func NewResonanceOperator(primeBasis []int, primeEngine *primes.PrimeEngine,
	resonanceNumber int, strength float64) *ResonanceOperator {
	return &ResonanceOperator{
		primeBasis:      primeBasis,
		primeEngine:     primeEngine,
		resonanceNumber: resonanceNumber,
		strength:        strength,
	}
}

// Apply applies the resonance operator R(n)|p⟩ = e^(2πi log_p n)|p⟩
func (ro *ResonanceOperator) Apply(state *hilbert.QuantumState) (*hilbert.QuantumState, error) {
	if state == nil {
		return nil, fmt.Errorf("state cannot be nil")
	}

	if len(state.Amplitudes) != len(ro.primeBasis) {
		return nil, fmt.Errorf("state dimension doesn't match operator")
	}

	newAmplitudes := make([]complex128, len(state.Amplitudes))

	for i, amp := range state.Amplitudes {
		prime := ro.primeBasis[i]

		// Calculate log_p(n) = ln(n)/ln(p)
		if prime <= 1 || ro.resonanceNumber <= 0 {
			newAmplitudes[i] = amp
			continue
		}

		logRatio := math.Log(float64(ro.resonanceNumber)) / math.Log(float64(prime))

		// Calculate resonance phase: 2π * log_p(n)
		phase := 2.0 * math.Pi * logRatio * ro.strength

		// Apply e^(iφ) = cos(φ) + i*sin(φ)
		resonanceFactor := cmplx.Exp(complex(0, phase))
		newAmplitudes[i] = amp * resonanceFactor
	}

	return &hilbert.QuantumState{
		Amplitudes: newAmplitudes,
		PrimeBasis: state.PrimeBasis,
		Normalized: state.Normalized, // Phase rotations preserve normalization
	}, nil
}

// GetMatrix returns the matrix representation of the resonance operator
func (ro *ResonanceOperator) GetMatrix(dimension int) ([][]complex128, error) {
	if dimension != len(ro.primeBasis) {
		return nil, fmt.Errorf("dimension mismatch")
	}

	matrix := make([][]complex128, dimension)
	for i := range matrix {
		matrix[i] = make([]complex128, dimension)

		prime := ro.primeBasis[i]
		if prime > 1 && ro.resonanceNumber > 0 {
			logRatio := math.Log(float64(ro.resonanceNumber)) / math.Log(float64(prime))
			phase := 2.0 * math.Pi * logRatio * ro.strength
			matrix[i][i] = cmplx.Exp(complex(0, phase))
		} else {
			matrix[i][i] = complex(1, 0)
		}
	}

	return matrix, nil
}

// GetName returns the operator name
func (ro *ResonanceOperator) GetName() string {
	return fmt.Sprintf("ResonanceOperator(n=%d)", ro.resonanceNumber)
}

// NewFactorizationOperator creates a new factorization operator
func NewFactorizationOperator(primeBasis []int, primeEngine *primes.PrimeEngine,
	targetNumber int) *FactorizationOperator {
	return &FactorizationOperator{
		primeBasis:   primeBasis,
		primeEngine:  primeEngine,
		targetNumber: targetNumber,
	}
}

// Apply applies the factorization operator F̂|n⟩ = Σᵢ√(aᵢ/A)|pᵢ⟩
func (fo *FactorizationOperator) Apply(state *hilbert.QuantumState) (*hilbert.QuantumState, error) {
	if state == nil {
		return nil, fmt.Errorf("state cannot be nil")
	}

	// Get prime factorization of target number
	factors := fo.primeEngine.Factorize(fo.targetNumber)
	if len(factors) == 0 {
		return state, nil // Identity for non-factorizable input
	}

	// Calculate total weight A = Σᵢaᵢ
	totalWeight := 0
	for _, factor := range factors {
		totalWeight += factor.Exponent
	}

	if totalWeight == 0 {
		return state, nil
	}

	// Create new amplitudes based on factorization
	newAmplitudes := make([]complex128, len(state.Amplitudes))

	// Create mapping from prime to its coefficient √(aᵢ/A)
	primeCoeffs := make(map[int]float64)
	for _, factor := range factors {
		coeff := math.Sqrt(float64(factor.Exponent) / float64(totalWeight))
		primeCoeffs[factor.Prime] = coeff
	}

	// Apply factorization: distribute amplitude to prime factors
	for i, amp := range state.Amplitudes {
		prime := fo.primeBasis[i]
		if coeff, exists := primeCoeffs[prime]; exists {
			newAmplitudes[i] = amp * complex(coeff, 0)
		} else {
			newAmplitudes[i] = complex(0, 0) // Zero for non-factors
		}
	}

	return &hilbert.QuantumState{
		Amplitudes: newAmplitudes,
		PrimeBasis: state.PrimeBasis,
		Normalized: false, // Needs renormalization
	}, nil
}

// GetMatrix returns the matrix representation of the factorization operator
func (fo *FactorizationOperator) GetMatrix(dimension int) ([][]complex128, error) {
	if dimension != len(fo.primeBasis) {
		return nil, fmt.Errorf("dimension mismatch")
	}

	factors := fo.primeEngine.Factorize(fo.targetNumber)
	if len(factors) == 0 {
		// Return identity matrix
		matrix := make([][]complex128, dimension)
		for i := range matrix {
			matrix[i] = make([]complex128, dimension)
			matrix[i][i] = complex(1, 0)
		}
		return matrix, nil
	}

	// Calculate coefficients
	totalWeight := 0
	for _, factor := range factors {
		totalWeight += factor.Exponent
	}

	primeCoeffs := make(map[int]float64)
	for _, factor := range factors {
		coeff := math.Sqrt(float64(factor.Exponent) / float64(totalWeight))
		primeCoeffs[factor.Prime] = coeff
	}

	// Create diagonal matrix with factorization coefficients
	matrix := make([][]complex128, dimension)
	for i := range matrix {
		matrix[i] = make([]complex128, dimension)
		prime := fo.primeBasis[i]
		if coeff, exists := primeCoeffs[prime]; exists {
			matrix[i][i] = complex(coeff, 0)
		}
		// Otherwise remains zero
	}

	return matrix, nil
}

// GetName returns the operator name
func (fo *FactorizationOperator) GetName() string {
	return fmt.Sprintf("FactorizationOperator(n=%d)", fo.targetNumber)
}

// NewEulerTransform creates a new Euler transform operator
func NewEulerTransform(primeBasis []int, primeEngine *primes.PrimeEngine) *EulerTransform {
	return &EulerTransform{
		primeBasis:  primeBasis,
		primeEngine: primeEngine,
	}
}

// Apply applies the Euler transform Ê|n⟩ = e^(2πi φ(n)/n)|n⟩
func (et *EulerTransform) Apply(state *hilbert.QuantumState) (*hilbert.QuantumState, error) {
	if state == nil {
		return nil, fmt.Errorf("state cannot be nil")
	}

	newAmplitudes := make([]complex128, len(state.Amplitudes))

	for i, amp := range state.Amplitudes {
		prime := et.primeBasis[i]

		// For primes, φ(p) = p-1, so φ(p)/p = (p-1)/p
		if prime <= 1 {
			newAmplitudes[i] = amp
			continue
		}

		eulerPhi := et.primeEngine.EulerPhi(prime)
		ratio := float64(eulerPhi) / float64(prime)

		// Apply phase e^(2πi φ(n)/n)
		phase := 2.0 * math.Pi * ratio
		phaseFactor := cmplx.Exp(complex(0, phase))

		newAmplitudes[i] = amp * phaseFactor
	}

	return &hilbert.QuantumState{
		Amplitudes: newAmplitudes,
		PrimeBasis: state.PrimeBasis,
		Normalized: state.Normalized, // Phase rotations preserve normalization
	}, nil
}

// GetMatrix returns the matrix representation of the Euler transform
func (et *EulerTransform) GetMatrix(dimension int) ([][]complex128, error) {
	if dimension != len(et.primeBasis) {
		return nil, fmt.Errorf("dimension mismatch")
	}

	matrix := make([][]complex128, dimension)
	for i := range matrix {
		matrix[i] = make([]complex128, dimension)

		prime := et.primeBasis[i]
		if prime > 1 {
			eulerPhi := et.primeEngine.EulerPhi(prime)
			ratio := float64(eulerPhi) / float64(prime)
			phase := 2.0 * math.Pi * ratio
			matrix[i][i] = cmplx.Exp(complex(0, phase))
		} else {
			matrix[i][i] = complex(1, 0)
		}
	}

	return matrix, nil
}

// GetName returns the operator name
func (et *EulerTransform) GetName() string {
	return "EulerTransform"
}

// NewMobiusTransform creates a new Möbius transform operator
func NewMobiusTransform(primeBasis []int, primeEngine *primes.PrimeEngine) *MobiusTransform {
	return &MobiusTransform{
		primeBasis:  primeBasis,
		primeEngine: primeEngine,
	}
}

// Apply applies the Möbius transform M̂|n⟩ = μ(n)|n⟩
func (mt *MobiusTransform) Apply(state *hilbert.QuantumState) (*hilbert.QuantumState, error) {
	if state == nil {
		return nil, fmt.Errorf("state cannot be nil")
	}

	newAmplitudes := make([]complex128, len(state.Amplitudes))

	for i, amp := range state.Amplitudes {
		prime := mt.primeBasis[i]

		// For primes, μ(p) = -1
		mobiusValue := mt.primeEngine.MobiusFunction(prime)

		newAmplitudes[i] = amp * complex(float64(mobiusValue), 0)
	}

	return &hilbert.QuantumState{
		Amplitudes: newAmplitudes,
		PrimeBasis: state.PrimeBasis,
		Normalized: false, // May need renormalization due to sign changes
	}, nil
}

// GetMatrix returns the matrix representation of the Möbius transform
func (mt *MobiusTransform) GetMatrix(dimension int) ([][]complex128, error) {
	if dimension != len(mt.primeBasis) {
		return nil, fmt.Errorf("dimension mismatch")
	}

	matrix := make([][]complex128, dimension)
	for i := range matrix {
		matrix[i] = make([]complex128, dimension)

		prime := mt.primeBasis[i]
		mobiusValue := mt.primeEngine.MobiusFunction(prime)
		matrix[i][i] = complex(float64(mobiusValue), 0)
	}

	return matrix, nil
}

// GetName returns the operator name
func (mt *MobiusTransform) GetName() string {
	return "MobiusTransform"
}

// NewResonanceLockingOperator creates a resonance locking evolution operator
func NewResonanceLockingOperator(hamiltonian [][]complex128, resonanceOp *ResonanceOperator,
	lambda, targetResonance, dt float64) *ResonanceLockingOperator {
	return &ResonanceLockingOperator{
		hamiltonian:     hamiltonian,
		resonanceOp:     resonanceOp,
		lambda:          lambda,
		targetResonance: targetResonance,
		dt:              dt,
	}
}

// Apply applies resonance locking evolution: d/dt|Ψ_C⟩ = iĤ|Ψ_C⟩ - λ(R̂-r_stable)|Ψ_C⟩
func (rlo *ResonanceLockingOperator) Apply(state *hilbert.QuantumState) (*hilbert.QuantumState, error) {
	if state == nil {
		return nil, fmt.Errorf("state cannot be nil")
	}

	dimension := len(state.Amplitudes)
	newAmplitudes := make([]complex128, dimension)

	// Apply iĤ|Ψ⟩ term (unitary evolution)
	for i := range newAmplitudes {
		for j := range state.Amplitudes {
			hamiltonianTerm := complex(0, 1) * rlo.hamiltonian[i][j] * state.Amplitudes[j]
			newAmplitudes[i] += hamiltonianTerm
		}
	}

	// Apply resonance locking term: -λ(R̂-r_stable)|Ψ⟩
	resonanceState, err := rlo.resonanceOp.Apply(state)
	if err != nil {
		return nil, fmt.Errorf("failed to apply resonance operator: %w", err)
	}

	for i := range newAmplitudes {
		// (R̂-r_stable)|Ψ⟩ = R̂|Ψ⟩ - r_stable|Ψ⟩
		resonanceTerm := resonanceState.Amplitudes[i] -
			complex(rlo.targetResonance, 0)*state.Amplitudes[i]

		lockingTerm := complex(-rlo.lambda, 0) * resonanceTerm
		newAmplitudes[i] += lockingTerm
	}

	// Apply time step: |Ψ(t+dt)⟩ = |Ψ(t)⟩ + dt * d/dt|Ψ⟩
	for i := range newAmplitudes {
		newAmplitudes[i] = state.Amplitudes[i] + complex(rlo.dt, 0)*newAmplitudes[i]
	}

	return &hilbert.QuantumState{
		Amplitudes: newAmplitudes,
		PrimeBasis: state.PrimeBasis,
		Normalized: false, // Needs renormalization after evolution
	}, nil
}

// GetMatrix returns the effective evolution matrix (complex due to non-unitary evolution)
func (rlo *ResonanceLockingOperator) GetMatrix(dimension int) ([][]complex128, error) {
	if len(rlo.hamiltonian) != dimension {
		return nil, fmt.Errorf("hamiltonian dimension mismatch")
	}

	// Get resonance operator matrix
	resonanceMatrix, err := rlo.resonanceOp.GetMatrix(dimension)
	if err != nil {
		return nil, fmt.Errorf("failed to get resonance matrix: %w", err)
	}

	// Construct evolution matrix: I + dt * (iH - λ(R - r_stable*I))
	matrix := make([][]complex128, dimension)
	for i := range matrix {
		matrix[i] = make([]complex128, dimension)

		for j := range matrix[i] {
			// Identity term
			if i == j {
				matrix[i][j] = complex(1, 0)
			}

			// dt * iH term
			hamiltonianTerm := complex(0, rlo.dt) * rlo.hamiltonian[i][j]
			matrix[i][j] += hamiltonianTerm

			// dt * (-λ)(R - r_stable*I) term
			resonanceTerm := resonanceMatrix[i][j]
			if i == j {
				resonanceTerm -= complex(rlo.targetResonance, 0)
			}
			lockingTerm := complex(-rlo.dt*rlo.lambda, 0) * resonanceTerm
			matrix[i][j] += lockingTerm
		}
	}

	return matrix, nil
}

// GetName returns the operator name
func (rlo *ResonanceLockingOperator) GetName() string {
	return fmt.Sprintf("ResonanceLockingOperator(λ=%.3f, r_target=%.3f)",
		rlo.lambda, rlo.targetResonance)
}

// CompositeOperator allows chaining multiple operators
type CompositeOperator struct {
	operators []Operator
	name      string
}

// NewCompositeOperator creates a composite operator from a sequence of operators
func NewCompositeOperator(operators []Operator, name string) *CompositeOperator {
	return &CompositeOperator{
		operators: operators,
		name:      name,
	}
}

// Apply applies all operators in sequence
func (co *CompositeOperator) Apply(state *hilbert.QuantumState) (*hilbert.QuantumState, error) {
	currentState := state
	var err error

	for i, op := range co.operators {
		currentState, err = op.Apply(currentState)
		if err != nil {
			return nil, fmt.Errorf("operator %d (%s) failed: %w", i, op.GetName(), err)
		}
	}

	return currentState, nil
}

// GetMatrix returns the product of all operator matrices
func (co *CompositeOperator) GetMatrix(dimension int) ([][]complex128, error) {
	if len(co.operators) == 0 {
		// Return identity matrix
		matrix := make([][]complex128, dimension)
		for i := range matrix {
			matrix[i] = make([]complex128, dimension)
			matrix[i][i] = complex(1, 0)
		}
		return matrix, nil
	}

	// Get first operator matrix
	result, err := co.operators[0].GetMatrix(dimension)
	if err != nil {
		return nil, fmt.Errorf("failed to get matrix for operator 0: %w", err)
	}

	// Multiply with subsequent operator matrices
	for i := 1; i < len(co.operators); i++ {
		nextMatrix, err := co.operators[i].GetMatrix(dimension)
		if err != nil {
			return nil, fmt.Errorf("failed to get matrix for operator %d: %w", i, err)
		}

		result = multiplyMatrices(result, nextMatrix)
	}

	return result, nil
}

// GetName returns the composite operator name
func (co *CompositeOperator) GetName() string {
	return co.name
}

// multiplyMatrices performs complex matrix multiplication
func multiplyMatrices(a, b [][]complex128) [][]complex128 {
	rows := len(a)
	cols := len(b[0])
	result := make([][]complex128, rows)

	for i := range result {
		result[i] = make([]complex128, cols)
		for j := range result[i] {
			for k := range b {
				result[i][j] += a[i][k] * b[k][j]
			}
		}
	}

	return result
}
