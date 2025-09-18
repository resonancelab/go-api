package core

import (
	"fmt"
	"math"
	"math/cmplx"
	"time"
)

// QuaternionicState implements ψq(x,t) = N⁻¹ψ̄q(x)·exp(iφ(x,t)) from Reson.net paper
// This represents a quaternionic quantum-like state for distributed synchronization
type QuaternionicState struct {
	// Spatial coordinates (x from paper)
	Position []float64 `json:"position"`

	// Time coordinate (t from paper)
	Time float64 `json:"time"`

	// Base amplitude ψ̄q(x) from paper
	BaseAmplitude complex128 `json:"base_amplitude"`

	// Dynamic phase φ(x,t) from paper
	Phase float64 `json:"phase"`

	// Normalization factor N from paper
	NormalizationFactor float64 `json:"normalization_factor"`

	// Gaussian coordinates from paper example
	GaussianCoords []float64 `json:"gaussian_coords"`

	// Eisenstein coordinates from paper example
	EisensteinCoords []float64 `json:"eisenstein_coords"`

	// Coherence measurement C(t) from paper
	Coherence float64 `json:"coherence"`

	// Timestamp for synchronization
	Timestamp time.Time `json:"timestamp"`

	// Associated prime oscillator (for resonance)
	PrimeOscillator *PrimeOscillator `json:"prime_oscillator,omitempty"`
}

// NewQuaternionicState creates a new quaternionic state with given parameters
func NewQuaternionicState(position []float64, baseAmplitude complex128, gaussian, eisenstein []float64) *QuaternionicState {
	qs := &QuaternionicState{
		Position:         make([]float64, len(position)),
		Time:             0.0,
		BaseAmplitude:    baseAmplitude,
		Phase:            0.0,
		GaussianCoords:   make([]float64, len(gaussian)),
		EisensteinCoords: make([]float64, len(eisenstein)),
		Timestamp:        time.Now(),
	}

	copy(qs.Position, position)
	copy(qs.GaussianCoords, gaussian)
	copy(qs.EisensteinCoords, eisenstein)

	// Compute initial normalization factor
	qs.computeNormalizationFactor()

	return qs
}

// ComputeQuaternionicAmplitude calculates ψq(x,t) = N⁻¹ψ̄q(x)·exp(iφ(x,t))
func (qs *QuaternionicState) ComputeQuaternionicAmplitude() complex128 {
	// exp(iφ(x,t)) phase factor
	phaseFactor := cmplx.Exp(complex(0, qs.Phase))

	// ψq(x,t) = N⁻¹ψ̄q(x)·exp(iφ(x,t))
	return complex(qs.NormalizationFactor, 0) * qs.BaseAmplitude * phaseFactor
}

// computeNormalizationFactor calculates N from the paper
func (qs *QuaternionicState) computeNormalizationFactor() {
	// N is computed based on the magnitude of the base amplitude
	// For simplicity, we use the magnitude, but this could be more complex
	// based on the specific quaternionic algebra being used
	magnitude := cmplx.Abs(qs.BaseAmplitude)
	if magnitude > 0 {
		qs.NormalizationFactor = 1.0 / magnitude
	} else {
		qs.NormalizationFactor = 1.0
	}
}

// UpdatePhase updates the dynamic phase φ(x,t) based on time evolution
func (qs *QuaternionicState) UpdatePhase(deltaTime float64, frequency float64) {
	// Phase evolution: φ(t+Δt) = φ(t) + 2π·f·Δt
	qs.Phase += 2.0 * math.Pi * frequency * deltaTime
	qs.Time += deltaTime
	qs.Timestamp = time.Now()
}

// MeasureCoherence calculates C(t) = Σᵢⱼwᵢⱼ·cos(Φᵢ(t) - Φⱼ(t)) from paper
func (qs *QuaternionicState) MeasureCoherence(otherStates []*QuaternionicState, weights [][]float64) float64 {
	if len(otherStates) == 0 {
		return 1.0 // Perfect coherence with self
	}

	coherence := 0.0
	totalWeight := 0.0

	// Include self in coherence calculation
	allStates := append([]*QuaternionicState{qs}, otherStates...)

	for i := 0; i < len(allStates); i++ {
		for j := i + 1; j < len(allStates); j++ {
			weight := 1.0 // Default weight
			if weights != nil && i < len(weights) && j < len(weights[i]) {
				weight = weights[i][j]
			}

			// cos(Φᵢ(t) - Φⱼ(t))
			phaseDiff := allStates[i].Phase - allStates[j].Phase
			coherence += weight * math.Cos(phaseDiff)
			totalWeight += weight
		}
	}

	if totalWeight > 0 {
		coherence /= totalWeight
	}

	qs.Coherence = coherence
	return coherence
}

// ComputePhaseDifference calculates ∆ψq ∈ [0, π] from paper
func (qs *QuaternionicState) ComputePhaseDifference(other *QuaternionicState) float64 {
	if other == nil {
		return 0.0
	}

	// Phase difference
	phaseDiff := math.Abs(qs.Phase - other.Phase)

	// Normalize to [0, π] as per paper
	for phaseDiff > math.Pi {
		phaseDiff -= 2.0 * math.Pi
	}
	for phaseDiff < 0 {
		phaseDiff += 2.0 * math.Pi
	}

	return math.Abs(phaseDiff)
}

// SynchronizeWith attempts to synchronize this state with another
func (qs *QuaternionicState) SynchronizeWith(other *QuaternionicState, couplingStrength float64) {
	if other == nil {
		return
	}

	// Phase coupling: adjust phase towards the other state
	phaseDiff := other.Phase - qs.Phase
	qs.Phase += couplingStrength * phaseDiff

	// Update timestamp
	qs.Timestamp = time.Now()
}

// Clone creates a deep copy of the quaternionic state
func (qs *QuaternionicState) Clone() *QuaternionicState {
	position := make([]float64, len(qs.Position))
	copy(position, qs.Position)

	gaussian := make([]float64, len(qs.GaussianCoords))
	copy(gaussian, qs.GaussianCoords)

	eisenstein := make([]float64, len(qs.EisensteinCoords))
	copy(eisenstein, qs.EisensteinCoords)

	clone := &QuaternionicState{
		Position:            position,
		Time:                qs.Time,
		BaseAmplitude:       qs.BaseAmplitude,
		Phase:               qs.Phase,
		NormalizationFactor: qs.NormalizationFactor,
		GaussianCoords:      gaussian,
		EisensteinCoords:    eisenstein,
		Coherence:           qs.Coherence,
		Timestamp:           qs.Timestamp,
	}

	if qs.PrimeOscillator != nil {
		clone.PrimeOscillator = qs.PrimeOscillator.Clone()
	}

	return clone
}

// String returns a string representation of the quaternionic state
func (qs *QuaternionicState) String() string {
	amplitude := qs.ComputeQuaternionicAmplitude()
	return fmt.Sprintf("QuaternionicState{pos=%v, t=%.3f, ψq=%.3f+%.3fi, φ=%.3f, C=%.3f}",
		qs.Position, qs.Time, real(amplitude), imag(amplitude), qs.Phase, qs.Coherence)
}

// Validate checks if the quaternionic state is in a valid state
func (qs *QuaternionicState) Validate() error {
	if len(qs.Position) == 0 {
		return fmt.Errorf("position coordinates cannot be empty")
	}

	if qs.NormalizationFactor <= 0 {
		return fmt.Errorf("normalization factor must be positive")
	}

	if qs.Coherence < -1.0 || qs.Coherence > 1.0 {
		return fmt.Errorf("coherence must be in range [-1, 1]")
	}

	return nil
}
