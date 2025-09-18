package core

import (
	"fmt"
	"math"
	"time"
)

// PrimeOscillator implements prime-indexed oscillators with fᵢ ∝ 1/pᵢ from Reson.net paper
type PrimeOscillator struct {
	// Prime number index (pᵢ from paper)
	Prime int `json:"prime"`

	// Frequency fᵢ ∝ 1/pᵢ from paper
	Frequency float64 `json:"frequency"`

	// Amplitude of oscillation
	Amplitude float64 `json:"amplitude"`

	// Current phase φ(t)
	Phase float64 `json:"phase"`

	// Damping factor λ from entropy model
	Damping float64 `json:"damping"`

	// Initial phase offset
	PhaseOffset float64 `json:"phase_offset"`

	// Timestamp of last update
	LastUpdate time.Time `json:"last_update"`

	// Associated quaternionic state (optional)
	QuaternionicState *QuaternionicState `json:"quaternionic_state,omitempty"`
}

// NewPrimeOscillator creates a new prime oscillator with given prime number
func NewPrimeOscillator(prime int, amplitude, phaseOffset, damping float64) *PrimeOscillator {
	if prime <= 1 {
		panic("prime must be greater than 1")
	}

	// Frequency ∝ 1/pᵢ as per paper
	frequency := 1.0 / float64(prime)

	return &PrimeOscillator{
		Prime:       prime,
		Frequency:   frequency,
		Amplitude:   amplitude,
		Phase:       phaseOffset,
		Damping:     damping,
		PhaseOffset: phaseOffset,
		LastUpdate:  time.Now(),
	}
}

// UpdatePhase evolves the oscillator phase over time
// φ(t+Δt) = φ(t) + 2π·fᵢ·Δt - λ·φ(t)·Δt (damped evolution)
func (po *PrimeOscillator) UpdatePhase(deltaTime float64) {
	// Natural frequency evolution: 2π·fᵢ·Δt
	frequencyTerm := 2.0 * math.Pi * po.Frequency * deltaTime

	// Damping term: -λ·φ(t)·Δt
	dampingTerm := -po.Damping * po.Phase * deltaTime

	// Update phase
	po.Phase += frequencyTerm + dampingTerm

	// Keep phase in reasonable range [-π, π]
	for po.Phase > math.Pi {
		po.Phase -= 2.0 * math.Pi
	}
	for po.Phase < -math.Pi {
		po.Phase += 2.0 * math.Pi
	}

	po.LastUpdate = time.Now()
}

// GetAmplitude returns the current amplitude (may include damping)
func (po *PrimeOscillator) GetAmplitude() float64 {
	// Apply amplitude damping over time
	timeSinceStart := time.Since(po.LastUpdate).Seconds()
	dampingFactor := math.Exp(-po.Damping * timeSinceStart)

	return po.Amplitude * dampingFactor
}

// GetComplexAmplitude returns the complex amplitude A·exp(iφ)
func (po *PrimeOscillator) GetComplexAmplitude() complex128 {
	amplitude := po.GetAmplitude()
	return complex(amplitude*math.Cos(po.Phase), amplitude*math.Sin(po.Phase))
}

// ComputeResonance computes resonance with another oscillator
// Based on phase difference and frequency relationship
func (po *PrimeOscillator) ComputeResonance(other *PrimeOscillator) float64 {
	if other == nil {
		return 0.0
	}

	// Phase difference contribution
	phaseDiff := math.Abs(po.Phase - other.Phase)
	phaseResonance := math.Cos(phaseDiff)

	// Frequency ratio contribution (harmonic relationships)
	freqRatio := po.Frequency / other.Frequency
	harmonicResonance := math.Cos(2.0 * math.Pi * math.Log(freqRatio))

	// Prime relationship contribution
	primeRatio := float64(po.Prime) / float64(other.Prime)
	primeResonance := 1.0 / (1.0 + math.Abs(math.Log(primeRatio)))

	// Combined resonance (weighted average)
	return 0.4*phaseResonance + 0.3*harmonicResonance + 0.3*primeResonance
}

// SynchronizeWith attempts to synchronize this oscillator with another
func (po *PrimeOscillator) SynchronizeWith(other *PrimeOscillator, couplingStrength float64) {
	if other == nil {
		return
	}

	// Phase coupling
	phaseDiff := other.Phase - po.Phase
	po.Phase += couplingStrength * phaseDiff

	// Frequency coupling (weaker)
	freqDiff := other.Frequency - po.Frequency
	po.Frequency += 0.1 * couplingStrength * freqDiff

	// Keep frequency positive
	if po.Frequency <= 0 {
		po.Frequency = 1.0 / float64(po.Prime)
	}

	po.LastUpdate = time.Now()
}

// Clone creates a deep copy of the prime oscillator
func (po *PrimeOscillator) Clone() *PrimeOscillator {
	clone := &PrimeOscillator{
		Prime:       po.Prime,
		Frequency:   po.Frequency,
		Amplitude:   po.Amplitude,
		Phase:       po.Phase,
		Damping:     po.Damping,
		PhaseOffset: po.PhaseOffset,
		LastUpdate:  po.LastUpdate,
	}

	if po.QuaternionicState != nil {
		clone.QuaternionicState = po.QuaternionicState.Clone()
	}

	return clone
}

// String returns a string representation of the prime oscillator
func (po *PrimeOscillator) String() string {
	complexAmp := po.GetComplexAmplitude()
	return fmt.Sprintf("PrimeOscillator{p=%d, f=%.6f, A=%.3f, φ=%.3f, ψ=%.3f+%.3fi}",
		po.Prime, po.Frequency, po.GetAmplitude(), po.Phase, real(complexAmp), imag(complexAmp))
}

// Validate checks if the oscillator is in a valid state
func (po *PrimeOscillator) Validate() error {
	if po.Prime <= 1 {
		return fmt.Errorf("prime must be greater than 1, got %d", po.Prime)
	}

	if po.Frequency <= 0 {
		return fmt.Errorf("frequency must be positive, got %f", po.Frequency)
	}

	if po.Amplitude < 0 {
		return fmt.Errorf("amplitude must be non-negative, got %f", po.Amplitude)
	}

	if po.Damping < 0 {
		return fmt.Errorf("damping must be non-negative, got %f", po.Damping)
	}

	return nil
}

// CreatePrimeOscillatorSet creates a set of oscillators for given primes
func CreatePrimeOscillatorSet(primes []int, baseAmplitude, damping float64) []*PrimeOscillator {
	oscillators := make([]*PrimeOscillator, len(primes))

	for i, prime := range primes {
		// Stagger initial phases for diversity
		phaseOffset := 2.0 * math.Pi * float64(i) / float64(len(primes))
		oscillators[i] = NewPrimeOscillator(prime, baseAmplitude, phaseOffset, damping)
	}

	return oscillators
}

// ComputeGlobalCoherence calculates coherence across a set of oscillators
func ComputeGlobalCoherence(oscillators []*PrimeOscillator) float64 {
	if len(oscillators) <= 1 {
		return 1.0
	}

	totalCoherence := 0.0
	pairCount := 0

	for i := 0; i < len(oscillators); i++ {
		for j := i + 1; j < len(oscillators); j++ {
			resonance := oscillators[i].ComputeResonance(oscillators[j])
			totalCoherence += resonance
			pairCount++
		}
	}

	if pairCount > 0 {
		return totalCoherence / float64(pairCount)
	}

	return 0.0
}
