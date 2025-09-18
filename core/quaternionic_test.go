package core

import (
	"math"
	"testing"
	"time"
)

func TestQuaternionicState(t *testing.T) {
	// Test basic quaternionic state creation
	position := []float64{1.0, 2.0, 3.0}
	baseAmplitude := complex(1.0, 0.5)
	gaussian := []float64{1.0, 2.0}
	eisenstein := []float64{3.0, 4.0}

	qs := NewQuaternionicState(position, baseAmplitude, gaussian, eisenstein)

	// Test validation
	if err := qs.Validate(); err != nil {
		t.Fatalf("QuaternionicState validation failed: %v", err)
	}

	// Test amplitude calculation ψq(x,t) = N⁻¹ψ̄q(x)·exp(iφ(x,t))
	amplitude := qs.ComputeQuaternionicAmplitude()
	if amplitude == 0 {
		t.Error("Quaternionic amplitude should not be zero")
	}

	// Test phase evolution
	initialPhase := qs.Phase
	qs.UpdatePhase(0.1, 1.0)
	if qs.Phase == initialPhase {
		t.Error("Phase should evolve over time")
	}
}

func TestPrimeOscillator(t *testing.T) {
	// Test oscillator creation
	osc := NewPrimeOscillator(13, 0.7, 1.0, 0.02)

	// Test validation
	if err := osc.Validate(); err != nil {
		t.Fatalf("PrimeOscillator validation failed: %v", err)
	}

	// Test frequency relationship f ∝ 1/p
	expectedFreq := 1.0 / 13.0
	if math.Abs(osc.Frequency-expectedFreq) > 1e-6 {
		t.Errorf("Expected frequency %.6f, got %.6f", expectedFreq, osc.Frequency)
	}

	// Test phase evolution
	initialPhase := osc.Phase
	osc.UpdatePhase(0.1)
	if osc.Phase == initialPhase {
		t.Error("Oscillator phase should evolve")
	}
}

func TestGlobalPhaseState(t *testing.T) {
	// Create global phase state
	gps := NewGlobalPhaseState(0.8, time.Second)

	// Add oscillators
	osc1 := NewPrimeOscillator(13, 0.7, 0.0, 0.02)
	osc2 := NewPrimeOscillator(17, 0.5, math.Pi/2, 0.02)

	if err := gps.AddOscillator(osc1); err != nil {
		t.Fatalf("Failed to add oscillator 1: %v", err)
	}
	if err := gps.AddOscillator(osc2); err != nil {
		t.Fatalf("Failed to add oscillator 2: %v", err)
	}

	// Test coherence calculation
	coherence := gps.UpdateGlobalCoherence()
	if coherence < -1.0 || coherence > 1.0 {
		t.Errorf("Coherence should be in [-1, 1], got %f", coherence)
	}

	// Test phase evolution
	gps.EvolvePhaseState(0.1)

	// Test synchronization
	gps.SynchronizePhases(0.1)

	// Test phase locking check
	locked := gps.CheckPhaseLocking()
	// Note: May not be locked initially, that's okay
	_ = locked
}

func TestResonanceComputation(t *testing.T) {
	osc1 := NewPrimeOscillator(13, 0.7, 0.0, 0.02)
	osc2 := NewPrimeOscillator(17, 0.5, 0.1, 0.02)

	resonance := osc1.ComputeResonance(osc2)
	if resonance < 0 || resonance > 1 {
		t.Errorf("Resonance should be in [0, 1], got %f", resonance)
	}
}

func TestQuaternionicCoherence(t *testing.T) {
	// Create multiple quaternionic states
	qs1 := NewQuaternionicState([]float64{1.0, 0.0}, complex(1.0, 0.0), []float64{1.0}, []float64{1.0})
	qs2 := NewQuaternionicState([]float64{0.0, 1.0}, complex(0.8, 0.2), []float64{2.0}, []float64{2.0})
	qs3 := NewQuaternionicState([]float64{1.0, 1.0}, complex(0.9, 0.1), []float64{1.5}, []float64{1.5})

	states := []*QuaternionicState{qs1, qs2, qs3}

	// Test coherence measurement
	weights := [][]float64{
		{0, 0.5, 0.5},
		{0.5, 0, 0.5},
		{0.5, 0.5, 0},
	}

	coherence := qs1.MeasureCoherence(states[1:], weights)
	if coherence < -1.0 || coherence > 1.0 {
		t.Errorf("Coherence should be in [-1, 1], got %f", coherence)
	}
}

func TestPhaseDifference(t *testing.T) {
	qs1 := NewQuaternionicState([]float64{0.0}, complex(1.0, 0.0), []float64{}, []float64{})
	qs2 := NewQuaternionicState([]float64{0.0}, complex(1.0, 0.0), []float64{}, []float64{})

	// Set different phases
	qs1.Phase = 0.0
	qs2.Phase = math.Pi / 2

	diff := qs1.ComputePhaseDifference(qs2)
	expected := math.Pi / 2

	if math.Abs(diff-expected) > 1e-6 {
		t.Errorf("Expected phase difference %.6f, got %.6f", expected, diff)
	}
}

func BenchmarkQuaternionicEvolution(b *testing.B) {
	qs := NewQuaternionicState([]float64{1.0, 2.0}, complex(1.0, 0.5), []float64{1.0, 2.0}, []float64{3.0, 4.0})

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		qs.UpdatePhase(0.01, 1.0)
		_ = qs.ComputeQuaternionicAmplitude()
	}
}

func BenchmarkGlobalCoherence(b *testing.B) {
	gps := NewGlobalPhaseState(0.8, time.Second)

	// Add multiple oscillators
	primes := []int{13, 17, 19, 23, 29}
	for _, prime := range primes {
		osc := NewPrimeOscillator(prime, 0.7, 0.0, 0.02)
		gps.AddOscillator(osc)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		gps.UpdateGlobalCoherence()
	}
}
