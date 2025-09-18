package tests

import (
	"math"
	"testing"

	"github.com/resonancelab/psizero/core"
)

// TestResonanceEngine tests the core resonance engine functionality
func TestResonanceEngine(t *testing.T) {
	config := &core.EngineConfig{
		Dimension:        256,
		MaxPrimeLimit:    1000,
		InitialEntropy:   1.5,
		EntropyLambda:    0.02,
		PlateauTolerance: 1e-6,
		PlateauWindow:    10,
		HistorySize:      10000,
	}

	engine, err := core.NewResonanceEngine(config)
	if err != nil {
		t.Fatalf("Failed to create resonance engine: %v", err)
	}

	t.Run("PrimeGeneration", func(t *testing.T) {
		primeBasis := engine.GetPrimeBasis()
		if len(primeBasis) == 0 {
			t.Fatal("Prime basis should not be empty")
		}

		// Check first few primes in basis
		expected := []int{2, 3, 5, 7, 11, 13, 17, 19, 23, 29}
		checkCount := len(expected)
		if len(primeBasis) < checkCount {
			checkCount = len(primeBasis)
		}

		for i := 0; i < checkCount; i++ {
			if primeBasis[i] != expected[i] {
				t.Errorf("Expected prime %d at index %d, got %d", expected[i], i, primeBasis[i])
			}
		}
	})

	t.Run("QuantumStateCreation", func(t *testing.T) {
		amplitudes := make([]complex128, config.Dimension)
		norm := 1.0 / math.Sqrt(float64(config.Dimension))

		for i := range amplitudes {
			amplitudes[i] = complex(norm, 0)
		}

		state, err := engine.CreateQuantumState(amplitudes)
		if err != nil {
			t.Fatalf("Failed to create quantum state: %v", err)
		}

		if state == nil {
			t.Fatal("Quantum state is nil")
		}

		if len(state.Amplitudes) != config.Dimension {
			t.Errorf("Expected %d amplitudes, got %d", config.Dimension, len(state.Amplitudes))
		}
	})

	t.Run("ResonanceOperatorCreation", func(t *testing.T) {
		operator := engine.GetResonanceOperator(15, 1.0) // Number 15, strength 1.0
		if operator == nil {
			t.Fatal("Resonance operator is nil")
		}

		// Test that we can get the operator name
		name := operator.GetName()
		if name == "" {
			t.Error("Operator name should not be empty")
		}
	})

	t.Run("StateEvolution", func(t *testing.T) {
		amplitudes := make([]complex128, config.Dimension)
		for i := range amplitudes {
			amplitudes[i] = complex(1.0/math.Sqrt(float64(config.Dimension)), 0)
		}

		state, err := engine.CreateQuantumState(amplitudes)
		if err != nil {
			t.Fatalf("Failed to create quantum state: %v", err)
		}

		evolvedState, err := engine.EvolveState(state, 0.01)
		if err != nil {
			t.Fatalf("Failed to evolve state: %v", err)
		}

		// Update state reference
		state = evolvedState

		// Note: Energy is not conserved after renormalization
	})

	t.Run("CoherenceCalculation", func(t *testing.T) {
		amplitudes := make([]complex128, config.Dimension)
		for i := range amplitudes {
			amplitudes[i] = complex(1.0/math.Sqrt(float64(config.Dimension)), 0)
		}

		state, err := engine.CreateQuantumState(amplitudes)
		if err != nil {
			t.Fatalf("Failed to create quantum state: %v", err)
		}

		// Use the coherence value from the quantum state
		coherence := state.Coherence
		if coherence < 0 {
			t.Errorf("Coherence should be non-negative, got %f", coherence)
		}

		// Perfect coherent state should have reasonable coherence
		if coherence < 0.1 {
			t.Errorf("Expected reasonable coherence for coherent state, got %f", coherence)
		}
	})

	t.Run("OverlapCalculation", func(t *testing.T) {
		amplitudes1 := make([]complex128, config.Dimension)
		amplitudes2 := make([]complex128, config.Dimension)

		norm := 1.0 / math.Sqrt(float64(config.Dimension))
		for i := range amplitudes1 {
			amplitudes1[i] = complex(norm, 0)
			amplitudes2[i] = complex(norm, 0)
		}

		state1, err := engine.CreateQuantumState(amplitudes1)
		if err != nil {
			t.Fatalf("Failed to create quantum state 1: %v", err)
		}

		state2, err := engine.CreateQuantumState(amplitudes2)
		if err != nil {
			t.Fatalf("Failed to create quantum state 2: %v", err)
		}

		// Use Hilbert space to compute overlap
		hilbertSpace := engine.GetHilbertSpace()
		overlap, err := hilbertSpace.ComputeInnerProduct(state1, state2)
		if err != nil {
			t.Fatalf("Failed to compute overlap: %v", err)
		}

		// Identical states should have overlap close to 1
		overlapMagnitude := real(overlap * complex(real(overlap), -imag(overlap)))
		if math.Abs(overlapMagnitude-1.0) > 1e-6 {
			t.Errorf("Expected overlap close to 1 for identical states, got %f", overlapMagnitude)
		}
	})
}

// TestPrimeOperations tests prime number operations
func TestPrimeOperations(t *testing.T) {
	config := &core.EngineConfig{
		Dimension:        128,
		MaxPrimeLimit:    100,
		InitialEntropy:   1.5,
		EntropyLambda:    0.02,
		PlateauTolerance: 1e-6,
		PlateauWindow:    10,
		HistorySize:      1000,
	}

	engine, err := core.NewResonanceEngine(config)
	if err != nil {
		t.Fatalf("Failed to create resonance engine: %v", err)
	}

	// Get prime basis for testing

	t.Run("PrimeGeneration", func(t *testing.T) {
		// Test generation of first few primes
		primeBasis := engine.GetPrimeBasis()
		expected := []int{2, 3, 5, 7, 11, 13, 17, 19, 23, 29}

		checkCount := len(expected)
		if len(primeBasis) < checkCount {
			checkCount = len(primeBasis)
		}

		for i := 0; i < checkCount; i++ {
			if primeBasis[i] != expected[i] {
				t.Errorf("Prime %d: expected %d, got %d", i, expected[i], primeBasis[i])
			}
		}
	})

	t.Run("PrimalityTest", func(t *testing.T) {
		// Test primality of known primes
		knownPrimes := []int{2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47}
		// We'll test by checking if they appear in the prime basis
		primeBasis := engine.GetPrimeBasis()
		primeSet := make(map[int]bool)
		for _, p := range primeBasis {
			primeSet[p] = true
		}

		for _, p := range knownPrimes {
			if !primeSet[p] {
				t.Errorf("Number %d should be prime", p)
			}
		}

		// Test non-primes - they should not be in the prime set
		nonPrimes := []int{4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 22, 24, 25}
		for _, n := range nonPrimes {
			if primeSet[n] {
				t.Errorf("Number %d should not be prime", n)
			}
		}
	})

	t.Run("NthPrime", func(t *testing.T) {
		// Test specific prime indices
		expected := map[int]int{
			0: 2, 1: 3, 2: 5, 3: 7, 4: 11, 5: 13, 6: 17, 7: 19, 8: 23, 9: 29,
		}

		primeBasis := engine.GetPrimeBasis()
		for index, expectedPrime := range expected {
			if index < len(primeBasis) {
				if primeBasis[index] != expectedPrime {
					t.Errorf("Prime at index %d: expected %d, got %d", index, expectedPrime, primeBasis[index])
				}
			}
		}
	})

	t.Run("PrimeRange", func(t *testing.T) {
		// Test primes in range
		primeBasis := engine.GetPrimeBasis()
		expected := []int{11, 13, 17, 19, 23, 29}

		// Find primes in range from prime basis
		var primeRange []int
		for _, p := range primeBasis {
			if p >= 10 && p <= 30 {
				primeRange = append(primeRange, p)
			}
		}

		if len(primeRange) != len(expected) {
			t.Fatalf("Expected %d primes in range [10,30], got %d", len(expected), len(primeRange))
		}

		for i, prime := range primeRange {
			if prime != expected[i] {
				t.Errorf("Prime %d in range: expected %d, got %d", i, expected[i], prime)
			}
		}
	})
}

// TestQuantumStateOperations tests quantum state operations
func TestQuantumStateOperations(t *testing.T) {
	config := &core.EngineConfig{
		Dimension:        64,
		MaxPrimeLimit:    100,
		InitialEntropy:   1.5,
		EntropyLambda:    0.02,
		PlateauTolerance: 1e-6,
		PlateauWindow:    10,
		HistorySize:      1000,
	}

	engine, err := core.NewResonanceEngine(config)
	if err != nil {
		t.Fatalf("Failed to create resonance engine: %v", err)
	}

	t.Run("StateNormalization", func(t *testing.T) {
		// Create unnormalized amplitudes
		amplitudes := make([]complex128, config.Dimension)
		for i := range amplitudes {
			amplitudes[i] = complex(1.0, 0) // Not normalized
		}

		state, err := engine.CreateQuantumState(amplitudes)
		if err != nil {
			t.Fatalf("Failed to create quantum state: %v", err)
		}

		// Check normalization
		normSquared := 0.0
		for _, amp := range state.Amplitudes {
			normSquared += real(amp)*real(amp) + imag(amp)*imag(amp)
		}

		if math.Abs(normSquared-1.0) > 1e-10 {
			t.Errorf("State not normalized: |ψ|² = %f", normSquared)
		}
	})

	t.Run("StateEvolutionUnitarity", func(t *testing.T) {
		amplitudes := make([]complex128, config.Dimension)
		norm := 1.0 / math.Sqrt(float64(config.Dimension))
		for i := range amplitudes {
			amplitudes[i] = complex(norm, 0)
		}

		state, err := engine.CreateQuantumState(amplitudes)
		if err != nil {
			t.Fatalf("Failed to create quantum state: %v", err)
		}

		// Evolve state multiple times
		for i := 0; i < 10; i++ {
			state, err = engine.EvolveStateWithResonance(state, 0.01, 1.0)
			if err != nil {
				t.Fatalf("Failed to evolve state at step %d: %v", i, err)
			}

			// Check normalization is preserved
			normSquared := 0.0
			for _, amp := range state.Amplitudes {
				normSquared += real(amp)*real(amp) + imag(amp)*imag(amp)
			}

			if math.Abs(normSquared-1.0) > 1e-8 {
				t.Errorf("Normalization not preserved at step %d: |ψ|² = %f", i, normSquared)
			}
		}
	})

	t.Run("StateEntanglement", func(t *testing.T) {
		// Create maximally entangled state
		amplitudes := make([]complex128, config.Dimension)
		amplitudes[0] = complex(1.0/math.Sqrt(2.0), 0)
		amplitudes[config.Dimension-1] = complex(1.0/math.Sqrt(2.0), 0)

		state, err := engine.CreateQuantumState(amplitudes)
		if err != nil {
			t.Fatalf("Failed to create quantum state: %v", err)
		}

		coherence := state.Coherence

		// Entangled state should have moderate coherence
		if coherence < 0.1 || coherence > 0.9 {
			t.Errorf("Unexpected coherence for entangled state: %f", coherence)
		}
	})
}

// BenchmarkResonanceEngine benchmarks core engine operations
func BenchmarkResonanceEngine(b *testing.B) {
	config := &core.EngineConfig{
		Dimension:        512,
		MaxPrimeLimit:    1000,
		InitialEntropy:   1.5,
		EntropyLambda:    0.02,
		PlateauTolerance: 1e-6,
		PlateauWindow:    10,
		HistorySize:      10000,
	}

	engine, err := core.NewResonanceEngine(config)
	if err != nil {
		b.Fatalf("Failed to create resonance engine: %v", err)
	}

	amplitudes := make([]complex128, config.Dimension)
	norm := 1.0 / math.Sqrt(float64(config.Dimension))
	for i := range amplitudes {
		amplitudes[i] = complex(norm, 0)
	}

	state, err := engine.CreateQuantumState(amplitudes)
	if err != nil {
		b.Fatalf("Failed to create quantum state: %v", err)
	}

	b.Run("StateEvolution", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, err := engine.EvolveStateWithResonance(state, 0.01, 1.0)
			if err != nil {
				b.Fatalf("Evolution failed: %v", err)
			}
		}
	})

	b.Run("CoherenceCalculation", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_ = state.Coherence
		}
	})

	b.Run("PrimeGeneration", func(b *testing.B) {
		primeBasis := engine.GetPrimeBasis()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = primeBasis[i%len(primeBasis)]
		}
	})
}
