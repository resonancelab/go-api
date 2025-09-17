package tests

import (
	"math"
	"testing"

	"github.com/resonancelab/psizero/api/core"
)

// TestResonanceEngine tests the core resonance engine functionality
func TestResonanceEngine(t *testing.T) {
	config := &core.ResonanceEngineConfig{
		Dimension:         256,
		MaxPrimes:         1000,
		TimeoutSeconds:    10,
		EvolutionSteps:    100,
		ResonanceStrength: 1.0,
	}

	engine, err := core.NewResonanceEngine(config)
	if err != nil {
		t.Fatalf("Failed to create resonance engine: %v", err)
	}

	t.Run("PrimeGeneration", func(t *testing.T) {
		primes := engine.GetPrimes()
		if primes == nil {
			t.Fatal("Primes generator is nil")
		}

		firstPrimes := primes.GetFirst(10)
		expected := []int{2, 3, 5, 7, 11, 13, 17, 19, 23, 29}

		if len(firstPrimes) != len(expected) {
			t.Fatalf("Expected %d primes, got %d", len(expected), len(firstPrimes))
		}

		for i, prime := range firstPrimes {
			if prime != expected[i] {
				t.Errorf("Expected prime %d at index %d, got %d", expected[i], i, prime)
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
		operator, err := engine.CreateResonanceOperator(1.0, 0.1)
		if err != nil {
			t.Fatalf("Failed to create resonance operator: %v", err)
		}

		if operator == nil {
			t.Fatal("Resonance operator is nil")
		}

		if operator.Strength != 1.0 {
			t.Errorf("Expected strength 1.0, got %f", operator.Strength)
		}

		if operator.Phase != 0.1 {
			t.Errorf("Expected phase 0.1, got %f", operator.Phase)
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

		initialEnergy := state.Energy

		err = engine.EvolveStateWithResonance(state, 0.1, 1.0)
		if err != nil {
			t.Fatalf("Failed to evolve state: %v", err)
		}

		// Energy should be conserved (approximately)
		energyDiff := math.Abs(state.Energy - initialEnergy)
		if energyDiff > 1e-10 {
			t.Errorf("Energy not conserved: initial=%f, final=%f, diff=%e",
				initialEnergy, state.Energy, energyDiff)
		}
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

		coherence := engine.CalculateCoherence(state)
		if coherence < 0 || coherence > 1 {
			t.Errorf("Coherence should be between 0 and 1, got %f", coherence)
		}

		// Perfect coherent state should have high coherence
		if coherence < 0.8 {
			t.Errorf("Expected high coherence for coherent state, got %f", coherence)
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

		overlap := engine.CalculateOverlap(state1, state2)

		// Identical states should have overlap close to 1
		if math.Abs(overlap-1.0) > 1e-10 {
			t.Errorf("Expected overlap close to 1 for identical states, got %f", overlap)
		}
	})
}

// TestPrimeOperations tests prime number operations
func TestPrimeOperations(t *testing.T) {
	config := &core.ResonanceEngineConfig{
		Dimension:         128,
		MaxPrimes:         100,
		TimeoutSeconds:    5,
		EvolutionSteps:    10,
		ResonanceStrength: 1.0,
	}

	engine, err := core.NewResonanceEngine(config)
	if err != nil {
		t.Fatalf("Failed to create resonance engine: %v", err)
	}

	primes := engine.GetPrimes()

	t.Run("PrimeGeneration", func(t *testing.T) {
		// Test generation of first few primes
		first10 := primes.GetFirst(10)
		expected := []int{2, 3, 5, 7, 11, 13, 17, 19, 23, 29}

		for i, prime := range first10 {
			if prime != expected[i] {
				t.Errorf("Prime %d: expected %d, got %d", i, expected[i], prime)
			}
		}
	})

	t.Run("PrimalityTest", func(t *testing.T) {
		// Test primality of known primes
		knownPrimes := []int{2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47}
		for _, p := range knownPrimes {
			if !primes.IsPrime(p) {
				t.Errorf("Number %d should be prime", p)
			}
		}

		// Test non-primes
		nonPrimes := []int{4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 22, 24, 25}
		for _, n := range nonPrimes {
			if primes.IsPrime(n) {
				t.Errorf("Number %d should not be prime", n)
			}
		}
	})

	t.Run("NthPrime", func(t *testing.T) {
		// Test specific prime indices
		expected := map[int]int{
			0: 2, 1: 3, 2: 5, 3: 7, 4: 11, 5: 13, 6: 17, 7: 19, 8: 23, 9: 29,
		}

		for index, expectedPrime := range expected {
			prime := primes.GetNthPrime(index)
			if prime != expectedPrime {
				t.Errorf("Prime at index %d: expected %d, got %d", index, expectedPrime, prime)
			}
		}
	})

	t.Run("PrimeRange", func(t *testing.T) {
		// Test primes in range
		primeRange := primes.GetPrimesInRange(10, 30)
		expected := []int{11, 13, 17, 19, 23, 29}

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
	config := &core.ResonanceEngineConfig{
		Dimension:         64,
		MaxPrimes:         100,
		TimeoutSeconds:    5,
		EvolutionSteps:    10,
		ResonanceStrength: 1.0,
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
			err = engine.EvolveStateWithResonance(state, 0.01, 1.0)
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

		coherence := engine.CalculateCoherence(state)

		// Entangled state should have moderate coherence
		if coherence < 0.1 || coherence > 0.9 {
			t.Errorf("Unexpected coherence for entangled state: %f", coherence)
		}
	})
}

// BenchmarkResonanceEngine benchmarks core engine operations
func BenchmarkResonanceEngine(b *testing.B) {
	config := &core.ResonanceEngineConfig{
		Dimension:         512,
		MaxPrimes:         1000,
		TimeoutSeconds:    30,
		EvolutionSteps:    100,
		ResonanceStrength: 1.0,
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
			err := engine.EvolveStateWithResonance(state, 0.01, 1.0)
			if err != nil {
				b.Fatalf("Evolution failed: %v", err)
			}
		}
	})

	b.Run("CoherenceCalculation", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_ = engine.CalculateCoherence(state)
		}
	})

	b.Run("PrimeGeneration", func(b *testing.B) {
		primes := engine.GetPrimes()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = primes.GetNthPrime(i % 100)
		}
	})
}
