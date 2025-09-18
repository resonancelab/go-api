package tests

import (
	"fmt"
	"math"
	"testing"
	"time"

	"github.com/resonancelab/psizero/core"
)

// BenchmarkResonanceEnginePerformance benchmarks core engine performance
func BenchmarkResonanceEnginePerformance(b *testing.B) {
	config := &core.EngineConfig{
		Dimension:        256,
		MaxPrimeLimit:    10000,
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

	// Benchmark quantum state creation
	b.Run("QuantumStateCreation", func(b *testing.B) {
		dimension := engine.GetDimension()
		b.ResetTimer()

		for i := 0; i < b.N; i++ {
			amplitudes := make([]complex128, dimension)
			norm := 1.0 / math.Sqrt(float64(dimension))

			for j := range amplitudes {
				amplitudes[j] = complex(norm, 0)
			}

			_, err := engine.CreateQuantumState(amplitudes)
			if err != nil {
				b.Fatalf("Failed to create quantum state: %v", err)
			}
		}
	})

	// Benchmark state evolution
	b.Run("StateEvolution", func(b *testing.B) {
		amplitudes := make([]complex128, engine.GetDimension())
		norm := 1.0 / math.Sqrt(float64(engine.GetDimension()))

		for i := range amplitudes {
			amplitudes[i] = complex(norm, 0)
		}

		state, err := engine.CreateQuantumState(amplitudes)
		if err != nil {
			b.Fatalf("Failed to create quantum state: %v", err)
		}

		b.ResetTimer()

		for i := 0; i < b.N; i++ {
			_, err := engine.EvolveState(state, 0.01)
			if err != nil {
				b.Fatalf("Failed to evolve state: %v", err)
			}
		}
	})

	// Benchmark resonance operator creation
	b.Run("ResonanceOperatorCreation", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			operator := engine.GetResonanceOperator(15+i%100, 1.0)
			if operator == nil {
				b.Fatal("Failed to create resonance operator")
			}
		}
	})

	// Benchmark prime basis access
	b.Run("PrimeBasisAccess", func(b *testing.B) {
		primes := engine.GetPrimeBasis()
		b.ResetTimer()

		for i := 0; i < b.N; i++ {
			_ = primes[i%len(primes)]
		}
	})
}

// BenchmarkDistributedOperations benchmarks distributed system performance
func BenchmarkDistributedOperations(b *testing.B) {
	config := &core.EngineConfig{
		Dimension:        128,
		MaxPrimeLimit:    1000,
		InitialEntropy:   1.5,
		EntropyLambda:    0.02,
		PlateauTolerance: 1e-6,
		PlateauWindow:    10,
		HistorySize:      1000,
	}

	engine, err := core.NewResonanceEngine(config)
	if err != nil {
		b.Fatalf("Failed to create resonance engine: %v", err)
	}

	// Benchmark concurrent state operations
	b.Run("ConcurrentStateOperations", func(b *testing.B) {
		b.RunParallel(func(pb *testing.PB) {
			localEngine, _ := core.NewResonanceEngine(config)

			for pb.Next() {
				amplitudes := make([]complex128, localEngine.GetDimension())
				norm := 1.0 / math.Sqrt(float64(localEngine.GetDimension()))

				for i := range amplitudes {
					amplitudes[i] = complex(norm, 0)
				}

				state, err := localEngine.CreateQuantumState(amplitudes)
				if err != nil {
					b.Fatalf("Failed to create quantum state: %v", err)
				}

				_, err = localEngine.EvolveState(state, 0.01)
				if err != nil {
					b.Fatalf("Failed to evolve state: %v", err)
				}
			}
		})
	})

	// Benchmark telemetry collection
	b.Run("TelemetryCollection", func(b *testing.B) {
		amplitudes := make([]complex128, engine.GetDimension())
		norm := 1.0 / math.Sqrt(float64(engine.GetDimension()))

		for i := range amplitudes {
			amplitudes[i] = complex(norm, 0)
		}

		state, err := engine.CreateQuantumState(amplitudes)
		if err != nil {
			b.Fatalf("Failed to create quantum state: %v", err)
		}

		b.ResetTimer()

		for i := 0; i < b.N; i++ {
			metrics := engine.ComputeStateMetrics(state)
			if len(metrics) == 0 {
				b.Fatal("No metrics computed")
			}
		}
	})
}

// BenchmarkScalability tests system scalability
func BenchmarkScalability(b *testing.B) {
	sizes := []int{64, 128, 256, 512}

	for _, size := range sizes {
		b.Run(fmt.Sprintf("Dimension_%d", size), func(b *testing.B) {
			config := &core.EngineConfig{
				Dimension:        size,
				MaxPrimeLimit:    1000,
				InitialEntropy:   1.5,
				EntropyLambda:    0.02,
				PlateauTolerance: 1e-6,
				PlateauWindow:    10,
				HistorySize:      1000,
			}

			engine, err := core.NewResonanceEngine(config)
			if err != nil {
				b.Fatalf("Failed to create resonance engine: %v", err)
			}

			amplitudes := make([]complex128, size)
			norm := 1.0 / math.Sqrt(float64(size))

			for i := range amplitudes {
				amplitudes[i] = complex(norm, 0)
			}

			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				state, err := engine.CreateQuantumState(amplitudes)
				if err != nil {
					b.Fatalf("Failed to create quantum state: %v", err)
				}

				_, err = engine.EvolveState(state, 0.01)
				if err != nil {
					b.Fatalf("Failed to evolve state: %v", err)
				}
			}
		})
	}
}

// BenchmarkMemoryUsage benchmarks memory efficiency
func BenchmarkMemoryUsage(b *testing.B) {
	config := &core.EngineConfig{
		Dimension:        256,
		MaxPrimeLimit:    1000,
		InitialEntropy:   1.5,
		EntropyLambda:    0.02,
		PlateauTolerance: 1e-6,
		PlateauWindow:    10,
		HistorySize:      1000,
	}

	engine, err := core.NewResonanceEngine(config)
	if err != nil {
		b.Fatalf("Failed to create resonance engine: %v", err)
	}

	b.Run("StateCreationMemory", func(b *testing.B) {
		b.ReportAllocs()

		for i := 0; i < b.N; i++ {
			amplitudes := make([]complex128, engine.GetDimension())
			norm := 1.0 / math.Sqrt(float64(engine.GetDimension()))

			for j := range amplitudes {
				amplitudes[j] = complex(norm, 0)
			}

			_, err := engine.CreateQuantumState(amplitudes)
			if err != nil {
				b.Fatalf("Failed to create quantum state: %v", err)
			}
		}
	})

	b.Run("EvolutionMemory", func(b *testing.B) {
		b.ReportAllocs()

		amplitudes := make([]complex128, engine.GetDimension())
		norm := 1.0 / math.Sqrt(float64(engine.GetDimension()))

		for i := range amplitudes {
			amplitudes[i] = complex(norm, 0)
		}

		state, err := engine.CreateQuantumState(amplitudes)
		if err != nil {
			b.Fatalf("Failed to create quantum state: %v", err)
		}

		b.ResetTimer()

		for i := 0; i < b.N; i++ {
			_, err := engine.EvolveState(state, 0.01)
			if err != nil {
				b.Fatalf("Failed to evolve state: %v", err)
			}
		}
	})
}

// TestPerformanceMetrics tests performance against paper specifications
func TestPerformanceMetrics(t *testing.T) {
	config := &core.EngineConfig{
		Dimension:        256,
		MaxPrimeLimit:    1000,
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

	// Test state creation performance
	t.Run("StateCreationPerformance", func(t *testing.T) {
		start := time.Now()

		// Create 100 states
		for i := 0; i < 100; i++ {
			amplitudes := make([]complex128, engine.GetDimension())
			norm := 1.0 / math.Sqrt(float64(engine.GetDimension()))

			for j := range amplitudes {
				amplitudes[j] = complex(norm, 0)
			}

			_, err := engine.CreateQuantumState(amplitudes)
			if err != nil {
				t.Fatalf("Failed to create quantum state: %v", err)
			}
		}

		duration := time.Since(start)
		avgTime := duration / 100

		// Should be able to create states in under 1ms each
		if avgTime > time.Millisecond {
			t.Errorf("State creation too slow: %v per state", avgTime)
		}

		t.Logf("State creation: 100 states in %v (avg: %v)", duration, avgTime)
	})

	// Test evolution performance
	t.Run("EvolutionPerformance", func(t *testing.T) {
		amplitudes := make([]complex128, engine.GetDimension())
		norm := 1.0 / math.Sqrt(float64(engine.GetDimension()))

		for i := range amplitudes {
			amplitudes[i] = complex(norm, 0)
		}

		state, err := engine.CreateQuantumState(amplitudes)
		if err != nil {
			t.Fatalf("Failed to create quantum state: %v", err)
		}

		start := time.Now()

		// Evolve state 1000 times
		for i := 0; i < 1000; i++ {
			state, err = engine.EvolveState(state, 0.01)
			if err != nil {
				t.Fatalf("Failed to evolve state: %v", err)
			}
		}

		duration := time.Since(start)
		avgTime := duration / 1000

		// Should be able to evolve states in under 1ms each (relaxed for complex operations)
		if avgTime > time.Millisecond {
			t.Errorf("State evolution too slow: %v per evolution", avgTime)
		}

		t.Logf("State evolution: 1000 evolutions in %v (avg: %v)", duration, avgTime)
	})

	// Test coherence calculation performance
	t.Run("CoherenceCalculationPerformance", func(t *testing.T) {
		amplitudes := make([]complex128, engine.GetDimension())
		norm := 1.0 / math.Sqrt(float64(engine.GetDimension()))

		for i := range amplitudes {
			amplitudes[i] = complex(norm, 0)
		}

		state, err := engine.CreateQuantumState(amplitudes)
		if err != nil {
			t.Fatalf("Failed to create quantum state: %v", err)
		}

		start := time.Now()

		// Calculate coherence 1000 times
		for i := 0; i < 1000; i++ {
			metrics := engine.ComputeStateMetrics(state)
			if coherence, exists := metrics["coherence"]; !exists {
				t.Errorf("Coherence metric missing")
			} else {
				// For now, just ensure coherence is a valid number
				if math.IsNaN(coherence) || math.IsInf(coherence, 0) {
					t.Errorf("Invalid coherence value: %v", coherence)
				}
			}
		}

		duration := time.Since(start)
		avgTime := duration / 1000

		// Should be able to calculate coherence in under 100μs each (relaxed for complex calculations)
		if avgTime > 100*time.Microsecond {
			t.Errorf("Coherence calculation too slow: %v per calculation", avgTime)
		}

		t.Logf("Coherence calculation: 1000 calculations in %v (avg: %v)", duration, avgTime)
	})

	// Test memory efficiency
	t.Run("MemoryEfficiency", func(t *testing.T) {
		// This is a basic memory test - in production would use more sophisticated tools
		initialStates := 10

		// Create states and track memory usage
		for i := 0; i < initialStates; i++ {
			amplitudes := make([]complex128, engine.GetDimension())
			norm := 1.0 / math.Sqrt(float64(engine.GetDimension()))

			for j := range amplitudes {
				amplitudes[j] = complex(norm, 0)
			}

			_, err := engine.CreateQuantumState(amplitudes)
			if err != nil {
				t.Fatalf("Failed to create quantum state: %v", err)
			}
		}

		// Memory should scale linearly with number of states
		// For 256-dimensional states, each should use approximately 256 * 16 * 2 = ~8KB
		// 10 states should use ~80KB
		expectedMemoryKB := float64(initialStates*engine.GetDimension()*16*2) / 1024.0

		t.Logf("Memory efficiency test: %d states created, estimated memory usage: %.1f KB",
			initialStates, expectedMemoryKB)
	})
}

// TestScalabilityMetrics tests system scalability metrics
func TestScalabilityMetrics(t *testing.T) {
	dimensions := []int{64, 128, 256}

	for _, dim := range dimensions {
		t.Run(fmt.Sprintf("Dimension_%d", dim), func(t *testing.T) {
			config := &core.EngineConfig{
				Dimension:        dim,
				MaxPrimeLimit:    1000,
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

			// Test that performance scales reasonably with dimension
			amplitudes := make([]complex128, dim)
			norm := 1.0 / math.Sqrt(float64(dim))

			for i := range amplitudes {
				amplitudes[i] = complex(norm, 0)
			}

			start := time.Now()

			// Create and evolve 10 states
			for i := 0; i < 10; i++ {
				state, err := engine.CreateQuantumState(amplitudes)
				if err != nil {
					t.Fatalf("Failed to create quantum state: %v", err)
				}

				_, err = engine.EvolveState(state, 0.01)
				if err != nil {
					t.Fatalf("Failed to evolve state: %v", err)
				}
			}

			duration := time.Since(start)
			avgTimePerOperation := duration / 20 // 10 creates + 10 evolves

			t.Logf("Scalability test (dim=%d): 20 operations in %v (avg: %v)",
				dim, duration, avgTimePerOperation)

			// Performance should scale roughly linearly with dimension
			// For dim=256 vs dim=64 (4x), time should be roughly 4x
			// Allow some overhead for fixed costs
			if dim == 256 && avgTimePerOperation > 100*time.Microsecond {
				t.Errorf("Performance scaling issue: dim=%d, avg time=%v", dim, avgTimePerOperation)
			}
		})
	}
}

// TestConcurrentPerformance tests concurrent operation performance
func TestConcurrentPerformance(t *testing.T) {
	config := &core.EngineConfig{
		Dimension:        128,
		MaxPrimeLimit:    1000,
		InitialEntropy:   1.5,
		EntropyLambda:    0.02,
		PlateauTolerance: 1e-6,
		PlateauWindow:    10,
		HistorySize:      1000,
	}

	concurrentOperations := 50

	t.Run("ConcurrentStateOperations", func(t *testing.T) {
		start := time.Now()

		// Launch concurrent operations
		results := make(chan error, concurrentOperations)

		for i := 0; i < concurrentOperations; i++ {
			go func(id int) {
				engine, err := core.NewResonanceEngine(config)
				if err != nil {
					results <- fmt.Errorf("failed to create engine %d: %v", id, err)
					return
				}

				amplitudes := make([]complex128, engine.GetDimension())
				norm := 1.0 / math.Sqrt(float64(engine.GetDimension()))

				for j := range amplitudes {
					amplitudes[j] = complex(norm, 0)
				}

				state, err := engine.CreateQuantumState(amplitudes)
				if err != nil {
					results <- fmt.Errorf("failed to create state %d: %v", id, err)
					return
				}

				// Perform some operations
				for k := 0; k < 10; k++ {
					state, err = engine.EvolveState(state, 0.01)
					if err != nil {
						results <- fmt.Errorf("failed to evolve state %d: %v", id, err)
						return
					}
				}

				results <- nil
			}(i)
		}

		// Collect results
		successCount := 0
		for i := 0; i < concurrentOperations; i++ {
			err := <-results
			if err != nil {
				t.Errorf("Concurrent operation failed: %v", err)
			} else {
				successCount++
			}
		}

		duration := time.Since(start)
		avgTimePerOperation := duration / time.Duration(concurrentOperations*10) // 10 operations per goroutine

		t.Logf("Concurrent performance: %d/%d operations successful in %v (avg: %v per operation)",
			successCount*10, concurrentOperations*10, duration, avgTimePerOperation)

		// Should complete within reasonable time
		if duration > 5*time.Second {
			t.Errorf("Concurrent operations took too long: %v", duration)
		}
	})
}
