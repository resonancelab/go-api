package tests

import (
	"testing"

	"github.com/resonancelab/psizero/engines/hqe"
	"github.com/resonancelab/psizero/engines/iching"
	"github.com/resonancelab/psizero/engines/nlc"
	"github.com/resonancelab/psizero/engines/qcr"
	"github.com/resonancelab/psizero/engines/qsem"
	"github.com/resonancelab/psizero/engines/srs"
	"github.com/resonancelab/psizero/engines/unified"
)

// TestSRSEngine tests the Symbolic AI Engine
func TestSRSEngine(t *testing.T) {
	engine, err := srs.NewSRSEngine()
	if err != nil {
		t.Fatalf("Failed to create SRS engine: %v", err)
	}

	t.Run("BasicEngineTest", func(t *testing.T) {
		state := engine.GetCurrentState()
		if state == nil {
			t.Fatal("State should not be nil")
		}
	})

	t.Run("SATSolving", func(t *testing.T) {
		// Test simple 3-SAT problem
		formula := map[string]interface{}{
			"variables": 3,
			"clauses": []interface{}{
				[]interface{}{
					map[string]interface{}{"var": 1, "neg": false},
					map[string]interface{}{"var": 2, "neg": false},
					map[string]interface{}{"var": 3, "neg": false},
				},
			},
		}

		result, telemetry, err := engine.SolveProblem("3sat", formula, nil)
		if err != nil {
			t.Fatalf("Failed to solve SAT problem: %v", err)
		}

		if result == nil {
			t.Fatal("Result should not be nil")
		}

		if len(telemetry) == 0 {
			t.Error("Should have some telemetry data")
		}
	})
}

// TestHQEEngine tests the Holographic Quantum Engine
func TestHQEEngine(t *testing.T) {
	engine, err := hqe.NewHQEEngine()
	if err != nil {
		t.Fatalf("Failed to create HQE engine: %v", err)
	}

	t.Run("BasicEngineTest", func(t *testing.T) {
		state := engine.GetCurrentState()
		if state == nil {
			t.Fatal("Engine state should not be nil")
		}
	})
}

// TestQSEMEngine tests the Semantic Encoding Engine
func TestQSEMEngine(t *testing.T) {
	engine, err := qsem.NewQSEMEngine()
	if err != nil {
		t.Fatalf("Failed to create QSEM engine: %v", err)
	}

	t.Run("BasicEngineTest", func(t *testing.T) {
		state := engine.GetCurrentState()
		if state == nil {
			t.Fatal("Engine state should not be nil")
		}
	})
}

// TestNLCEngine tests the Non-Local Communication Engine
func TestNLCEngine(t *testing.T) {
	engine, err := nlc.NewNLCEngine()
	if err != nil {
		t.Fatalf("Failed to create NLC engine: %v", err)
	}

	t.Run("BasicEngineTest", func(t *testing.T) {
		// Test basic communication setup
		participants := []string{"node_a", "node_b"}
		result, telemetry, err := engine.EstablishNonLocalCommunication("teleportation", participants, nil)
		if err != nil {
			t.Fatalf("Failed to establish communication: %v", err)
		}

		if result == nil {
			t.Fatal("Communication result should not be nil")
		}

		if len(telemetry) == 0 {
			t.Error("Should have telemetry data")
		}
	})
}

// TestQCREngine tests the Consciousness Resonance Engine
func TestQCREngine(t *testing.T) {
	engine, err := qcr.NewQCREngine()
	if err != nil {
		t.Fatalf("Failed to create QCR engine: %v", err)
	}

	t.Run("BasicEngineTest", func(t *testing.T) {
		// Test consciousness simulation
		parameters := map[string]interface{}{
			"entity_count": 2,
		}
		result, telemetry, err := engine.SimulateConsciousness("self_awareness", parameters, nil)
		if err != nil {
			t.Fatalf("Failed to simulate consciousness: %v", err)
		}

		if result == nil {
			t.Fatal("Consciousness result should not be nil")
		}

		if len(telemetry) == 0 {
			t.Error("Should have telemetry data")
		}
	})
}

// TestIChingEngine tests the Quantum Oracle Engine
func TestIChingEngine(t *testing.T) {
	engine, err := iching.NewIChingEngine()
	if err != nil {
		t.Fatalf("Failed to create I-Ching engine: %v", err)
	}

	t.Run("BasicEngineTest", func(t *testing.T) {
		// Test oracle consultation
		result, telemetry, err := engine.ConsultOracle("What is the nature of reality?", "philosophical", "seeker", nil)
		if err != nil {
			t.Fatalf("Failed to consult oracle: %v", err)
		}

		if result == nil {
			t.Fatal("Oracle result should not be nil")
		}

		if len(telemetry) == 0 {
			t.Error("Should have telemetry data")
		}
	})
}

// TestUnifiedEngine tests the Unified Physics Engine
func TestUnifiedEngine(t *testing.T) {
	engine, err := unified.NewUnifiedEngine()
	if err != nil {
		t.Fatalf("Failed to create Unified engine: %v", err)
	}

	t.Run("BasicEngineTest", func(t *testing.T) {
		// Test physics simulation
		initialConditions := map[string]interface{}{
			"particle_count": 3,
		}
		result, telemetry, err := engine.SimulateUnifiedPhysics("particle_interaction", initialConditions, nil)
		if err != nil {
			t.Fatalf("Failed to simulate physics: %v", err)
		}

		if result == nil {
			t.Fatal("Physics result should not be nil")
		}

		if len(telemetry) == 0 {
			t.Error("Should have telemetry data")
		}
	})
}

// BenchmarkEngines benchmarks all engines
func BenchmarkEngines(b *testing.B) {
	b.Run("SRSEngine", func(b *testing.B) {
		engine, err := srs.NewSRSEngine()
		if err != nil {
			b.Fatalf("Failed to create SRS engine: %v", err)
		}

		formula := map[string]interface{}{
			"variables": 2,
			"clauses": []interface{}{
				[]interface{}{
					map[string]interface{}{"var": 1, "neg": false},
					map[string]interface{}{"var": 2, "neg": false},
				},
			},
		}

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, _, _ = engine.SolveProblem("3sat", formula, nil)
		}
	})

	b.Run("QSEMEngine", func(b *testing.B) {
		engine, err := qsem.NewQSEMEngine()
		if err != nil {
			b.Fatalf("Failed to create QSEM engine: %v", err)
		}

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			// Just create engine instances for benchmarking
			_ = engine
		}
	})
}
