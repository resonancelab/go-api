package tests

import (
	"testing"
	"time"

	"github.com/resonancelab/psizero/api/core"
	"github.com/resonancelab/psizero/api/engines/hqe"
	"github.com/resonancelab/psizero/api/engines/iching"
	"github.com/resonancelab/psizero/api/engines/nlc"
	"github.com/resonancelab/psizero/api/engines/qcr"
	"github.com/resonancelab/psizero/api/engines/qsem"
	"github.com/resonancelab/psizero/api/engines/srs"
	"github.com/resonancelab/psizero/api/engines/unified"
)

// TestSRSEngine tests the Symbolic AI Engine
func TestSRSEngine(t *testing.T) {
	coreConfig := &core.ResonanceEngineConfig{
		Dimension:         256,
		MaxPrimes:         1000,
		TimeoutSeconds:    10,
		EvolutionSteps:    100,
		ResonanceStrength: 1.0,
	}

	coreEngine, err := core.NewResonanceEngine(coreConfig)
	if err != nil {
		t.Fatalf("Failed to create core engine: %v", err)
	}

	config := &srs.SRSEngineConfig{
		ProblemComplexity:  "NP",
		MaxVariables:       100,
		MaxClauses:         300,
		TimeoutSeconds:     30,
		SolverAlgorithm:    "quantum_resonance",
		OptimizationLevel:  "high",
		ParallelProcessing: true,
		CacheSize:          1000,
		EnablePredictions:  true,
	}

	engine, err := srs.NewSRSEngine(coreEngine, config)
	if err != nil {
		t.Fatalf("Failed to create SRS engine: %v", err)
	}

	t.Run("EngineName", func(t *testing.T) {
		if engine.GetID() == "" {
			t.Error("Engine ID should not be empty")
		}
		if engine.GetType() != "srs" {
			t.Errorf("Expected engine type 'srs', got '%s'", engine.GetType())
		}
	})

	t.Run("SATSolving", func(t *testing.T) {
		// Test simple 3-SAT problem: (x1 ∨ x2 ∨ x3) ∧ (¬x1 ∨ x2 ∨ ¬x3) ∧ (x1 ∨ ¬x2 ∨ x3)
		formula := map[string]interface{}{
			"type":      "3SAT",
			"variables": 3,
			"clauses": [][]int{
				{1, 2, 3},   // (x1 ∨ x2 ∨ x3)
				{-1, 2, -3}, // (¬x1 ∨ x2 ∨ ¬x3)
				{1, -2, 3},  // (x1 ∨ ¬x2 ∨ x3)
			},
		}

		result, err := engine.SolveProblem(formula)
		if err != nil {
			t.Fatalf("Failed to solve SAT problem: %v", err)
		}

		if result == nil {
			t.Fatal("Result should not be nil")
		}

		// Check if solution is found
		if !result.Satisfiable {
			t.Error("Problem should be satisfiable")
		}

		if len(result.Solution) != 3 {
			t.Errorf("Expected 3 variables in solution, got %d", len(result.Solution))
		}
	})

	t.Run("NPCompleteProblem", func(t *testing.T) {
		// Test more complex NP-complete problem
		formula := map[string]interface{}{
			"type":     "graph_coloring",
			"vertices": 4,
			"edges":    [][]int{{0, 1}, {1, 2}, {2, 3}, {3, 0}}, // Square graph
			"colors":   3,
		}

		result, err := engine.SolveProblem(formula)
		if err != nil {
			t.Fatalf("Failed to solve graph coloring: %v", err)
		}

		if !result.Satisfiable {
			t.Error("Graph coloring should be satisfiable")
		}
	})

	t.Run("StateManagement", func(t *testing.T) {
		state, err := engine.GetState()
		if err != nil {
			t.Fatalf("Failed to get engine state: %v", err)
		}

		if state.ID != engine.GetID() {
			t.Errorf("State ID mismatch: expected %s, got %s", engine.GetID(), state.ID)
		}

		if state.Type != "srs" {
			t.Errorf("State type mismatch: expected 'srs', got %s", state.Type)
		}
	})
}

// TestHQEEngine tests the Holographic Quantum Engine
func TestHQEEngine(t *testing.T) {
	coreConfig := &core.ResonanceEngineConfig{
		Dimension:         256,
		MaxPrimes:         1000,
		TimeoutSeconds:    10,
		EvolutionSteps:    100,
		ResonanceStrength: 1.0,
	}

	coreEngine, err := core.NewResonanceEngine(coreConfig)
	if err != nil {
		t.Fatalf("Failed to create core engine: %v", err)
	}

	config := &hqe.HQEEngineConfig{
		BulkDimensions:     5,
		BoundaryDimensions: 4,
		AdSRadius:          1.0,
		HolographicScale:   1.0,
		QuantumCorrections: true,
		MaxEvolutionSteps:  1000,
		TimeoutSeconds:     30,
		SimulationType:     "AdS_CFT",
	}

	engine, err := hqe.NewHQEEngine(coreEngine, config)
	if err != nil {
		t.Fatalf("Failed to create HQE engine: %v", err)
	}

	t.Run("HolographicSimulation", func(t *testing.T) {
		params := map[string]interface{}{
			"boundary_state": "thermal",
			"temperature":    1.0,
			"coupling":       0.1,
		}

		result, err := engine.RunSimulation(params)
		if err != nil {
			t.Fatalf("Failed to run holographic simulation: %v", err)
		}

		if result == nil {
			t.Fatal("Simulation result should not be nil")
		}

		// Check if bulk geometry was computed
		if result.BulkGeometry == nil {
			t.Error("Bulk geometry should be computed")
		}

		// Check if boundary state exists
		if result.BoundaryState == nil {
			t.Error("Boundary state should exist")
		}
	})

	t.Run("AdSCFTCorrespondence", func(t *testing.T) {
		params := map[string]interface{}{
			"check_correspondence": true,
			"precision":            1e-6,
		}

		result, err := engine.RunSimulation(params)
		if err != nil {
			t.Fatalf("Failed to verify AdS/CFT correspondence: %v", err)
		}

		// Check correspondence validity
		if !result.CorrespondenceValid {
			t.Error("AdS/CFT correspondence should be valid")
		}
	})
}

// TestQSEMEngine tests the Semantic Encoding Engine
func TestQSEMEngine(t *testing.T) {
	coreConfig := &core.ResonanceEngineConfig{
		Dimension:         256,
		MaxPrimes:         1000,
		TimeoutSeconds:    10,
		EvolutionSteps:    100,
		ResonanceStrength: 1.0,
	}

	coreEngine, err := core.NewResonanceEngine(coreConfig)
	if err != nil {
		t.Fatalf("Failed to create core engine: %v", err)
	}

	config := &qsem.QSEMEngineConfig{
		VectorDimension:    256,
		MaxConcepts:        1000,
		SemanticThreshold:  0.7,
		LearningRate:       0.01,
		ContextWindow:      10,
		EnableOntologyTree: true,
		MaxDepth:           5,
		TimeoutSeconds:     30,
	}

	engine, err := qsem.NewQSEMEngine(coreEngine, config)
	if err != nil {
		t.Fatalf("Failed to create QSEM engine: %v", err)
	}

	t.Run("ConceptEncoding", func(t *testing.T) {
		concepts := []string{"quantum", "resonance", "consciousness", "prime", "mathematics"}

		result, err := engine.EncodeConcepts(concepts)
		if err != nil {
			t.Fatalf("Failed to encode concepts: %v", err)
		}

		if len(result.ConceptVectors) != len(concepts) {
			t.Errorf("Expected %d concept vectors, got %d", len(concepts), len(result.ConceptVectors))
		}

		// Check vector dimensions
		for concept, vector := range result.ConceptVectors {
			if len(vector) != config.VectorDimension {
				t.Errorf("Concept '%s' vector has wrong dimension: expected %d, got %d",
					concept, config.VectorDimension, len(vector))
			}
		}
	})

	t.Run("SemanticSimilarity", func(t *testing.T) {
		query := map[string]interface{}{
			"concept1": "quantum",
			"concept2": "physics",
			"method":   "cosine_similarity",
		}

		result, err := engine.CalculateSimilarity(query)
		if err != nil {
			t.Fatalf("Failed to calculate semantic similarity: %v", err)
		}

		// Similarity should be between 0 and 1
		if result.Similarity < 0 || result.Similarity > 1 {
			t.Errorf("Similarity should be between 0 and 1, got %f", result.Similarity)
		}
	})

	t.Run("MeaningSpace", func(t *testing.T) {
		params := map[string]interface{}{
			"query":       "consciousness and quantum mechanics",
			"max_results": 10,
		}

		result, err := engine.QueryMeaningSpace(params)
		if err != nil {
			t.Fatalf("Failed to query meaning space: %v", err)
		}

		if len(result.Results) == 0 {
			t.Error("Should return some results from meaning space")
		}
	})
}

// TestNLCEngine tests the Non-Local Communication Engine
func TestNLCEngine(t *testing.T) {
	coreConfig := &core.ResonanceEngineConfig{
		Dimension:         256,
		MaxPrimes:         1000,
		TimeoutSeconds:    10,
		EvolutionSteps:    100,
		ResonanceStrength: 1.0,
	}

	coreEngine, err := core.NewResonanceEngine(coreConfig)
	if err != nil {
		t.Fatalf("Failed to create core engine: %v", err)
	}

	config := &nlc.NLCEngineConfig{
		MaxEntanglementPairs: 100,
		FidelityThreshold:    0.9,
		DecoherenceTime:      10.0,
		ChannelNoise:         0.01,
		ProtocolType:         "BB84",
		TimeoutSeconds:       30,
		EnableTeleportation:  true,
		QuantumRepeaters:     true,
	}

	engine, err := nlc.NewNLCEngine(coreEngine, config)
	if err != nil {
		t.Fatalf("Failed to create NLC engine: %v", err)
	}

	t.Run("QuantumEntanglement", func(t *testing.T) {
		params := map[string]interface{}{
			"pairs":           10,
			"target_fidelity": 0.95,
		}

		result, err := engine.CreateEntanglement(params)
		if err != nil {
			t.Fatalf("Failed to create entanglement: %v", err)
		}

		if result.EntangledPairs != 10 {
			t.Errorf("Expected 10 entangled pairs, got %d", result.EntangledPairs)
		}

		if result.AverageFidelity < config.FidelityThreshold {
			t.Errorf("Fidelity below threshold: %f < %f", result.AverageFidelity, config.FidelityThreshold)
		}
	})

	t.Run("QuantumTeleportation", func(t *testing.T) {
		params := map[string]interface{}{
			"state_type": "random",
			"distance":   1000.0, // km
		}

		result, err := engine.PerformTeleportation(params)
		if err != nil {
			t.Fatalf("Failed to perform teleportation: %v", err)
		}

		if !result.Success {
			t.Error("Teleportation should succeed")
		}

		if result.Fidelity < 0.8 {
			t.Errorf("Teleportation fidelity too low: %f", result.Fidelity)
		}
	})

	t.Run("BellTest", func(t *testing.T) {
		params := map[string]interface{}{
			"measurements": 1000,
			"test_type":    "CHSH",
		}

		result, err := engine.PerformBellTest(params)
		if err != nil {
			t.Fatalf("Failed to perform Bell test: %v", err)
		}

		// CHSH inequality should be violated for entangled states
		if result.CHSHValue <= 2.0 {
			t.Errorf("CHSH value should violate classical bound (2.0), got %f", result.CHSHValue)
		}

		// Maximum quantum value is 2√2 ≈ 2.828
		if result.CHSHValue > 2.828 {
			t.Errorf("CHSH value exceeds quantum bound (2.828), got %f", result.CHSHValue)
		}
	})
}

// TestQCREngine tests the Consciousness Resonance Engine
func TestQCREngine(t *testing.T) {
	coreConfig := &core.ResonanceEngineConfig{
		Dimension:         256,
		MaxPrimes:         1000,
		TimeoutSeconds:    10,
		EvolutionSteps:    100,
		ResonanceStrength: 1.0,
	}

	coreEngine, err := core.NewResonanceEngine(coreConfig)
	if err != nil {
		t.Fatalf("Failed to create core engine: %v", err)
	}

	config := &qcr.QCREngineConfig{
		ConsciousnessLevel:    "human",
		IntegratedInformation: 0.8,
		CoherenceThreshold:    0.7,
		MemoryCapacity:        1000,
		ProcessingSpeed:       100.0,
		EmotionalRange:        1.0,
		CreativityFactor:      0.5,
		TimeoutSeconds:        30,
	}

	engine, err := qcr.NewQCREngine(coreEngine, config)
	if err != nil {
		t.Fatalf("Failed to create QCR engine: %v", err)
	}

	t.Run("ConsciousnessSimulation", func(t *testing.T) {
		params := map[string]interface{}{
			"scenario": "self_awareness_test",
			"duration": 1.0, // seconds
			"stimuli":  []string{"visual", "auditory", "cognitive"},
		}

		result, err := engine.SimulateConsciousness(params)
		if err != nil {
			t.Fatalf("Failed to simulate consciousness: %v", err)
		}

		// Check consciousness metrics
		if result.PhiValue < 0 || result.PhiValue > 1 {
			t.Errorf("Phi value should be between 0 and 1, got %f", result.PhiValue)
		}

		if result.CoherenceLevel < 0 || result.CoherenceLevel > 1 {
			t.Errorf("Coherence level should be between 0 and 1, got %f", result.CoherenceLevel)
		}
	})

	t.Run("CognitiveProcessing", func(t *testing.T) {
		params := map[string]interface{}{
			"task":       "pattern_recognition",
			"complexity": "medium",
			"data":       []float64{1, 2, 3, 5, 8, 13, 21}, // Fibonacci sequence
		}

		result, err := engine.ProcessCognitive(params)
		if err != nil {
			t.Fatalf("Failed to process cognitive task: %v", err)
		}

		// Should recognize the Fibonacci pattern
		if !result.PatternRecognized {
			t.Error("Should recognize Fibonacci pattern")
		}
	})

	t.Run("EmotionalResponse", func(t *testing.T) {
		params := map[string]interface{}{
			"stimulus_type": "positive",
			"intensity":     0.8,
			"context":       "achievement",
		}

		result, err := engine.GenerateEmotion(params)
		if err != nil {
			t.Fatalf("Failed to generate emotional response: %v", err)
		}

		// Positive stimulus should generate positive emotion
		if result.Valence <= 0 {
			t.Errorf("Positive stimulus should generate positive valence, got %f", result.Valence)
		}

		if result.Arousal < 0 || result.Arousal > 1 {
			t.Errorf("Arousal should be between 0 and 1, got %f", result.Arousal)
		}
	})
}

// TestIChingEngine tests the Quantum Oracle Engine
func TestIChingEngine(t *testing.T) {
	coreConfig := &core.ResonanceEngineConfig{
		Dimension:         256,
		MaxPrimes:         1000,
		TimeoutSeconds:    10,
		EvolutionSteps:    100,
		ResonanceStrength: 1.0,
	}

	coreEngine, err := core.NewResonanceEngine(coreConfig)
	if err != nil {
		t.Fatalf("Failed to create core engine: %v", err)
	}

	config := &iching.IChingEngineConfig{
		DivinerExperience:  "master",
		CosmicAlignment:    0.8,
		QuantumRandomness:  true,
		ReadingDepth:       "deep",
		CulturalContext:    "traditional",
		TimeoutSeconds:     30,
		WisdomAccumulation: true,
		SynchronicityLevel: 0.7,
	}

	engine, err := iching.NewIChingEngine(coreEngine, config)
	if err != nil {
		t.Fatalf("Failed to create I-Ching engine: %v", err)
	}

	t.Run("HexagramGeneration", func(t *testing.T) {
		params := map[string]interface{}{
			"question":    "What is the nature of quantum consciousness?",
			"method":      "yarrow_stalks",
			"cosmic_time": time.Now(),
		}

		result, err := engine.CastHexagram(params)
		if err != nil {
			t.Fatalf("Failed to cast hexagram: %v", err)
		}

		// Should generate valid hexagram (1-64)
		if result.PrimaryHexagram < 1 || result.PrimaryHexagram > 64 {
			t.Errorf("Invalid primary hexagram: %d", result.PrimaryHexagram)
		}

		if result.SecondaryHexagram != 0 && (result.SecondaryHexagram < 1 || result.SecondaryHexagram > 64) {
			t.Errorf("Invalid secondary hexagram: %d", result.SecondaryHexagram)
		}

		// Should have interpretation
		if result.Interpretation == "" {
			t.Error("Interpretation should not be empty")
		}
	})

	t.Run("QuantumDivination", func(t *testing.T) {
		params := map[string]interface{}{
			"quantum_source": "true_random",
			"entanglement":   true,
			"observers":      2,
		}

		result, err := engine.QuantumDivination(params)
		if err != nil {
			t.Fatalf("Failed to perform quantum divination: %v", err)
		}

		// Check quantum properties
		if result.QuantumCoherence < 0 || result.QuantumCoherence > 1 {
			t.Errorf("Quantum coherence should be between 0 and 1, got %f", result.QuantumCoherence)
		}

		if result.Synchronicity < 0 || result.Synchronicity > 1 {
			t.Errorf("Synchronicity should be between 0 and 1, got %f", result.Synchronicity)
		}
	})

	t.Run("WisdomAccumulation", func(t *testing.T) {
		// Perform multiple readings to accumulate wisdom
		for i := 0; i < 5; i++ {
			params := map[string]interface{}{
				"question": "Test question " + string(rune(i+'0')),
				"method":   "coin_toss",
			}

			_, err := engine.CastHexagram(params)
			if err != nil {
				t.Fatalf("Failed to cast hexagram %d: %v", i, err)
			}
		}

		wisdom := engine.GetAccumulatedWisdom()

		// Wisdom should increase with readings
		if wisdom.TotalReadings != 5 {
			t.Errorf("Expected 5 total readings, got %d", wisdom.TotalReadings)
		}

		if wisdom.WisdomLevel <= 0 {
			t.Errorf("Wisdom level should be positive, got %f", wisdom.WisdomLevel)
		}
	})
}

// TestUnifiedEngine tests the Unified Physics Engine
func TestUnifiedEngine(t *testing.T) {
	coreConfig := &core.ResonanceEngineConfig{
		Dimension:         256,
		MaxPrimes:         1000,
		TimeoutSeconds:    10,
		EvolutionSteps:    100,
		ResonanceStrength: 1.0,
	}

	coreEngine, err := core.NewResonanceEngine(coreConfig)
	if err != nil {
		t.Fatalf("Failed to create core engine: %v", err)
	}

	config := &unified.UnifiedEngineConfig{
		SimulationType:      "standard_model",
		MaxParticles:        100,
		SpacetimeDimensions: 4,
		PlanckLength:        1.616e-35,
		PlanckTime:          5.391e-44,
		TimeStep:            1e-18,
		EvolutionSteps:      1000,
		TimeoutSeconds:      60,
	}

	engine, err := unified.NewUnifiedEngine(coreEngine, config)
	if err != nil {
		t.Fatalf("Failed to create Unified engine: %v", err)
	}

	t.Run("ParticlePhysicsSimulation", func(t *testing.T) {
		params := map[string]interface{}{
			"simulation_type": "particle_physics",
			"particles": []map[string]interface{}{
				{"type": "electron", "energy": 0.511}, // MeV
				{"type": "proton", "energy": 938.3},   // MeV
			},
		}

		result, err := engine.RunPhysicsSimulation(params)
		if err != nil {
			t.Fatalf("Failed to run physics simulation: %v", err)
		}

		// Check if particles were created
		if result.ParticleSummary == nil {
			t.Error("Particle summary should not be nil")
		}

		// Check energy conservation
		if totalEnergy, exists := result.PhysicsMetrics["total_energy"]; exists {
			if totalEnergy.(float64) <= 0 {
				t.Error("Total energy should be positive")
			}
		}
	})

	t.Run("GravitationalFields", func(t *testing.T) {
		params := map[string]interface{}{
			"simulation_type": "black_hole",
			"mass":            10.0, // Solar masses
		}

		result, err := engine.RunPhysicsSimulation(params)
		if err != nil {
			t.Fatalf("Failed to run gravitational simulation: %v", err)
		}

		// Check spacetime curvature
		if curvature, exists := result.PhysicsMetrics["spacetime_curvature"]; exists {
			if curvature.(float64) == 0 {
				t.Error("Spacetime curvature should be non-zero for black hole")
			}
		}
	})

	t.Run("UnifiedFieldTheory", func(t *testing.T) {
		params := map[string]interface{}{
			"unification_test": true,
			"energy_scale":     1e16, // GUT scale
		}

		result, err := engine.RunPhysicsSimulation(params)
		if err != nil {
			t.Fatalf("Failed to test unified field theory: %v", err)
		}

		// Check unification degree
		if unification, exists := result.PhysicsMetrics["unification_degree"]; exists {
			unificationDegree := unification.(float64)
			if unificationDegree < 0 || unificationDegree > 1 {
				t.Errorf("Unification degree should be between 0 and 1, got %f", unificationDegree)
			}
		}
	})
}

// BenchmarkEngines benchmarks all engines
func BenchmarkEngines(b *testing.B) {
	coreConfig := &core.ResonanceEngineConfig{
		Dimension:         128,
		MaxPrimes:         500,
		TimeoutSeconds:    5,
		EvolutionSteps:    50,
		ResonanceStrength: 1.0,
	}

	coreEngine, err := core.NewResonanceEngine(coreConfig)
	if err != nil {
		b.Fatalf("Failed to create core engine: %v", err)
	}

	b.Run("SRSEngine", func(b *testing.B) {
		config := &srs.SRSEngineConfig{
			ProblemComplexity: "P",
			MaxVariables:      50,
			MaxClauses:        100,
			TimeoutSeconds:    10,
			SolverAlgorithm:   "quantum_resonance",
			OptimizationLevel: "medium",
		}

		engine, err := srs.NewSRSEngine(coreEngine, config)
		if err != nil {
			b.Fatalf("Failed to create SRS engine: %v", err)
		}

		formula := map[string]interface{}{
			"type":      "2SAT",
			"variables": 10,
			"clauses":   [][]int{{1, 2}, {-1, 3}, {2, -3}},
		}

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, err := engine.SolveProblem(formula)
			if err != nil {
				b.Fatalf("Solve failed: %v", err)
			}
		}
	})

	b.Run("QSEMEngine", func(b *testing.B) {
		config := &qsem.QSEMEngineConfig{
			VectorDimension:   128,
			MaxConcepts:       100,
			SemanticThreshold: 0.7,
			TimeoutSeconds:    10,
		}

		engine, err := qsem.NewQSEMEngine(coreEngine, config)
		if err != nil {
			b.Fatalf("Failed to create QSEM engine: %v", err)
		}

		concepts := []string{"test", "benchmark", "performance"}

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, err := engine.EncodeConcepts(concepts)
			if err != nil {
				b.Fatalf("Encoding failed: %v", err)
			}
		}
	})
}
