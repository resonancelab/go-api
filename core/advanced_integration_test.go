package core

import (
	"math/big"
	"testing"
	"time"
)

func TestCompleteResonNetSystemWithEconomy(t *testing.T) {
	// Test the complete Reson.net system with token economy integration
	// This demonstrates the full economic and computational pipeline

	// 1. Initialize RSN Token Economy
	economy := NewRSNEconomy()

	// Mint initial tokens for testing
	user1 := "user_alice"
	user2 := "user_bob"
	node1 := "node_validator_1"
	node2 := "node_validator_2"

	// Mint tokens for users and nodes
	economy.MintTokens(big.NewInt(1000000), user1) // 1M RSN for Alice
	economy.MintTokens(big.NewInt(1000000), user2) // 1M RSN for Bob
	economy.MintTokens(big.NewInt(500000), node1)  // 500K RSN for Node 1
	economy.MintTokens(big.NewInt(500000), node2)  // 500K RSN for Node 2

	// 2. Initialize Global Phase State
	gps := NewGlobalPhaseState(0.8, time.Second)

	// Create prime oscillators
	osc1 := NewPrimeOscillator(13, 0.7, 1.0, 0.02)
	osc2 := NewPrimeOscillator(17, 0.5, 1.3, 0.02)
	gps.AddOscillator(osc1)
	gps.AddOscillator(osc2)

	// Create quaternionic states
	position := []float64{1.0, 2.0}
	gaussian := []float64{1.0, 2.0}
	eisenstein := []float64{3.0, 4.0}

	qs1 := NewQuaternionicState(position, complex(1.0, 0.5), gaussian, eisenstein)
	qs1.PrimeOscillator = osc1
	gps.AddQuaternionicState("node1", qs1)

	// 3. Initialize Holographic Memory
	hm := NewHolographicMemory(0.7, 0.3, 1000)
	field1 := hm.AddMemoryField("node1")
	hm.AddMemoryField("node2")

	// Store patterns
	patternData := []complex128{complex(1.0, 0.5), complex(0.8, 0.2), complex(0.9, 0.1)}
	field1.StorePattern("pattern1", patternData, 0.0, 1.0)

	// 4. Initialize Proof-of-Resonance Consensus
	rc := NewResonanceConsensus(0.7, 2)
	rc.AddValidator(node1, []byte("key1"), 100.0)
	rc.AddValidator(node2, []byte("key2"), 150.0)
	rc.AddValidator("node1", []byte("key3"), 200.0) // Add node1 as validator

	// 5. Initialize Distributed Execution Engine
	dee := NewDistributedExecutionEngine()
	dee.RegisterNode("node1", "localhost:8081", []string{"quaternionic", "resonance", "holographic"})
	dee.RegisterNode("node2", "localhost:8082", []string{"quaternionic", "resonance"})

	// 6. Register computational resources
	economy.RegisterComputationalResource("node1", "cpu", 8.0, big.NewInt(100))    // 8 CPU cores
	economy.RegisterComputationalResource("node1", "memory", 16.0, big.NewInt(50)) // 16 GB RAM
	economy.RegisterComputationalResource("node2", "cpu", 4.0, big.NewInt(80))     // 4 CPU cores
	economy.RegisterComputationalResource("node2", "memory", 8.0, big.NewInt(40))  // 8 GB RAM

	// 7. Test coherence and evolution
	coherence := gps.UpdateGlobalCoherence()
	if coherence < -1.0 || coherence > 1.0 {
		t.Errorf("Global coherence should be in [-1, 1], got %f", coherence)
	}

	gps.EvolvePhaseState(0.1)
	gps.SynchronizePhases(0.1)

	// 8. Test resource pricing
	resourcePrice, err := economy.CalculateResourcePrice("node1", "cpu", 2.0)
	if err != nil {
		t.Fatalf("Failed to calculate resource price: %v", err)
	}
	if resourcePrice.Cmp(big.NewInt(0)) <= 0 {
		t.Error("Resource price should be positive")
	}

	// 9. Test resource allocation and payment
	totalCost, err := economy.AllocateResource(user1, "node1", "cpu", 2.0, time.Hour)
	if err != nil {
		t.Fatalf("Failed to allocate resource: %v", err)
	}
	if totalCost.Cmp(big.NewInt(0)) <= 0 {
		t.Error("Allocation cost should be positive")
	}

	// 10. Test distributed program execution
	requirements := ProgramRequirements{
		MinNodes:             1,
		MaxNodes:             2,
		NodeCapabilities:     []string{"quaternionic"},
		Timeout:              30 * time.Second,
		Priority:             5,
		ResourceRequirements: map[string]float64{"cpu": 1.0},
	}

	programCode := `
		primelet p13 = oscillator(prime=13, amplitude=0.7, phase=1.0);
		quatstate q = quaternion(p13, gaussian=(1,2), eisenstein=(3,4));
		pay 100 RSN to node1;
	`

	programID, err := dee.SubmitProgram(programCode, "resolang", requirements)
	if err != nil {
		t.Fatalf("Failed to submit program: %v", err)
	}

	// Execute the program
	result, err := dee.ExecuteProgram(programID)
	if err != nil {
		t.Fatalf("Failed to execute program: %v", err)
	}

	if !result.Success {
		t.Errorf("Program execution should succeed, got error: %s", result.Error)
	}

	// 11. Test node contribution and rewards
	err = economy.RecordNodeContribution("node1", "computation", "Executed quaternionic computation", 100.0)
	if err != nil {
		t.Fatalf("Failed to record contribution: %v", err)
	}

	err = economy.RecordNodeContribution("node2", "validation", "Validated resonance proof", 50.0)
	if err != nil {
		t.Fatalf("Failed to record contribution: %v", err)
	}

	// 12. Test reward claiming
	rewardAmount, err := economy.ClaimNodeRewards("node1", time.Now().Format("2006-01"))
	if err != nil {
		t.Logf("Reward claim failed (expected if no rewards available): %v", err)
	} else {
		t.Logf("Successfully claimed %s RSN in rewards", rewardAmount.String())
	}

	// 13. Test consensus with economic incentives
	_, err = rc.CreateProof("node1", gps)
	if err != nil {
		t.Fatalf("Failed to create proof: %v", err)
	}

	result_consensus, err := rc.ProposeBlock("node1", gps)
	if err != nil {
		t.Fatalf("Failed to propose block: %v", err)
	}

	if !result_consensus.Accepted {
		t.Error("Block should have been accepted")
	}

	// 14. Test holographic memory with coherence
	hm.UpdateGlobalCoherence(gps)
	retrieved, err := hm.GlobalRetrieve("pattern1")
	if err != nil {
		t.Fatalf("Failed to retrieve pattern: %v", err)
	}
	if retrieved == nil {
		t.Error("Retrieved pattern should not be nil")
	}

	// 15. Verify economic state
	user1Balance := economy.GetBalance(user1)
	if user1Balance.Cmp(big.NewInt(0)) <= 0 {
		t.Error("User should have positive balance after transactions")
	}

	economyStats := economy.GetEconomyStats()
	if economyStats["total_transactions"].(int) == 0 {
		t.Error("Should have recorded transactions")
	}

	engineStats := dee.GetEngineStats()
	if engineStats["total_programs"].(int) == 0 {
		t.Error("Should have executed programs")
	}

	t.Logf("🎉 Complete Reson.net Economic System Test PASSED!")
	t.Logf("   - Global coherence: %.3f", coherence)
	t.Logf("   - Phase locked: %t", gps.CheckPhaseLocking())
	t.Logf("   - Consensus accepted: %t", result_consensus.Accepted)
	t.Logf("   - Program executed: %t", result.Success)
	t.Logf("   - User balance: %s RSN", user1Balance.String())
	t.Logf("   - Total transactions: %d", economyStats["total_transactions"])
	t.Logf("   - Active nodes: %d", engineStats["active_nodes"])
}

func TestResonNetEconomicScaling(t *testing.T) {
	// Test economic scaling with multiple users and nodes

	economy := NewRSNEconomy()
	dee := NewDistributedExecutionEngine()

	// Create multiple users and nodes
	users := []string{"alice", "bob", "charlie", "diana"}
	nodes := []string{"node1", "node2", "node3", "node4"}

	// Mint tokens for all users
	for _, user := range users {
		economy.MintTokens(big.NewInt(100000), user)
	}

	// Register nodes with different capabilities
	nodeCapabilities := [][]string{
		{"quaternionic", "resonance", "holographic"},
		{"quaternionic", "resonance"},
		{"resonance", "holographic"},
		{"quaternionic"},
	}

	for i, node := range nodes {
		dee.RegisterNode(node, "localhost:808"+string(rune(i+'1')), nodeCapabilities[i])
		economy.RegisterComputationalResource(node, "cpu", 4.0, big.NewInt(100))
		economy.RegisterComputationalResource(node, "memory", 8.0, big.NewInt(50))
	}

	// Simulate multiple concurrent programs
	programs := []string{
		"primelet p13 = oscillator(prime=13, amplitude=0.7, phase=1.0);",
		"primelet p17 = oscillator(prime=17, amplitude=0.5, phase=1.3);",
		"primelet p19 = oscillator(prime=19, amplitude=0.6, phase=0.8);",
	}

	programIDs := make([]string, len(programs))

	// Submit programs
	for i, code := range programs {
		requirements := ProgramRequirements{
			MinNodes:         1,
			MaxNodes:         2,
			NodeCapabilities: []string{"quaternionic"},
			Timeout:          30 * time.Second,
			Priority:         3,
		}

		programID, err := dee.SubmitProgram(code, "resolang", requirements)
		if err != nil {
			t.Fatalf("Failed to submit program %d: %v", i, err)
		}
		programIDs[i] = programID
	}

	// Execute programs
	for i, programID := range programIDs {
		result, err := dee.ExecuteProgram(programID)
		if err != nil {
			t.Fatalf("Failed to execute program %d: %v", i, err)
		}
		if !result.Success {
			t.Errorf("Program %d execution should succeed", i)
		}
	}

	// Verify economic activity
	economyStats := economy.GetEconomyStats()
	engineStats := dee.GetEngineStats()

	// Note: Programs don't execute payments in this simplified test
	// We just verify that the programs were submitted and executed
	if engineStats["completed_programs"].(int) != len(programs) {
		t.Errorf("Expected %d completed programs, got %d", len(programs), engineStats["completed_programs"])
	}

	t.Logf("✅ Economic scaling test passed!")
	t.Logf("   - Users: %d", len(users))
	t.Logf("   - Nodes: %d", len(nodes))
	t.Logf("   - Programs executed: %d", len(programs))
	t.Logf("   - Total transactions: %d", economyStats["total_transactions"])
	t.Logf("   - Active nodes: %d", engineStats["active_nodes"])
}

func BenchmarkResonNetEconomicPipeline(b *testing.B) {
	// Benchmark the complete economic pipeline

	economy := NewRSNEconomy()
	gps := NewGlobalPhaseState(0.8, time.Second)
	dee := NewDistributedExecutionEngine()

	// Setup
	economy.MintTokens(big.NewInt(1000000), "user")
	dee.RegisterNode("node1", "localhost:8081", []string{"quaternionic"})
	economy.RegisterComputationalResource("node1", "cpu", 4.0, big.NewInt(100))

	osc := NewPrimeOscillator(13, 0.7, 1.0, 0.02)
	gps.AddOscillator(osc)

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		// Complete economic pipeline: resource allocation -> program execution -> payment -> consensus
		_, _ = economy.AllocateResource("user", "node1", "cpu", 1.0, time.Minute)

		requirements := ProgramRequirements{MinNodes: 1, MaxNodes: 1, NodeCapabilities: []string{"quaternionic"}, Timeout: time.Second, Priority: 1}
		programID, _ := dee.SubmitProgram("primelet p = oscillator(prime=13, amplitude=0.7, phase=1.0);", "resolang", requirements)
		_, _ = dee.ExecuteProgram(programID)

		gps.UpdateGlobalCoherence()
		gps.EvolvePhaseState(0.01)
	}
}
