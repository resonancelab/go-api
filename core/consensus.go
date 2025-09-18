package core

import (
	"crypto/sha256"
	"fmt"
	"math"
	"sort"
	"sync"
	"time"
)

// ProofOfResonance represents a proof-of-resonance validation from Reson.net paper
type ProofOfResonance struct {
	NodeID            string             `json:"node_id"`
	PhaseMeasurements []PhaseMeasurement `json:"phase_measurements"`
	CoherenceScore    float64            `json:"coherence_score"`
	ResonanceStrength float64            `json:"resonance_strength"`
	Timestamp         time.Time          `json:"timestamp"`
	Signature         []byte             `json:"signature"`
	QuaternionicState *QuaternionicState `json:"quaternionic_state"`
	BlockHeight       int                `json:"block_height"`
	PreviousHash      string             `json:"previous_hash"`
}

// PhaseMeasurement represents a phase measurement at a specific time
type PhaseMeasurement struct {
	Prime     int       `json:"prime"`
	Phase     float64   `json:"phase"`
	Amplitude float64   `json:"amplitude"`
	Timestamp time.Time `json:"timestamp"`
}

// ResonanceConsensus manages proof-of-resonance consensus
type ResonanceConsensus struct {
	Validators         []*ValidatorNode    `json:"validators"`
	PoRChain           []*ProofOfResonance `json:"por_chain"`
	CurrentBlock       *ProofOfResonance   `json:"current_block"`
	ConsensusThreshold float64             `json:"consensus_threshold"`
	MinValidators      int                 `json:"min_validators"`
	mu                 sync.RWMutex        `json:"-"`
}

// ValidatorNode represents a validator in the consensus network
type ValidatorNode struct {
	NodeID      string    `json:"node_id"`
	PublicKey   []byte    `json:"public_key"`
	Stake       float64   `json:"stake"` // RSN stake
	LastActive  time.Time `json:"last_active"`
	Reliability float64   `json:"reliability"` // 0-1 reliability score
}

// ConsensusResult represents the result of a consensus round
type ConsensusResult struct {
	Accepted       bool              `json:"accepted"`
	BlockHash      string            `json:"block_hash"`
	ValidatorCount int               `json:"validator_count"`
	CoherenceScore float64           `json:"coherence_score"`
	Timestamp      time.Time         `json:"timestamp"`
	Signatures     map[string][]byte `json:"signatures"`
}

// NewResonanceConsensus creates a new proof-of-resonance consensus system
func NewResonanceConsensus(threshold float64, minValidators int) *ResonanceConsensus {
	return &ResonanceConsensus{
		Validators:         make([]*ValidatorNode, 0),
		PoRChain:           make([]*ProofOfResonance, 0),
		ConsensusThreshold: threshold,
		MinValidators:      minValidators,
	}
}

// AddValidator adds a validator to the consensus network
func (rc *ResonanceConsensus) AddValidator(nodeID string, publicKey []byte, stake float64) {
	rc.mu.Lock()
	defer rc.mu.Unlock()

	validator := &ValidatorNode{
		NodeID:      nodeID,
		PublicKey:   publicKey,
		Stake:       stake,
		LastActive:  time.Now(),
		Reliability: 1.0, // Start with perfect reliability
	}

	rc.Validators = append(rc.Validators, validator)
}

// CreateProof creates a new proof-of-resonance for the current state
func (rc *ResonanceConsensus) CreateProof(nodeID string, globalPhaseState *GlobalPhaseState) (*ProofOfResonance, error) {
	rc.mu.RLock()
	defer rc.mu.RUnlock()

	// Collect phase measurements from global state
	measurements := rc.collectPhaseMeasurements(globalPhaseState)

	// Calculate coherence score
	coherenceScore := rc.calculateCoherenceScore(measurements)

	// Calculate resonance strength
	resonanceStrength := rc.calculateResonanceStrength(measurements)

	// Get current quaternionic state (use first available)
	var quaternionicState *QuaternionicState
	for _, state := range globalPhaseState.QuaternionicStates {
		quaternionicState = state
		break
	}

	// Create proof
	proof := &ProofOfResonance{
		NodeID:            nodeID,
		PhaseMeasurements: measurements,
		CoherenceScore:    coherenceScore,
		ResonanceStrength: resonanceStrength,
		Timestamp:         time.Now(),
		QuaternionicState: quaternionicState,
		BlockHeight:       len(rc.PoRChain),
	}

	// Set previous hash
	if len(rc.PoRChain) > 0 {
		proof.PreviousHash = rc.PoRChain[len(rc.PoRChain)-1].Hash()
	} else {
		proof.PreviousHash = "genesis"
	}

	// Sign the proof
	signature, err := rc.signProof(proof, nodeID)
	if err != nil {
		return nil, fmt.Errorf("failed to sign proof: %w", err)
	}
	proof.Signature = signature

	return proof, nil
}

// collectPhaseMeasurements collects phase measurements from global state
func (rc *ResonanceConsensus) collectPhaseMeasurements(gps *GlobalPhaseState) []PhaseMeasurement {
	measurements := make([]PhaseMeasurement, 0, len(gps.Oscillators))

	for prime, oscillator := range gps.Oscillators {
		measurement := PhaseMeasurement{
			Prime:     prime,
			Phase:     oscillator.Phase,
			Amplitude: oscillator.GetAmplitude(),
			Timestamp: time.Now(),
		}
		measurements = append(measurements, measurement)
	}

	// Sort by prime for consistent ordering
	sort.Slice(measurements, func(i, j int) bool {
		return measurements[i].Prime < measurements[j].Prime
	})

	return measurements
}

// calculateCoherenceScore calculates the coherence score from measurements
func (rc *ResonanceConsensus) calculateCoherenceScore(measurements []PhaseMeasurement) float64 {
	if len(measurements) <= 1 {
		return 1.0
	}

	coherence := 0.0
	pairCount := 0

	for i := 0; i < len(measurements); i++ {
		for j := i + 1; j < len(measurements); j++ {
			// Phase coherence: cos(φᵢ - φⱼ)
			phaseDiff := measurements[i].Phase - measurements[j].Phase
			coherence += math.Cos(phaseDiff)
			pairCount++
		}
	}

	if pairCount > 0 {
		coherence /= float64(pairCount)
	}

	return math.Max(0, coherence) // Ensure non-negative
}

// calculateResonanceStrength calculates resonance strength from measurements
func (rc *ResonanceConsensus) calculateResonanceStrength(measurements []PhaseMeasurement) float64 {
	if len(measurements) == 0 {
		return 0.0
	}

	// Resonance strength based on amplitude-weighted coherence
	totalAmplitude := 0.0
	weightedCoherence := 0.0

	for _, measurement := range measurements {
		totalAmplitude += measurement.Amplitude
	}

	if totalAmplitude == 0 {
		return 0.0
	}

	for _, measurement := range measurements {
		weight := measurement.Amplitude / totalAmplitude
		// Use phase alignment as resonance indicator
		phaseCoherence := math.Cos(measurement.Phase)
		weightedCoherence += weight * phaseCoherence
	}

	return weightedCoherence
}

// ValidateProof validates a proof-of-resonance
func (rc *ResonanceConsensus) ValidateProof(proof *ProofOfResonance) error {
	// Check coherence threshold
	if proof.CoherenceScore < rc.ConsensusThreshold {
		return fmt.Errorf("coherence score %.3f below threshold %.3f",
			proof.CoherenceScore, rc.ConsensusThreshold)
	}

	// Check resonance strength
	if proof.ResonanceStrength < 0.1 { // Minimum resonance threshold
		return fmt.Errorf("resonance strength %.3f too low", proof.ResonanceStrength)
	}

	// Verify signature
	if err := rc.verifySignature(proof); err != nil {
		return fmt.Errorf("invalid signature: %w", err)
	}

	// Check timestamp (not too old)
	if time.Since(proof.Timestamp) > 5*time.Minute {
		return fmt.Errorf("proof timestamp too old")
	}

	return nil
}

// ProposeBlock proposes a new block for consensus
func (rc *ResonanceConsensus) ProposeBlock(nodeID string, globalPhaseState *GlobalPhaseState) (*ConsensusResult, error) {
	// Create proof
	proof, err := rc.CreateProof(nodeID, globalPhaseState)
	if err != nil {
		return nil, fmt.Errorf("failed to create proof: %w", err)
	}

	// Validate proof
	if err := rc.ValidateProof(proof); err != nil {
		return nil, fmt.Errorf("proof validation failed: %w", err)
	}

	// Collect signatures from validators
	signatures, validatorCount, err := rc.collectSignatures(proof)
	if err != nil {
		return nil, fmt.Errorf("failed to collect signatures: %w", err)
	}

	// Check minimum validator count
	if validatorCount < rc.MinValidators {
		return nil, fmt.Errorf("insufficient validators: %d < %d", validatorCount, rc.MinValidators)
	}

	// Create consensus result
	result := &ConsensusResult{
		Accepted:       true,
		BlockHash:      proof.Hash(),
		ValidatorCount: validatorCount,
		CoherenceScore: proof.CoherenceScore,
		Timestamp:      time.Now(),
		Signatures:     signatures,
	}

	// Add to chain
	rc.mu.Lock()
	rc.PoRChain = append(rc.PoRChain, proof)
	rc.CurrentBlock = proof
	rc.mu.Unlock()

	return result, nil
}

// collectSignatures collects signatures from validators
func (rc *ResonanceConsensus) collectSignatures(proof *ProofOfResonance) (map[string][]byte, int, error) {
	signatures := make(map[string][]byte)
	validSignatures := 0

	for _, validator := range rc.Validators {
		// In a real implementation, this would send the proof to validators
		// and collect their signatures. For now, we'll simulate.

		// Simulate signature collection (replace with actual network calls)
		signature, err := rc.signProof(proof, validator.NodeID)
		if err == nil {
			signatures[validator.NodeID] = signature
			validSignatures++
		}
	}

	return signatures, validSignatures, nil
}

// signProof signs a proof (simplified - use proper crypto in production)
func (rc *ResonanceConsensus) signProof(proof *ProofOfResonance, nodeID string) ([]byte, error) {
	// Create a hash of the proof
	hash := proof.Hash()

	// In production, use proper digital signatures (ECDSA, Ed25519, etc.)
	// For now, create a simple signature
	signature := sha256.Sum256(append([]byte(nodeID), []byte(hash)...))

	return signature[:], nil
}

// verifySignature verifies a proof signature
func (rc *ResonanceConsensus) verifySignature(proof *ProofOfResonance) error {
	// Find validator
	var validator *ValidatorNode
	for _, v := range rc.Validators {
		if v.NodeID == proof.NodeID {
			validator = v
			break
		}
	}

	if validator == nil {
		return fmt.Errorf("validator %s not found", proof.NodeID)
	}

	// In production, verify against validator's public key
	// For now, just check signature format
	if len(proof.Signature) != 32 {
		return fmt.Errorf("invalid signature length")
	}

	return nil
}

// Hash returns the hash of the proof
func (por *ProofOfResonance) Hash() string {
	// Create a deterministic representation for hashing
	data := fmt.Sprintf("%s:%d:%.6f:%.6f:%s",
		por.NodeID,
		por.BlockHeight,
		por.CoherenceScore,
		por.ResonanceStrength,
		por.Timestamp.Format(time.RFC3339Nano),
	)

	hash := sha256.Sum256([]byte(data))
	return fmt.Sprintf("%x", hash)
}

// GetLatestBlock returns the latest block in the chain
func (rc *ResonanceConsensus) GetLatestBlock() *ProofOfResonance {
	rc.mu.RLock()
	defer rc.mu.RUnlock()

	if len(rc.PoRChain) == 0 {
		return nil
	}

	return rc.PoRChain[len(rc.PoRChain)-1]
}

// GetChainLength returns the current chain length
func (rc *ResonanceConsensus) GetChainLength() int {
	rc.mu.RLock()
	defer rc.mu.RUnlock()

	return len(rc.PoRChain)
}

// GetConsensusStats returns consensus statistics
func (rc *ResonanceConsensus) GetConsensusStats() map[string]interface{} {
	rc.mu.RLock()
	defer rc.mu.RUnlock()

	avgCoherence := 0.0
	avgResonance := 0.0

	if len(rc.PoRChain) > 0 {
		for _, proof := range rc.PoRChain {
			avgCoherence += proof.CoherenceScore
			avgResonance += proof.ResonanceStrength
		}
		avgCoherence /= float64(len(rc.PoRChain))
		avgResonance /= float64(len(rc.PoRChain))
	}

	return map[string]interface{}{
		"chain_length":        len(rc.PoRChain),
		"validator_count":     len(rc.Validators),
		"avg_coherence":       avgCoherence,
		"avg_resonance":       avgResonance,
		"consensus_threshold": rc.ConsensusThreshold,
		"min_validators":      rc.MinValidators,
	}
}

// ValidateChain validates the entire proof-of-resonance chain
func (rc *ResonanceConsensus) ValidateChain() error {
	rc.mu.RLock()
	defer rc.mu.RUnlock()

	for i, proof := range rc.PoRChain {
		// Validate individual proof
		if err := rc.ValidateProof(proof); err != nil {
			return fmt.Errorf("invalid proof at height %d: %w", i, err)
		}

		// Validate chain continuity
		if i > 0 {
			expectedPrevHash := rc.PoRChain[i-1].Hash()
			if proof.PreviousHash != expectedPrevHash {
				return fmt.Errorf("chain continuity broken at height %d", i)
			}
		}
	}

	return nil
}
