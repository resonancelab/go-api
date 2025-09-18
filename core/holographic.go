package core

import (
	"fmt"
	"math"
	"math/cmplx"
	"sync"
	"time"
)

// MemoryField represents a local holographic memory field from Reson.net paper
// Stores patterns as M(r) with holographic retrieval capabilities
type MemoryField struct {
	NodeID             string                           `json:"node_id"`
	Patterns           map[string]*Pattern              `json:"patterns"`
	Strengths          map[string]float64               `json:"strengths"`
	InterferenceMatrix map[string]map[string]complex128 `json:"interference_matrix"`
	LastUpdate         time.Time                        `json:"last_update"`
	mu                 sync.RWMutex                     `json:"-"`
}

// Pattern represents a stored memory pattern
type Pattern struct {
	ID          string       `json:"id"`
	Data        []complex128 `json:"data"`      // Pattern data
	Phase       float64      `json:"phase"`     // Phase offset
	Amplitude   float64      `json:"amplitude"` // Amplitude scaling
	Timestamp   time.Time    `json:"timestamp"`
	AccessCount int          `json:"access_count"`
}

// HolographicMemory implements distributed holographic memory from Reson.net paper
// Uses global coherence C(t) for retrieval and reconstruction
type HolographicMemory struct {
	LocalFields           map[string]*MemoryField `json:"local_fields"`
	GlobalCoherence       float64                 `json:"global_coherence"`
	RetrievalThreshold    float64                 `json:"retrieval_threshold"`
	InterferenceThreshold float64                 `json:"interference_threshold"`
	MaxPatterns           int                     `json:"max_patterns"`
	mu                    sync.RWMutex            `json:"-"`
}

// NewHolographicMemory creates a new holographic memory system
func NewHolographicMemory(retrievalThreshold, interferenceThreshold float64, maxPatterns int) *HolographicMemory {
	return &HolographicMemory{
		LocalFields:           make(map[string]*MemoryField),
		GlobalCoherence:       0.0,
		RetrievalThreshold:    retrievalThreshold,
		InterferenceThreshold: interferenceThreshold,
		MaxPatterns:           maxPatterns,
	}
}

// AddMemoryField adds a memory field for a specific node
func (hm *HolographicMemory) AddMemoryField(nodeID string) *MemoryField {
	hm.mu.Lock()
	defer hm.mu.Unlock()

	field := &MemoryField{
		NodeID:             nodeID,
		Patterns:           make(map[string]*Pattern),
		Strengths:          make(map[string]float64),
		InterferenceMatrix: make(map[string]map[string]complex128),
		LastUpdate:         time.Now(),
	}

	hm.LocalFields[nodeID] = field
	return field
}

// StorePattern stores a pattern in the holographic memory
func (mf *MemoryField) StorePattern(id string, data []complex128, phase, amplitude float64) error {
	mf.mu.Lock()
	defer mf.mu.Unlock()

	// Check capacity
	if len(mf.Patterns) >= 1000 { // Reasonable limit per field
		return fmt.Errorf("memory field capacity exceeded")
	}

	pattern := &Pattern{
		ID:          id,
		Data:        make([]complex128, len(data)),
		Phase:       phase,
		Amplitude:   amplitude,
		Timestamp:   time.Now(),
		AccessCount: 0,
	}

	copy(pattern.Data, data)
	mf.Patterns[id] = pattern
	mf.Strengths[id] = amplitude
	mf.LastUpdate = time.Now()

	// Update interference matrix
	mf.updateInterferenceMatrix(id, pattern)

	return nil
}

// updateInterferenceMatrix updates the interference relationships between patterns
func (mf *MemoryField) updateInterferenceMatrix(newID string, newPattern *Pattern) {
	// Initialize interference row for new pattern
	mf.InterferenceMatrix[newID] = make(map[string]complex128)

	// Calculate interference with existing patterns
	for existingID, existingPattern := range mf.Patterns {
		if existingID == newID {
			continue
		}

		// Compute holographic interference: ⟨pattern₁|pattern₂⟩
		interference := mf.computeInterference(newPattern.Data, existingPattern.Data)
		mf.InterferenceMatrix[newID][existingID] = interference
		mf.InterferenceMatrix[existingID][newID] = cmplx.Conj(interference)
	}
}

// computeInterference calculates the inner product between two patterns
func (mf *MemoryField) computeInterference(pattern1, pattern2 []complex128) complex128 {
	if len(pattern1) != len(pattern2) {
		return complex(0, 0)
	}

	interference := complex(0, 0)
	for i := range pattern1 {
		interference += pattern1[i] * cmplx.Conj(pattern2[i])
	}

	return interference
}

// RetrievePattern retrieves a pattern using holographic reconstruction
func (mf *MemoryField) RetrievePattern(id string) (*Pattern, error) {
	mf.mu.RLock()
	defer mf.mu.RUnlock()

	pattern, exists := mf.Patterns[id]
	if !exists {
		return nil, fmt.Errorf("pattern %s not found", id)
	}

	// Increment access count
	pattern.AccessCount++

	// Create a copy to avoid external modification
	retrieved := &Pattern{
		ID:          pattern.ID,
		Data:        make([]complex128, len(pattern.Data)),
		Phase:       pattern.Phase,
		Amplitude:   pattern.Amplitude,
		Timestamp:   time.Now(),
		AccessCount: pattern.AccessCount,
	}
	copy(retrieved.Data, pattern.Data)

	return retrieved, nil
}

// RetrieveWithInterference retrieves a pattern considering interference effects
func (mf *MemoryField) RetrieveWithInterference(id string, coherence float64) (*Pattern, error) {
	pattern, err := mf.RetrievePattern(id)
	if err != nil {
		return nil, err
	}

	// Apply interference correction based on coherence
	if coherence < mf.InterferenceMatrixThreshold() {
		// High interference - apply correction
		pattern = mf.applyInterferenceCorrection(pattern, coherence)
	}

	return pattern, nil
}

// applyInterferenceCorrection applies holographic interference correction
func (mf *MemoryField) applyInterferenceCorrection(pattern *Pattern, coherence float64) *Pattern {
	corrected := &Pattern{
		ID:          pattern.ID,
		Data:        make([]complex128, len(pattern.Data)),
		Phase:       pattern.Phase,
		Amplitude:   pattern.Amplitude * coherence, // Reduce amplitude due to interference
		Timestamp:   time.Now(),
		AccessCount: pattern.AccessCount,
	}

	// Apply phase correction based on interference matrix
	interferenceCorrection := complex(coherence, 0)
	for i := range pattern.Data {
		corrected.Data[i] = pattern.Data[i] * interferenceCorrection
	}

	return corrected
}

// InterferenceMatrixThreshold returns the threshold for interference correction
func (mf *MemoryField) InterferenceMatrixThreshold() float64 {
	// Simple threshold based on pattern density
	patternCount := len(mf.Patterns)
	if patternCount == 0 {
		return 1.0
	}

	// Higher density means more interference
	density := float64(patternCount) / 1000.0 // Normalized to capacity
	return math.Max(0.1, 1.0-density*0.5)
}

// GetPatternStrength returns the storage strength of a pattern
func (mf *MemoryField) GetPatternStrength(id string) (float64, bool) {
	mf.mu.RLock()
	defer mf.mu.RUnlock()

	strength, exists := mf.Strengths[id]
	return strength, exists
}

// ListPatterns returns all pattern IDs in the memory field
func (mf *MemoryField) ListPatterns() []string {
	mf.mu.RLock()
	defer mf.mu.RUnlock()

	ids := make([]string, 0, len(mf.Patterns))
	for id := range mf.Patterns {
		ids = append(ids, id)
	}

	return ids
}

// GetStats returns memory field statistics
func (mf *MemoryField) GetStats() map[string]interface{} {
	mf.mu.RLock()
	defer mf.mu.RUnlock()

	return map[string]interface{}{
		"node_id":          mf.NodeID,
		"pattern_count":    len(mf.Patterns),
		"total_strength":   mf.computeTotalStrength(),
		"last_update":      mf.LastUpdate,
		"avg_access_count": mf.computeAvgAccessCount(),
	}
}

// computeTotalStrength calculates the total storage strength
func (mf *MemoryField) computeTotalStrength() float64 {
	total := 0.0
	for _, strength := range mf.Strengths {
		total += strength
	}
	return total
}

// computeAvgAccessCount calculates average pattern access count
func (mf *MemoryField) computeAvgAccessCount() float64 {
	if len(mf.Patterns) == 0 {
		return 0.0
	}

	total := 0
	for _, pattern := range mf.Patterns {
		total += pattern.AccessCount
	}

	return float64(total) / float64(len(mf.Patterns))
}

// GlobalRetrieve performs global holographic retrieval across all memory fields
func (hm *HolographicMemory) GlobalRetrieve(patternID string) (*Pattern, error) {
	hm.mu.RLock()
	defer hm.mu.RUnlock()

	var bestPattern *Pattern
	var maxCoherence float64

	// Try retrieval from each memory field
	for _, field := range hm.LocalFields {
		pattern, err := field.RetrieveWithInterference(patternID, hm.GlobalCoherence)
		if err != nil {
			continue // Pattern not in this field
		}

		// Use coherence to determine best retrieval
		fieldCoherence := hm.computeFieldCoherence(field, patternID)
		if fieldCoherence > maxCoherence {
			maxCoherence = fieldCoherence
			bestPattern = pattern
		}
	}

	if bestPattern == nil {
		return nil, fmt.Errorf("pattern %s not found in any memory field", patternID)
	}

	// Check if coherence meets retrieval threshold
	if maxCoherence < hm.RetrievalThreshold {
		return nil, fmt.Errorf("insufficient coherence for retrieval: %.3f < %.3f",
			maxCoherence, hm.RetrievalThreshold)
	}

	return bestPattern, nil
}

// computeFieldCoherence computes coherence for a specific field and pattern
func (hm *HolographicMemory) computeFieldCoherence(field *MemoryField, patternID string) float64 {
	strength, exists := field.GetPatternStrength(patternID)
	if !exists {
		return 0.0
	}

	// Coherence based on storage strength and global coherence
	fieldCoherence := strength * hm.GlobalCoherence

	// Apply interference penalty
	interferencePenalty := field.InterferenceMatrixThreshold()
	fieldCoherence *= interferencePenalty

	return fieldCoherence
}

// UpdateGlobalCoherence updates the global coherence for holographic retrieval
func (hm *HolographicMemory) UpdateGlobalCoherence(globalPhaseState *GlobalPhaseState) {
	hm.mu.Lock()
	defer hm.mu.Unlock()

	// Use the global coherence from the phase state
	hm.GlobalCoherence = globalPhaseState.GlobalCoherence
}

// GetGlobalStats returns global memory statistics
func (hm *HolographicMemory) GetGlobalStats() map[string]interface{} {
	hm.mu.RLock()
	defer hm.mu.RUnlock()

	totalPatterns := 0
	totalFields := len(hm.LocalFields)
	totalStrength := 0.0

	for _, field := range hm.LocalFields {
		stats := field.GetStats()
		totalPatterns += stats["pattern_count"].(int)
		totalStrength += stats["total_strength"].(float64)
	}

	return map[string]interface{}{
		"total_fields":           totalFields,
		"total_patterns":         totalPatterns,
		"total_strength":         totalStrength,
		"global_coherence":       hm.GlobalCoherence,
		"retrieval_threshold":    hm.RetrievalThreshold,
		"avg_patterns_per_field": float64(totalPatterns) / float64(totalFields),
	}
}

// Cleanup removes patterns with low access counts or old timestamps
func (hm *HolographicMemory) Cleanup(maxAge time.Duration, minAccessCount int) int {
	hm.mu.Lock()
	defer hm.mu.Unlock()

	removed := 0
	cutoff := time.Now().Add(-maxAge)

	for _, field := range hm.LocalFields {
		field.mu.Lock()

		for id, pattern := range field.Patterns {
			// Remove old or rarely accessed patterns
			if pattern.Timestamp.Before(cutoff) || pattern.AccessCount < minAccessCount {
				delete(field.Patterns, id)
				delete(field.Strengths, id)
				delete(field.InterferenceMatrix, id)

				// Remove from interference matrix of other patterns
				for otherID := range field.InterferenceMatrix {
					if field.InterferenceMatrix[otherID] != nil {
						delete(field.InterferenceMatrix[otherID], id)
					}
				}

				removed++
			}
		}

		field.LastUpdate = time.Now()
		field.mu.Unlock()
	}

	return removed
}
