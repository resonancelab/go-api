package core

import (
	"crypto/sha256"
	"fmt"
	"math"
	"math/big"
	"sync"
	"time"
)

// RSNToken represents the Reson (RSN) token from the Reson.net paper
type RSNToken struct {
	Amount    *big.Int  `json:"amount"` // Token amount in smallest unit
	Owner     string    `json:"owner"`  // Token owner address
	Timestamp time.Time `json:"timestamp"`
	Signature []byte    `json:"signature"`
}

// Transaction represents an RSN token transaction
type Transaction struct {
	ID          string    `json:"id"`
	From        string    `json:"from"`
	To          string    `json:"to"`
	Amount      *big.Int  `json:"amount"`
	Fee         *big.Int  `json:"fee"`
	Timestamp   time.Time `json:"timestamp"`
	Signature   []byte    `json:"signature"`
	BlockHeight int       `json:"block_height"`
	Status      TxStatus  `json:"status"`
}

// TxStatus represents transaction status
type TxStatus int

const (
	TxStatusPending TxStatus = iota
	TxStatusConfirmed
	TxStatusFailed
)

// ComputationalResource represents a computational resource for pricing
type ComputationalResource struct {
	NodeID       string    `json:"node_id"`
	ResourceType string    `json:"resource_type"` // "cpu", "memory", "storage", "network"
	Capacity     float64   `json:"capacity"`      // Total capacity
	Available    float64   `json:"available"`     // Available capacity
	BasePrice    *big.Int  `json:"base_price"`    // Base price in RSN
	DynamicPrice *big.Int  `json:"dynamic_price"` // Current dynamic price
	LastUpdate   time.Time `json:"last_update"`
	Utilization  float64   `json:"utilization"` // Current utilization 0-1
}

// NodeReward represents rewards earned by a node
type NodeReward struct {
	NodeID        string         `json:"node_id"`
	Period        string         `json:"period"` // Reward period (e.g., "2024-01")
	Contributions []Contribution `json:"contributions"`
	TotalReward   *big.Int       `json:"total_reward"`
	Claimed       bool           `json:"claimed"`
	ClaimedAt     *time.Time     `json:"claimed_at,omitempty"`
}

// Contribution represents a node's contribution to the network
type Contribution struct {
	Type        string    `json:"type"`   // "computation", "storage", "validation"
	Amount      float64   `json:"amount"` // Contribution amount
	Reward      *big.Int  `json:"reward"` // RSN reward for this contribution
	Timestamp   time.Time `json:"timestamp"`
	Description string    `json:"description"`
}

// RSNEconomy manages the Reson token economy
type RSNEconomy struct {
	// Token management
	TotalSupply       *big.Int            `json:"total_supply"`
	CirculatingSupply *big.Int            `json:"circulating_supply"`
	Balances          map[string]*big.Int `json:"balances"`
	Transactions      []*Transaction      `json:"transactions"`

	// Resource pricing
	Resources map[string]*ComputationalResource `json:"resources"`

	// Reward system
	NodeRewards map[string][]*NodeReward `json:"node_rewards"`
	RewardPool  *big.Int                 `json:"reward_pool"` // Total RSN available for rewards
	RewardRate  float64                  `json:"reward_rate"` // RSN per unit contribution

	// Economic parameters
	InflationRate   float64 `json:"inflation_rate"`   // Annual inflation rate
	HalvingInterval int     `json:"halving_interval"` // Blocks between halvings
	CurrentBlock    int     `json:"current_block"`

	// Thread safety
	mu sync.RWMutex `json:"-"`
}

// NewRSNEconomy creates a new RSN token economy
func NewRSNEconomy() *RSNEconomy {
	initialSupply := new(big.Int).Mul(big.NewInt(1000000000), big.NewInt(1000000000)) // 1 trillion RSN

	return &RSNEconomy{
		TotalSupply:       initialSupply,
		CirculatingSupply: big.NewInt(0),
		Balances:          make(map[string]*big.Int),
		Transactions:      make([]*Transaction, 0),
		Resources:         make(map[string]*ComputationalResource),
		NodeRewards:       make(map[string][]*NodeReward),
		RewardPool:        new(big.Int).Div(initialSupply, big.NewInt(10)), // 10% for rewards
		RewardRate:        0.001,                                           // 0.001 RSN per unit contribution
		InflationRate:     0.02,                                            // 2% annual inflation
		HalvingInterval:   210000,                                          // Bitcoin-style halving
		CurrentBlock:      0,
	}
}

// MintTokens mints new RSN tokens (governed process)
func (rse *RSNEconomy) MintTokens(amount *big.Int, recipient string) error {
	rse.mu.Lock()
	defer rse.mu.Unlock()

	// Check if minting would exceed total supply
	newSupply := new(big.Int).Add(rse.CirculatingSupply, amount)
	if newSupply.Cmp(rse.TotalSupply) > 0 {
		return fmt.Errorf("minting would exceed total supply")
	}

	// Update balances
	if rse.Balances[recipient] == nil {
		rse.Balances[recipient] = big.NewInt(0)
	}
	rse.Balances[recipient].Add(rse.Balances[recipient], amount)
	rse.CirculatingSupply.Add(rse.CirculatingSupply, amount)

	return nil
}

// TransferTokens transfers RSN tokens between addresses
func (rse *RSNEconomy) TransferTokens(from, to string, amount *big.Int) (*Transaction, error) {
	rse.mu.Lock()
	defer rse.mu.Unlock()

	// Check balance
	fromBalance := rse.Balances[from]
	if fromBalance == nil || fromBalance.Cmp(amount) < 0 {
		return nil, fmt.Errorf("insufficient balance")
	}

	// Calculate fee (0.1% of transaction)
	fee := new(big.Int).Div(amount, big.NewInt(1000))
	if fee.Cmp(big.NewInt(1)) < 0 {
		fee = big.NewInt(1) // Minimum fee
	}

	totalDeduct := new(big.Int).Add(amount, fee)

	// Check balance again with fee
	if fromBalance.Cmp(totalDeduct) < 0 {
		return nil, fmt.Errorf("insufficient balance including fee")
	}

	// Update balances
	fromBalance.Sub(fromBalance, totalDeduct)

	if rse.Balances[to] == nil {
		rse.Balances[to] = big.NewInt(0)
	}
	rse.Balances[to].Add(rse.Balances[to], amount)

	// Add fee to reward pool
	rse.RewardPool.Add(rse.RewardPool, fee)

	// Create transaction
	tx := &Transaction{
		ID:          rse.generateTransactionID(from, to, amount),
		From:        from,
		To:          to,
		Amount:      amount,
		Fee:         fee,
		Timestamp:   time.Now(),
		BlockHeight: rse.CurrentBlock,
		Status:      TxStatusConfirmed,
	}

	rse.Transactions = append(rse.Transactions, tx)
	return tx, nil
}

// RegisterComputationalResource registers a computational resource for pricing
func (rse *RSNEconomy) RegisterComputationalResource(nodeID, resourceType string, capacity float64, basePrice *big.Int) error {
	rse.mu.Lock()
	defer rse.mu.Unlock()

	resourceID := fmt.Sprintf("%s:%s", nodeID, resourceType)

	resource := &ComputationalResource{
		NodeID:       nodeID,
		ResourceType: resourceType,
		Capacity:     capacity,
		Available:    capacity,
		BasePrice:    basePrice,
		DynamicPrice: new(big.Int).Set(basePrice),
		LastUpdate:   time.Now(),
		Utilization:  0.0,
	}

	rse.Resources[resourceID] = resource
	return nil
}

// CalculateResourcePrice calculates the dynamic price for a computational resource
func (rse *RSNEconomy) CalculateResourcePrice(nodeID, resourceType string, requestedAmount float64) (*big.Int, error) {
	rse.mu.RLock()
	defer rse.mu.RUnlock()

	resourceID := fmt.Sprintf("%s:%s", nodeID, resourceType)
	resource, exists := rse.Resources[resourceID]
	if !exists {
		return nil, fmt.Errorf("resource not found: %s", resourceID)
	}

	// Update utilization
	resource.Utilization = (resource.Capacity - resource.Available) / resource.Capacity

	// Calculate dynamic price based on utilization
	utilizationFactor := 1.0 + resource.Utilization*2.0 // 1x to 3x multiplier

	// Demand factor based on requested amount
	demandFactor := 1.0
	if requestedAmount > resource.Available {
		demandFactor = 2.0 // Premium for over-subscription
	}

	// Calculate final price
	priceMultiplier := utilizationFactor * demandFactor
	dynamicPrice := new(big.Int).Mul(resource.BasePrice, big.NewInt(int64(math.Ceil(priceMultiplier*100))))
	dynamicPrice.Div(dynamicPrice, big.NewInt(100))

	resource.DynamicPrice = dynamicPrice
	resource.LastUpdate = time.Now()

	return dynamicPrice, nil
}

// AllocateResource allocates computational resources and charges RSN tokens
func (rse *RSNEconomy) AllocateResource(userID, nodeID, resourceType string, amount float64, duration time.Duration) (*big.Int, error) {
	// Calculate price
	price, err := rse.CalculateResourcePrice(nodeID, resourceType, amount)
	if err != nil {
		return nil, err
	}

	// Scale price by duration (assuming hourly billing)
	hours := duration.Hours()
	if hours < 1 {
		hours = 1 // Minimum 1 hour
	}
	totalPrice := new(big.Int).Mul(price, big.NewInt(int64(math.Ceil(hours))))

	// Transfer payment
	_, err = rse.TransferTokens(userID, nodeID, totalPrice)
	if err != nil {
		return nil, fmt.Errorf("payment failed: %w", err)
	}

	// Update resource availability
	resourceID := fmt.Sprintf("%s:%s", nodeID, resourceType)
	if resource, exists := rse.Resources[resourceID]; exists {
		resource.Available -= amount
		if resource.Available < 0 {
			resource.Available = 0
		}
	}

	return totalPrice, nil
}

// RecordNodeContribution records a node's contribution for reward calculation
func (rse *RSNEconomy) RecordNodeContribution(nodeID, contributionType, description string, amount float64) error {
	rse.mu.Lock()
	defer rse.mu.Unlock()

	// Calculate reward
	rewardAmount := int64(amount * rse.RewardRate * 1000000) // Convert to smallest RSN unit
	reward := big.NewInt(rewardAmount)

	// Check reward pool
	if reward.Cmp(rse.RewardPool) > 0 {
		return fmt.Errorf("insufficient reward pool")
	}

	contribution := &Contribution{
		Type:        contributionType,
		Amount:      amount,
		Reward:      reward,
		Timestamp:   time.Now(),
		Description: description,
	}

	// Get or create current period rewards
	period := time.Now().Format("2006-01")
	if rse.NodeRewards[nodeID] == nil {
		rse.NodeRewards[nodeID] = make([]*NodeReward, 0)
	}

	var currentReward *NodeReward
	for _, nr := range rse.NodeRewards[nodeID] {
		if nr.Period == period && !nr.Claimed {
			currentReward = nr
			break
		}
	}

	if currentReward == nil {
		currentReward = &NodeReward{
			NodeID:        nodeID,
			Period:        period,
			Contributions: make([]Contribution, 0),
			TotalReward:   big.NewInt(0),
			Claimed:       false,
		}
		rse.NodeRewards[nodeID] = append(rse.NodeRewards[nodeID], currentReward)
	}

	// Add contribution
	currentReward.Contributions = append(currentReward.Contributions, *contribution)
	currentReward.TotalReward.Add(currentReward.TotalReward, reward)

	// Deduct from reward pool
	rse.RewardPool.Sub(rse.RewardPool, reward)

	return nil
}

// ClaimNodeRewards allows a node to claim accumulated rewards
func (rse *RSNEconomy) ClaimNodeRewards(nodeID, period string) (*big.Int, error) {
	rse.mu.Lock()

	rewards := rse.NodeRewards[nodeID]
	if rewards == nil {
		rse.mu.Unlock()
		return nil, fmt.Errorf("no rewards found for node %s", nodeID)
	}

	var targetReward *NodeReward
	for _, reward := range rewards {
		if reward.Period == period && !reward.Claimed {
			targetReward = reward
			break
		}
	}

	if targetReward == nil {
		rse.mu.Unlock()
		return nil, fmt.Errorf("no unclaimed rewards found for period %s", period)
	}

	// Get reward amount before releasing lock
	rewardAmount := new(big.Int).Set(targetReward.TotalReward)

	// Mark as claimed
	now := time.Now()
	targetReward.Claimed = true
	targetReward.ClaimedAt = &now

	rse.mu.Unlock()

	// Mint reward tokens to node (outside of lock to avoid deadlock)
	err := rse.MintTokens(rewardAmount, nodeID)
	if err != nil {
		// If minting fails, revert the claimed status
		rse.mu.Lock()
		targetReward.Claimed = false
		targetReward.ClaimedAt = nil
		rse.mu.Unlock()
		return nil, fmt.Errorf("failed to mint reward tokens: %w", err)
	}

	return rewardAmount, nil
}

// GetBalance returns the RSN balance for an address
func (rse *RSNEconomy) GetBalance(address string) *big.Int {
	rse.mu.RLock()
	defer rse.mu.RUnlock()

	balance := rse.Balances[address]
	if balance == nil {
		return big.NewInt(0)
	}

	return new(big.Int).Set(balance)
}

// GetResourcePrice returns the current price for a computational resource
func (rse *RSNEconomy) GetResourcePrice(nodeID, resourceType string) (*big.Int, error) {
	rse.mu.RLock()
	defer rse.mu.RUnlock()

	resourceID := fmt.Sprintf("%s:%s", nodeID, resourceType)
	resource, exists := rse.Resources[resourceID]
	if !exists {
		return nil, fmt.Errorf("resource not found: %s", resourceID)
	}

	return resource.DynamicPrice, nil
}

// GetNodeRewards returns all rewards for a node
func (rse *RSNEconomy) GetNodeRewards(nodeID string) []*NodeReward {
	rse.mu.RLock()
	defer rse.mu.RUnlock()

	rewards := rse.NodeRewards[nodeID]
	if rewards == nil {
		return []*NodeReward{}
	}

	// Return copy to avoid external modification
	result := make([]*NodeReward, len(rewards))
	for i, reward := range rewards {
		result[i] = &NodeReward{
			NodeID:        reward.NodeID,
			Period:        reward.Period,
			Contributions: make([]Contribution, len(reward.Contributions)),
			TotalReward:   new(big.Int).Set(reward.TotalReward),
			Claimed:       reward.Claimed,
		}
		copy(result[i].Contributions, reward.Contributions)
		if reward.ClaimedAt != nil {
			claimedAt := *reward.ClaimedAt
			result[i].ClaimedAt = &claimedAt
		}
	}

	return result
}

// GetEconomyStats returns comprehensive economy statistics
func (rse *RSNEconomy) GetEconomyStats() map[string]interface{} {
	rse.mu.RLock()
	defer rse.mu.RUnlock()

	totalBalances := big.NewInt(0)
	activeAddresses := 0

	for _, balance := range rse.Balances {
		if balance.Cmp(big.NewInt(0)) > 0 {
			totalBalances.Add(totalBalances, balance)
			activeAddresses++
		}
	}

	totalResources := len(rse.Resources)
	totalRewards := big.NewInt(0)

	for _, rewards := range rse.NodeRewards {
		for _, reward := range rewards {
			totalRewards.Add(totalRewards, reward.TotalReward)
		}
	}

	return map[string]interface{}{
		"total_supply":       rse.TotalSupply.String(),
		"circulating_supply": rse.CirculatingSupply.String(),
		"reward_pool":        rse.RewardPool.String(),
		"active_addresses":   activeAddresses,
		"total_transactions": len(rse.Transactions),
		"total_resources":    totalResources,
		"total_rewards":      totalRewards.String(),
		"inflation_rate":     rse.InflationRate,
		"reward_rate":        rse.RewardRate,
		"current_block":      rse.CurrentBlock,
	}
}

// generateTransactionID generates a unique transaction ID
func (rse *RSNEconomy) generateTransactionID(from, to string, amount *big.Int) string {
	data := fmt.Sprintf("%s:%s:%s:%d", from, to, amount.String(), time.Now().UnixNano())
	hash := sha256.Sum256([]byte(data))
	return fmt.Sprintf("%x", hash[:16]) // First 16 bytes as hex
}

// ProcessResonancePayment processes payment for resonance-based computations
func (rse *RSNEconomy) ProcessResonancePayment(userID string, computation *ResonanceComputation) (*big.Int, error) {
	// Calculate computational complexity
	complexity := rse.calculateComputationComplexity(computation)

	// Get base price per complexity unit
	basePrice := big.NewInt(1000) // 1000 RSN per complexity unit

	// Calculate total price
	totalPrice := new(big.Int).Mul(basePrice, big.NewInt(int64(complexity)))

	// Apply resonance discount (better coherence = lower price)
	coherenceDiscount := computation.GlobalCoherence * 0.5 // Up to 50% discount
	discountAmount := new(big.Int).Mul(totalPrice, big.NewInt(int64(coherenceDiscount*100)))
	discountAmount.Div(discountAmount, big.NewInt(100))

	finalPrice := new(big.Int).Sub(totalPrice, discountAmount)

	// Process payment
	_, err := rse.TransferTokens(userID, "resonance_pool", finalPrice)
	if err != nil {
		return nil, fmt.Errorf("payment processing failed: %w", err)
	}

	return finalPrice, nil
}

// calculateComputationComplexity calculates the complexity of a resonance computation
func (rse *RSNEconomy) calculateComputationComplexity(computation *ResonanceComputation) float64 {
	// Base complexity from number of oscillators and states
	baseComplexity := float64(len(computation.Oscillators) + len(computation.QuaternionicStates))

	// Add complexity from coherence calculations
	coherenceComplexity := 1.0 / (1.0 + computation.GlobalCoherence) // Higher coherence = lower complexity

	// Add time complexity
	timeComplexity := computation.Duration.Seconds() / 60.0 // Per minute

	return baseComplexity * coherenceComplexity * timeComplexity
}

// ResonanceComputation represents a resonance-based computation for pricing
type ResonanceComputation struct {
	Oscillators        []*PrimeOscillator   `json:"oscillators"`
	QuaternionicStates []*QuaternionicState `json:"quaternionic_states"`
	GlobalCoherence    float64              `json:"global_coherence"`
	Duration           time.Duration        `json:"duration"`
	Complexity         float64              `json:"complexity"`
}
