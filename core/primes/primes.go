package primes

import (
	"fmt"
	"math"
	"sort"
	"sync"
)

// PrimeFactor represents a prime factorization component
type PrimeFactor struct {
	Prime    int `json:"prime"`
	Exponent int `json:"exponent"`
}

// PrimeEngine provides prime number operations for the Ψ0=1 formalism
type PrimeEngine struct {
	sieve      *Sieve
	primeCache []int
	mu         sync.RWMutex
	factorCache map[int][]PrimeFactor
}

// Sieve implements the Sieve of Eratosthenes for efficient prime generation
type Sieve struct {
	limit  int
	primes []bool
}

// NewPrimeEngine creates a new prime engine with specified limit
func NewPrimeEngine(limit int) *PrimeEngine {
	sieve := NewSieve(limit)
	primes := sieve.GeneratePrimes()
	
	return &PrimeEngine{
		sieve:       sieve,
		primeCache:  primes,
		factorCache: make(map[int][]PrimeFactor),
	}
}

// NewSieve creates a new Sieve of Eratosthenes
func NewSieve(limit int) *Sieve {
	primes := make([]bool, limit+1)
	for i := 2; i <= limit; i++ {
		primes[i] = true
	}
	
	for i := 2; i*i <= limit; i++ {
		if primes[i] {
			for j := i * i; j <= limit; j += i {
				primes[j] = false
			}
		}
	}
	
	return &Sieve{
		limit:  limit,
		primes: primes,
	}
}

// GeneratePrimes returns all primes up to the sieve limit
func (s *Sieve) GeneratePrimes() []int {
	var primes []int
	for i := 2; i <= s.limit; i++ {
		if s.primes[i] {
			primes = append(primes, i)
		}
	}
	return primes
}

// IsPrime checks if a number is prime
func (s *Sieve) IsPrime(n int) bool {
	if n <= s.limit {
		return s.primes[n]
	}
	
	// For numbers beyond sieve limit, use trial division
	if n < 2 {
		return false
	}
	if n == 2 {
		return true
	}
	if n%2 == 0 {
		return false
	}
	
	sqrt := int(math.Sqrt(float64(n)))
	for i := 3; i <= sqrt; i += 2 {
		if n%i == 0 {
			return false
		}
	}
	return true
}

// GetPrimeBasis returns the first n primes for Hilbert space basis
func (pe *PrimeEngine) GetPrimeBasis(n int) []int {
	pe.mu.RLock()
	defer pe.mu.RUnlock()
	
	if n <= len(pe.primeCache) {
		return pe.primeCache[:n]
	}
	
	// If we need more primes, extend the cache
	pe.mu.RUnlock()
	pe.mu.Lock()
	defer pe.mu.Unlock()
	
	// Double-check after acquiring write lock
	if n <= len(pe.primeCache) {
		return pe.primeCache[:n]
	}
	
	// Generate more primes
	limit := pe.sieve.limit
	for len(pe.primeCache) < n {
		limit *= 2
		newSieve := NewSieve(limit)
		pe.sieve = newSieve
		pe.primeCache = newSieve.GeneratePrimes()
	}
	
	return pe.primeCache[:n]
}

// Factorize returns the prime factorization of n
func (pe *PrimeEngine) Factorize(n int) []PrimeFactor {
	if n < 2 {
		return []PrimeFactor{}
	}
	
	pe.mu.RLock()
	if cached, exists := pe.factorCache[n]; exists {
		pe.mu.RUnlock()
		return cached
	}
	pe.mu.RUnlock()
	
	// Compute factorization
	factors := pe.computeFactorization(n)
	
	// Cache the result
	pe.mu.Lock()
	pe.factorCache[n] = factors
	pe.mu.Unlock()
	
	return factors
}

// computeFactorization performs the actual factorization
func (pe *PrimeEngine) computeFactorization(n int) []PrimeFactor {
	var factors []PrimeFactor
	temp := n
	
	// Try all primes in our cache first
	for _, prime := range pe.primeCache {
		if prime*prime > temp {
			break
		}
		
		exponent := 0
		for temp%prime == 0 {
			temp /= prime
			exponent++
		}
		
		if exponent > 0 {
			factors = append(factors, PrimeFactor{
				Prime:    prime,
				Exponent: exponent,
			})
		}
		
		if temp == 1 {
			break
		}
	}
	
	// If temp > 1, it's a prime factor we haven't found
	if temp > 1 {
		factors = append(factors, PrimeFactor{
			Prime:    temp,
			Exponent: 1,
		})
	}
	
	return factors
}

// ComputePrimeResonance calculates resonance between two primes
// Based on R(n)|p⟩ = e^(2πi log_p n)|p⟩
func (pe *PrimeEngine) ComputePrimeResonance(p1, p2 int) float64 {
	if !pe.sieve.IsPrime(p1) || !pe.sieve.IsPrime(p2) {
		return 0.0
	}
	
	if p1 == p2 {
		return 1.0
	}
	
	// Compute log_p1(p2) = ln(p2)/ln(p1)
	logRatio := math.Log(float64(p2)) / math.Log(float64(p1))
	
	// Resonance phase: 2π * log_p1(p2)
	phase := 2.0 * math.Pi * logRatio
	
	// Return the magnitude of the complex exponential (always 1 for prime resonance)
	// But we'll use a more nuanced resonance based on prime gap relationships
	gap1 := pe.findPrimeGap(p1)
	gap2 := pe.findPrimeGap(p2)
	
	// Resonance strength based on gap similarity and ratio relationships
	gapSimilarity := 1.0 - math.Abs(float64(gap1-gap2))/math.Max(float64(gap1), float64(gap2))
	phaseResonance := math.Cos(phase)
	
	return 0.5 * (gapSimilarity + math.Abs(phaseResonance))
}

// findPrimeGap finds the gap to the next prime
func (pe *PrimeEngine) findPrimeGap(p int) int {
	idx := sort.SearchInts(pe.primeCache, p)
	if idx < len(pe.primeCache)-1 {
		return pe.primeCache[idx+1] - p
	}
	
	// If not in cache, find next prime manually
	next := p + 1
	if p == 2 {
		next = 3
	} else {
		next = p + 2 // Skip even numbers for odd primes
	}
	
	for !pe.sieve.IsPrime(next) {
		if p == 2 {
			next++
		} else {
			next += 2
		}
	}
	
	return next - p
}

// GenerateCompositeState creates composite number state from prime factorization
// |n⟩ = Σᵢ √(aᵢ/A)|pᵢ⟩ where n = ∏pᵢ^aᵢ and A = Σᵢaᵢ
func (pe *PrimeEngine) GenerateCompositeState(n int) ([]int, []float64) {
	factors := pe.Factorize(n)
	if len(factors) == 0 {
		return []int{}, []float64{}
	}
	
	// Calculate total weight A = Σᵢaᵢ
	totalWeight := 0
	for _, factor := range factors {
		totalWeight += factor.Exponent
	}
	
	// Generate state coefficients
	primes := make([]int, len(factors))
	coefficients := make([]float64, len(factors))
	
	for i, factor := range factors {
		primes[i] = factor.Prime
		coefficients[i] = math.Sqrt(float64(factor.Exponent) / float64(totalWeight))
	}
	
	return primes, coefficients
}

// EulerPhi computes Euler's totient function φ(n)
func (pe *PrimeEngine) EulerPhi(n int) int {
	if n <= 1 {
		return 0
	}
	
	factors := pe.Factorize(n)
	result := n
	
	for _, factor := range factors {
		result = result * (factor.Prime - 1) / factor.Prime
	}
	
	return result
}

// MobiusFunction computes the Möbius function μ(n)
func (pe *PrimeEngine) MobiusFunction(n int) int {
	if n == 1 {
		return 1
	}
	
	factors := pe.Factorize(n)
	
	// Check for squared prime factors
	for _, factor := range factors {
		if factor.Exponent > 1 {
			return 0
		}
	}
	
	// If all exponents are 1, return (-1)^k where k is number of prime factors
	if len(factors)%2 == 0 {
		return 1
	}
	return -1
}

// VonMangoldtFunction computes the von Mangoldt function Λ(n)
func (pe *PrimeEngine) VonMangoldtFunction(n int) float64 {
	if n <= 1 {
		return 0.0
	}
	
	factors := pe.Factorize(n)
	
	// Λ(n) = ln(p) if n = p^k for some prime p and k ≥ 1, otherwise 0
	if len(factors) == 1 {
		return math.Log(float64(factors[0].Prime))
	}
	
	return 0.0
}

// GetPrimeCount returns the number of primes up to n
func (pe *PrimeEngine) GetPrimeCount(n int) int {
	count := 0
	for _, prime := range pe.primeCache {
		if prime <= n {
			count++
		} else {
			break
		}
	}
	return count
}

// ValidatePrimeBasis ensures the given primes form a valid basis
func (pe *PrimeEngine) ValidatePrimeBasis(primes []int) error {
	for i, p := range primes {
		if !pe.sieve.IsPrime(p) {
			return fmt.Errorf("element %d at index %d is not prime", p, i)
		}
	}
	
	// Check for duplicates
	seen := make(map[int]bool)
	for i, p := range primes {
		if seen[p] {
			return fmt.Errorf("duplicate prime %d at index %d", p, i)
		}
		seen[p] = true
	}
	
	return nil
}

// ComputeResonanceFrequency calculates the resonance frequency for a prime
// Based on the prime's position in the sequence and harmonic relationships
func (pe *PrimeEngine) ComputeResonanceFrequency(prime int) float64 {
	if !pe.sieve.IsPrime(prime) {
		return 0.0
	}
	
	// Find position of prime in sequence
	position := 0
	for i, p := range pe.primeCache {
		if p == prime {
			position = i + 1 // 1-indexed
			break
		}
	}
	
	if position == 0 {
		return 0.0
	}
	
	// Base frequency scaled by prime position and logarithmic relationship
	baseFreq := 1.0
	frequency := baseFreq * float64(position) / math.Log(float64(prime))
	
	return frequency
}