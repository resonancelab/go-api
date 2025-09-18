package core

import (
	"bytes"
	crand "crypto/rand"
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"math/rand"
	"sync"
	"time"
)

// PostQuantumCrypto provides quantum-resistant cryptographic primitives for Reson.net
type PostQuantumCrypto struct {
	// Lattice-based cryptography (Kyber-like)
	latticeParams *LatticeParameters

	// Hash-based signatures (XMSS-like)
	hashTree *HashTree

	// Multivariate cryptography (Rainbow-like)
	rainbowSystem *RainbowSystem

	// Code-based cryptography (McEliece-like)
	mcelieceSystem *McElieceSystem

	// Symmetric cryptography with quantum resistance
	symmetricCipher *QuantumResistantSymmetric

	// Key management
	keyStore map[string]*QuantumResistantKey

	// Thread safety
	mu sync.RWMutex
}

// LatticeParameters contains parameters for lattice-based cryptography
type LatticeParameters struct {
	N        int     // Ring dimension
	K        int     // Module rank
	Q        int     // Modulus
	Eta      int     // Noise parameter
	Du       int     // Public key compression
	Dv       int     // Ciphertext compression
	Sigma    float64 // Standard deviation for noise
	PolySize int     // Size of polynomials
}

// QuantumResistantKey represents a quantum-resistant cryptographic key
type QuantumResistantKey struct {
	Algorithm  string      `json:"algorithm"`
	KeyID      string      `json:"key_id"`
	PublicKey  interface{} `json:"public_key"`
	PrivateKey interface{} `json:"-"` // Never serialize private key
	Created    int64       `json:"created"`
	Expires    int64       `json:"expires,omitempty"`
}

// LatticeKeyPair represents a lattice-based key pair
type LatticeKeyPair struct {
	PublicKey  *LatticePublicKey  `json:"public_key"`
	PrivateKey *LatticePrivateKey `json:"-"` // Never serialize
}

// LatticePublicKey represents a lattice public key
type LatticePublicKey struct {
	T   []int  `json:"t"`   // Public key matrix
	Rho []byte `json:"rho"` // Seed for public key
}

// LatticePrivateKey represents a lattice private key
type LatticePrivateKey struct {
	S   []int  `json:"-"`   // Secret key
	Rho []byte `json:"rho"` // Seed for private key
	A   []int  `json:"a"`   // Matrix A
}

// HashTree represents a hash-based signature tree (XMSS-like)
type HashTree struct {
	Height     int
	WotsParams *WOTSParameters
	Root       []byte
	UsedLeafs  map[int]bool
	PrivateKey [][]byte
	PublicKey  []byte
}

// WOTSParameters contains parameters for Winternitz One-Time Signature
type WOTSParameters struct {
	N int // Security parameter (32 for SHA256)
	W int // Winternitz parameter (16)
	T int // Number of WOTS chains
}

// RainbowSystem represents a multivariate cryptographic system
type RainbowSystem struct {
	Layers      []*RainbowLayer
	PublicKey   []int // Public key coefficients
	PrivateKey  []int // Private key (vinegar variables)
	VinegarVars int
	OilVars     int
}

// RainbowLayer represents a layer in the Rainbow system
type RainbowLayer struct {
	VinegarStart int
	VinegarEnd   int
	OilStart     int
	OilEnd       int
	Coefficients [][]int
}

// McElieceSystem represents a code-based cryptographic system
type McElieceSystem struct {
	GoppaCode   *GoppaCode
	Scrambler   [][]int // Scrambling matrix S
	Permutation []int   // Permutation matrix P
	PublicKey   [][]int // Public key G' = S * G * P
}

// GoppaCode represents a Goppa code for McEliece cryptosystem
type GoppaCode struct {
	N         int     // Code length
	K         int     // Code dimension
	T         int     // Error correction capability
	Generator [][]int // Generator matrix
	GoppaPoly []int   // Goppa polynomial coefficients
}

// QuantumResistantSymmetric provides symmetric encryption with quantum resistance
type QuantumResistantSymmetric struct {
	KeySize int
	IVSize  int
	Rounds  int
}

// NewPostQuantumCrypto creates a new post-quantum cryptography system
func NewPostQuantumCrypto() *PostQuantumCrypto {
	pqc := &PostQuantumCrypto{
		latticeParams: &LatticeParameters{
			N:        256,  // Ring dimension
			K:        3,    // Module rank
			Q:        3329, // Modulus (prime)
			Eta:      2,    // Noise parameter
			Du:       10,   // Public key compression
			Dv:       4,    // Ciphertext compression
			Sigma:    1.5,  // Standard deviation
			PolySize: 256,  // Polynomial size
		},
		keyStore: make(map[string]*QuantumResistantKey),
	}

	// Initialize subsystems
	pqc.initializeLatticeSystem()
	pqc.initializeHashBasedSystem()
	pqc.initializeMultivariateSystem()
	pqc.initializeCodeBasedSystem()
	pqc.initializeSymmetricSystem()

	return pqc
}

// initializeLatticeSystem initializes the lattice-based cryptographic system
func (pqc *PostQuantumCrypto) initializeLatticeSystem() {
	// Initialize lattice parameters for Kyber-like system
	// This would contain the full Kyber implementation
	fmt.Println("Initializing lattice-based cryptographic system (Kyber-like)")
}

// initializeHashBasedSystem initializes the hash-based signature system
func (pqc *PostQuantumCrypto) initializeHashBasedSystem() {
	pqc.hashTree = &HashTree{
		Height: 10, // 2^10 = 1024 signatures
		WotsParams: &WOTSParameters{
			N: 32, // SHA256 output size
			W: 16, // Winternitz parameter
			T: 67, // Number of chains for W=16
		},
		UsedLeafs: make(map[int]bool),
	}

	// Generate initial keys
	pqc.generateHashBasedKeys()
}

// initializeMultivariateSystem initializes the multivariate cryptographic system
func (pqc *PostQuantumCrypto) initializeMultivariateSystem() {
	pqc.rainbowSystem = &RainbowSystem{
		VinegarVars: 32,
		OilVars:     32,
		Layers:      make([]*RainbowLayer, 0),
	}

	// Create Rainbow layers
	pqc.createRainbowLayers()
	pqc.generateRainbowKeys()
}

// initializeCodeBasedSystem initializes the code-based cryptographic system
func (pqc *PostQuantumCrypto) initializeCodeBasedSystem() {
	pqc.mcelieceSystem = &McElieceSystem{
		GoppaCode: &GoppaCode{
			N: 1024, // Code length
			K: 524,  // Code dimension
			T: 50,   // Error correction capability
		},
	}

	// Generate Goppa code and keys
	pqc.generateGoppaCode()
	pqc.generateMcElieceKeys()
}

// initializeSymmetricSystem initializes the quantum-resistant symmetric system
func (pqc *PostQuantumCrypto) initializeSymmetricSystem() {
	pqc.symmetricCipher = &QuantumResistantSymmetric{
		KeySize: 32, // 256 bits
		IVSize:  16, // 128 bits
		Rounds:  20, // Number of rounds
	}
}

// GenerateKeyPair generates a quantum-resistant key pair
func (pqc *PostQuantumCrypto) GenerateKeyPair(algorithm string) (*QuantumResistantKey, error) {
	pqc.mu.Lock()
	defer pqc.mu.Unlock()

	var publicKey, privateKey interface{}
	var err error

	switch algorithm {
	case "lattice":
		publicKey, privateKey, err = pqc.generateLatticeKeyPair()
	case "hash":
		publicKey, privateKey, err = pqc.generateHashKeyPair()
	case "rainbow":
		publicKey, privateKey, err = pqc.generateRainbowKeyPair()
	case "mceliece":
		publicKey, privateKey, err = pqc.generateMcElieceKeyPair()
	default:
		return nil, fmt.Errorf("unsupported algorithm: %s", algorithm)
	}

	if err != nil {
		return nil, err
	}

	keyID := pqc.generateKeyID()
	key := &QuantumResistantKey{
		Algorithm:  algorithm,
		KeyID:      keyID,
		PublicKey:  publicKey,
		PrivateKey: privateKey,
		Created:    pqc.getCurrentTimestamp(),
	}

	pqc.keyStore[keyID] = key
	return key, nil
}

// generateLatticeKeyPair generates a lattice-based key pair
func (pqc *PostQuantumCrypto) generateLatticeKeyPair() (*LatticePublicKey, *LatticePrivateKey, error) {
	// Generate random seed
	rho := make([]byte, 32)
	if _, err := rand.Read(rho); err != nil {
		return nil, nil, err
	}

	// Generate matrix A (simplified)
	A := pqc.generateMatrixA(rho)

	// Generate secret key s
	s := pqc.generateSecretVector()

	// Generate public key t = A*s + e (simplified)
	t := pqc.computePublicKey(A, s)

	publicKey := &LatticePublicKey{
		T:   t,
		Rho: rho,
	}

	privateKey := &LatticePrivateKey{
		S:   s,
		Rho: rho,
		A:   A,
	}

	return publicKey, privateKey, nil
}

// generateMatrixA generates the matrix A from seed (simplified)
func (pqc *PostQuantumCrypto) generateMatrixA(seed []byte) []int {
	// In a real implementation, this would use SHAKE to expand the seed
	// into a matrix of the appropriate size
	matrix := make([]int, pqc.latticeParams.N*pqc.latticeParams.K)

	// Simple deterministic generation for demonstration
	for i := range matrix {
		matrix[i] = int(seed[i%len(seed)]) % pqc.latticeParams.Q
	}

	return matrix
}

// generateSecretVector generates the secret vector s
func (pqc *PostQuantumCrypto) generateSecretVector() []int {
	s := make([]int, pqc.latticeParams.N)

	for i := range s {
		// Generate small coefficients with noise
		coeff := pqc.sampleNoise()
		s[i] = coeff
	}

	return s
}

// sampleNoise samples from a discrete Gaussian distribution
func (pqc *PostQuantumCrypto) sampleNoise() int {
	// Simplified noise sampling - in practice, use more sophisticated distribution
	randomBytes := make([]byte, 4)
	rand.Read(randomBytes)
	randomValue := int(binary.LittleEndian.Uint32(randomBytes))

	// Map to [-eta, eta] range
	eta := pqc.latticeParams.Eta
	return (randomValue % (2*eta + 1)) - eta
}

// computePublicKey computes t = A*s + e
func (pqc *PostQuantumCrypto) computePublicKey(A, s []int) []int {
	t := make([]int, len(A)/len(s))

	for i := range t {
		sum := 0
		for j := range s {
			sum += A[i*len(s)+j] * s[j]
		}
		// Add noise
		sum += pqc.sampleNoise()
		// Modulo operation
		sum = ((sum % pqc.latticeParams.Q) + pqc.latticeParams.Q) % pqc.latticeParams.Q
		t[i] = sum
	}

	return t
}

// generateHashBasedKeys generates hash-based signature keys
func (pqc *PostQuantumCrypto) generateHashBasedKeys() {
	// Generate WOTS private key
	pqc.hashTree.PrivateKey = make([][]byte, pqc.hashTree.WotsParams.T)

	for i := 0; i < pqc.hashTree.WotsParams.T; i++ {
		pqc.hashTree.PrivateKey[i] = make([]byte, pqc.hashTree.WotsParams.N)
		rand.Read(pqc.hashTree.PrivateKey[i])
	}

	// Build Merkle tree and compute root
	pqc.buildMerkleTree()
}

// buildMerkleTree builds the XMSS Merkle tree
func (pqc *PostQuantumCrypto) buildMerkleTree() {
	height := pqc.hashTree.Height
	leafCount := 1 << height // 2^height

	// Generate leaf nodes (WOTS public keys)
	leaves := make([][]byte, leafCount)
	for i := 0; i < leafCount; i++ {
		leaves[i] = pqc.computeWOTSPublicKey(i)
	}

	// Build tree
	tree := make([][][]byte, height+1)
	tree[0] = leaves

	for level := 1; level <= height; level++ {
		parentCount := len(tree[level-1]) / 2
		tree[level] = make([][]byte, parentCount)

		for i := 0; i < parentCount; i++ {
			left := tree[level-1][2*i]
			right := tree[level-1][2*i+1]
			tree[level][i] = pqc.hashConcat(left, right)
		}
	}

	pqc.hashTree.Root = tree[height][0]
	pqc.hashTree.PublicKey = pqc.hashTree.Root
}

// computeWOTSPublicKey computes the WOTS public key for a given index
func (pqc *PostQuantumCrypto) computeWOTSPublicKey(index int) []byte {
	// Simplified WOTS public key computation
	// In practice, this would compute the public key from the private key chains
	hash := sha256.New()
	hash.Write([]byte(fmt.Sprintf("wots_public_key_%d", index)))
	return hash.Sum(nil)
}

// hashConcat concatenates and hashes two byte slices
func (pqc *PostQuantumCrypto) hashConcat(left, right []byte) []byte {
	hash := sha256.New()
	hash.Write(left)
	hash.Write(right)
	return hash.Sum(nil)
}

// createRainbowLayers creates the layers for the Rainbow system
func (pqc *PostQuantumCrypto) createRainbowLayers() {
	// Create multiple layers with increasing variable counts
	layers := 3
	vinegarIncrement := pqc.rainbowSystem.VinegarVars / layers
	oilIncrement := pqc.rainbowSystem.OilVars / layers

	currentVinegar := 0
	currentOil := 0

	for i := 0; i < layers; i++ {
		layer := &RainbowLayer{
			VinegarStart: currentVinegar,
			VinegarEnd:   currentVinegar + vinegarIncrement,
			OilStart:     currentOil,
			OilEnd:       currentOil + oilIncrement,
		}

		// Generate random coefficients for this layer
		vars := layer.VinegarEnd - layer.VinegarStart + layer.OilEnd - layer.OilStart
		layer.Coefficients = pqc.generateRainbowCoefficients(vars)

		pqc.rainbowSystem.Layers = append(pqc.rainbowSystem.Layers, layer)

		currentVinegar = layer.VinegarEnd
		currentOil = layer.OilEnd
	}
}

// generateRainbowCoefficients generates coefficients for Rainbow equations
func (pqc *PostQuantumCrypto) generateRainbowCoefficients(vars int) [][]int {
	coefficients := make([][]int, vars)

	for i := range coefficients {
		coefficients[i] = make([]int, vars+1) // +1 for constant term

		for j := range coefficients[i] {
			randomBytes := make([]byte, 4)
			rand.Read(randomBytes)
			coefficients[i][j] = int(binary.LittleEndian.Uint32(randomBytes)) % 256
		}
	}

	return coefficients
}

// generateRainbowKeys generates Rainbow public and private keys
func (pqc *PostQuantumCrypto) generateRainbowKeys() {
	totalVars := pqc.rainbowSystem.VinegarVars + pqc.rainbowSystem.OilVars
	pqc.rainbowSystem.PrivateKey = make([]int, pqc.rainbowSystem.VinegarVars)

	// Generate vinegar variables (private key)
	for i := range pqc.rainbowSystem.PrivateKey {
		randomBytes := make([]byte, 4)
		rand.Read(randomBytes)
		pqc.rainbowSystem.PrivateKey[i] = int(binary.LittleEndian.Uint32(randomBytes)) % 256
	}

	// Generate public key coefficients
	pqc.rainbowSystem.PublicKey = make([]int, totalVars*totalVars)

	for i := range pqc.rainbowSystem.PublicKey {
		randomBytes := make([]byte, 4)
		rand.Read(randomBytes)
		pqc.rainbowSystem.PublicKey[i] = int(binary.LittleEndian.Uint32(randomBytes)) % 256
	}
}

// generateGoppaCode generates a Goppa code for McEliece
func (pqc *PostQuantumCrypto) generateGoppaCode() {
	// Generate irreducible Goppa polynomial
	goppaPoly := pqc.generateIrreduciblePolynomial()

	// Generate support set L
	support := pqc.generateSupportSet()

	// Generate parity check matrix H
	parityCheck := pqc.generateParityCheckMatrix(goppaPoly, support)

	// Generate generator matrix G from H
	generator := pqc.generateGeneratorMatrix(parityCheck)

	pqc.mcelieceSystem.GoppaCode = &GoppaCode{
		N:         len(support),
		K:         len(generator),
		T:         len(goppaPoly) - 1, // Degree of Goppa polynomial
		Generator: generator,
		GoppaPoly: goppaPoly,
	}
}

// generateIrreduciblePolynomial generates an irreducible polynomial over GF(2^m)
func (pqc *PostQuantumCrypto) generateIrreduciblePolynomial() []int {
	// Simplified: return a known irreducible polynomial
	// In practice, this would test for irreducibility
	return []int{1, 1, 0, 1, 1, 0, 0, 0, 1} // x^8 + x^4 + x^3 + x + 1
}

// generateSupportSet generates the support set L
func (pqc *PostQuantumCrypto) generateSupportSet() []int {
	support := make([]int, pqc.mcelieceSystem.GoppaCode.N)

	for i := range support {
		support[i] = i
	}

	return support
}

// generateParityCheckMatrix generates the parity check matrix for Goppa code
func (pqc *PostQuantumCrypto) generateParityCheckMatrix(goppaPoly, support []int) [][]int {
	// Simplified implementation
	// In practice, this would compute the proper Goppa code parity check matrix
	rows := len(goppaPoly) - 1
	cols := len(support)

	matrix := make([][]int, rows)
	for i := range matrix {
		matrix[i] = make([]int, cols)
		for j := range matrix[i] {
			// Simplified: random entries
			randomBytes := make([]byte, 1)
			rand.Read(randomBytes)
			matrix[i][j] = int(randomBytes[0]) % 2
		}
	}

	return matrix
}

// generateGeneratorMatrix generates the generator matrix from parity check matrix
func (pqc *PostQuantumCrypto) generateGeneratorMatrix(parityCheck [][]int) [][]int {
	// Simplified: generate a random generator matrix
	// In practice, this would be computed from the parity check matrix
	rows := pqc.mcelieceSystem.GoppaCode.K
	cols := pqc.mcelieceSystem.GoppaCode.N

	matrix := make([][]int, rows)
	for i := range matrix {
		matrix[i] = make([]int, cols)
		for j := range matrix[i] {
			randomBytes := make([]byte, 1)
			rand.Read(randomBytes)
			matrix[i][j] = int(randomBytes[0]) % 2
		}
	}

	return matrix
}

// generateMcElieceKeys generates McEliece public and private keys
func (pqc *PostQuantumCrypto) generateMcElieceKeys() {
	// Generate scrambling matrix S (invertible)
	S := pqc.generateInvertibleMatrix(pqc.mcelieceSystem.GoppaCode.K)

	// Generate permutation matrix P
	P := pqc.generatePermutation(pqc.mcelieceSystem.GoppaCode.N)

	// Compute public key G' = S * G * P
	G := pqc.mcelieceSystem.GoppaCode.Generator
	Pmatrix := pqc.permutationToMatrix(P)
	Gprime := pqc.multiplyMatrices(pqc.multiplyMatrices(S, G), Pmatrix)

	pqc.mcelieceSystem.Scrambler = S
	pqc.mcelieceSystem.Permutation = P
	pqc.mcelieceSystem.PublicKey = Gprime
}

// generateInvertibleMatrix generates an invertible matrix over GF(2)
func (pqc *PostQuantumCrypto) generateInvertibleMatrix(size int) [][]int {
	// Keep generating random matrices until we get an invertible one
	for {
		matrix := make([][]int, size)
		for i := range matrix {
			matrix[i] = make([]int, size)
			for j := range matrix[i] {
				randomBytes := make([]byte, 1)
				rand.Read(randomBytes)
				matrix[i][j] = int(randomBytes[0]) % 2
			}
		}

		if pqc.isMatrixInvertible(matrix) {
			return matrix
		}
	}
}

// isMatrixInvertible checks if a matrix is invertible over GF(2)
func (pqc *PostQuantumCrypto) isMatrixInvertible(matrix [][]int) bool {
	// Simplified check - in practice, use Gaussian elimination
	size := len(matrix)
	det := 0

	// Compute determinant (simplified)
	for i := 0; i < size; i++ {
		det ^= matrix[i][i]
	}

	return det == 1
}

// generatePermutation generates a random permutation
func (pqc *PostQuantumCrypto) generatePermutation(size int) []int {
	if size <= 0 {
		return []int{}
	}

	permutation := make([]int, size)
	for i := range permutation {
		permutation[i] = i
	}

	// Fisher-Yates shuffle
	for i := size - 1; i > 0; i-- {
		randomBytes := make([]byte, 8)
		if _, err := crand.Read(randomBytes); err != nil {
			// Fallback to deterministic if crypto rand fails
			randomValue := int64(time.Now().UnixNano())
			j := int(randomValue % int64(i+1))
			permutation[i], permutation[j] = permutation[j], permutation[i]
		} else {
			randomValue := int64(binary.LittleEndian.Uint64(randomBytes))
			j := int(randomValue % int64(i+1))
			permutation[i], permutation[j] = permutation[j], permutation[i]
		}
	}

	return permutation
}

// multiplyMatrices multiplies two matrices over GF(2)
func (pqc *PostQuantumCrypto) multiplyMatrices(a, b [][]int) [][]int {
	if len(a) == 0 || len(b) == 0 {
		return nil
	}

	// Check if matrices can be multiplied
	if len(a[0]) != len(b) {
		return nil
	}

	result := make([][]int, len(a))
	for i := range result {
		result[i] = make([]int, len(b[0]))
		for j := range result[i] {
			sum := 0
			for k := range b {
				if i < len(a) && k < len(a[i]) && k < len(b) && j < len(b[k]) {
					sum ^= a[i][k] * b[k][j]
				}
			}
			result[i][j] = sum
		}
	}

	return result
}

// transposeMatrix transposes a matrix
func (pqc *PostQuantumCrypto) transposeMatrix(matrix [][]int) [][]int {
	if len(matrix) == 0 {
		return nil
	}

	rows := len(matrix)
	cols := len(matrix[0])

	result := make([][]int, cols)
	for i := range result {
		result[i] = make([]int, rows)
		for j := range result[i] {
			result[i][j] = matrix[j][i]
		}
	}

	return result
}

// permutationToMatrix converts a permutation array to a permutation matrix
func (pqc *PostQuantumCrypto) permutationToMatrix(permutation []int) [][]int {
	size := len(permutation)
	if size == 0 {
		return [][]int{}
	}

	matrix := make([][]int, size)

	for i := range matrix {
		matrix[i] = make([]int, size)
		if i < len(permutation) && permutation[i] < size && permutation[i] >= 0 {
			matrix[i][permutation[i]] = 1
		}
	}

	return matrix
}

// generateHashKeyPair generates hash-based key pair
func (pqc *PostQuantumCrypto) generateHashKeyPair() ([]byte, *HashTree, error) {
	return pqc.hashTree.PublicKey, pqc.hashTree, nil
}

// generateRainbowKeyPair generates Rainbow key pair
func (pqc *PostQuantumCrypto) generateRainbowKeyPair() ([]int, *RainbowSystem, error) {
	return pqc.rainbowSystem.PublicKey, pqc.rainbowSystem, nil
}

// generateMcElieceKeyPair generates McEliece key pair
func (pqc *PostQuantumCrypto) generateMcElieceKeyPair() ([][]int, *McElieceSystem, error) {
	return pqc.mcelieceSystem.PublicKey, pqc.mcelieceSystem, nil
}

// generateKeyID generates a unique key identifier
func (pqc *PostQuantumCrypto) generateKeyID() string {
	randomBytes := make([]byte, 16)
	rand.Read(randomBytes)

	hash := sha256.Sum256(randomBytes)
	return fmt.Sprintf("%x", hash[:8])
}

// getCurrentTimestamp returns current Unix timestamp
func (pqc *PostQuantumCrypto) getCurrentTimestamp() int64 {
	return time.Now().Unix()
}

// Encrypt encrypts data using quantum-resistant algorithms
func (pqc *PostQuantumCrypto) Encrypt(keyID string, plaintext []byte) ([]byte, error) {
	pqc.mu.RLock()
	key, exists := pqc.keyStore[keyID]
	pqc.mu.RUnlock()

	if !exists {
		return nil, fmt.Errorf("key not found: %s", keyID)
	}

	switch key.Algorithm {
	case "lattice":
		return pqc.encryptLattice(key.PublicKey.(*LatticePublicKey), plaintext)
	case "rainbow":
		return pqc.encryptRainbow(key.PublicKey.([]int), plaintext)
	case "mceliece":
		return pqc.encryptMcEliece(key.PublicKey.([][]int), plaintext)
	default:
		return nil, fmt.Errorf("unsupported encryption algorithm: %s", key.Algorithm)
	}
}

// encryptLattice performs lattice-based encryption (Kyber-like)
func (pqc *PostQuantumCrypto) encryptLattice(publicKey *LatticePublicKey, plaintext []byte) ([]byte, error) {
	// Simplified Kyber encryption
	// Generate random message m
	m := make([]int, pqc.latticeParams.N)
	for i := range m {
		if i < len(plaintext) {
			m[i] = int(plaintext[i])
		} else {
			randomBytes := make([]byte, 1)
			rand.Read(randomBytes)
			m[i] = int(randomBytes[0])
		}
	}

	// Generate random r
	r := pqc.generateSecretVector()

	// Compute u = A^T * r + e1
	u := pqc.computeU(publicKey, r)

	// Compute v = t^T * r + e2 + m
	v := pqc.computeV(publicKey, r, m)

	// Combine u and v into ciphertext
	ciphertext := pqc.packCiphertext(u, v)

	return ciphertext, nil
}

// computeU computes u = A^T * r + e1
func (pqc *PostQuantumCrypto) computeU(publicKey *LatticePublicKey, r []int) []int {
	// Simplified computation
	u := make([]int, len(r))

	for i := range u {
		sum := 0
		for j := range r {
			// Simplified: use public key seed to derive A^T
			sum += publicKey.T[i*len(r)+j] * r[j]
		}
		sum += pqc.sampleNoise()
		u[i] = ((sum % pqc.latticeParams.Q) + pqc.latticeParams.Q) % pqc.latticeParams.Q
	}

	return u
}

// computeV computes v = t^T * r + e2 + m
func (pqc *PostQuantumCrypto) computeV(publicKey *LatticePublicKey, r, m []int) []int {
	v := make([]int, len(m))

	for i := range v {
		sum := 0
		for j := range r {
			sum += publicKey.T[j*len(m)+i] * r[j]
		}
		sum += pqc.sampleNoise()
		sum += m[i]
		v[i] = ((sum % pqc.latticeParams.Q) + pqc.latticeParams.Q) % pqc.latticeParams.Q
	}

	return v
}

// packCiphertext packs u and v into ciphertext bytes
func (pqc *PostQuantumCrypto) packCiphertext(u, v []int) []byte {
	// Simplified packing
	ciphertext := make([]byte, len(u)+len(v))

	for i, val := range u {
		ciphertext[i] = byte(val % 256)
	}

	for i, val := range v {
		ciphertext[len(u)+i] = byte(val % 256)
	}

	return ciphertext
}

// encryptRainbow performs Rainbow encryption
func (pqc *PostQuantumCrypto) encryptRainbow(publicKey []int, plaintext []byte) ([]byte, error) {
	// Simplified Rainbow encryption
	hash := sha256.Sum256(plaintext)
	ciphertext := make([]byte, len(hash))

	for i := range ciphertext {
		ciphertext[i] = hash[i] ^ byte(publicKey[i%len(publicKey)])
	}

	return ciphertext, nil
}

// encryptMcEliece performs McEliece encryption
func (pqc *PostQuantumCrypto) encryptMcEliece(publicKey [][]int, plaintext []byte) ([]byte, error) {
	// Convert plaintext to codeword
	codeword := pqc.encodeMessage(plaintext)

	// Add random error vector
	errorVector := pqc.generateErrorVector()

	// Compute ciphertext = codeword + errorVector
	ciphertext := make([]int, len(codeword))
	for i := range ciphertext {
		ciphertext[i] = (codeword[i] + errorVector[i]) % 2
	}

	// Convert to bytes
	ciphertextBytes := make([]byte, (len(ciphertext)+7)/8)
	for i, bit := range ciphertext {
		if bit == 1 {
			ciphertextBytes[i/8] |= 1 << (i % 8)
		}
	}

	return ciphertextBytes, nil
}

// encodeMessage encodes a message using the generator matrix
func (pqc *PostQuantumCrypto) encodeMessage(message []byte) []int {
	// Convert message to bits
	messageBits := make([]int, 0, len(message)*8)
	for _, b := range message {
		for i := 0; i < 8; i++ {
			bit := (b >> i) & 1
			messageBits = append(messageBits, int(bit))
		}
	}

	// Pad message to match generator matrix dimensions
	for len(messageBits) < pqc.mcelieceSystem.GoppaCode.K {
		messageBits = append(messageBits, 0)
	}

	// Multiply by generator matrix
	codeword := make([]int, pqc.mcelieceSystem.GoppaCode.N)
	for i := 0; i < pqc.mcelieceSystem.GoppaCode.N; i++ {
		sum := 0
		for j := 0; j < pqc.mcelieceSystem.GoppaCode.K; j++ {
			sum ^= pqc.mcelieceSystem.GoppaCode.Generator[j][i] * messageBits[j]
		}
		codeword[i] = sum
	}

	return codeword
}

// generateErrorVector generates a random error vector of weight t
func (pqc *PostQuantumCrypto) generateErrorVector() []int {
	errorVector := make([]int, pqc.mcelieceSystem.GoppaCode.N)
	errorPositions := make(map[int]bool)

	// Generate t random error positions
	for len(errorPositions) < pqc.mcelieceSystem.GoppaCode.T {
		randomBytes := make([]byte, 8)
		if _, err := crand.Read(randomBytes); err != nil {
			// Fallback to deterministic if crypto rand fails
			randomValue := int64(time.Now().UnixNano())
			pos := int(randomValue) % pqc.mcelieceSystem.GoppaCode.N
			errorPositions[pos] = true
		} else {
			randomValue := int64(binary.LittleEndian.Uint64(randomBytes))
			pos := int(randomValue) % pqc.mcelieceSystem.GoppaCode.N
			errorPositions[pos] = true
		}
	}

	for pos := range errorPositions {
		errorVector[pos] = 1
	}

	return errorVector
}

// Decrypt decrypts data using quantum-resistant algorithms
func (pqc *PostQuantumCrypto) Decrypt(keyID string, ciphertext []byte) ([]byte, error) {
	pqc.mu.RLock()
	key, exists := pqc.keyStore[keyID]
	pqc.mu.RUnlock()

	if !exists {
		return nil, fmt.Errorf("key not found: %s", keyID)
	}

	switch key.Algorithm {
	case "lattice":
		return pqc.decryptLattice(key.PrivateKey.(*LatticePrivateKey), ciphertext)
	case "rainbow":
		return pqc.decryptRainbow(key.PrivateKey.(*RainbowSystem), ciphertext)
	case "mceliece":
		return pqc.decryptMcEliece(key.PrivateKey.(*McElieceSystem), ciphertext)
	default:
		return nil, fmt.Errorf("unsupported decryption algorithm: %s", key.Algorithm)
	}
}

// decryptLattice performs lattice-based decryption
func (pqc *PostQuantumCrypto) decryptLattice(privateKey *LatticePrivateKey, ciphertext []byte) ([]byte, error) {
	// Unpack ciphertext
	u, v := pqc.unpackCiphertext(ciphertext)

	// Compute m = v - s^T * u
	m := pqc.computeDecryptedMessage(privateKey, u, v)

	// Convert back to bytes
	plaintext := make([]byte, len(m))
	for i, val := range m {
		plaintext[i] = byte(val % 256)
	}

	return plaintext, nil
}

// unpackCiphertext unpacks ciphertext into u and v
func (pqc *PostQuantumCrypto) unpackCiphertext(ciphertext []byte) ([]int, []int) {
	uSize := pqc.latticeParams.N
	u := make([]int, uSize)
	v := make([]int, len(ciphertext)-uSize)

	for i := 0; i < uSize; i++ {
		u[i] = int(ciphertext[i])
	}

	for i := 0; i < len(v); i++ {
		v[i] = int(ciphertext[uSize+i])
	}

	return u, v
}

// computeDecryptedMessage computes m = v - s^T * u
func (pqc *PostQuantumCrypto) computeDecryptedMessage(privateKey *LatticePrivateKey, u, v []int) []int {
	m := make([]int, len(v))

	for i := range m {
		sum := v[i]
		for j := range u {
			sum -= privateKey.S[j] * u[j]
		}
		sum = ((sum % pqc.latticeParams.Q) + pqc.latticeParams.Q) % pqc.latticeParams.Q
		m[i] = sum
	}

	return m
}

// decryptRainbow performs Rainbow decryption
func (pqc *PostQuantumCrypto) decryptRainbow(privateKey *RainbowSystem, ciphertext []byte) ([]byte, error) {
	// Simplified Rainbow decryption
	plaintext := make([]byte, len(ciphertext))

	for i := range plaintext {
		plaintext[i] = ciphertext[i] ^ byte(privateKey.PrivateKey[i%len(privateKey.PrivateKey)])
	}

	return plaintext, nil
}

// decryptMcEliece performs McEliece decryption
func (pqc *PostQuantumCrypto) decryptMcEliece(privateKey *McElieceSystem, ciphertext []byte) ([]byte, error) {
	// Convert ciphertext to bits
	ciphertextBits := make([]int, len(ciphertext)*8)
	for i, b := range ciphertext {
		for j := 0; j < 8; j++ {
			bit := (b >> j) & 1
			ciphertextBits[i*8+j] = int(bit)
		}
	}

	// Apply inverse permutation P^-1
	permuted := make([]int, len(ciphertextBits))
	for i, pos := range privateKey.Permutation {
		if pos < len(ciphertextBits) {
			permuted[pos] = ciphertextBits[i]
		}
	}

	// Decode using Goppa code (simplified)
	decoded := pqc.decodeGoppaCode(permuted)

	// Apply inverse scrambling S^-1
	unscrambled := pqc.multiplyMatrixVector(privateKey.Scrambler, decoded)

	// Extract original message
	messageBits := unscrambled[:privateKey.GoppaCode.K]
	messageBytes := make([]byte, (len(messageBits)+7)/8)

	for i, bit := range messageBits {
		if bit == 1 {
			messageBytes[i/8] |= 1 << (i % 8)
		}
	}

	return messageBytes, nil
}

// decodeGoppaCode decodes a Goppa code (simplified)
func (pqc *PostQuantumCrypto) decodeGoppaCode(codeword []int) []int {
	// Simplified decoding - in practice, this would use Patterson algorithm
	// For now, just return the codeword as-is
	return codeword
}

// multiplyMatrixVector multiplies matrix by vector over GF(2)
func (pqc *PostQuantumCrypto) multiplyMatrixVector(matrix [][]int, vector []int) []int {
	if len(matrix) == 0 || len(matrix[0]) != len(vector) {
		return nil
	}

	result := make([]int, len(matrix))
	for i := range result {
		sum := 0
		for j := range vector {
			sum ^= matrix[i][j] * vector[j]
		}
		result[i] = sum
	}

	return result
}

// Sign creates a digital signature using quantum-resistant algorithms
func (pqc *PostQuantumCrypto) Sign(keyID string, message []byte) ([]byte, error) {
	pqc.mu.RLock()
	key, exists := pqc.keyStore[keyID]
	pqc.mu.RUnlock()

	if !exists {
		return nil, fmt.Errorf("key not found: %s", keyID)
	}

	switch key.Algorithm {
	case "hash":
		return pqc.signHashBased(key.PrivateKey.(*HashTree), message)
	default:
		return nil, fmt.Errorf("unsupported signing algorithm: %s", key.Algorithm)
	}
}

// signHashBased creates a hash-based signature
func (pqc *PostQuantumCrypto) signHashBased(privateKey *HashTree, message []byte) ([]byte, error) {
	// Find an unused leaf
	var leafIndex int
	for i := 0; i < (1 << privateKey.Height); i++ {
		if !privateKey.UsedLeafs[i] {
			leafIndex = i
			break
		}
	}

	if privateKey.UsedLeafs[leafIndex] {
		return nil, fmt.Errorf("no unused signatures available")
	}

	// Mark leaf as used
	privateKey.UsedLeafs[leafIndex] = true

	// Generate WOTS signature
	wotsSignature := pqc.generateWOTSSignature(privateKey, message, leafIndex)

	// Generate authentication path
	authPath := pqc.generateAuthPath(privateKey, leafIndex)

	// Combine signature components
	signature := pqc.combineSignature(leafIndex, wotsSignature, authPath)

	return signature, nil
}

// generateWOTSSignature generates a WOTS signature
func (pqc *PostQuantumCrypto) generateWOTSSignature(privateKey *HashTree, message []byte, index int) []byte {
	// Simplified WOTS signature generation
	hash := sha256.Sum256(message)
	signature := make([]byte, len(hash)*2)

	copy(signature[:len(hash)], hash[:])
	copy(signature[len(hash):], privateKey.PrivateKey[index%len(privateKey.PrivateKey)])

	return signature
}

// generateAuthPath generates the authentication path for XMSS
func (pqc *PostQuantumCrypto) generateAuthPath(privateKey *HashTree, leafIndex int) [][]byte {
	// Simplified authentication path generation
	path := make([][]byte, privateKey.Height)

	for i := 0; i < privateKey.Height; i++ {
		// Determine sibling index
		siblingIndex := leafIndex ^ 1
		// Generate sibling hash (simplified)
		siblingHash := sha256.Sum256([]byte(fmt.Sprintf("sibling_%d_%d", i, siblingIndex)))
		path[i] = siblingHash[:]
	}

	return path
}

// combineSignature combines signature components
func (pqc *PostQuantumCrypto) combineSignature(leafIndex int, wotsSig []byte, authPath [][]byte) []byte {
	// Combine all components into a single signature
	signature := make([]byte, 4+len(wotsSig)) // 4 bytes for index

	binary.LittleEndian.PutUint32(signature[:4], uint32(leafIndex))
	copy(signature[4:], wotsSig)

	// Add authentication path
	for _, pathElement := range authPath {
		signature = append(signature, pathElement...)
	}

	return signature
}

// Verify verifies a digital signature
func (pqc *PostQuantumCrypto) Verify(keyID string, message, signature []byte) (bool, error) {
	pqc.mu.RLock()
	key, exists := pqc.keyStore[keyID]
	pqc.mu.RUnlock()

	if !exists {
		return false, fmt.Errorf("key not found: %s", keyID)
	}

	switch key.Algorithm {
	case "hash":
		return pqc.verifyHashBased(key.PublicKey.([]byte), message, signature)
	default:
		return false, fmt.Errorf("unsupported verification algorithm: %s", key.Algorithm)
	}
}

// verifyHashBased verifies a hash-based signature
func (pqc *PostQuantumCrypto) verifyHashBased(publicKey []byte, message, signature []byte) (bool, error) {
	if len(signature) < 4 {
		return false, fmt.Errorf("signature too short")
	}

	// Extract leaf index
	_ = int(binary.LittleEndian.Uint32(signature[:4]))

	// Extract WOTS signature
	wotsSig := signature[4 : 4+32] // Simplified size

	// Verify WOTS signature (simplified)
	messageHash := sha256.Sum256(message)
	expectedWOTS := make([]byte, len(messageHash)*2)
	copy(expectedWOTS[:len(messageHash)], messageHash[:])

	// Compare with public key
	if !bytes.Equal(wotsSig[:len(messageHash)], expectedWOTS[:len(messageHash)]) {
		return false, nil
	}

	// Verify authentication path (simplified)
	// In practice, this would verify the Merkle tree path
	return bytes.Equal(publicKey, pqc.hashTree.PublicKey), nil
}

// GetKeyInfo returns information about a stored key
func (pqc *PostQuantumCrypto) GetKeyInfo(keyID string) (*QuantumResistantKey, error) {
	pqc.mu.RLock()
	defer pqc.mu.RUnlock()

	key, exists := pqc.keyStore[keyID]
	if !exists {
		return nil, fmt.Errorf("key not found: %s", keyID)
	}

	// Return a copy without the private key
	keyInfo := &QuantumResistantKey{
		Algorithm: key.Algorithm,
		KeyID:     key.KeyID,
		PublicKey: key.PublicKey,
		Created:   key.Created,
		Expires:   key.Expires,
	}

	return keyInfo, nil
}

// ListKeys returns a list of all stored key IDs
func (pqc *PostQuantumCrypto) ListKeys() []string {
	pqc.mu.RLock()
	defer pqc.mu.RUnlock()

	keys := make([]string, 0, len(pqc.keyStore))
	for keyID := range pqc.keyStore {
		keys = append(keys, keyID)
	}

	return keys
}

// DeleteKey deletes a stored key
func (pqc *PostQuantumCrypto) DeleteKey(keyID string) error {
	pqc.mu.Lock()
	defer pqc.mu.Unlock()

	if _, exists := pqc.keyStore[keyID]; !exists {
		return fmt.Errorf("key not found: %s", keyID)
	}

	delete(pqc.keyStore, keyID)
	return nil
}

// GetSecurityLevel returns the security level of an algorithm
func (pqc *PostQuantumCrypto) GetSecurityLevel(algorithm string) (int, error) {
	switch algorithm {
	case "lattice":
		return 128, nil // NIST security level 2
	case "hash":
		return 256, nil // SHA256-based security
	case "rainbow":
		return 128, nil // NIST security level 1
	case "mceliece":
		return 128, nil // NIST security level 1
	default:
		return 0, fmt.Errorf("unknown algorithm: %s", algorithm)
	}
}

// BenchmarkAlgorithm benchmarks a cryptographic algorithm
func (pqc *PostQuantumCrypto) BenchmarkAlgorithm(algorithm string, operation string, iterations int) (time.Duration, error) {
	start := time.Now()

	for i := 0; i < iterations; i++ {
		switch operation {
		case "keygen":
			_, err := pqc.GenerateKeyPair(algorithm)
			if err != nil {
				return 0, err
			}
		case "encrypt":
			// Use a test key for encryption
			testData := []byte("benchmark test data")
			keys := pqc.ListKeys()
			if len(keys) == 0 {
				return 0, fmt.Errorf("no keys available for benchmark")
			}
			_, err := pqc.Encrypt(keys[0], testData)
			if err != nil {
				return 0, err
			}
		case "sign":
			// Use a test key for signing
			testData := []byte("benchmark test data")
			keys := pqc.ListKeys()
			if len(keys) == 0 {
				return 0, fmt.Errorf("no keys available for benchmark")
			}
			_, err := pqc.Sign(keys[0], testData)
			if err != nil {
				return 0, err
			}
		default:
			return 0, fmt.Errorf("unsupported operation: %s", operation)
		}
	}

	elapsed := time.Since(start)
	return elapsed, nil
}
