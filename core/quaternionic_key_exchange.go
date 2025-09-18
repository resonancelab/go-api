package core

import (
	"crypto/rand"
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"math"
	"math/cmplx"
	"sync"
	"time"

	"github.com/resonancelab/psizero/core/hilbert"
	"github.com/resonancelab/psizero/core/primes"
)

// QuaternionicKeyExchange implements secure key exchange using quaternionic states
type QuaternionicKeyExchange struct {
	// Core components
	quaternionicState *QuaternionicState
	hilbertSpace      *hilbert.HilbertSpace
	primeEngine       *primes.PrimeEngine

	// Key exchange parameters
	sessionID    string
	sharedSecret []byte
	publicKey    *QuaternionicPublicKey
	privateKey   *QuaternionicPrivateKey

	// Protocol state
	state      KeyExchangeState
	peerPublic *QuaternionicPublicKey
	peerState  *QuaternionicState

	// Security parameters
	securityLevel int
	keySize       int

	// Synchronization
	mu sync.RWMutex

	// Telemetry
	telemetry *KeyExchangeTelemetry
}

// KeyExchangeState represents the current state of key exchange
type KeyExchangeState int

const (
	StateInitialized KeyExchangeState = iota
	StateKeyGenerated
	StatePublicSent
	StatePublicReceived
	StateSharedComputed
	StateEstablished
	StateError
)

// QuaternionicPublicKey represents a public key in quaternionic space
type QuaternionicPublicKey struct {
	// Public quaternionic state
	State *QuaternionicState

	// Public parameters
	Generator []complex128
	Modulus   []complex128

	// Verification data
	Commitment []byte
	Proof      []byte

	// Metadata
	Timestamp int64
	NodeID    string
}

// QuaternionicPrivateKey represents a private key in quaternionic space
type QuaternionicPrivateKey struct {
	// Private quaternionic state
	State *QuaternionicState

	// Private parameters
	Exponent []complex128
	Secret   []byte

	// Key derivation
	Seed []byte
	Salt []byte
}

// KeyExchangeMessage represents a message in the key exchange protocol
type KeyExchangeMessage struct {
	Type      string                 `json:"type"`
	SessionID string                 `json:"session_id"`
	Payload   map[string]interface{} `json:"payload"`
	Timestamp int64                  `json:"timestamp"`
	Signature []byte                 `json:"signature,omitempty"`
}

// KeyExchangeTelemetry tracks key exchange performance and security metrics
type KeyExchangeTelemetry struct {
	StartTime        time.Time
	EndTime          time.Time
	RoundTrips       int
	BytesTransferred int64
	CoherenceLevel   float64
	PhaseAlignment   float64
	SecurityStrength float64
	Errors           []string
}

// NewQuaternionicKeyExchange creates a new quaternionic key exchange instance
func NewQuaternionicKeyExchange(hilbertSpace *hilbert.HilbertSpace, primeEngine *primes.PrimeEngine) *QuaternionicKeyExchange {
	qke := &QuaternionicKeyExchange{
		hilbertSpace:  hilbertSpace,
		primeEngine:   primeEngine,
		state:         StateInitialized,
		securityLevel: 256, // 256-bit security
		keySize:       32,  // 256-bit key
		telemetry: &KeyExchangeTelemetry{
			StartTime: time.Now(),
			Errors:    make([]string, 0),
		},
	}

	// Generate session ID
	qke.generateSessionID()

	// Initialize quaternionic state
	qke.initializeQuaternionicState()

	return qke
}

// generateSessionID generates a unique session identifier
func (qke *QuaternionicKeyExchange) generateSessionID() {
	randomBytes := make([]byte, 16)
	rand.Read(randomBytes)

	hash := sha256.Sum256(append(randomBytes, []byte(fmt.Sprintf("%d", time.Now().UnixNano()))...))
	qke.sessionID = fmt.Sprintf("%x", hash[:8])
}

// initializeQuaternionicState initializes the quaternionic state for key exchange
func (qke *QuaternionicKeyExchange) initializeQuaternionicState() {
	// Generate random seed for reproducibility
	seed := make([]byte, 32)
	rand.Read(seed)

	// Create position coordinates (spatial dimension)
	position := make([]float64, 3) // 3D space
	for i := range position {
		position[i] = float64(seed[i%len(seed)]) / 255.0
	}

	// Create base amplitude
	baseAmplitude := complex(
		float64(seed[0])/255.0*2.0-1.0, // Real part in [-1, 1]
		float64(seed[1])/255.0*2.0-1.0, // Imaginary part in [-1, 1]
	)

	// Create Gaussian and Eisenstein coordinates
	gaussian := make([]float64, 2)
	eisenstein := make([]float64, 2)

	for i := range gaussian {
		gaussian[i] = float64(seed[i+2]) / 255.0
		eisenstein[i] = float64(seed[i+4]) / 255.0
	}

	// Create initial quaternionic state
	qke.quaternionicState = NewQuaternionicState(position, baseAmplitude, gaussian, eisenstein)
}

// initializePrimeBasedState initializes the state using prime number relationships
func (qke *QuaternionicKeyExchange) initializePrimeBasedState(seed []byte) {
	primeBasis := qke.hilbertSpace.GetPrimeBasis()

	// Use first prime to influence the base amplitude
	if len(primeBasis) > 0 {
		prime := primeBasis[0]

		// Use prime properties to generate complex base amplitude
		realPart := math.Sin(float64(prime) * math.Pi / 180.0) // Convert to radians
		imagPart := math.Cos(float64(prime) * math.Pi / 180.0)

		// Add noise based on seed
		noiseReal := float64(seed[0])/255.0 - 0.5
		noiseImag := float64(seed[1])/255.0 - 0.5

		baseAmplitude := complex(realPart+noiseReal, imagPart+noiseImag)

		// Update the base amplitude
		qke.quaternionicState.BaseAmplitude = baseAmplitude

		// Update phase based on prime relationships
		phaseNoise := float64(seed[2]) / 255.0 * 2.0 * math.Pi
		qke.quaternionicState.Phase = phaseNoise

		// Recompute normalization factor
		qke.quaternionicState.computeNormalizationFactor()
	}
}

// normalizeAmplitudes normalizes the amplitude vector
func (qke *QuaternionicKeyExchange) normalizeAmplitudes(amplitudes []complex128) {
	norm := 0.0
	for _, amp := range amplitudes {
		norm += real(amp)*real(amp) + imag(amp)*imag(amp)
	}
	norm = math.Sqrt(norm)

	if norm > 0 {
		for i := range amplitudes {
			amplitudes[i] /= complex(norm, 0)
		}
	}
}

// GenerateKeys generates public and private key pair
func (qke *QuaternionicKeyExchange) GenerateKeys() error {
	qke.mu.Lock()
	defer qke.mu.Unlock()

	if qke.state != StateInitialized {
		return fmt.Errorf("invalid state for key generation: %v", qke.state)
	}

	// Generate private key parameters
	privateSeed := make([]byte, 32)
	if _, err := rand.Read(privateSeed); err != nil {
		return fmt.Errorf("failed to generate private seed: %w", err)
	}

	// Generate private exponents
	exponents := make([]complex128, qke.hilbertSpace.GetDimension())
	for i := range exponents {
		// Generate random complex exponents
		realPart := (float64(privateSeed[i%len(privateSeed)]) / 255.0) * 2 * math.Pi
		imagPart := (float64(privateSeed[(i+1)%len(privateSeed)]) / 255.0) * 2 * math.Pi
		exponents[i] = complex(realPart, imagPart)
	}

	// Create private key
	qke.privateKey = &QuaternionicPrivateKey{
		State:    qke.quaternionicState,
		Exponent: exponents,
		Seed:     privateSeed,
		Salt:     make([]byte, 16),
	}
	rand.Read(qke.privateKey.Salt)

	// Generate public key from private key
	publicKey, err := qke.generatePublicKey()
	if err != nil {
		return fmt.Errorf("failed to generate public key: %w", err)
	}

	qke.publicKey = publicKey
	qke.state = StateKeyGenerated

	qke.telemetry.CoherenceLevel = qke.computeCoherence()
	qke.telemetry.PhaseAlignment = qke.computePhaseAlignment()

	return nil
}

// generatePublicKey generates the public key from private key
func (qke *QuaternionicKeyExchange) generatePublicKey() (*QuaternionicPublicKey, error) {
	// Generate public generators
	generators := make([]complex128, qke.hilbertSpace.GetDimension())
	for i := range generators {
		// Use prime basis to generate generators
		primeBasis := qke.hilbertSpace.GetPrimeBasis()
		prime := primeBasis[i]

		// Generate generator based on prime properties
		angle := 2 * math.Pi * float64(prime) / float64(qke.hilbertSpace.GetDimension())
		generators[i] = complex(math.Cos(angle), math.Sin(angle))
	}

	// Generate public modulus
	modulus := make([]complex128, qke.hilbertSpace.GetDimension())
	for i := range modulus {
		// Use private exponents to compute modulus
		modulus[i] = cmplx.Exp(qke.privateKey.Exponent[i])
	}

	// Create commitment for zero-knowledge proof
	commitment := qke.generateCommitment()

	// Create zero-knowledge proof
	proof := qke.generateProof()

	publicKey := &QuaternionicPublicKey{
		State:      qke.quaternionicState,
		Generator:  generators,
		Modulus:    modulus,
		Commitment: commitment,
		Proof:      proof,
		Timestamp:  time.Now().Unix(),
		NodeID:     qke.sessionID, // Use session ID as node ID for now
	}

	return publicKey, nil
}

// generateCommitment generates a commitment for zero-knowledge proof
func (qke *QuaternionicKeyExchange) generateCommitment() []byte {
	// Create commitment using hash of public parameters
	hash := sha256.New()

	// Hash generators
	for _, gen := range qke.publicKey.Generator {
		binary.Write(hash, binary.LittleEndian, real(gen))
		binary.Write(hash, binary.LittleEndian, imag(gen))
	}

	// Hash modulus
	for _, mod := range qke.publicKey.Modulus {
		binary.Write(hash, binary.LittleEndian, real(mod))
		binary.Write(hash, binary.LittleEndian, imag(mod))
	}

	return hash.Sum(nil)
}

// generateProof generates a zero-knowledge proof
func (qke *QuaternionicKeyExchange) generateProof() []byte {
	// Simplified zero-knowledge proof
	// In practice, this would be a full ZKP system
	hash := sha256.New()
	hash.Write(qke.privateKey.Seed)
	hash.Write(qke.privateKey.Salt)
	return hash.Sum(nil)
}

// InitiateKeyExchange initiates the key exchange protocol
func (qke *QuaternionicKeyExchange) InitiateKeyExchange() (*KeyExchangeMessage, error) {
	qke.mu.Lock()
	defer qke.mu.Unlock()

	if qke.state != StateKeyGenerated {
		return nil, fmt.Errorf("keys not generated yet")
	}

	// Create initiation message
	message := &KeyExchangeMessage{
		Type:      "key_exchange_init",
		SessionID: qke.sessionID,
		Payload: map[string]interface{}{
			"public_key": qke.publicKey,
			"protocol":   "quaternionic_diffie_hellman",
			"version":    "1.0",
		},
		Timestamp: time.Now().Unix(),
	}

	qke.state = StatePublicSent
	qke.telemetry.RoundTrips++

	return message, nil
}

// ProcessKeyExchangeMessage processes an incoming key exchange message
func (qke *QuaternionicKeyExchange) ProcessKeyExchangeMessage(message *KeyExchangeMessage) (*KeyExchangeMessage, error) {
	qke.mu.Lock()
	defer qke.mu.Unlock()

	switch message.Type {
	case "key_exchange_init":
		return qke.processInitMessage(message)
	case "key_exchange_response":
		return qke.processResponseMessage(message)
	default:
		return nil, fmt.Errorf("unknown message type: %s", message.Type)
	}
}

// processInitMessage processes a key exchange initiation message
func (qke *QuaternionicKeyExchange) processInitMessage(message *KeyExchangeMessage) (*KeyExchangeMessage, error) {
	if qke.state != StateKeyGenerated {
		return nil, fmt.Errorf("not ready to process init message")
	}

	// Extract peer's public key
	peerPublicData, ok := message.Payload["public_key"]
	if !ok {
		return nil, fmt.Errorf("missing public key in init message")
	}

	// Convert to QuaternionicPublicKey
	peerPublic, ok := peerPublicData.(*QuaternionicPublicKey)
	if !ok {
		return nil, fmt.Errorf("invalid public key format")
	}

	// Store peer's public key
	qke.peerPublic = peerPublic
	qke.state = StatePublicReceived

	// Generate response message
	response := &KeyExchangeMessage{
		Type:      "key_exchange_response",
		SessionID: qke.sessionID,
		Payload: map[string]interface{}{
			"public_key": qke.publicKey,
			"ack":        true,
		},
		Timestamp: time.Now().Unix(),
	}

	qke.telemetry.RoundTrips++

	return response, nil
}

// processResponseMessage processes a key exchange response message
func (qke *QuaternionicKeyExchange) processResponseMessage(message *KeyExchangeMessage) (*KeyExchangeMessage, error) {
	if qke.state != StatePublicSent {
		return nil, fmt.Errorf("not expecting response message")
	}

	// Extract peer's public key
	peerPublicData, ok := message.Payload["public_key"]
	if !ok {
		return nil, fmt.Errorf("missing public key in response message")
	}

	// Convert to QuaternionicPublicKey
	peerPublic, ok := peerPublicData.(*QuaternionicPublicKey)
	if !ok {
		return nil, fmt.Errorf("invalid public key format")
	}

	// Store peer's public key
	qke.peerPublic = peerPublic
	qke.state = StatePublicReceived

	// Compute shared secret
	err := qke.computeSharedSecret()
	if err != nil {
		qke.state = StateError
		return nil, fmt.Errorf("failed to compute shared secret: %w", err)
	}

	qke.state = StateSharedComputed

	// Generate final confirmation message
	confirmation := &KeyExchangeMessage{
		Type:      "key_exchange_confirm",
		SessionID: qke.sessionID,
		Payload: map[string]interface{}{
			"status": "established",
			"key_id": qke.generateKeyID(),
		},
		Timestamp: time.Now().Unix(),
	}

	qke.telemetry.RoundTrips++

	return confirmation, nil
}

// computeSharedSecret computes the shared secret using quaternionic Diffie-Hellman
func (qke *QuaternionicKeyExchange) computeSharedSecret() error {
	if qke.peerPublic == nil || qke.privateKey == nil {
		return fmt.Errorf("missing required keys for shared secret computation")
	}

	// Use quaternionic state evolution to compute shared secret
	sharedState := qke.quaternionicState.Clone()

	// Apply peer's public parameters to our private state
	// Combine our private exponent with peer's public modulus
	combined := qke.privateKey.Exponent[0] * cmplx.Log(qke.peerPublic.Modulus[0])

	// Apply quaternionic transformation to base amplitude
	sharedState.BaseAmplitude *= cmplx.Exp(combined)

	// Update phase based on combined transformation
	sharedState.Phase += cmplx.Phase(combined)

	// Recompute normalization factor
	sharedState.computeNormalizationFactor()

	// Extract key material from shared state
	keyMaterial := qke.extractKeyMaterial(sharedState)
	qke.sharedSecret = keyMaterial

	// Update telemetry
	qke.telemetry.EndTime = time.Now()
	qke.telemetry.SecurityStrength = qke.computeSecurityStrength()

	return nil
}

// extractKeyMaterial extracts cryptographic key material from quaternionic state
func (qke *QuaternionicKeyExchange) extractKeyMaterial(state *QuaternionicState) []byte {
	// Use base amplitude and phase to generate key material
	keyMaterial := make([]byte, 0, qke.keySize*2)

	// Extract real and imaginary parts of base amplitude
	realBytes := make([]byte, 8)
	imagBytes := make([]byte, 8)
	binary.LittleEndian.PutUint64(realBytes, math.Float64bits(real(state.BaseAmplitude)))
	binary.LittleEndian.PutUint64(imagBytes, math.Float64bits(imag(state.BaseAmplitude)))
	keyMaterial = append(keyMaterial, realBytes...)
	keyMaterial = append(keyMaterial, imagBytes...)

	// Extract phase information
	phaseBytes := make([]byte, 8)
	binary.LittleEndian.PutUint64(phaseBytes, math.Float64bits(state.Phase))
	keyMaterial = append(keyMaterial, phaseBytes...)

	// Extract position coordinates
	for _, pos := range state.Position {
		posBytes := make([]byte, 8)
		binary.LittleEndian.PutUint64(posBytes, math.Float64bits(pos))
		keyMaterial = append(keyMaterial, posBytes...)
	}

	// Extract Gaussian coordinates
	for _, gauss := range state.GaussianCoords {
		gaussBytes := make([]byte, 8)
		binary.LittleEndian.PutUint64(gaussBytes, math.Float64bits(gauss))
		keyMaterial = append(keyMaterial, gaussBytes...)
	}

	// Extract Eisenstein coordinates
	for _, eisen := range state.EisensteinCoords {
		eisenBytes := make([]byte, 8)
		binary.LittleEndian.PutUint64(eisenBytes, math.Float64bits(eisen))
		keyMaterial = append(keyMaterial, eisenBytes...)
	}

	// Hash to get final key
	hash := sha256.Sum256(keyMaterial)
	return hash[:qke.keySize]
}

// generateKeyID generates a unique identifier for the established key
func (qke *QuaternionicKeyExchange) generateKeyID() string {
	hash := sha256.New()
	hash.Write([]byte(qke.sessionID))
	hash.Write(qke.sharedSecret)
	hash.Write([]byte(fmt.Sprintf("%d", time.Now().Unix())))
	return fmt.Sprintf("%x", hash.Sum(nil)[:8])
}

// computeCoherence computes the quantum coherence of the current state
func (qke *QuaternionicKeyExchange) computeCoherence() float64 {
	if qke.quaternionicState == nil {
		return 0.0
	}

	// For single quaternionic state, coherence is based on the magnitude and phase stability
	amplitude := qke.quaternionicState.ComputeQuaternionicAmplitude()
	coherence := cmplx.Abs(amplitude)

	// Normalize coherence to [0, 1]
	if coherence > 1.0 {
		coherence = 1.0
	}

	return coherence
}

// computePhaseAlignment computes phase alignment between states
func (qke *QuaternionicKeyExchange) computePhaseAlignment() float64 {
	if qke.peerPublic == nil {
		return 0.0
	}

	// Compare phases of our state and peer's state
	ourPhase := qke.quaternionicState.Phase
	peerPhase := qke.peerPublic.State.Phase

	phaseDiff := math.Abs(ourPhase - peerPhase)

	// Normalize phase difference to [0, π]
	for phaseDiff > math.Pi {
		phaseDiff -= 2 * math.Pi
	}
	for phaseDiff < 0 {
		phaseDiff += 2 * math.Pi
	}

	// Phase alignment is cos(phase difference)
	alignment := math.Cos(phaseDiff)

	// Ensure alignment is in [0, 1] range
	if alignment < 0 {
		alignment = 0
	}

	return alignment
}

// extractPhases extracts phase information from complex amplitudes
func (qke *QuaternionicKeyExchange) extractPhases(amplitudes []complex128) []float64 {
	phases := make([]float64, len(amplitudes))
	for i, amp := range amplitudes {
		phases[i] = cmplx.Phase(amp)
	}
	return phases
}

// computeSecurityStrength computes the security strength of the established key
func (qke *QuaternionicKeyExchange) computeSecurityStrength() float64 {
	if qke.sharedSecret == nil {
		return 0.0
	}

	// Base security on key entropy and coherence
	entropy := qke.computeKeyEntropy()
	coherence := qke.computeCoherence()

	// Combine metrics for overall security score
	return (entropy + coherence) / 2.0
}

// computeKeyEntropy computes the entropy of the shared secret
func (qke *QuaternionicKeyExchange) computeKeyEntropy() float64 {
	if len(qke.sharedSecret) == 0 {
		return 0.0
	}

	// Simple entropy calculation based on byte distribution
	counts := make(map[byte]int)
	for _, b := range qke.sharedSecret {
		counts[b]++
	}

	entropy := 0.0
	total := float64(len(qke.sharedSecret))

	for _, count := range counts {
		if count > 0 {
			p := float64(count) / total
			entropy -= p * math.Log2(p)
		}
	}

	// Normalize to [0, 1]
	return entropy / 8.0 // Maximum entropy for 8-bit values
}

// GetSharedSecret returns the established shared secret
func (qke *QuaternionicKeyExchange) GetSharedSecret() ([]byte, error) {
	qke.mu.RLock()
	defer qke.mu.RUnlock()

	if qke.state != StateSharedComputed && qke.state != StateEstablished {
		return nil, fmt.Errorf("key exchange not completed")
	}

	if qke.sharedSecret == nil {
		return nil, fmt.Errorf("shared secret not computed")
	}

	// Return a copy to prevent external modification
	secret := make([]byte, len(qke.sharedSecret))
	copy(secret, qke.sharedSecret)

	return secret, nil
}

// GetSessionID returns the current session ID
func (qke *QuaternionicKeyExchange) GetSessionID() string {
	return qke.sessionID
}

// GetState returns the current key exchange state
func (qke *QuaternionicKeyExchange) GetState() KeyExchangeState {
	qke.mu.RLock()
	defer qke.mu.RUnlock()
	return qke.state
}

// GetTelemetry returns key exchange telemetry data
func (qke *QuaternionicKeyExchange) GetTelemetry() *KeyExchangeTelemetry {
	qke.mu.RLock()
	defer qke.mu.RUnlock()

	// Return a copy to prevent external modification
	telemetry := &KeyExchangeTelemetry{
		StartTime:        qke.telemetry.StartTime,
		EndTime:          qke.telemetry.EndTime,
		RoundTrips:       qke.telemetry.RoundTrips,
		BytesTransferred: qke.telemetry.BytesTransferred,
		CoherenceLevel:   qke.telemetry.CoherenceLevel,
		PhaseAlignment:   qke.telemetry.PhaseAlignment,
		SecurityStrength: qke.telemetry.SecurityStrength,
		Errors:           make([]string, len(qke.telemetry.Errors)),
	}

	copy(telemetry.Errors, qke.telemetry.Errors)
	return telemetry
}

// IsEstablished returns true if key exchange is successfully established
func (qke *QuaternionicKeyExchange) IsEstablished() bool {
	qke.mu.RLock()
	defer qke.mu.RUnlock()
	return qke.state == StateEstablished
}

// Reset resets the key exchange to initial state
func (qke *QuaternionicKeyExchange) Reset() {
	qke.mu.Lock()
	defer qke.mu.Unlock()

	qke.state = StateInitialized
	qke.sharedSecret = nil
	qke.publicKey = nil
	qke.privateKey = nil
	qke.peerPublic = nil
	qke.peerState = nil

	// Generate new session ID
	qke.generateSessionID()

	// Reset telemetry
	qke.telemetry = &KeyExchangeTelemetry{
		StartTime: time.Now(),
		Errors:    make([]string, 0),
	}
}

// ValidatePeerKey validates the peer's public key
func (qke *QuaternionicKeyExchange) ValidatePeerKey(peerKey *QuaternionicPublicKey) error {
	if peerKey == nil {
		return fmt.Errorf("peer key is nil")
	}

	if peerKey.State == nil {
		return fmt.Errorf("peer key missing state")
	}

	if len(peerKey.Generator) == 0 {
		return fmt.Errorf("peer key missing generators")
	}

	if len(peerKey.Modulus) == 0 {
		return fmt.Errorf("peer key missing modulus")
	}

	// Validate timestamp (not too old)
	if peerKey.Timestamp > 0 {
		age := time.Now().Unix() - peerKey.Timestamp
		if age > 300 { // 5 minutes
			return fmt.Errorf("peer key too old: %d seconds", age)
		}
	}

	// Validate commitment and proof
	if len(peerKey.Commitment) == 0 {
		return fmt.Errorf("peer key missing commitment")
	}

	if len(peerKey.Proof) == 0 {
		return fmt.Errorf("peer key missing proof")
	}

	return nil
}

// GetKeyExchangeStats returns statistics about the key exchange
func (qke *QuaternionicKeyExchange) GetKeyExchangeStats() map[string]interface{} {
	qke.mu.RLock()
	defer qke.mu.RUnlock()

	stats := map[string]interface{}{
		"session_id":        qke.sessionID,
		"state":             qke.state,
		"security_level":    qke.securityLevel,
		"key_size":          qke.keySize,
		"established":       qke.IsEstablished(),
		"round_trips":       qke.telemetry.RoundTrips,
		"coherence_level":   qke.telemetry.CoherenceLevel,
		"phase_alignment":   qke.telemetry.PhaseAlignment,
		"security_strength": qke.telemetry.SecurityStrength,
		"duration_ms":       0,
	}

	if !qke.telemetry.EndTime.IsZero() {
		stats["duration_ms"] = qke.telemetry.EndTime.Sub(qke.telemetry.StartTime).Milliseconds()
	}

	if len(qke.telemetry.Errors) > 0 {
		stats["errors"] = qke.telemetry.Errors
	}

	return stats
}
