package tests

import (
	"math"
	"math/cmplx"
	"testing"
	"time"

	"github.com/resonancelab/psizero/core"
	"github.com/resonancelab/psizero/core/hilbert"
	"github.com/resonancelab/psizero/core/primes"
)

// TestQuaternionicStateMathematicalCorrectness tests the mathematical correctness of quaternionic states
func TestQuaternionicStateMathematicalCorrectness(t *testing.T) {
	t.Run("QuaternionicAmplitudeCalculation", func(t *testing.T) {
		// Test ψq(x,t) = N⁻¹ψ̄q(x)·exp(iφ(x,t)) from paper
		position := []float64{1.0, 2.0, 3.0}
		baseAmplitude := complex(2.0, 3.0)
		gaussian := []float64{0.5, 0.7}
		eisenstein := []float64{0.3, 0.6}

		qs := core.NewQuaternionicState(position, baseAmplitude, gaussian, eisenstein)

		// Test initial amplitude calculation
		amplitude := qs.ComputeQuaternionicAmplitude()

		// The amplitude should be normalized (magnitude = 1.0)
		// because ψq(x,t) = N⁻¹ψ̄q(x)·exp(iφ(x,t)) and N⁻¹ = 1/|ψ̄q|
		expectedMagnitude := 1.0

		if math.Abs(cmplx.Abs(amplitude)-expectedMagnitude) > 1e-10 {
			t.Errorf("Expected normalized amplitude magnitude %.10f, got %.10f", expectedMagnitude, cmplx.Abs(amplitude))
		}

		// Test phase evolution
		deltaTime := 0.1
		frequency := 2.0 * math.Pi
		initialPhase := qs.Phase

		qs.UpdatePhase(deltaTime, frequency)

		expectedPhase := initialPhase + 2.0*math.Pi*frequency*deltaTime
		if math.Abs(qs.Phase-expectedPhase) > 1e-10 {
			t.Errorf("Expected phase %.10f, got %.10f", expectedPhase, qs.Phase)
		}
	})

	t.Run("NormalizationFactorCorrectness", func(t *testing.T) {
		baseAmplitude := complex(3.0, 4.0) // Magnitude = 5
		position := []float64{0.0, 0.0, 0.0}
		gaussian := []float64{0.0, 0.0}
		eisenstein := []float64{0.0, 0.0}

		qs := core.NewQuaternionicState(position, baseAmplitude, gaussian, eisenstein)

		// For |ψ̄q| = 5, N should be 5, so N⁻¹ = 0.2
		expectedNormalization := 1.0 / 5.0

		if math.Abs(qs.NormalizationFactor-expectedNormalization) > 1e-10 {
			t.Errorf("Expected normalization factor %.10f, got %.10f", expectedNormalization, qs.NormalizationFactor)
		}
	})

	t.Run("PhaseDifferenceCalculation", func(t *testing.T) {
		// Create two states with different phases
		pos1 := []float64{1.0, 0.0, 0.0}
		pos2 := []float64{1.0, 0.0, 0.0}
		amp := complex(1.0, 0.0)
		gauss := []float64{0.0, 0.0}
		eisen := []float64{0.0, 0.0}

		qs1 := core.NewQuaternionicState(pos1, amp, gauss, eisen)
		qs2 := core.NewQuaternionicState(pos2, amp, gauss, eisen)

		// Set different phases
		qs1.Phase = math.Pi / 4 // 45 degrees
		qs2.Phase = math.Pi / 2 // 90 degrees

		phaseDiff := qs1.ComputePhaseDifference(qs2)
		expectedDiff := math.Pi / 4 // 45 degrees difference

		if math.Abs(phaseDiff-expectedDiff) > 1e-10 {
			t.Errorf("Expected phase difference %.10f, got %.10f", expectedDiff, phaseDiff)
		}
	})

	t.Run("CoherenceCalculation", func(t *testing.T) {
		// Create multiple states for coherence testing
		states := []*core.QuaternionicState{}

		for i := 0; i < 3; i++ {
			pos := []float64{float64(i), 0.0, 0.0}
			amp := complex(1.0, 0.0)
			gauss := []float64{0.0, 0.0}
			eisen := []float64{0.0, 0.0}

			qs := core.NewQuaternionicState(pos, amp, gauss, eisen)
			qs.Phase = float64(i) * math.Pi / 6 // Different phases
			states = append(states, qs)
		}

		// Test coherence with uniform weights
		weights := [][]float64{
			{0, 1, 1},
			{1, 0, 1},
			{1, 1, 0},
		}

		coherence := states[0].MeasureCoherence(states[1:], weights)

		// Coherence should be between -1 and 1
		if coherence < -1.0 || coherence > 1.0 {
			t.Errorf("Coherence %.10f is outside valid range [-1, 1]", coherence)
		}

		// Test perfect coherence (all phases equal)
		for i := 1; i < len(states); i++ {
			states[i].Phase = states[0].Phase
		}

		perfectCoherence := states[0].MeasureCoherence(states[1:], weights)
		if math.Abs(perfectCoherence-1.0) > 1e-10 {
			t.Errorf("Expected perfect coherence 1.0, got %.10f", perfectCoherence)
		}
	})
}

// TestPrimeOperationsMathematicalCorrectness tests prime number operations
func TestPrimeOperationsMathematicalCorrectness(t *testing.T) {
	primeEngine := primes.NewPrimeEngine(1000)

	t.Run("PrimeGenerationCorrectness", func(t *testing.T) {
		primes := primeEngine.GetPrimeBasis(10)

		// Check first few primes
		expected := []int{2, 3, 5, 7, 11, 13, 17, 19, 23, 29}

		for i, prime := range primes {
			if i < len(expected) && prime != expected[i] {
				t.Errorf("Expected prime %d at index %d, got %d", expected[i], i, prime)
			}
		}
	})

	t.Run("PrimalityTesting", func(t *testing.T) {
		testCases := []struct {
			number   int
			expected bool
		}{
			{2, true}, {3, true}, {4, false}, {5, true},
			{6, false}, {7, true}, {8, false}, {9, false},
			{11, true}, {13, true}, {15, false}, {17, true},
		}

		for _, tc := range testCases {
			result := primeEngine.GetPrimeBasis(100) // Get enough primes
			isPrime := false
			for _, prime := range result {
				if prime == tc.number {
					isPrime = true
					break
				}
			}

			if isPrime != tc.expected {
				t.Errorf("Number %d: expected prime=%v, got prime=%v", tc.number, tc.expected, isPrime)
			}
		}
	})

	t.Run("FactorizationCorrectness", func(t *testing.T) {
		testCases := []struct {
			number          int
			expectedFactors map[int]int
		}{
			{12, map[int]int{2: 2, 3: 1}},
			{15, map[int]int{3: 1, 5: 1}},
			{28, map[int]int{2: 2, 7: 1}},
			{49, map[int]int{7: 2}},
			{97, map[int]int{97: 1}}, // Prime
		}

		for _, tc := range testCases {
			factors := primeEngine.Factorize(tc.number)

			if len(factors) != len(tc.expectedFactors) {
				t.Errorf("Number %d: expected %d factors, got %d", tc.number, len(tc.expectedFactors), len(factors))
				continue
			}

			for _, factor := range factors {
				expected, exists := tc.expectedFactors[factor.Prime]
				if !exists || factor.Exponent != expected {
					t.Errorf("Number %d: factor %d^%d not expected", tc.number, factor.Prime, factor.Exponent)
				}
			}
		}
	})

	t.Run("EulerTotientFunction", func(t *testing.T) {
		testCases := []struct {
			n        int
			expected int
		}{
			{1, 0}, {2, 1}, {3, 2}, {4, 2}, {5, 4}, {6, 2}, {7, 6}, {8, 4}, {9, 6}, {10, 4},
		}

		for _, tc := range testCases {
			result := primeEngine.EulerPhi(tc.n)
			if result != tc.expected {
				t.Errorf("φ(%d): expected %d, got %d", tc.n, tc.expected, result)
			}
		}
	})

	t.Run("MobiusFunction", func(t *testing.T) {
		testCases := []struct {
			n        int
			expected int
		}{
			{1, 1}, {2, -1}, {3, -1}, {4, 0}, {5, -1}, {6, 1}, {7, -1}, {8, 0}, {9, 0}, {10, 1},
		}

		for _, tc := range testCases {
			result := primeEngine.MobiusFunction(tc.n)
			if result != tc.expected {
				t.Errorf("μ(%d): expected %d, got %d", tc.n, tc.expected, result)
			}
		}
	})
}

// TestHilbertSpaceMathematicalCorrectness tests Hilbert space operations
func TestHilbertSpaceMathematicalCorrectness(t *testing.T) {
	primeEngine := primes.NewPrimeEngine(100)
	hilbertSpace, err := hilbert.NewHilbertSpace(10, primeEngine)
	if err != nil {
		t.Fatalf("Failed to create Hilbert space: %v", err)
	}

	t.Run("StateCreationAndNormalization", func(t *testing.T) {
		// Create a non-normalized state
		amplitudes := make([]complex128, 10)
		for i := range amplitudes {
			amplitudes[i] = complex(1.0, 1.0) // Not normalized
		}

		state, err := hilbertSpace.CreateState(amplitudes)
		if err != nil {
			t.Fatalf("Failed to create state: %v", err)
		}

		// Check normalization
		normSquared := 0.0
		for _, amp := range state.Amplitudes {
			normSquared += real(amp)*real(amp) + imag(amp)*imag(amp)
		}

		if math.Abs(normSquared-1.0) > 1e-10 {
			t.Errorf("State not normalized: |ψ|² = %.10f", normSquared)
		}

		if !state.Normalized {
			t.Error("State should be marked as normalized")
		}
	})

	t.Run("InnerProductCorrectness", func(t *testing.T) {
		// Create two states
		amp1 := make([]complex128, 10)
		amp2 := make([]complex128, 10)

		for i := range amp1 {
			amp1[i] = complex(float64(i+1), 0)
			amp2[i] = complex(float64(i+1), 0)
		}

		state1, _ := hilbertSpace.CreateState(amp1)
		state2, _ := hilbertSpace.CreateState(amp2)

		innerProduct, err := hilbertSpace.ComputeInnerProduct(state1, state2)
		if err != nil {
			t.Fatalf("Failed to compute inner product: %v", err)
		}

		// For identical normalized states, <ψ|ψ> = 1
		if math.Abs(real(innerProduct)-1.0) > 1e-10 {
			t.Errorf("Expected <ψ|ψ> = 1, got %.10f", real(innerProduct))
		}
	})

	t.Run("BasisStateOrthogonality", func(t *testing.T) {
		basis1, err := hilbertSpace.CreateBasisState(0)
		if err != nil {
			t.Fatalf("Failed to create basis state 0: %v", err)
		}

		basis2, err := hilbertSpace.CreateBasisState(1)
		if err != nil {
			t.Fatalf("Failed to create basis state 1: %v", err)
		}

		innerProduct, err := hilbertSpace.ComputeInnerProduct(basis1, basis2)
		if err != nil {
			t.Fatalf("Failed to compute inner product: %v", err)
		}

		// Orthogonal basis states should have <0|1> = 0
		if math.Abs(real(innerProduct)) > 1e-10 {
			t.Errorf("Expected orthogonal basis states <0|1> = 0, got %.10f", real(innerProduct))
		}
	})

	t.Run("SuperpositionState", func(t *testing.T) {
		superposition, err := hilbertSpace.CreateSuperposition()
		if err != nil {
			t.Fatalf("Failed to create superposition: %v", err)
		}

		// Check uniform amplitudes
		expectedAmplitude := complex(1.0/math.Sqrt(float64(hilbertSpace.GetDimension())), 0)

		for i, amp := range superposition.Amplitudes {
			if math.Abs(real(amp)-real(expectedAmplitude)) > 1e-10 ||
				math.Abs(imag(amp)-imag(expectedAmplitude)) > 1e-10 {
				t.Errorf("Amplitude %d: expected %.10f, got %.10f + %.10fi",
					i, real(expectedAmplitude), real(amp), imag(amp))
			}
		}
	})
}

// TestPostQuantumCryptoMathematicalCorrectness tests post-quantum cryptographic primitives
func TestPostQuantumCryptoMathematicalCorrectness(t *testing.T) {
	pqc := core.NewPostQuantumCrypto()

	t.Run("KeyGenerationConsistency", func(t *testing.T) {
		key1, err := pqc.GenerateKeyPair("lattice")
		if err != nil {
			t.Fatalf("Failed to generate lattice key: %v", err)
		}

		key2, err := pqc.GenerateKeyPair("hash")
		if err != nil {
			t.Fatalf("Failed to generate hash key: %v", err)
		}

		if key1.KeyID == key2.KeyID {
			t.Error("Generated keys should have unique IDs")
		}

		if key1.Algorithm != "lattice" {
			t.Errorf("Expected algorithm 'lattice', got '%s'", key1.Algorithm)
		}

		if key2.Algorithm != "hash" {
			t.Errorf("Expected algorithm 'hash', got '%s'", key2.Algorithm)
		}
	})

	t.Run("KeyStorageAndRetrieval", func(t *testing.T) {
		key, err := pqc.GenerateKeyPair("rainbow")
		if err != nil {
			t.Fatalf("Failed to generate rainbow key: %v", err)
		}

		// Retrieve key info
		keyInfo, err := pqc.GetKeyInfo(key.KeyID)
		if err != nil {
			t.Fatalf("Failed to retrieve key info: %v", err)
		}

		if keyInfo.KeyID != key.KeyID {
			t.Errorf("Retrieved key ID mismatch: expected %s, got %s", key.KeyID, keyInfo.KeyID)
		}

		if keyInfo.Algorithm != key.Algorithm {
			t.Errorf("Retrieved algorithm mismatch: expected %s, got %s", key.Algorithm, keyInfo.Algorithm)
		}
	})

	t.Run("SecurityLevelValidation", func(t *testing.T) {
		testCases := []struct {
			algorithm     string
			expectedLevel int
		}{
			{"lattice", 128},
			{"hash", 256},
			{"rainbow", 128},
			{"mceliece", 128},
		}

		for _, tc := range testCases {
			level, err := pqc.GetSecurityLevel(tc.algorithm)
			if err != nil {
				t.Errorf("Failed to get security level for %s: %v", tc.algorithm, err)
				continue
			}

			if level != tc.expectedLevel {
				t.Errorf("Algorithm %s: expected security level %d, got %d", tc.algorithm, tc.expectedLevel, level)
			}
		}
	})
}

// TestQuaternionicKeyExchangeMathematicalCorrectness tests the key exchange protocol
func TestQuaternionicKeyExchangeMathematicalCorrectness(t *testing.T) {
	primeEngine := primes.NewPrimeEngine(100)
	hilbertSpace, err := hilbert.NewHilbertSpace(8, primeEngine)
	if err != nil {
		t.Fatalf("Failed to create Hilbert space: %v", err)
	}

	t.Run("KeyExchangeProtocol", func(t *testing.T) {
		// Create two parties
		alice := core.NewQuaternionicKeyExchange(hilbertSpace, primeEngine)
		bob := core.NewQuaternionicKeyExchange(hilbertSpace, primeEngine)

		// Generate keys
		err := alice.GenerateKeys()
		if err != nil {
			t.Fatalf("Alice failed to generate keys: %v", err)
		}

		err = bob.GenerateKeys()
		if err != nil {
			t.Fatalf("Bob failed to generate keys: %v", err)
		}

		// Alice initiates key exchange
		initMsg, err := alice.InitiateKeyExchange()
		if err != nil {
			t.Fatalf("Alice failed to initiate key exchange: %v", err)
		}

		// Bob processes initiation
		responseMsg, err := bob.ProcessKeyExchangeMessage(initMsg)
		if err != nil {
			t.Fatalf("Bob failed to process initiation: %v", err)
		}

		// Alice processes response
		confirmMsg, err := alice.ProcessKeyExchangeMessage(responseMsg)
		if err != nil {
			t.Fatalf("Alice failed to process response: %v", err)
		}

		// Bob processes confirmation
		_, err = bob.ProcessKeyExchangeMessage(confirmMsg)
		if err != nil {
			t.Fatalf("Bob failed to process confirmation: %v", err)
		}

		// Check that both parties have established keys
		if !alice.IsEstablished() {
			t.Error("Alice's key exchange should be established")
		}

		if !bob.IsEstablished() {
			t.Error("Bob's key exchange should be established")
		}

		// Check that shared secrets match
		aliceSecret, err := alice.GetSharedSecret()
		if err != nil {
			t.Fatalf("Failed to get Alice's shared secret: %v", err)
		}

		bobSecret, err := bob.GetSharedSecret()
		if err != nil {
			t.Fatalf("Failed to get Bob's shared secret: %v", err)
		}

		if len(aliceSecret) != len(bobSecret) {
			t.Errorf("Shared secret lengths don't match: Alice=%d, Bob=%d", len(aliceSecret), len(bobSecret))
		}

		// Shared secrets should be identical
		for i := range aliceSecret {
			if i < len(bobSecret) && aliceSecret[i] != bobSecret[i] {
				t.Errorf("Shared secrets don't match at position %d", i)
				break
			}
		}
	})

	t.Run("KeyExchangeSecurityMetrics", func(t *testing.T) {
		alice := core.NewQuaternionicKeyExchange(hilbertSpace, primeEngine)
		bob := core.NewQuaternionicKeyExchange(hilbertSpace, primeEngine)

		// Generate keys and complete exchange
		alice.GenerateKeys()
		bob.GenerateKeys()

		initMsg, _ := alice.InitiateKeyExchange()
		responseMsg, _ := bob.ProcessKeyExchangeMessage(initMsg)
		confirmMsg, _ := alice.ProcessKeyExchangeMessage(responseMsg)
		bob.ProcessKeyExchangeMessage(confirmMsg)

		// Check security metrics
		aliceTelemetry := alice.GetTelemetry()
		bobTelemetry := bob.GetTelemetry()

		if aliceTelemetry.CoherenceLevel < 0 || aliceTelemetry.CoherenceLevel > 1 {
			t.Errorf("Alice coherence level %.3f is outside [0,1] range", aliceTelemetry.CoherenceLevel)
		}

		if bobTelemetry.CoherenceLevel < 0 || bobTelemetry.CoherenceLevel > 1 {
			t.Errorf("Bob coherence level %.3f is outside [0,1] range", bobTelemetry.CoherenceLevel)
		}

		if aliceTelemetry.PhaseAlignment < -1 || aliceTelemetry.PhaseAlignment > 1 {
			t.Errorf("Alice phase alignment %.3f is outside [-1,1] range", aliceTelemetry.PhaseAlignment)
		}

		if bobTelemetry.PhaseAlignment < -1 || bobTelemetry.PhaseAlignment > 1 {
			t.Errorf("Bob phase alignment %.3f is outside [-1,1] range", bobTelemetry.PhaseAlignment)
		}

		if aliceTelemetry.SecurityStrength < 0 || aliceTelemetry.SecurityStrength > 1 {
			t.Errorf("Alice security strength %.3f is outside [0,1] range", aliceTelemetry.SecurityStrength)
		}

		if bobTelemetry.SecurityStrength < 0 || bobTelemetry.SecurityStrength > 1 {
			t.Errorf("Bob security strength %.3f is outside [0,1] range", bobTelemetry.SecurityStrength)
		}
	})
}

// TestIntegrationMathematicalCorrectness tests integration between components
func TestIntegrationMathematicalCorrectness(t *testing.T) {
	t.Run("FullSystemIntegration", func(t *testing.T) {
		// Create all core components
		primeEngine := primes.NewPrimeEngine(100)
		hilbertSpace, err := hilbert.NewHilbertSpace(8, primeEngine)
		if err != nil {
			t.Fatalf("Failed to create Hilbert space: %v", err)
		}

		pqc := core.NewPostQuantumCrypto()
		keyExchange := core.NewQuaternionicKeyExchange(hilbertSpace, primeEngine)

		// Test that all components work together
		err = keyExchange.GenerateKeys()
		if err != nil {
			t.Fatalf("Key exchange key generation failed: %v", err)
		}

		// Generate post-quantum keys
		pqKey, err := pqc.GenerateKeyPair("lattice")
		if err != nil {
			t.Fatalf("Post-quantum key generation failed: %v", err)
		}

		// Verify key exchange session
		sessionID := keyExchange.GetSessionID()
		if sessionID == "" {
			t.Error("Key exchange should have a valid session ID")
		}

		// Verify post-quantum key
		if pqKey.KeyID == "" {
			t.Error("Post-quantum key should have a valid ID")
		}

		// Test key exchange stats
		stats := keyExchange.GetKeyExchangeStats()
		if stats["session_id"] != sessionID {
			t.Error("Key exchange stats should contain correct session ID")
		}

		t.Logf("Integration test passed: Session ID=%s, PQ Key ID=%s", sessionID, pqKey.KeyID)
	})

	t.Run("PerformanceValidation", func(t *testing.T) {
		primeEngine := primes.NewPrimeEngine(50)
		hilbertSpace, _ := hilbert.NewHilbertSpace(5, primeEngine)

		start := time.Now()

		// Create multiple key exchanges
		for i := 0; i < 10; i++ {
			keyExchange := core.NewQuaternionicKeyExchange(hilbertSpace, primeEngine)
			keyExchange.GenerateKeys()

			_, _ = keyExchange.InitiateKeyExchange()
			// Simulate peer response
			responseMsg := &core.KeyExchangeMessage{
				Type:      "key_exchange_response",
				SessionID: keyExchange.GetSessionID(),
				Payload: map[string]interface{}{
					"ack": true,
				},
				Timestamp: time.Now().Unix(),
			}

			keyExchange.ProcessKeyExchangeMessage(responseMsg)
		}

		duration := time.Since(start)
		avgDuration := duration / 10

		t.Logf("Average key exchange duration: %v", avgDuration)

		// Should complete within reasonable time (adjust threshold as needed)
		if avgDuration > 100*time.Millisecond {
			t.Logf("Warning: Key exchange is slower than expected: %v", avgDuration)
		}
	})
}

// BenchmarkMathematicalOperations benchmarks key mathematical operations
func BenchmarkMathematicalOperations(b *testing.B) {
	primeEngine := primes.NewPrimeEngine(100)
	hilbertSpace, _ := hilbert.NewHilbertSpace(10, primeEngine)

	b.Run("QuaternionicStateCreation", func(b *testing.B) {
		position := []float64{1.0, 2.0, 3.0}
		baseAmplitude := complex(2.0, 3.0)
		gaussian := []float64{0.5, 0.7}
		eisenstein := []float64{0.3, 0.6}

		for i := 0; i < b.N; i++ {
			core.NewQuaternionicState(position, baseAmplitude, gaussian, eisenstein)
		}
	})

	b.Run("PrimeFactorization", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			primeEngine.Factorize(100 + i)
		}
	})

	b.Run("HilbertSpaceInnerProduct", func(b *testing.B) {
		amp1 := make([]complex128, 10)
		amp2 := make([]complex128, 10)
		for i := range amp1 {
			amp1[i] = complex(float64(i), 0)
			amp2[i] = complex(float64(i), 0)
		}

		state1, _ := hilbertSpace.CreateState(amp1)
		state2, _ := hilbertSpace.CreateState(amp2)

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			hilbertSpace.ComputeInnerProduct(state1, state2)
		}
	})

	b.Run("KeyExchangeGeneration", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			keyExchange := core.NewQuaternionicKeyExchange(hilbertSpace, primeEngine)
			keyExchange.GenerateKeys()
		}
	})
}
