package nlc

import (
	"fmt"
	"math"
	"time"

	"github.com/resonancelab/psizero/shared/types"
)

// Complete the calculateQuantumCorrelation function
func (nlc *NLCEngine) calculateQuantumCorrelation(bellPair *BellState) float64 {
	// Quantum correlation based on entanglement measure and coherence
	correlation := bellPair.Entanglement * bellPair.Coherence

	// Adjust for decoherence
	timeSinceCreation := time.Since(bellPair.CreatedAt).Seconds()
	decoherenceFactor := math.Exp(-timeSinceCreation / nlc.config.DecoherenceTime)

	correlation *= decoherenceFactor
	return math.Min(correlation, 1.0)
}

// recordTeleportationTelemetry records telemetry for teleportation attempts
func (nlc *NLCEngine) recordTeleportationTelemetry(attempt int, fidelity, correlation float64, success bool) {
	successRate := 0.0
	if success {
		successRate = 1.0
	}

	point := types.TelemetryPoint{
		Step:              attempt,
		SymbolicEntropy:   1.0 - fidelity, // Higher entropy for lower fidelity
		LyapunovMetric:    correlation,    // Use correlation as stability metric
		SatisfactionRate:  successRate,
		ResonanceStrength: nlc.entanglementNetwork.Connectivity,
		Dominance:         fidelity,
		Timestamp:         time.Now(),
	}

	nlc.telemetryPoints = append(nlc.telemetryPoints, point)
}

// Protocol implementation methods
func (nlc *NLCEngine) executeQuantumTeleportation(message *QuantumMessage) error {
	// Implementation of quantum teleportation protocol

	// Find sender and receiver nodes
	var senderNode, receiverNode *CommunicationNode
	for _, node := range nlc.communicationNodes {
		if node.ID == message.SenderID {
			senderNode = node
		} else if node.ID == message.ReceiverID {
			receiverNode = node
		}
	}

	if senderNode == nil || receiverNode == nil {
		return fmt.Errorf("sender or receiver node not found")
	}

	// Find available Bell pair
	var bellPair *BellState
	for _, bell := range nlc.bellStates {
		if !nlc.isBellStateExpired(bell) {
			if (bell.ParticleA.NodeID == message.SenderID && bell.ParticleB.NodeID == message.ReceiverID) ||
				(bell.ParticleA.NodeID == message.ReceiverID && bell.ParticleB.NodeID == message.SenderID) {
				bellPair = bell
				break
			}
		}
	}

	if bellPair == nil {
		return fmt.Errorf("no available Bell pair for teleportation")
	}

	// Perform Bell measurement and classical communication
	measurement := nlc.performBellMeasurement(bellPair)

	// Apply correction operations at receiver based on measurement
	correctionSuccess := nlc.applyCorrectionOperations(receiverNode, measurement)

	if correctionSuccess {
		message.Status = "transmitted"
	} else {
		message.Status = "failed"
	}

	// Record transmission
	record := &TransmissionRecord{
		ID:               fmt.Sprintf("teleport_%d", time.Now().UnixNano()),
		Timestamp:        time.Now(),
		Protocol:         "teleportation",
		SenderID:         message.SenderID,
		ReceiverID:       message.ReceiverID,
		MessageSize:      len(message.Content) * 8, // Convert to bits
		TransmissionTime: time.Duration(100) * time.Microsecond,
		Fidelity:         bellPair.Fidelity,
		ErrorRate:        1.0 - bellPair.Fidelity,
		Success:          correctionSuccess,
	}

	nlc.transmissionHistory = append(nlc.transmissionHistory, record)

	return nil
}

func (nlc *NLCEngine) executeSuperdenseCoding(message *QuantumMessage) error {
	// Implementation of superdense coding protocol
	// Allows sending 2 classical bits using 1 entangled qubit

	// Find sender node
	var senderNode *CommunicationNode
	for _, node := range nlc.communicationNodes {
		if node.ID == message.SenderID {
			senderNode = node
			break
		}
	}

	if senderNode == nil {
		return fmt.Errorf("sender node not found")
	}

	// Encode 2 bits into qubit using Pauli operations
	encodingSuccess := nlc.encodeBitsIntoQubit(senderNode, message.Content)

	if encodingSuccess {
		message.Status = "transmitted"
	} else {
		message.Status = "failed"
	}

	return nil
}

func (nlc *NLCEngine) executeEntanglementSwapping(message *QuantumMessage) error {
	// Implementation of entanglement swapping to extend entanglement distance

	// Find intermediate nodes for swapping
	intermediateNodes := nlc.findIntermediateNodes(message.SenderID, message.ReceiverID)

	if len(intermediateNodes) == 0 {
		return fmt.Errorf("no intermediate nodes available for entanglement swapping")
	}

	// Perform Bell measurements at intermediate nodes
	swappingSuccess := true
	for _, nodeID := range intermediateNodes {
		success := nlc.performEntanglementSwapAtNode(nodeID)
		if !success {
			swappingSuccess = false
			break
		}
	}

	if swappingSuccess {
		message.Status = "transmitted"
	} else {
		message.Status = "failed"
	}

	return nil
}

// Helper methods

func (nlc *NLCEngine) isBellStateExpired(bellState *BellState) bool {
	return time.Now().After(bellState.ExpiresAt)
}

func (nlc *NLCEngine) applyCorrectionOperations(node *CommunicationNode, measurement *MeasurementRecord) bool {
	// Apply quantum correction operations based on Bell measurement results
	// This would involve applying Pauli X and Z operations

	// Simplified implementation
	correctionFidelity := measurement.Confidence * (1.0 - nlc.config.NoiseLevel)
	return correctionFidelity > nlc.config.FidelityThreshold
}

func (nlc *NLCEngine) encodeBitsIntoQubit(node *CommunicationNode, data []byte) bool {
	// Encode classical bits into quantum state using Pauli operations
	// I: 00, X: 01, Z: 10, Y: 11

	if len(data) == 0 {
		return false
	}

	// Simplified encoding simulation
	encodingFidelity := node.QuantumState.Coherence * (1.0 - nlc.config.NoiseLevel)
	return encodingFidelity > nlc.config.FidelityThreshold
}

func (nlc *NLCEngine) findIntermediateNodes(senderID, receiverID string) []string {
	// Find nodes that can serve as intermediates for entanglement swapping
	intermediates := make([]string, 0)

	for _, node := range nlc.entanglementNetwork.Nodes {
		// Check if node is entangled with both sender and receiver (or their neighbors)
		if node.ID != senderID && node.ID != receiverID {
			hasConnectionToSender := false
			hasConnectionToReceiver := false

			for _, entangledWith := range node.EntangledWith {
				if entangledWith == senderID {
					hasConnectionToSender = true
				}
				if entangledWith == receiverID {
					hasConnectionToReceiver = true
				}
			}

			if hasConnectionToSender || hasConnectionToReceiver {
				intermediates = append(intermediates, node.ID)
			}
		}
	}

	return intermediates
}

func (nlc *NLCEngine) performEntanglementSwapAtNode(nodeID string) bool {
	// Perform Bell measurement to swap entanglement

	// Find the node
	var node *EntanglementNode
	for _, n := range nlc.entanglementNetwork.Nodes {
		if n.ID == nodeID {
			node = n
			break
		}
	}

	if node == nil {
		return false
	}

	// Simulate successful swapping based on node coherence
	swapFidelity := node.Coherence * (1.0 - nlc.config.NoiseLevel)
	return swapFidelity > nlc.config.FidelityThreshold
}

// Utility and management methods

// SendQuantumMessage queues a message for transmission
func (nlc *NLCEngine) SendQuantumMessage(senderID, receiverID, protocol string, content []byte, priority int) (*QuantumMessage, error) {
	nlc.mu.Lock()
	defer nlc.mu.Unlock()

	// Find sender node
	var senderNode *CommunicationNode
	for _, node := range nlc.communicationNodes {
		if node.ID == senderID {
			senderNode = node
			break
		}
	}

	if senderNode == nil {
		return nil, fmt.Errorf("sender node not found: %s", senderID)
	}

	// Create quantum message
	message := &QuantumMessage{
		ID:            fmt.Sprintf("msg_%s_%d", senderID, time.Now().UnixNano()),
		SenderID:      senderID,
		ReceiverID:    receiverID,
		Protocol:      protocol,
		Content:       content,
		Priority:      priority,
		SecurityLevel: nlc.config.SecurityLevel,
		CreatedAt:     time.Now(),
		ExpiresAt:     time.Now().Add(time.Duration(nlc.config.ProtocolTimeout) * time.Second),
		Status:        "pending",
	}

	// Encode content in quantum state
	if len(content) > 0 {
		amplitudes := nlc.encodeContentToAmplitudes(content)
		quantumState, err := nlc.resonanceEngine.CreateQuantumState(amplitudes)
		if err != nil {
			return nil, fmt.Errorf("failed to create quantum encoding: %w", err)
		}
		message.QuantumEncoding = quantumState
	}

	// Add to sender's message queue
	senderNode.MessageQueue = append(senderNode.MessageQueue, message)

	return message, nil
}

// encodeContentToAmplitudes encodes classical content into quantum amplitudes
func (nlc *NLCEngine) encodeContentToAmplitudes(content []byte) []complex128 {
	dimension := nlc.resonanceEngine.GetDimension()
	amplitudes := make([]complex128, dimension)

	normFactor := 0.0

	// Encode content bits into quantum superposition
	for i := 0; i < dimension; i++ {
		amplitude := 0.0

		if i < len(content)*8 { // Each byte has 8 bits
			byteIndex := i / 8
			bitIndex := i % 8

			if byteIndex < len(content) {
				bit := (content[byteIndex] >> bitIndex) & 1
				if bit == 1 {
					amplitude = 1.0
				}
			}
		}

		// Add quantum phase
		phase := float64(i) * math.Pi / float64(dimension)
		amplitudes[i] = complex(amplitude*math.Cos(phase), amplitude*math.Sin(phase))
		normFactor += amplitude * amplitude
	}

	// Normalize
	if normFactor > 0 {
		normFactor = math.Sqrt(normFactor)
		for i := range amplitudes {
			amplitudes[i] /= complex(normFactor, 0)
		}
	}

	return amplitudes
}

// GetNetworkStatus returns current network status
func (nlc *NLCEngine) GetNetworkStatus() map[string]interface{} {
	nlc.mu.RLock()
	defer nlc.mu.RUnlock()

	status := map[string]interface{}{
		"network_topology":     nlc.config.NetworkTopology,
		"active_nodes":         len(nlc.communicationNodes),
		"entanglement_links":   len(nlc.entanglementNetwork.Links),
		"active_bell_states":   nlc.countActiveBellStates(),
		"network_coherence":    nlc.entanglementNetwork.Coherence,
		"network_connectivity": nlc.entanglementNetwork.Connectivity,
		"total_transmissions":  len(nlc.transmissionHistory),
	}

	// Node performance summary
	if len(nlc.communicationNodes) > 0 {
		totalSuccessRate := 0.0
		totalThroughput := 0.0

		for _, node := range nlc.communicationNodes {
			totalSuccessRate += node.Performance.SuccessRate
			totalThroughput += node.Performance.Throughput
		}

		status["average_success_rate"] = totalSuccessRate / float64(len(nlc.communicationNodes))
		status["total_throughput"] = totalThroughput
	}

	return status
}

// countActiveBellStates counts non-expired Bell states
func (nlc *NLCEngine) countActiveBellStates() int {
	count := 0
	for _, bell := range nlc.bellStates {
		if !nlc.isBellStateExpired(bell) {
			count++
		}
	}
	return count
}

// GetTelemetry returns current telemetry data
func (nlc *NLCEngine) GetTelemetry() []types.TelemetryPoint {
	nlc.mu.RLock()
	defer nlc.mu.RUnlock()

	telemetry := make([]types.TelemetryPoint, len(nlc.telemetryPoints))
	copy(telemetry, nlc.telemetryPoints)
	return telemetry
}

// GetTransmissionHistory returns communication history
func (nlc *NLCEngine) GetTransmissionHistory() []*TransmissionRecord {
	nlc.mu.RLock()
	defer nlc.mu.RUnlock()

	history := make([]*TransmissionRecord, len(nlc.transmissionHistory))
	copy(history, nlc.transmissionHistory)
	return history
}

// GetCurrentState returns the current state of the NLC engine
func (nlc *NLCEngine) GetCurrentState() map[string]interface{} {
	nlc.mu.RLock()
	defer nlc.mu.RUnlock()

	state := map[string]interface{}{
		"current_protocol":    nlc.currentProtocol,
		"protocol_step":       nlc.protocolStep,
		"elapsed_time":        time.Since(nlc.startTime).Seconds(),
		"communication_nodes": len(nlc.communicationNodes),
		"quantum_channels":    len(nlc.quantumChannels),
		"bell_states":         len(nlc.bellStates),
	}

	if nlc.entanglementNetwork != nil {
		state["entanglement_network"] = map[string]interface{}{
			"topology":     nlc.entanglementNetwork.Topology,
			"node_count":   len(nlc.entanglementNetwork.Nodes),
			"link_count":   len(nlc.entanglementNetwork.Links),
			"coherence":    nlc.entanglementNetwork.Coherence,
			"connectivity": nlc.entanglementNetwork.Connectivity,
			"max_distance": nlc.entanglementNetwork.MaxDistance,
		}
	}

	// Recent transmission statistics
	recentTransmissions := 0
	successfulTransmissions := 0
	totalFidelity := 0.0

	cutoff := time.Now().Add(-time.Hour) // Last hour
	for _, record := range nlc.transmissionHistory {
		if record.Timestamp.After(cutoff) {
			recentTransmissions++
			if record.Success {
				successfulTransmissions++
			}
			totalFidelity += record.Fidelity
		}
	}

	state["recent_stats"] = map[string]interface{}{
		"transmissions":    recentTransmissions,
		"success_rate":     float64(successfulTransmissions) / math.Max(float64(recentTransmissions), 1.0),
		"average_fidelity": totalFidelity / math.Max(float64(recentTransmissions), 1.0),
	}

	return state
}

// Reset resets the NLC engine to initial state
func (nlc *NLCEngine) Reset() {
	nlc.mu.Lock()
	defer nlc.mu.Unlock()

	nlc.entanglementNetwork = nil
	nlc.communicationNodes = make([]*CommunicationNode, 0)
	nlc.quantumChannels = make([]*QuantumCommunicationChannel, 0)
	nlc.bellStates = make([]*BellState, 0)
	nlc.currentProtocol = ""
	nlc.protocolStep = 0
	nlc.telemetryPoints = make([]types.TelemetryPoint, 0)
	nlc.transmissionHistory = make([]*TransmissionRecord, 0)
}

// SetConfig updates the NLC configuration
func (nlc *NLCEngine) SetConfig(config *NLCConfig) {
	nlc.mu.Lock()
	defer nlc.mu.Unlock()

	if config != nil {
		nlc.config = config
	}
}

// GetConfig returns the current NLC configuration
func (nlc *NLCEngine) GetConfig() *NLCConfig {
	nlc.mu.RLock()
	defer nlc.mu.RUnlock()

	configCopy := *nlc.config
	return &configCopy
}

// ProcessPendingMessages processes queued messages for all nodes
func (nlc *NLCEngine) ProcessPendingMessages() error {
	nlc.mu.Lock()
	defer nlc.mu.Unlock()

	for _, node := range nlc.communicationNodes {
		for i := len(node.MessageQueue) - 1; i >= 0; i-- {
			message := node.MessageQueue[i]

			// Check if message has expired
			if time.Now().After(message.ExpiresAt) {
				message.Status = "expired"
				// Remove from queue
				node.MessageQueue = append(node.MessageQueue[:i], node.MessageQueue[i+1:]...)
				continue
			}

			// Process message based on protocol
			if protocol, exists := node.Protocols[message.Protocol]; exists {
				if err := protocol.Implementation(nlc, message); err != nil {
					message.Status = "failed"
				}

				// Remove processed message from queue
				node.MessageQueue = append(node.MessageQueue[:i], node.MessageQueue[i+1:]...)
			}
		}
	}

	return nil
}

// MaintainEntanglementNetwork performs network maintenance
func (nlc *NLCEngine) MaintainEntanglementNetwork() error {
	nlc.mu.Lock()
	defer nlc.mu.Unlock()

	// Remove expired Bell states
	activeBellStates := make([]*BellState, 0)
	for _, bell := range nlc.bellStates {
		if !nlc.isBellStateExpired(bell) {
			activeBellStates = append(activeBellStates, bell)
		}
	}
	nlc.bellStates = activeBellStates

	// Update network metrics
	nlc.calculateNetworkMetrics()

	// Update node coherence due to decoherence
	for _, node := range nlc.entanglementNetwork.Nodes {
		decayFactor := math.Exp(-time.Since(node.LastMeasurement).Seconds() / nlc.config.DecoherenceTime)
		node.Coherence *= decayFactor

		// Update fidelity
		node.Fidelity *= decayFactor
	}

	return nil
}
