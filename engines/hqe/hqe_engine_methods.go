package hqe

import (
	"fmt"
	"math"
	"math/cmplx"
	"math/rand"
	"time"

	"github.com/resonancelab/psizero/shared/types"
)

// initializeEntanglementGraph creates the tensor network structure
func (hqe *HQEEngine) initializeEntanglementGraph() error {
	nodeCount := len(hqe.boundaryStates) + len(hqe.holographicSlices)*10 // All boundary states + slice states

	hqe.entanglementGraph = &EntanglementGraph{
		Nodes: make([]*EntanglementNode, nodeCount),
		Edges: make([]*EntanglementEdge, 0),
	}

	nodeIndex := 0

	// Create nodes for boundary states
	for i, boundaryState := range hqe.boundaryStates {
		node := &EntanglementNode{
			ID:           nodeIndex,
			Position:     append([]float64{}, boundaryState.Position...),
			QuantumState: boundaryState.QuantumState,
			LocalEntropy: boundaryState.QuantumState.Entropy,
			Connectivity: make([]int, 0),
		}
		hqe.entanglementGraph.Nodes[nodeIndex] = node
		nodeIndex++
	}

	// Create nodes for bulk states
	for _, slice := range hqe.holographicSlices {
		for j, state := range slice.QuantumStates {
			position := []float64{
				slice.RadialDepth,
				float64(j) * 0.1,               // Angular position
				rand.Float64() * 2.0 * math.Pi, // Azimuthal position
			}

			node := &EntanglementNode{
				ID:           nodeIndex,
				Position:     position,
				QuantumState: state,
				LocalEntropy: state.Entropy,
				Connectivity: make([]int, 0),
			}
			hqe.entanglementGraph.Nodes[nodeIndex] = node
			nodeIndex++
		}
	}

	// Create entanglement edges based on spatial proximity and quantum correlations
	if err := hqe.createEntanglementEdges(); err != nil {
		return fmt.Errorf("failed to create entanglement edges: %w", err)
	}

	// Update graph metrics
	hqe.updateEntanglementGraphMetrics()

	return nil
}

// createEntanglementEdges establishes entanglement connections between nodes
func (hqe *HQEEngine) createEntanglementEdges() error {
	nodes := hqe.entanglementGraph.Nodes
	threshold := hqe.config.EntanglementThreshold

	for i, nodeA := range nodes {
		for j, nodeB := range nodes {
			if i >= j {
				continue
			}

			// Calculate entanglement strength based on quantum correlation
			entanglementStrength, err := hqe.calculateEntanglementStrength(nodeA, nodeB)
			if err != nil {
				continue // Skip if calculation fails
			}

			// Create edge if above threshold
			if entanglementStrength > threshold {
				mutualInfo := hqe.calculateMutualInformation(nodeA, nodeB)

				// Create quantum channel
				channel := &QuantumChannel{
					Capacity:   entanglementStrength,
					Fidelity:   0.9 + 0.1*entanglementStrength,
					NoiseLevel: 0.1 * (1.0 - entanglementStrength),
				}

				edge := &EntanglementEdge{
					NodeA:                i,
					NodeB:                j,
					EntanglementStrength: entanglementStrength,
					MutualInformation:    mutualInfo,
					QuantumChannel:       channel,
				}

				hqe.entanglementGraph.Edges = append(hqe.entanglementGraph.Edges, edge)

				// Update node connectivity
				nodeA.Connectivity = append(nodeA.Connectivity, j)
				nodeB.Connectivity = append(nodeB.Connectivity, i)
				nodeA.EntanglementDegree++
				nodeB.EntanglementDegree++
			}
		}
	}

	return nil
}

// calculateEntanglementStrength computes entanglement between two quantum states
func (hqe *HQEEngine) calculateEntanglementStrength(nodeA, nodeB *EntanglementNode) (float64, error) {
	// Calculate quantum state overlap
	overlap, err := hqe.resonanceEngine.GetHilbertSpace().ComputeInnerProduct(
		nodeA.QuantumState, nodeB.QuantumState)
	if err != nil {
		return 0.0, err
	}

	// Calculate spatial correlation factor
	distance := hqe.calculateNodeDistance(nodeA, nodeB)
	spatialFactor := math.Exp(-distance / hqe.config.AdSRadius)

	// Entanglement strength combines quantum and spatial correlations
	quantumOverlap := real(overlap * cmplx.Conj(overlap))
	strength := math.Sqrt(quantumOverlap * spatialFactor)

	return math.Min(strength, 1.0), nil
}

// calculateNodeDistance computes distance between two nodes
func (hqe *HQEEngine) calculateNodeDistance(nodeA, nodeB *EntanglementNode) float64 {
	distance := 0.0
	minDim := len(nodeA.Position)
	if len(nodeB.Position) < minDim {
		minDim = len(nodeB.Position)
	}

	for i := 0; i < minDim; i++ {
		diff := nodeA.Position[i] - nodeB.Position[i]
		distance += diff * diff
	}

	return math.Sqrt(distance)
}

// calculateMutualInformation computes mutual information between two nodes
func (hqe *HQEEngine) calculateMutualInformation(nodeA, nodeB *EntanglementNode) float64 {
	// Mutual information: I(A:B) = S(A) + S(B) - S(A,B)
	entropyA := nodeA.LocalEntropy
	entropyB := nodeB.LocalEntropy

	// Estimate joint entropy (simplified model)
	jointEntropy := entropyA + entropyB - 0.5*math.Min(entropyA, entropyB)

	mutualInfo := entropyA + entropyB - jointEntropy
	return math.Max(0.0, mutualInfo)
}

// updateEntanglementGraphMetrics updates global graph metrics
func (hqe *HQEEngine) updateEntanglementGraphMetrics() {
	if len(hqe.entanglementGraph.Nodes) == 0 {
		return
	}

	// Calculate total entropy
	totalEntropy := 0.0
	for _, node := range hqe.entanglementGraph.Nodes {
		totalEntropy += node.LocalEntropy
	}
	hqe.entanglementGraph.TotalEntropy = totalEntropy

	// Calculate average connectivity
	totalConnections := 0
	for _, node := range hqe.entanglementGraph.Nodes {
		totalConnections += node.EntanglementDegree
	}
	hqe.entanglementGraph.AverageConnectivity = float64(totalConnections) / float64(len(hqe.entanglementGraph.Nodes))

	// Calculate clustering coefficient
	hqe.entanglementGraph.ClusteringCoeff = hqe.calculateClusteringCoefficient()
}

// calculateClusteringCoefficient computes the clustering coefficient of the graph
func (hqe *HQEEngine) calculateClusteringCoefficient() float64 {
	totalClustering := 0.0
	nodeCount := 0

	for _, node := range hqe.entanglementGraph.Nodes {
		if node.EntanglementDegree < 2 {
			continue // Need at least 2 connections for clustering
		}

		// Count triangles involving this node
		triangles := 0
		possibleTriangles := node.EntanglementDegree * (node.EntanglementDegree - 1) / 2

		for i, neighborA := range node.Connectivity {
			for j, neighborB := range node.Connectivity {
				if i >= j {
					continue
				}

				// Check if neighborA and neighborB are connected
				if hqe.areNodesConnected(neighborA, neighborB) {
					triangles++
				}
			}
		}

		if possibleTriangles > 0 {
			totalClustering += float64(triangles) / float64(possibleTriangles)
			nodeCount++
		}
	}

	if nodeCount > 0 {
		return totalClustering / float64(nodeCount)
	}
	return 0.0
}

// areNodesConnected checks if two nodes are connected by an edge
func (hqe *HQEEngine) areNodesConnected(nodeA, nodeB int) bool {
	for _, edge := range hqe.entanglementGraph.Edges {
		if (edge.NodeA == nodeA && edge.NodeB == nodeB) ||
			(edge.NodeA == nodeB && edge.NodeB == nodeA) {
			return true
		}
	}
	return false
}

// evolveHolographicSystem runs the main holographic evolution loop
func (hqe *HQEEngine) evolveHolographicSystem() (*HolographicSimulationResult, error) {
	timeout := time.Duration(hqe.config.TimeoutSeconds) * time.Second

	for hqe.currentStep = 0; hqe.currentStep < hqe.config.MaxIterations; hqe.currentStep++ {
		// Check timeout
		if time.Since(hqe.startTime) > timeout {
			break
		}

		// Evolve bulk geometry
		if err := hqe.evolveBulkGeometry(); err != nil {
			return nil, fmt.Errorf("bulk evolution failed at step %d: %w", hqe.currentStep, err)
		}

		// Evolve boundary states
		if err := hqe.evolveBoundaryStates(); err != nil {
			return nil, fmt.Errorf("boundary evolution failed at step %d: %w", hqe.currentStep, err)
		}

		// Update entanglement structure
		if err := hqe.updateEntanglementStructure(); err != nil {
			return nil, fmt.Errorf("entanglement update failed at step %d: %w", hqe.currentStep, err)
		}

		// Record telemetry
		hqe.recordTelemetry()

		// Check for convergence
		if hqe.checkHolographicConvergence() {
			break
		}
	}

	// Generate final result
	result := hqe.generateSimulationResult()
	return result, nil
}

// evolveBulkGeometry evolves the AdS bulk geometry
func (hqe *HQEEngine) evolveBulkGeometry() error {
	dt := hqe.config.TimeStep

	// Evolve each holographic slice
	for _, slice := range hqe.holographicSlices {
		// Evolve quantum states in the slice
		for _, state := range slice.QuantumStates {
			if err := hqe.resonanceEngine.EvolveStateWithResonance(
				state, dt, 0.3); err != nil {
				return fmt.Errorf("quantum evolution failed: %w", err)
			}
		}

		// Update slice metrics
		slice.Energy = hqe.calculateSliceEnergy(slice.QuantumStates)
		slice.Entropy = hqe.calculateSliceEntropy(slice.QuantumStates)
		slice.Temperature = hqe.calculateSliceTemperature(slice.RadialDepth)

		// Update geometry tensor based on energy-momentum
		hqe.updateSliceGeometry(slice)
	}

	// Update bulk metric and curvature
	hqe.updateBulkMetric()

	return nil
}

// updateSliceGeometry updates the geometry tensor of a slice
func (hqe *HQEEngine) updateSliceGeometry(slice *HolographicSlice) {
	// Einstein field equations: G_μν = 8πT_μν
	energyDensity := slice.Energy
	pressure := slice.Temperature * slice.Entropy / slice.Volume

	for i := 0; i < hqe.config.BulkDimension; i++ {
		for j := 0; j < hqe.config.BulkDimension; j++ {
			if i == j {
				if i == 0 {
					// Time-time component
					slice.GeometryTensor[i][j] = complex(-energyDensity, 0)
				} else {
					// Spatial components
					slice.GeometryTensor[i][j] = complex(pressure, 0)
				}
			}
		}
	}
}

// updateBulkMetric updates the overall bulk metric
func (hqe *HQEEngine) updateBulkMetric() {
	// Average geometry over all slices to update bulk metric
	totalEnergy := 0.0
	totalEntropy := 0.0

	for _, slice := range hqe.holographicSlices {
		totalEnergy += slice.Energy
		totalEntropy += slice.Entropy
	}

	hqe.bulkGeometry.Entropy = totalEntropy

	// Update Hawking temperature
	if hqe.bulkGeometry.HorizonRadius > 0 {
		hqe.bulkGeometry.Temperature = 1.0 / (4.0 * math.Pi * hqe.bulkGeometry.HorizonRadius)
	}
}

// evolveBoundaryStates evolves the CFT boundary states
func (hqe *HQEEngine) evolveBoundaryStates() error {
	dt := hqe.config.TimeStep

	for _, boundaryState := range hqe.boundaryStates {
		// Evolve quantum state
		if err := hqe.resonanceEngine.EvolveStateWithResonance(
			boundaryState.QuantumState, dt, 0.5); err != nil {
			return fmt.Errorf("boundary state evolution failed: %w", err)
		}

		// Update energy
		boundaryState.Energy = boundaryState.QuantumState.Energy

		// Update correlation functions
		boundaryState.CorrelationFunc = hqe.generateCorrelationFunctions(boundaryState.Position)

		// Update conformal weight
		boundaryState.ConformalWeight = hqe.calculateConformalWeight(
			boundaryState.Position, boundaryState.QuantumState)
	}

	return nil
}

// updateEntanglementStructure updates the entanglement graph
func (hqe *HQEEngine) updateEntanglementStructure() error {
	// Recalculate entanglement strengths
	for _, edge := range hqe.entanglementGraph.Edges {
		nodeA := hqe.entanglementGraph.Nodes[edge.NodeA]
		nodeB := hqe.entanglementGraph.Nodes[edge.NodeB]

		newStrength, err := hqe.calculateEntanglementStrength(nodeA, nodeB)
		if err != nil {
			continue
		}

		edge.EntanglementStrength = newStrength
		edge.MutualInformation = hqe.calculateMutualInformation(nodeA, nodeB)

		// Update quantum channel properties
		edge.QuantumChannel.Capacity = newStrength
		edge.QuantumChannel.Fidelity = 0.9 + 0.1*newStrength
		edge.QuantumChannel.NoiseLevel = 0.1 * (1.0 - newStrength)
	}

	// Update global graph metrics
	hqe.updateEntanglementGraphMetrics()

	return nil
}

// recordTelemetry records current state for telemetry
func (hqe *HQEEngine) recordTelemetry() {
	// Calculate system-wide metrics
	totalBulkEntropy := 0.0
	avgBulkEnergy := 0.0
	maxTemperature := 0.0

	for _, slice := range hqe.holographicSlices {
		totalBulkEntropy += slice.Entropy
		avgBulkEnergy += slice.Energy
		if slice.Temperature > maxTemperature {
			maxTemperature = slice.Temperature
		}
	}

	if len(hqe.holographicSlices) > 0 {
		avgBulkEnergy /= float64(len(hqe.holographicSlices))
	}

	// Calculate boundary metrics
	totalBoundaryEnergy := 0.0
	avgConformalWeight := 0.0

	for _, state := range hqe.boundaryStates {
		totalBoundaryEnergy += state.Energy
		avgConformalWeight += state.ConformalWeight
	}

	if len(hqe.boundaryStates) > 0 {
		avgConformalWeight /= float64(len(hqe.boundaryStates))
	}

	// Calculate holographic complexity
	complexity := hqe.calculateHolographicComplexity()

	// Calculate geometry stability
	stability := hqe.calculateGeometryStability()

	point := types.TelemetryPoint{
		Step:              hqe.currentStep,
		SymbolicEntropy:   totalBulkEntropy,
		LyapunovMetric:    stability,
		SatisfactionRate:  complexity,
		ResonanceStrength: hqe.entanglementGraph.AverageConnectivity,
		Dominance:         avgConformalWeight,
		Timestamp:         time.Now(),
	}

	hqe.telemetryPoints = append(hqe.telemetryPoints, point)
}

// calculateHolographicComplexity computes the holographic complexity
func (hqe *HQEEngine) calculateHolographicComplexity() float64 {
	// Complexity related to entanglement structure and bulk geometry
	entanglementComplexity := hqe.entanglementGraph.TotalEntropy * hqe.entanglementGraph.AverageConnectivity

	geometryComplexity := 0.0
	for _, slice := range hqe.holographicSlices {
		geometryComplexity += math.Abs(slice.CurvatureScalar) * slice.Volume
	}

	return (entanglementComplexity + geometryComplexity) / 2.0
}

// calculateGeometryStability computes the stability of the bulk geometry
func (hqe *HQEEngine) calculateGeometryStability() float64 {
	if len(hqe.holographicSlices) < 2 {
		return 1.0
	}

	// Calculate variance in energy density across slices
	avgEnergy := 0.0
	for _, slice := range hqe.holographicSlices {
		avgEnergy += slice.Energy
	}
	avgEnergy /= float64(len(hqe.holographicSlices))

	variance := 0.0
	for _, slice := range hqe.holographicSlices {
		diff := slice.Energy - avgEnergy
		variance += diff * diff
	}
	variance /= float64(len(hqe.holographicSlices))

	// Stability is inverse of variance (higher variance = lower stability)
	stability := 1.0 / (1.0 + variance)
	return stability
}

// checkHolographicConvergence checks if the holographic system has converged
func (hqe *HQEEngine) checkHolographicConvergence() bool {
	if len(hqe.telemetryPoints) < 50 {
		return false
	}

	// Check convergence based on entropy and complexity plateaus
	recent := hqe.telemetryPoints[len(hqe.telemetryPoints)-20:]

	entropyVariance := 0.0
	complexityVariance := 0.0

	avgEntropy := 0.0
	avgComplexity := 0.0
	for _, point := range recent {
		avgEntropy += point.SymbolicEntropy
		avgComplexity += point.SatisfactionRate
	}
	avgEntropy /= float64(len(recent))
	avgComplexity /= float64(len(recent))

	for _, point := range recent {
		entropyDiff := point.SymbolicEntropy - avgEntropy
		complexityDiff := point.SatisfactionRate - avgComplexity
		entropyVariance += entropyDiff * entropyDiff
		complexityVariance += complexityDiff * complexityDiff
	}

	entropyVariance /= float64(len(recent))
	complexityVariance /= float64(len(recent))

	tolerance := hqe.config.ReconstructionTolerance
	return entropyVariance < tolerance && complexityVariance < tolerance
}

// generateSimulationResult creates the final simulation result
func (hqe *HQEEngine) generateSimulationResult() *HolographicSimulationResult {
	// Calculate final metrics
	totalEntropy := 0.0
	for _, slice := range hqe.holographicSlices {
		totalEntropy += slice.Entropy
	}

	complexity := hqe.calculateHolographicComplexity()
	stability := hqe.calculateGeometryStability()

	avgBoundaryEnergy := 0.0
	for _, state := range hqe.boundaryStates {
		avgBoundaryEnergy += state.Energy
	}
	if len(hqe.boundaryStates) > 0 {
		avgBoundaryEnergy /= float64(len(hqe.boundaryStates))
	}

	// Generate boundary correlations matrix
	correlations := hqe.generateBoundaryCorrelations()

	// Check if system is in thermal state
	thermalState := hqe.checkThermalState()

	// Check convergence
	converged := hqe.checkHolographicConvergence()

	result := &HolographicSimulationResult{
		BulkReconstruction:    hqe.reconstructionData,
		BoundaryCorrelations:  correlations,
		EntanglementEntropy:   totalEntropy,
		HolographicComplexity: complexity,
		GeometryStability:     stability,
		CFTEnergyDensity:      avgBoundaryEnergy,
		BlackHoleEntropy:      hqe.bulkGeometry.Entropy,
		ThermalState:          thermalState,
		ComputeTime:           time.Since(hqe.startTime).Seconds(),
		Converged:             converged,
	}

	// Store reconstruction data
	hqe.reconstructionData["bulk_entropy"] = totalEntropy
	hqe.reconstructionData["complexity"] = complexity
	hqe.reconstructionData["stability"] = stability
	hqe.reconstructionData["converged"] = converged

	return result
}

// generateBoundaryCorrelations computes CFT boundary correlation matrix
func (hqe *HQEEngine) generateBoundaryCorrelations() [][]complex128 {
	n := len(hqe.boundaryStates)
	correlations := make([][]complex128, n)

	for i := 0; i < n; i++ {
		correlations[i] = make([]complex128, n)
		for j := 0; j < n; j++ {
			if i == j {
				correlations[i][j] = complex(1.0, 0)
			} else {
				// Calculate correlation between boundary states
				overlap, err := hqe.resonanceEngine.GetHilbertSpace().ComputeInnerProduct(
					hqe.boundaryStates[i].QuantumState,
					hqe.boundaryStates[j].QuantumState)
				if err != nil {
					correlations[i][j] = complex(0, 0)
				} else {
					correlations[i][j] = overlap
				}
			}
		}
	}

	return correlations
}

// checkThermalState determines if the system is in thermal equilibrium
func (hqe *HQEEngine) checkThermalState() bool {
	if len(hqe.holographicSlices) < 2 {
		return false
	}

	// Check if temperature is approximately constant across slices
	avgTemp := 0.0
	for _, slice := range hqe.holographicSlices {
		avgTemp += slice.Temperature
	}
	avgTemp /= float64(len(hqe.holographicSlices))

	tempVariance := 0.0
	for _, slice := range hqe.holographicSlices {
		diff := slice.Temperature - avgTemp
		tempVariance += diff * diff
	}
	tempVariance /= float64(len(hqe.holographicSlices))

	// System is thermal if temperature variance is small
	return tempVariance < 0.01
}

// GetTelemetry returns current telemetry data
func (hqe *HQEEngine) GetTelemetry() []types.TelemetryPoint {
	hqe.mu.RLock()
	defer hqe.mu.RUnlock()

	telemetry := make([]types.TelemetryPoint, len(hqe.telemetryPoints))
	copy(telemetry, hqe.telemetryPoints)
	return telemetry
}

// GetCurrentState returns the current state of the HQE engine
func (hqe *HQEEngine) GetCurrentState() map[string]interface{} {
	hqe.mu.RLock()
	defer hqe.mu.RUnlock()

	state := map[string]interface{}{
		"step":           hqe.currentStep,
		"slice_count":    len(hqe.holographicSlices),
		"boundary_count": len(hqe.boundaryStates),
		"elapsed_time":   time.Since(hqe.startTime).Seconds(),
	}

	if hqe.bulkGeometry != nil {
		state["bulk_geometry"] = map[string]interface{}{
			"dimension":      hqe.bulkGeometry.Dimension,
			"ads_radius":     hqe.bulkGeometry.AdSRadius,
			"horizon_radius": hqe.bulkGeometry.HorizonRadius,
			"temperature":    hqe.bulkGeometry.Temperature,
			"entropy":        hqe.bulkGeometry.Entropy,
		}
	}

	if hqe.entanglementGraph != nil {
		state["entanglement_graph"] = map[string]interface{}{
			"node_count":       len(hqe.entanglementGraph.Nodes),
			"edge_count":       len(hqe.entanglementGraph.Edges),
			"total_entropy":    hqe.entanglementGraph.TotalEntropy,
			"avg_connectivity": hqe.entanglementGraph.AverageConnectivity,
			"clustering_coeff": hqe.entanglementGraph.ClusteringCoeff,
		}
	}

	return state
}

// Reset resets the HQE engine to initial state
func (hqe *HQEEngine) Reset() {
	hqe.mu.Lock()
	defer hqe.mu.Unlock()

	hqe.holographicSlices = make([]*HolographicSlice, 0)
	hqe.boundaryStates = make([]*BoundaryState, 0)
	hqe.bulkGeometry = nil
	hqe.entanglementGraph = nil
	hqe.currentStep = 0
	hqe.telemetryPoints = make([]types.TelemetryPoint, 0)
	hqe.reconstructionData = make(map[string]interface{})
}

// SetConfig updates the HQE configuration
func (hqe *HQEEngine) SetConfig(config *HQEConfig) {
	hqe.mu.Lock()
	defer hqe.mu.Unlock()

	if config != nil {
		hqe.config = config
	}
}

// GetConfig returns the current HQE configuration
func (hqe *HQEEngine) GetConfig() *HQEConfig {
	hqe.mu.RLock()
	defer hqe.mu.RUnlock()

	configCopy := *hqe.config
	return &configCopy
}
