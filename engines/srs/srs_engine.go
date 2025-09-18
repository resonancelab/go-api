package srs

import (
	"fmt"
	"math"
	"math/rand"
	"sync"
	"time"

	"github.com/resonancelab/psizero/core"
	"github.com/resonancelab/psizero/core/hilbert"
	"github.com/resonancelab/psizero/shared/types"
)

// SRSEngine implements the Symbolic Resonance Solver for P=NP problems
type SRSEngine struct {
	resonanceEngine *core.ResonanceEngine
	particles       []*EntropyParticle
	constraints     []Constraint
	problemSpace    *ProblemSpace
	config          *SRSConfig
	mu              sync.RWMutex

	// Evolution tracking
	currentIteration int
	startTime        time.Time
	bestSolution     *Solution
	telemetryPoints  []types.TelemetryPoint
}

// EntropyParticle represents a symbolic entropy particle in the problem space
type EntropyParticle struct {
	ID           int                   `json:"id"`
	Position     []float64             `json:"position"`    // Position in problem space
	Velocity     []float64             `json:"velocity"`    // Velocity vector
	Mass         float64               `json:"mass"`        // Particle mass
	Energy       float64               `json:"energy"`      // Current energy
	Entropy      float64               `json:"entropy"`     // Local entropy
	Assignment   []int                 `json:"assignment"`  // Current variable assignment
	Constraints  []int                 `json:"constraints"` // Constraint indices affecting this particle
	Satisfied    int                   `json:"satisfied"`   // Number of satisfied constraints
	Resonance    float64               `json:"resonance"`   // Resonance strength with solution space
	QuantumState *hilbert.QuantumState `json:"-"`           // Associated quantum state
}

// Constraint represents a constraint in the NP problem
type Constraint interface {
	Evaluate(assignment []int) bool
	GetVariables() []int
	GetWeight() float64
	GetType() string
}

// SATClause represents a SAT clause constraint
type SATClause struct {
	Literals []Literal `json:"literals"`
	Weight   float64   `json:"weight"`
}

// Literal represents a literal in a SAT clause
type Literal struct {
	Variable int  `json:"variable"`
	Negated  bool `json:"negated"`
}

// SubsetSumConstraint represents a subset sum constraint
type SubsetSumConstraint struct {
	Numbers []float64 `json:"numbers"`
	Target  float64   `json:"target"`
	Weight  float64   `json:"weight"`
}

// ProblemSpace defines the search space for the NP problem
type ProblemSpace struct {
	Dimensions  int                    `json:"dimensions"`
	Variables   int                    `json:"variables"`
	Bounds      [][2]float64           `json:"bounds"`
	ProblemType string                 `json:"problem_type"`
	Metadata    map[string]interface{} `json:"metadata"`
}

// SRSConfig contains configuration for the SRS engine
type SRSConfig struct {
	ParticleCount     int     `json:"particle_count"`
	MaxIterations     int     `json:"max_iterations"`
	PlateauThreshold  float64 `json:"plateau_threshold"`
	EntropyLambda     float64 `json:"entropy_lambda"`
	ResonanceStrength float64 `json:"resonance_strength"`
	InertiaWeight     float64 `json:"inertia_weight"`
	CognitiveFactor   float64 `json:"cognitive_factor"`
	SocialFactor      float64 `json:"social_factor"`
	QuantumFactor     float64 `json:"quantum_factor"`
	TimeoutSeconds    int     `json:"timeout_seconds"`
}

// Solution represents a solution to the NP problem
type Solution struct {
	Assignment  []int   `json:"assignment"`
	Feasible    bool    `json:"feasible"`
	Objective   float64 `json:"objective"`
	Satisfied   int     `json:"satisfied"`
	Total       int     `json:"total"`
	Energy      float64 `json:"energy"`
	Entropy     float64 `json:"entropy"`
	Confidence  float64 `json:"confidence"`
	FoundAt     int     `json:"found_at"`
	ComputeTime float64 `json:"compute_time"`
}

// NewSRSEngine creates a new Symbolic Resonance Solver engine
func NewSRSEngine() (*SRSEngine, error) {
	// Initialize core resonance engine
	config := core.DefaultEngineConfig()
	config.Dimension = 100      // Larger space for complex problems
	config.InitialEntropy = 2.0 // Higher initial entropy for exploration

	resonanceEngine, err := core.NewResonanceEngine(config)
	if err != nil {
		return nil, fmt.Errorf("failed to create resonance engine: %w", err)
	}

	return &SRSEngine{
		resonanceEngine: resonanceEngine,
		particles:       make([]*EntropyParticle, 0),
		constraints:     make([]Constraint, 0),
		config:          DefaultSRSConfig(),
		telemetryPoints: make([]types.TelemetryPoint, 0),
	}, nil
}

// DefaultSRSConfig returns default SRS configuration
func DefaultSRSConfig() *SRSConfig {
	return &SRSConfig{
		ParticleCount:     50,
		MaxIterations:     5000,
		PlateauThreshold:  1e-6,
		EntropyLambda:     0.02,
		ResonanceStrength: 0.8,
		InertiaWeight:     0.9,
		CognitiveFactor:   2.0,
		SocialFactor:      2.0,
		QuantumFactor:     0.5,
		TimeoutSeconds:    300,
	}
}

// SolveProblem solves an NP-complete problem using symbolic resonance
func (srs *SRSEngine) SolveProblem(problemType string, spec map[string]interface{}, config *SRSConfig) (*Solution, []types.TelemetryPoint, error) {
	srs.mu.Lock()
	defer srs.mu.Unlock()

	if config != nil {
		srs.config = config
	}

	srs.startTime = time.Now()
	srs.currentIteration = 0
	srs.bestSolution = nil
	srs.telemetryPoints = make([]types.TelemetryPoint, 0)

	// DEBUG: Log start of problem solving
	fmt.Printf("[SRS] Starting %s problem solving with max_iterations=%d, particles=%d\n",
		problemType, srs.config.MaxIterations, srs.config.ParticleCount)

	// Parse problem specification
	if err := srs.parseProblemSpec(problemType, spec); err != nil {
		return nil, nil, fmt.Errorf("failed to parse problem: %w", err)
	}

	// Initialize entropy particles
	if err := srs.initializeParticles(); err != nil {
		return nil, nil, fmt.Errorf("failed to initialize particles: %w", err)
	}

	// Run symbolic resonance evolution
	solution, err := srs.evolveParticles()
	if err != nil {
		return nil, nil, fmt.Errorf("evolution failed: %w", err)
	}

	return solution, srs.telemetryPoints, nil
}

// parseProblemSpec parses the problem specification and sets up constraints
func (srs *SRSEngine) parseProblemSpec(problemType string, spec map[string]interface{}) error {
	switch problemType {
	case "3sat":
		return srs.parse3SAT(spec)
	case "ksat":
		return srs.parseKSAT(spec)
	case "subsetsum":
		return srs.parseSubsetSum(spec)
	default:
		return fmt.Errorf("unsupported problem type: %s", problemType)
	}
}

// parse3SAT parses a 3-SAT problem specification
func (srs *SRSEngine) parse3SAT(spec map[string]interface{}) error {
	var variables float64
	switch v := spec["variables"].(type) {
	case float64:
		variables = v
	case int:
		variables = float64(v)
	default:
		return fmt.Errorf("variables field missing or invalid")
	}

	clausesInterface, ok := spec["clauses"]
	if !ok {
		return fmt.Errorf("clauses field missing")
	}

	clauses, ok := clausesInterface.([]interface{})
	if !ok {
		return fmt.Errorf("clauses field invalid format")
	}

	// Create problem space
	srs.problemSpace = &ProblemSpace{
		Dimensions:  int(variables),
		Variables:   int(variables),
		Bounds:      make([][2]float64, int(variables)),
		ProblemType: "3sat",
		Metadata: map[string]interface{}{
			"clause_count": len(clauses),
		},
	}

	// Set variable bounds [0, 1] for boolean variables
	for i := 0; i < int(variables); i++ {
		srs.problemSpace.Bounds[i] = [2]float64{0, 1}
	}

	// Parse clauses
	srs.constraints = make([]Constraint, 0, len(clauses))

	for i, clauseInterface := range clauses {
		clauseArray, ok := clauseInterface.([]interface{})
		if !ok {
			return fmt.Errorf("clause %d has invalid format", i)
		}

		literals := make([]Literal, len(clauseArray))
		for j, literalInterface := range clauseArray {
			literalMap, ok := literalInterface.(map[string]interface{})
			if !ok {
				return fmt.Errorf("literal %d in clause %d has invalid format", j, i)
			}

			var variable float64
			switch v := literalMap["var"].(type) {
			case float64:
				variable = v
			case int:
				variable = float64(v)
			default:
				return fmt.Errorf("variable field missing in literal %d of clause %d", j, i)
			}

			negated, ok := literalMap["neg"].(bool)
			if !ok {
				negated = false
			}

			literals[j] = Literal{
				Variable: int(variable) - 1, // Convert to 0-indexed
				Negated:  negated,
			}
		}

		clause := &SATClause{
			Literals: literals,
			Weight:   1.0,
		}

		srs.constraints = append(srs.constraints, clause)
	}

	return nil
}

// parseKSAT parses a general k-SAT problem
func (srs *SRSEngine) parseKSAT(spec map[string]interface{}) error {
	// Similar to 3-SAT but handles variable clause lengths
	return srs.parse3SAT(spec) // For now, use same parsing logic
}

// parseSubsetSum parses a subset sum problem
func (srs *SRSEngine) parseSubsetSum(spec map[string]interface{}) error {
	// Extract numbers array
	numbersInterface, ok := spec["numbers"]
	if !ok {
		return fmt.Errorf("numbers field missing")
	}

	numbersArray, ok := numbersInterface.([]interface{})
	if !ok {
		return fmt.Errorf("numbers field invalid format")
	}

	// Extract target sum
	targetInterface, ok := spec["target"]
	if !ok {
		return fmt.Errorf("target field missing")
	}

	var target float64
	switch v := targetInterface.(type) {
	case float64:
		target = v
	case int:
		target = float64(v)
	default:
		return fmt.Errorf("target field invalid format")
	}

	// Convert numbers to float64
	numbers := make([]float64, len(numbersArray))
	for i, numInterface := range numbersArray {
		switch v := numInterface.(type) {
		case float64:
			numbers[i] = v
		case int:
			numbers[i] = float64(v)
		default:
			return fmt.Errorf("number at index %d has invalid format", i)
		}
	}

	// Create problem space
	srs.problemSpace = &ProblemSpace{
		Dimensions:  len(numbers),
		Variables:   len(numbers),
		Bounds:      make([][2]float64, len(numbers)),
		ProblemType: "subsetsum",
		Metadata: map[string]interface{}{
			"target_sum":   target,
			"number_count": len(numbers),
		},
	}

	// Set variable bounds [0, 1] for binary selection (include/exclude)
	for i := 0; i < len(numbers); i++ {
		srs.problemSpace.Bounds[i] = [2]float64{0, 1}
	}

	// Create subset sum constraint
	constraint := &SubsetSumConstraint{
		Numbers: numbers,
		Target:  target,
		Weight:  1.0,
	}

	srs.constraints = []Constraint{constraint}

	return nil
}

// initializeParticles creates and initializes entropy particles
func (srs *SRSEngine) initializeParticles() error {
	srs.particles = make([]*EntropyParticle, srs.config.ParticleCount)

	for i := 0; i < srs.config.ParticleCount; i++ {
		particle, err := srs.createParticle(i)
		if err != nil {
			return fmt.Errorf("failed to create particle %d: %w", i, err)
		}
		srs.particles[i] = particle
	}

	return nil
}

// createParticle creates a single entropy particle
func (srs *SRSEngine) createParticle(id int) (*EntropyParticle, error) {
	dimensions := srs.problemSpace.Dimensions

	// Initialize position randomly within bounds
	position := make([]float64, dimensions)
	velocity := make([]float64, dimensions)
	assignment := make([]int, dimensions)

	for d := 0; d < dimensions; d++ {
		bounds := srs.problemSpace.Bounds[d]
		position[d] = bounds[0] + rand.Float64()*(bounds[1]-bounds[0])
		velocity[d] = (rand.Float64() - 0.5) * 0.1 // Small initial velocity

		// Convert continuous position to discrete assignment for boolean variables
		if srs.problemSpace.ProblemType == "3sat" || srs.problemSpace.ProblemType == "ksat" {
			assignment[d] = 0
			if position[d] > 0.5 {
				assignment[d] = 1
			}
		} else {
			assignment[d] = int(math.Round(position[d]))
		}
	}

	// Create associated quantum state
	amplitudes := srs.positionToAmplitudes(position)
	quantumState, err := srs.resonanceEngine.CreateQuantumState(amplitudes)
	if err != nil {
		return nil, fmt.Errorf("failed to create quantum state: %w", err)
	}

	// Calculate initial metrics
	satisfied := srs.evaluateAssignment(assignment)
	energy := srs.calculateParticleEnergy(position, assignment)
	entropy := quantumState.Entropy

	particle := &EntropyParticle{
		ID:           id,
		Position:     position,
		Velocity:     velocity,
		Mass:         1.0 + rand.Float64()*0.2, // Slight mass variation
		Energy:       energy,
		Entropy:      entropy,
		Assignment:   assignment,
		Constraints:  srs.getAffectingConstraints(assignment),
		Satisfied:    satisfied,
		Resonance:    0.0,
		QuantumState: quantumState,
	}

	return particle, nil
}

// positionToAmplitudes converts particle position to quantum state amplitudes
func (srs *SRSEngine) positionToAmplitudes(position []float64) []complex128 {
	dimension := srs.resonanceEngine.GetDimension()
	amplitudes := make([]complex128, dimension)

	// Map position vector to prime basis amplitudes
	// Use normalized position components with quantum phase
	normFactor := 0.0
	for _, pos := range position {
		normFactor += pos * pos
	}
	normFactor = math.Sqrt(normFactor)

	if normFactor == 0 {
		normFactor = 1.0
	}

	for i := 0; i < dimension && i < len(position); i++ {
		normalizedPos := position[i] / normFactor

		// Add quantum phase based on position
		phase := 2.0 * math.Pi * normalizedPos
		amplitudes[i] = complex(normalizedPos*math.Cos(phase), normalizedPos*math.Sin(phase))
	}

	// Fill remaining amplitudes with small random values
	for i := len(position); i < dimension; i++ {
		amplitudes[i] = complex(0.01*rand.Float64(), 0.01*rand.Float64())
	}

	return amplitudes
}

// evaluateAssignment counts satisfied constraints for an assignment
func (srs *SRSEngine) evaluateAssignment(assignment []int) int {
	satisfied := 0
	for _, constraint := range srs.constraints {
		if constraint.Evaluate(assignment) {
			satisfied++
		}
	}
	return satisfied
}

// calculateParticleEnergy computes the energy of a particle
func (srs *SRSEngine) calculateParticleEnergy(position []float64, assignment []int) float64 {
	// Energy is based on constraint violations and quantum energy
	satisfied := srs.evaluateAssignment(assignment)
	violationEnergy := float64(len(srs.constraints) - satisfied)

	// Add kinetic energy component
	kineticEnergy := 0.0
	for _, pos := range position {
		kineticEnergy += pos * pos
	}

	return violationEnergy + 0.1*kineticEnergy
}

// getAffectingConstraints returns indices of constraints that affect given assignment
func (srs *SRSEngine) getAffectingConstraints(assignment []int) []int {
	var affecting []int
	for i, constraint := range srs.constraints {
		variables := constraint.GetVariables()
		for _, v := range variables {
			if v < len(assignment) {
				affecting = append(affecting, i)
				break
			}
		}
	}
	return affecting
}

// evolveParticles runs the main evolution loop
func (srs *SRSEngine) evolveParticles() (*Solution, error) {
	timeout := time.Duration(srs.config.TimeoutSeconds) * time.Second

	fmt.Printf("[SRS] Starting evolution loop: max_iterations=%d, timeout=%v\n",
		srs.config.MaxIterations, timeout)

	for srs.currentIteration = 0; srs.currentIteration < srs.config.MaxIterations; srs.currentIteration++ {
		iterStart := time.Now()

		// Check timeout
		if time.Since(srs.startTime) > timeout {
			fmt.Printf("[SRS] Timeout reached at iteration %d after %v\n",
				srs.currentIteration, time.Since(srs.startTime))
			break
		}

		// Update particles
		if err := srs.updateParticles(); err != nil {
			return nil, fmt.Errorf("failed to update particles at iteration %d: %w", srs.currentIteration, err)
		}

		// Record telemetry
		srs.recordTelemetry()

		// Log progress every 10 iterations
		if srs.currentIteration%10 == 0 {
			iterTime := time.Since(iterStart)
			elapsed := time.Since(srs.startTime)
			fmt.Printf("[SRS] Iteration %d: took %v, total elapsed %v\n",
				srs.currentIteration, iterTime, elapsed)
		}

		// Check for solution
		if solution := srs.checkForSolution(); solution != nil {
			fmt.Printf("[SRS] Solution found at iteration %d after %v\n",
				srs.currentIteration, time.Since(srs.startTime))
			solution.ComputeTime = time.Since(srs.startTime).Seconds()
			return solution, nil
		}

		// Check for convergence
		if srs.checkConvergence() {
			fmt.Printf("[SRS] Convergence detected at iteration %d after %v\n",
				srs.currentIteration, time.Since(srs.startTime))
			break
		}
	}

	// Return best solution found
	if srs.bestSolution != nil {
		srs.bestSolution.ComputeTime = time.Since(srs.startTime).Seconds()
		return srs.bestSolution, nil
	}

	// No solution found
	return &Solution{
		Assignment:  make([]int, srs.problemSpace.Variables),
		Feasible:    false,
		Objective:   math.Inf(1),
		Satisfied:   0,
		Total:       len(srs.constraints),
		ComputeTime: time.Since(srs.startTime).Seconds(),
	}, nil
}

// updateParticles updates all particles using symbolic resonance dynamics
func (srs *SRSEngine) updateParticles() error {
	// Apply quantum evolution to all particles
	for _, particle := range srs.particles {
		if err := srs.updateParticle(particle); err != nil {
			return fmt.Errorf("failed to update particle %d: %w", particle.ID, err)
		}
	}

	// Apply inter-particle resonance
	srs.applyInterParticleResonance()

	return nil
}

// updateParticle updates a single particle
func (srs *SRSEngine) updateParticle(particle *EntropyParticle) error {
	// Apply quantum evolution
	dt := 0.01
	evolvedState, err := srs.resonanceEngine.EvolveStateWithResonance(
		particle.QuantumState, dt, srs.config.QuantumFactor)
	if err != nil {
		return fmt.Errorf("quantum evolution failed: %w", err)
	}
	particle.QuantumState = evolvedState

	// Update particle metrics
	srs.resonanceEngine.RecordEntropyPoint(particle.QuantumState, srs.currentIteration)

	// Extract new position from quantum state
	newPosition := srs.amplitudesToPosition(particle.QuantumState.Amplitudes)

	// Calculate velocity update using PSO-style dynamics
	srs.updateParticleVelocity(particle, newPosition)

	// Update position
	for i := range particle.Position {
		particle.Position[i] += particle.Velocity[i]

		// Apply bounds
		bounds := srs.problemSpace.Bounds[i]
		if particle.Position[i] < bounds[0] {
			particle.Position[i] = bounds[0]
			particle.Velocity[i] = 0
		} else if particle.Position[i] > bounds[1] {
			particle.Position[i] = bounds[1]
			particle.Velocity[i] = 0
		}
	}

	// Update assignment
	srs.updateParticleAssignment(particle)

	// Update metrics
	particle.Satisfied = srs.evaluateAssignment(particle.Assignment)
	particle.Energy = srs.calculateParticleEnergy(particle.Position, particle.Assignment)
	particle.Entropy = particle.QuantumState.Entropy

	return nil
}

// amplitudesToPosition converts quantum amplitudes back to position
func (srs *SRSEngine) amplitudesToPosition(amplitudes []complex128) []float64 {
	position := make([]float64, srs.problemSpace.Dimensions)

	for i := 0; i < len(position) && i < len(amplitudes); i++ {
		// Extract position from amplitude magnitude and phase
		magnitude := real(amplitudes[i] * complex(real(amplitudes[i]), -imag(amplitudes[i])))
		phase := math.Atan2(imag(amplitudes[i]), real(amplitudes[i]))

		// Map to position space
		position[i] = magnitude * (1.0 + 0.1*math.Cos(phase))
	}

	return position
}

// updateParticleVelocity updates particle velocity using PSO dynamics
func (srs *SRSEngine) updateParticleVelocity(particle *EntropyParticle, quantumPosition []float64) {
	for i := range particle.Velocity {
		// Inertia term
		inertia := srs.config.InertiaWeight * particle.Velocity[i]

		// Cognitive term (attraction to quantum position)
		cognitive := srs.config.CognitiveFactor * rand.Float64() *
			(quantumPosition[i] - particle.Position[i])

		// Social term (attraction to best known solution)
		social := 0.0
		if srs.bestSolution != nil && i < len(srs.bestSolution.Assignment) {
			targetPos := float64(srs.bestSolution.Assignment[i])
			social = srs.config.SocialFactor * rand.Float64() *
				(targetPos - particle.Position[i])
		}

		// Update velocity
		particle.Velocity[i] = inertia + cognitive + social

		// Velocity clamping
		maxVel := 0.1
		if particle.Velocity[i] > maxVel {
			particle.Velocity[i] = maxVel
		} else if particle.Velocity[i] < -maxVel {
			particle.Velocity[i] = -maxVel
		}
	}
}

// updateParticleAssignment updates discrete assignment from continuous position
func (srs *SRSEngine) updateParticleAssignment(particle *EntropyParticle) {
	for i := range particle.Assignment {
		if srs.problemSpace.ProblemType == "3sat" || srs.problemSpace.ProblemType == "ksat" {
			// Boolean assignment
			particle.Assignment[i] = 0
			if particle.Position[i] > 0.5 {
				particle.Assignment[i] = 1
			}
		} else {
			// Integer assignment
			particle.Assignment[i] = int(math.Round(particle.Position[i]))
		}
	}
}

// applyInterParticleResonance applies resonance between particles
func (srs *SRSEngine) applyInterParticleResonance() {
	for i, particle1 := range srs.particles {
		for j, particle2 := range srs.particles {
			if i >= j {
				continue
			}

			// Calculate resonance between particles
			resonance := srs.calculateParticleResonance(particle1, particle2)

			// Apply attractive force if particles are in resonance
			if resonance > 0.7 {
				srs.applyAttractiveForcex(particle1, particle2, resonance)
			}
		}
	}
}

// calculateParticleResonance calculates resonance between two particles
func (srs *SRSEngine) calculateParticleResonance(p1, p2 *EntropyParticle) float64 {
	// Calculate quantum state overlap
	overlap, err := srs.resonanceEngine.GetHilbertSpace().ComputeInnerProduct(
		p1.QuantumState, p2.QuantumState)
	if err != nil {
		return 0.0
	}

	// Calculate assignment similarity
	agreementCount := 0
	for i := range p1.Assignment {
		if i < len(p2.Assignment) && p1.Assignment[i] == p2.Assignment[i] {
			agreementCount++
		}
	}

	assignmentSimilarity := float64(agreementCount) / float64(len(p1.Assignment))
	quantumOverlap := real(overlap * complex(real(overlap), -imag(overlap)))

	return 0.7*quantumOverlap + 0.3*assignmentSimilarity
}

// applyAttractiveForcex applies attractive force between resonant particles
func (srs *SRSEngine) applyAttractiveForcex(p1, p2 *EntropyParticle, resonance float64) {
	force := resonance * 0.01

	for i := range p1.Velocity {
		if i < len(p2.Position) {
			direction := p2.Position[i] - p1.Position[i]
			p1.Velocity[i] += force * direction / p1.Mass
			p2.Velocity[i] -= force * direction / p2.Mass
		}
	}
}

// recordTelemetry records current state for telemetry
func (srs *SRSEngine) recordTelemetry() {
	if len(srs.particles) == 0 {
		return
	}

	// Calculate average metrics
	avgEntropy := 0.0
	avgSatisfaction := 0.0
	maxSatisfied := 0
	avgResonance := 0.0

	for _, particle := range srs.particles {
		avgEntropy += particle.Entropy
		avgSatisfaction += float64(particle.Satisfied) / float64(len(srs.constraints))
		if particle.Satisfied > maxSatisfied {
			maxSatisfied = particle.Satisfied
		}
		avgResonance += particle.Resonance
	}

	count := float64(len(srs.particles))
	avgEntropy /= count
	avgSatisfaction /= count
	avgResonance /= count

	// Calculate Lyapunov metric (stability)
	lyapunov := srs.calculateLyapunovMetric()

	point := types.TelemetryPoint{
		Step:              srs.currentIteration,
		SymbolicEntropy:   avgEntropy,
		LyapunovMetric:    lyapunov,
		SatisfactionRate:  avgSatisfaction,
		ResonanceStrength: avgResonance,
		Dominance:         float64(maxSatisfied) / float64(len(srs.constraints)),
		Timestamp:         time.Now(),
	}

	srs.telemetryPoints = append(srs.telemetryPoints, point)
}

// calculateLyapunovMetric calculates system stability metric
func (srs *SRSEngine) calculateLyapunovMetric() float64 {
	if len(srs.particles) < 2 {
		return 0.0
	}

	// Calculate average distance between particles in solution space
	totalDistance := 0.0
	pairCount := 0

	for i, p1 := range srs.particles {
		for j, p2 := range srs.particles {
			if i >= j {
				continue
			}

			distance := 0.0
			for k := range p1.Assignment {
				if k < len(p2.Assignment) {
					diff := float64(p1.Assignment[k] - p2.Assignment[k])
					distance += diff * diff
				}
			}

			totalDistance += math.Sqrt(distance)
			pairCount++
		}
	}

	if pairCount == 0 {
		return 0.0
	}

	avgDistance := totalDistance / float64(pairCount)

	// Lyapunov metric: inverse of average distance (higher = more stable)
	// Add small epsilon to avoid division by zero
	lyapunov := 1.0 / (avgDistance + 1e-6)

	// Normalize to [0, 1] range
	return math.Min(lyapunov, 1.0)
}

// checkForSolution checks if a complete solution has been found
func (srs *SRSEngine) checkForSolution() *Solution {
	for _, particle := range srs.particles {
		if particle.Satisfied == len(srs.constraints) {
			// Found a complete solution
			solution := &Solution{
				Assignment: make([]int, len(particle.Assignment)),
				Feasible:   true,
				Objective:  0.0,
				Satisfied:  particle.Satisfied,
				Total:      len(srs.constraints),
				Energy:     particle.Energy,
				Entropy:    particle.Entropy,
				Confidence: 1.0,
				FoundAt:    srs.currentIteration,
			}

			copy(solution.Assignment, particle.Assignment)

			// Update best solution
			srs.bestSolution = solution
			return solution
		}

		// Track best partial solution
		if srs.bestSolution == nil || particle.Satisfied > srs.bestSolution.Satisfied {
			srs.bestSolution = &Solution{
				Assignment: make([]int, len(particle.Assignment)),
				Feasible:   particle.Satisfied == len(srs.constraints),
				Objective:  float64(len(srs.constraints) - particle.Satisfied),
				Satisfied:  particle.Satisfied,
				Total:      len(srs.constraints),
				Energy:     particle.Energy,
				Entropy:    particle.Entropy,
				Confidence: float64(particle.Satisfied) / float64(len(srs.constraints)),
				FoundAt:    srs.currentIteration,
			}

			copy(srs.bestSolution.Assignment, particle.Assignment)
		}
	}

	return nil
}

// checkConvergence checks if the system has converged
func (srs *SRSEngine) checkConvergence() bool {
	// Need at least 100 iterations of telemetry for meaningful convergence check
	if len(srs.telemetryPoints) < 100 {
		return false
	}

	// Check if entropy has plateaued (indicating convergence)
	recentPoints := srs.telemetryPoints[len(srs.telemetryPoints)-50:]

	entropyVariance := 0.0
	satisfactionVariance := 0.0

	// Calculate variance in recent entropy and satisfaction
	avgEntropy := 0.0
	avgSatisfaction := 0.0

	for _, point := range recentPoints {
		avgEntropy += point.SymbolicEntropy
		avgSatisfaction += point.SatisfactionRate
	}

	avgEntropy /= float64(len(recentPoints))
	avgSatisfaction /= float64(len(recentPoints))

	for _, point := range recentPoints {
		entropyDiff := point.SymbolicEntropy - avgEntropy
		satisfactionDiff := point.SatisfactionRate - avgSatisfaction
		entropyVariance += entropyDiff * entropyDiff
		satisfactionVariance += satisfactionDiff * satisfactionDiff
	}

	entropyVariance /= float64(len(recentPoints))
	satisfactionVariance /= float64(len(recentPoints))

	// System has converged if both entropy and satisfaction have low variance
	entropyConverged := entropyVariance < srs.config.PlateauThreshold
	satisfactionConverged := satisfactionVariance < srs.config.PlateauThreshold*0.1

	return entropyConverged && satisfactionConverged
}

// Evaluate implements the Constraint interface for SATClause
func (c *SATClause) Evaluate(assignment []int) bool {
	for _, literal := range c.Literals {
		if literal.Variable >= len(assignment) {
			continue // Skip invalid variables
		}

		value := assignment[literal.Variable]

		// Check if literal is satisfied
		if (!literal.Negated && value == 1) || (literal.Negated && value == 0) {
			return true // Clause is satisfied if any literal is true
		}
	}

	return false // All literals are false
}

// GetVariables implements the Constraint interface for SATClause
func (c *SATClause) GetVariables() []int {
	variables := make([]int, len(c.Literals))
	for i, literal := range c.Literals {
		variables[i] = literal.Variable
	}
	return variables
}

// GetWeight implements the Constraint interface for SATClause
func (c *SATClause) GetWeight() float64 {
	return c.Weight
}

// GetType implements the Constraint interface for SATClause
func (c *SATClause) GetType() string {
	return "sat_clause"
}

// Evaluate implements the Constraint interface for SubsetSumConstraint
func (ssc *SubsetSumConstraint) Evaluate(assignment []int) bool {
	sum := 0.0
	for i, include := range assignment {
		if include == 1 && i < len(ssc.Numbers) {
			sum += ssc.Numbers[i]
		}
	}

	// Allow small tolerance for floating point comparison
	tolerance := 1e-6
	return math.Abs(sum-ssc.Target) < tolerance
}

// GetVariables implements the Constraint interface for SubsetSumConstraint
func (ssc *SubsetSumConstraint) GetVariables() []int {
	variables := make([]int, len(ssc.Numbers))
	for i := range variables {
		variables[i] = i
	}
	return variables
}

// GetWeight implements the Constraint interface for SubsetSumConstraint
func (ssc *SubsetSumConstraint) GetWeight() float64 {
	return ssc.Weight
}

// GetType implements the Constraint interface for SubsetSumConstraint
func (ssc *SubsetSumConstraint) GetType() string {
	return "subset_sum"
}

// GetTelemetry returns current telemetry data
func (srs *SRSEngine) GetTelemetry() []types.TelemetryPoint {
	srs.mu.RLock()
	defer srs.mu.RUnlock()

	// Return a copy to avoid race conditions
	telemetry := make([]types.TelemetryPoint, len(srs.telemetryPoints))
	copy(telemetry, srs.telemetryPoints)

	return telemetry
}

// GetCurrentState returns the current state of the SRS engine
func (srs *SRSEngine) GetCurrentState() map[string]interface{} {
	srs.mu.RLock()
	defer srs.mu.RUnlock()

	state := map[string]interface{}{
		"iteration":        srs.currentIteration,
		"particle_count":   len(srs.particles),
		"constraint_count": len(srs.constraints),
		"elapsed_time":     time.Since(srs.startTime).Seconds(),
	}

	if srs.bestSolution != nil {
		state["best_solution"] = map[string]interface{}{
			"satisfied":  srs.bestSolution.Satisfied,
			"total":      srs.bestSolution.Total,
			"feasible":   srs.bestSolution.Feasible,
			"confidence": srs.bestSolution.Confidence,
		}
	}

	if srs.problemSpace != nil {
		state["problem_space"] = map[string]interface{}{
			"dimensions":   srs.problemSpace.Dimensions,
			"variables":    srs.problemSpace.Variables,
			"problem_type": srs.problemSpace.ProblemType,
		}
	}

	// Add particle statistics
	if len(srs.particles) > 0 {
		avgEntropy := 0.0
		avgSatisfied := 0.0
		avgEnergy := 0.0

		for _, particle := range srs.particles {
			avgEntropy += particle.Entropy
			avgSatisfied += float64(particle.Satisfied)
			avgEnergy += particle.Energy
		}

		count := float64(len(srs.particles))
		state["particle_stats"] = map[string]interface{}{
			"avg_entropy":   avgEntropy / count,
			"avg_satisfied": avgSatisfied / count,
			"avg_energy":    avgEnergy / count,
		}
	}

	return state
}

// Reset resets the SRS engine to initial state
func (srs *SRSEngine) Reset() {
	srs.mu.Lock()
	defer srs.mu.Unlock()

	srs.particles = make([]*EntropyParticle, 0)
	srs.constraints = make([]Constraint, 0)
	srs.problemSpace = nil
	srs.currentIteration = 0
	srs.bestSolution = nil
	srs.telemetryPoints = make([]types.TelemetryPoint, 0)
}

// SetConfig updates the SRS configuration
func (srs *SRSEngine) SetConfig(config *SRSConfig) {
	srs.mu.Lock()
	defer srs.mu.Unlock()

	if config != nil {
		srs.config = config
	}
}

// GetConfig returns the current SRS configuration
func (srs *SRSEngine) GetConfig() *SRSConfig {
	srs.mu.RLock()
	defer srs.mu.RUnlock()

	// Return a copy to avoid external modification
	configCopy := *srs.config
	return &configCopy
}
