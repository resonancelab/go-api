package resolang

import (
	"fmt"
	"math"
	"sync"
	"time"

	"github.com/resonancelab/psizero/core"
	"github.com/resonancelab/psizero/core/hilbert"
	"github.com/resonancelab/psizero/shared/types"
)

// Runtime represents the ResoLang distributed execution runtime
type Runtime struct {
	resonanceEngine *core.ResonanceEngine
	programs        map[string]*Program
	nodes           map[string]*Node
	channels        map[string]*QuantumChannel
	config          *RuntimeConfig
	mu              sync.RWMutex

	// Execution state
	currentEpoch    int
	startTime       time.Time
	telemetryPoints []types.TelemetryPoint
	executionStats  map[string]interface{}
}

// Program represents a compiled ResoLang program
type Program struct {
	ID           string                 `json:"id"`
	Code         string                 `json:"code"`
	AST          *ASTNode               `json:"-"`
	QuantumState *hilbert.QuantumState  `json:"-"`
	Variables    map[string]interface{} `json:"variables"`
	Functions    map[string]*Function   `json:"-"`
	Status       ExecutionStatus        `json:"status"`
	Created      time.Time              `json:"created"`
}

// Node represents a computational node in the distributed network
type Node struct {
	ID            string                `json:"id"`
	Address       string                `json:"address"`
	QuantumState  *hilbert.QuantumState `json:"-"`
	Programs      map[string]*Program   `json:"programs"`
	Resonance     float64               `json:"resonance"`
	Load          float64               `json:"load"`
	LastHeartbeat time.Time             `json:"last_heartbeat"`
	Capabilities  []string              `json:"capabilities"`
}

// QuantumChannel represents a quantum communication channel between nodes
type QuantumChannel struct {
	ID        string                `json:"id"`
	FromNode  string                `json:"from_node"`
	ToNode    string                `json:"to_node"`
	Capacity  float64               `json:"capacity"`
	Coherence float64               `json:"coherence"`
	Latency   time.Duration         `json:"latency"`
	State     *hilbert.QuantumState `json:"-"`
}

// RuntimeConfig contains runtime configuration
type RuntimeConfig struct {
	MaxPrograms         int           `json:"max_programs"`
	MaxNodes            int           `json:"max_nodes"`
	HeartbeatInterval   time.Duration `json:"heartbeat_interval"`
	ExecutionTimeout    time.Duration `json:"execution_timeout"`
	QuantumSyncInterval time.Duration `json:"quantum_sync_interval"`
	ResonanceThreshold  float64       `json:"resonance_threshold"`
	LoadBalancing       bool          `json:"load_balancing"`
}

// ExecutionStatus represents the status of program execution
type ExecutionStatus string

const (
	StatusPending   ExecutionStatus = "pending"
	StatusRunning   ExecutionStatus = "running"
	StatusCompleted ExecutionStatus = "completed"
	StatusFailed    ExecutionStatus = "failed"
	StatusSuspended ExecutionStatus = "suspended"
)

// Execution status constants for convenience
const (
	ExecutionRunning   ExecutionStatus = StatusRunning
	ExecutionCompleted ExecutionStatus = StatusCompleted
	ExecutionFailed    ExecutionStatus = StatusFailed
)

// ASTNode represents a node in the abstract syntax tree
type ASTNode struct {
	Type       string                 `json:"type"`
	Value      interface{}            `json:"value"`
	Children   []*ASTNode             `json:"children"`
	Line       int                    `json:"line"`
	Column     int                    `json:"column"`
	Attributes map[string]interface{} `json:"attributes"`
}

// Function represents a ResoLang function
type Function struct {
	Name       string   `json:"name"`
	Parameters []string `json:"parameters"`
	Body       *ASTNode `json:"body"`
	ReturnType string   `json:"return_type"`
	IsQuantum  bool     `json:"is_quantum"`
}

// NewRuntime creates a new ResoLang runtime
func NewRuntime() (*Runtime, error) {
	// Initialize core resonance engine
	config := core.DefaultEngineConfig()
	config.Dimension = 500 // Large space for complex programs

	resonanceEngine, err := core.NewResonanceEngine(config)
	if err != nil {
		return nil, fmt.Errorf("failed to create resonance engine: %w", err)
	}

	return &Runtime{
		resonanceEngine: resonanceEngine,
		programs:        make(map[string]*Program),
		nodes:           make(map[string]*Node),
		channels:        make(map[string]*QuantumChannel),
		config:          DefaultRuntimeConfig(),
		telemetryPoints: make([]types.TelemetryPoint, 0),
		executionStats:  make(map[string]interface{}),
	}, nil
}

// DefaultRuntimeConfig returns default runtime configuration
func DefaultRuntimeConfig() *RuntimeConfig {
	return &RuntimeConfig{
		MaxPrograms:         1000,
		MaxNodes:            100,
		HeartbeatInterval:   30 * time.Second,
		ExecutionTimeout:    300 * time.Second,
		QuantumSyncInterval: 10 * time.Second,
		ResonanceThreshold:  0.7,
		LoadBalancing:       true,
	}
}

// ExecuteProgram executes a ResoLang program in distributed mode
func (rt *Runtime) ExecuteProgram(programID string, input map[string]interface{}) (*ExecutionResult, error) {
	rt.mu.Lock()
	defer rt.mu.Unlock()

	program, exists := rt.programs[programID]
	if !exists {
		return nil, fmt.Errorf("program not found: %s", programID)
	}

	rt.startTime = time.Now()
	rt.currentEpoch = 0

	// Initialize execution context
	context := &ExecutionContext{
		Program:     program,
		Input:       input,
		Variables:   make(map[string]interface{}),
		QuantumVars: make(map[string]*hilbert.QuantumState),
		NodeStates:  make(map[string]*NodeState),
	}

	// Distribute execution across nodes
	result, err := rt.distributeExecution(context)
	if err != nil {
		return nil, fmt.Errorf("distributed execution failed: %w", err)
	}

	return result, nil
}

// distributeExecution distributes program execution across available nodes
func (rt *Runtime) distributeExecution(context *ExecutionContext) (*ExecutionResult, error) {
	// Find suitable nodes for execution
	suitableNodes := rt.findSuitableNodes(context)

	if len(suitableNodes) == 0 {
		return nil, fmt.Errorf("no suitable nodes found for execution")
	}

	// Create execution plan
	plan := rt.createExecutionPlan(context, suitableNodes)

	// Execute plan across nodes
	result := rt.executePlan(plan)

	return result, nil
}

// findSuitableNodes finds nodes suitable for executing the given program
func (rt *Runtime) findSuitableNodes(context *ExecutionContext) []*Node {
	var suitableNodes []*Node

	for _, node := range rt.nodes {
		if rt.isNodeSuitable(node, context) {
			suitableNodes = append(suitableNodes, node)
		}
	}

	// Sort by resonance and load
	for i := 0; i < len(suitableNodes)-1; i++ {
		for j := i + 1; j < len(suitableNodes); j++ {
			if rt.nodeScore(suitableNodes[j]) > rt.nodeScore(suitableNodes[i]) {
				suitableNodes[i], suitableNodes[j] = suitableNodes[j], suitableNodes[i]
			}
		}
	}

	return suitableNodes
}

// isNodeSuitable checks if a node is suitable for executing a program
func (rt *Runtime) isNodeSuitable(node *Node, context *ExecutionContext) bool {
	// Check if node is alive
	if time.Since(node.LastHeartbeat) > 2*rt.config.HeartbeatInterval {
		return false
	}

	// Check load
	if node.Load > 0.9 {
		return false
	}

	// Check capabilities
	requiredCapabilities := rt.extractRequiredCapabilities(context)
	for _, required := range requiredCapabilities {
		found := false
		for _, capability := range node.Capabilities {
			if capability == required {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}

	// Check resonance threshold
	if node.Resonance < rt.config.ResonanceThreshold {
		return false
	}

	return true
}

// nodeScore calculates a score for node selection
func (rt *Runtime) nodeScore(node *Node) float64 {
	// Higher resonance and lower load = higher score
	resonanceScore := node.Resonance
	loadScore := 1.0 - node.Load

	// Age penalty for old heartbeats
	age := time.Since(node.LastHeartbeat).Seconds()
	agePenalty := math.Max(0, 1.0-age/60.0) // Penalty after 1 minute

	return (resonanceScore*0.6 + loadScore*0.3) * agePenalty
}

// extractRequiredCapabilities extracts required capabilities from program context
func (rt *Runtime) extractRequiredCapabilities(context *ExecutionContext) []string {
	capabilities := []string{"basic_execution"}

	// Check for quantum operations
	if rt.hasQuantumOperations(context.Program) {
		capabilities = append(capabilities, "quantum_execution")
	}

	// Check for distributed operations
	if rt.hasDistributedOperations(context.Program) {
		capabilities = append(capabilities, "distributed_execution")
	}

	return capabilities
}

// hasQuantumOperations checks if program has quantum operations
func (rt *Runtime) hasQuantumOperations(program *Program) bool {
	// Simple check - in real implementation would analyze AST
	return len(program.Functions) > 0
}

// hasDistributedOperations checks if program has distributed operations
func (rt *Runtime) hasDistributedOperations(program *Program) bool {
	// Simple check - in real implementation would analyze AST
	return program.Code != ""
}

// createExecutionPlan creates an execution plan for distributed execution
func (rt *Runtime) createExecutionPlan(context *ExecutionContext, nodes []*Node) *ExecutionPlan {
	plan := &ExecutionPlan{
		ProgramID: context.Program.ID,
		Nodes:     make([]string, len(nodes)),
		Tasks:     make([]*Task, 0),
		Channels:  make([]string, 0),
		StartTime: time.Now(),
	}

	// Assign nodes
	for i, node := range nodes {
		plan.Nodes[i] = node.ID
	}

	// Create tasks from program AST
	if context.Program.AST != nil {
		plan.Tasks = rt.createTasksFromAST(context.Program.AST, nodes)
	}

	// Create communication channels
	for i := 0; i < len(nodes)-1; i++ {
		channelID := fmt.Sprintf("chan_%s_%s", nodes[i].ID, nodes[i+1].ID)
		plan.Channels = append(plan.Channels, channelID)
	}

	return plan
}

// createTasksFromAST creates execution tasks from AST
func (rt *Runtime) createTasksFromAST(ast *ASTNode, nodes []*Node) []*Task {
	tasks := make([]*Task, 0)

	// Simple task creation - in real implementation would traverse AST
	task := &Task{
		ID:           "main_task",
		Type:         "execute",
		NodeID:       nodes[0].ID,
		ASTNode:      ast,
		Dependencies: make([]string, 0),
		Status:       TaskPending,
	}

	tasks = append(tasks, task)
	return tasks
}

// executePlan executes the execution plan across nodes
func (rt *Runtime) executePlan(plan *ExecutionPlan) *ExecutionResult {
	result := &ExecutionResult{
		ProgramID:   plan.ProgramID,
		Status:      ExecutionRunning,
		StartTime:   plan.StartTime,
		NodeResults: make(map[string]*NodeResult),
		Output:      make(map[string]interface{}),
	}

	// Initialize node results
	for _, nodeID := range plan.Nodes {
		result.NodeResults[nodeID] = &NodeResult{
			NodeID:    nodeID,
			Status:    NodePending,
			StartTime: time.Now(),
		}
	}

	// Execute tasks
	for _, task := range plan.Tasks {
		rt.executeTask(task, result)
	}

	result.Status = ExecutionCompleted
	result.EndTime = time.Now()
	result.Duration = result.EndTime.Sub(result.StartTime)

	return result
}

// executeTask executes a single task on its assigned node
func (rt *Runtime) executeTask(task *Task, result *ExecutionResult) {
	nodeResult := result.NodeResults[task.NodeID]
	nodeResult.Status = NodeRunning

	// Simulate task execution
	time.Sleep(100 * time.Millisecond) // Simulate work

	// Update node result
	nodeResult.Status = NodeCompleted
	nodeResult.EndTime = time.Now()
	nodeResult.Duration = nodeResult.EndTime.Sub(nodeResult.StartTime)

	task.Status = TaskCompleted
}

// ExecutionContext represents the execution context for a program
type ExecutionContext struct {
	Program     *Program                         `json:"program"`
	Input       map[string]interface{}           `json:"input"`
	Variables   map[string]interface{}           `json:"variables"`
	QuantumVars map[string]*hilbert.QuantumState `json:"quantum_vars"`
	NodeStates  map[string]*NodeState            `json:"node_states"`
}

// ExecutionPlan represents a distributed execution plan
type ExecutionPlan struct {
	ProgramID string    `json:"program_id"`
	Nodes     []string  `json:"nodes"`
	Tasks     []*Task   `json:"tasks"`
	Channels  []string  `json:"channels"`
	StartTime time.Time `json:"start_time"`
}

// Task represents a single execution task
type Task struct {
	ID           string     `json:"id"`
	Type         string     `json:"type"`
	NodeID       string     `json:"node_id"`
	ASTNode      *ASTNode   `json:"ast_node"`
	Dependencies []string   `json:"dependencies"`
	Status       TaskStatus `json:"status"`
	StartTime    time.Time  `json:"start_time"`
	EndTime      time.Time  `json:"end_time"`
}

// TaskStatus represents the status of a task
type TaskStatus string

const (
	TaskPending   TaskStatus = "pending"
	TaskRunning   TaskStatus = "running"
	TaskCompleted TaskStatus = "completed"
	TaskFailed    TaskStatus = "failed"
)

// ExecutionResult represents the result of program execution
type ExecutionResult struct {
	ProgramID   string                 `json:"program_id"`
	Status      ExecutionStatus        `json:"status"`
	StartTime   time.Time              `json:"start_time"`
	EndTime     time.Time              `json:"end_time"`
	Duration    time.Duration          `json:"duration"`
	NodeResults map[string]*NodeResult `json:"node_results"`
	Output      map[string]interface{} `json:"output"`
	Error       string                 `json:"error,omitempty"`
}

// NodeResult represents the result from a single node
type NodeResult struct {
	NodeID    string        `json:"node_id"`
	Status    NodeStatus    `json:"status"`
	StartTime time.Time     `json:"start_time"`
	EndTime   time.Time     `json:"end_time"`
	Duration  time.Duration `json:"duration"`
	Output    interface{}   `json:"output,omitempty"`
	Error     string        `json:"error,omitempty"`
}

// NodeStatus represents the status of a node execution
type NodeStatus string

const (
	NodePending   NodeStatus = "pending"
	NodeRunning   NodeStatus = "running"
	NodeCompleted NodeStatus = "completed"
	NodeFailed    NodeStatus = "failed"
)

// NodeState represents the state of a node during execution
type NodeState struct {
	NodeID       string                 `json:"node_id"`
	QuantumState *hilbert.QuantumState  `json:"quantum_state"`
	Variables    map[string]interface{} `json:"variables"`
	LastUpdate   time.Time              `json:"last_update"`
}

// RegisterNode registers a new node in the distributed network
func (rt *Runtime) RegisterNode(nodeID, address string, capabilities []string) error {
	rt.mu.Lock()
	defer rt.mu.Unlock()

	if len(rt.nodes) >= rt.config.MaxNodes {
		return fmt.Errorf("maximum number of nodes reached")
	}

	// Create quantum state for node
	amplitudes := make([]complex128, rt.resonanceEngine.GetDimension())
	for i := range amplitudes {
		amplitudes[i] = complex(0.1, 0.0) // Small initial amplitude
	}

	quantumState, err := rt.resonanceEngine.CreateQuantumState(amplitudes)
	if err != nil {
		return fmt.Errorf("failed to create quantum state for node: %w", err)
	}

	node := &Node{
		ID:            nodeID,
		Address:       address,
		QuantumState:  quantumState,
		Programs:      make(map[string]*Program),
		Resonance:     0.5, // Initial resonance
		Load:          0.0, // Initial load
		LastHeartbeat: time.Now(),
		Capabilities:  capabilities,
	}

	rt.nodes[nodeID] = node
	return nil
}

// UnregisterNode removes a node from the distributed network
func (rt *Runtime) UnregisterNode(nodeID string) error {
	rt.mu.Lock()
	defer rt.mu.Unlock()

	if _, exists := rt.nodes[nodeID]; !exists {
		return fmt.Errorf("node not found: %s", nodeID)
	}

	delete(rt.nodes, nodeID)
	return nil
}

// UpdateNodeHeartbeat updates the heartbeat timestamp for a node
func (rt *Runtime) UpdateNodeHeartbeat(nodeID string) error {
	rt.mu.Lock()
	defer rt.mu.Unlock()

	node, exists := rt.nodes[nodeID]
	if !exists {
		return fmt.Errorf("node not found: %s", nodeID)
	}

	node.LastHeartbeat = time.Now()
	return nil
}

// GetNodeStatus returns the status of all nodes
func (rt *Runtime) GetNodeStatus() map[string]interface{} {
	rt.mu.RLock()
	defer rt.mu.RUnlock()

	status := map[string]interface{}{
		"total_nodes": len(rt.nodes),
		"nodes":       make(map[string]interface{}),
	}

	for nodeID, node := range rt.nodes {
		status["nodes"].(map[string]interface{})[nodeID] = map[string]interface{}{
			"address":        node.Address,
			"resonance":      node.Resonance,
			"load":           node.Load,
			"last_heartbeat": node.LastHeartbeat,
			"capabilities":   node.Capabilities,
			"program_count":  len(node.Programs),
		}
	}

	return status
}

// GetTelemetry returns current telemetry data
func (rt *Runtime) GetTelemetry() []types.TelemetryPoint {
	rt.mu.RLock()
	defer rt.mu.RUnlock()

	telemetry := make([]types.TelemetryPoint, len(rt.telemetryPoints))
	copy(telemetry, rt.telemetryPoints)
	return telemetry
}

// GetExecutionStats returns execution statistics
func (rt *Runtime) GetExecutionStats() map[string]interface{} {
	rt.mu.RLock()
	defer rt.mu.RUnlock()

	stats := make(map[string]interface{})
	for k, v := range rt.executionStats {
		stats[k] = v
	}

	return stats
}

// SyncQuantumStates synchronizes quantum states across nodes
func (rt *Runtime) SyncQuantumStates() error {
	rt.mu.Lock()
	defer rt.mu.Unlock()

	// Synchronize quantum states between connected nodes
	for _, channel := range rt.channels {
		if err := rt.syncChannelStates(channel); err != nil {
			return fmt.Errorf("failed to sync channel %s: %w", channel.ID, err)
		}
	}

	return nil
}

// syncChannelStates synchronizes quantum states through a channel
func (rt *Runtime) syncChannelStates(channel *QuantumChannel) error {
	fromNode, fromExists := rt.nodes[channel.FromNode]
	toNode, toExists := rt.nodes[channel.ToNode]

	if !fromExists || !toExists {
		return fmt.Errorf("channel nodes not found")
	}

	// Simple state synchronization - in real implementation would use quantum entanglement
	// Average the quantum states
	fromAmplitudes := fromNode.QuantumState.Amplitudes
	toAmplitudes := toNode.QuantumState.Amplitudes

	if len(fromAmplitudes) == len(toAmplitudes) {
		for i := range fromAmplitudes {
			// Weighted average based on coherence
			fromWeight := fromNode.Resonance
			toWeight := toNode.Resonance
			totalWeight := fromWeight + toWeight

			if totalWeight > 0 {
				fromNode.QuantumState.Amplitudes[i] =
					(fromAmplitudes[i]*complex(fromWeight, 0) +
						toAmplitudes[i]*complex(toWeight, 0)) / complex(totalWeight, 0)
			}
		}
	}

	return nil
}

// LoadBalance redistributes programs across nodes for optimal performance
func (rt *Runtime) LoadBalance() error {
	rt.mu.Lock()
	defer rt.mu.Unlock()

	if !rt.config.LoadBalancing {
		return nil
	}

	// Simple load balancing - move programs from high-load to low-load nodes
	highLoadNodes := make([]*Node, 0)
	lowLoadNodes := make([]*Node, 0)

	for _, node := range rt.nodes {
		if node.Load > 0.8 {
			highLoadNodes = append(highLoadNodes, node)
		} else if node.Load < 0.5 {
			lowLoadNodes = append(lowLoadNodes, node)
		}
	}

	// Redistribute programs
	for _, highLoadNode := range highLoadNodes {
		for _, lowLoadNode := range lowLoadNodes {
			if err := rt.migratePrograms(highLoadNode, lowLoadNode); err != nil {
				return fmt.Errorf("failed to migrate programs: %w", err)
			}
		}
	}

	return nil
}

// migratePrograms migrates programs from one node to another
func (rt *Runtime) migratePrograms(fromNode, toNode *Node) error {
	// Simple migration - move one program at a time
	for programID, program := range fromNode.Programs {
		if len(toNode.Programs) < 10 { // Arbitrary limit
			// Move program
			toNode.Programs[programID] = program
			delete(fromNode.Programs, programID)

			// Update loads
			fromNode.Load = math.Max(0, fromNode.Load-0.1)
			toNode.Load = math.Min(1.0, toNode.Load+0.1)

			break // Move only one program per migration call
		}
	}

	return nil
}
