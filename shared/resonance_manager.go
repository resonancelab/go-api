package shared

import (
	"fmt"
	"sync"
	"time"

	"github.com/resonancelab/psizero/api/core"
	"github.com/resonancelab/psizero/shared/types"
)

// ResonanceManager coordinates all engine operations and manages shared state
type ResonanceManager struct {
	// Core engine
	resonanceEngine *core.ResonanceEngine

	// Engine registry
	engines      map[string]ResonanceEngine
	engineStates map[string]*EngineState
	engineMutex  sync.RWMutex

	// Global state
	globalState *GlobalResonanceState
	stateMutex  sync.RWMutex

	// Telemetry
	telemetryCollector *TelemetryCollector
	telemetryBuffer    []types.TelemetryPoint
	telemetryMutex     sync.RWMutex

	// Configuration
	config *ResonanceManagerConfig

	// Database connection
	dbConnection DatabaseConnection

	// Cache
	cache *ResonanceCache

	// Performance monitoring
	performanceMonitor *PerformanceMonitor

	// Event system
	eventBus *EventBus

	// Synchronization
	shutdown chan bool
	running  bool
	runMutex sync.Mutex
}

// ResonanceEngine interface that all engines must implement
type ResonanceEngine interface {
	GetID() string
	GetType() string
	GetState() (*EngineState, error)
	SetState(*EngineState) error
	Synchronize(*GlobalResonanceState) error
	GetTelemetryHistory() []types.TelemetryPoint
	Cleanup() error
}

// EngineState represents the state of an individual engine
type EngineState struct {
	ID              string                 `json:"id"`
	Type            string                 `json:"type"`
	Status          string                 `json:"status"`
	LastUpdate      time.Time              `json:"last_update"`
	Metrics         map[string]float64     `json:"metrics"`
	Configuration   map[string]interface{} `json:"configuration"`
	QuantumState    *core.QuantumState     `json:"quantum_state,omitempty"`
	ResonanceLevel  float64                `json:"resonance_level"`
	Coherence       float64                `json:"coherence"`
	EntanglementMap map[string]float64     `json:"entanglement_map"`
}

// GlobalResonanceState represents the global state across all engines
type GlobalResonanceState struct {
	// Core resonance metrics
	GlobalResonance   float64 `json:"global_resonance"`
	SystemCoherence   float64 `json:"system_coherence"`
	UnificationDegree float64 `json:"unification_degree"`

	// Quantum state
	MasterQuantumState *core.QuantumState `json:"master_quantum_state"`
	EntanglementMatrix [][]complex128     `json:"entanglement_matrix"`

	// Engine synchronization
	EngineStates  map[string]*EngineState `json:"engine_states"`
	SyncTimestamp time.Time               `json:"sync_timestamp"`

	// Shared resources
	SharedPrimes       []int                     `json:"shared_primes"`
	ResonanceOperators []*core.ResonanceOperator `json:"resonance_operators"`

	// Telemetry aggregation
	AggregatedMetrics map[string]float64     `json:"aggregated_metrics"`
	TelemetryHistory  []types.TelemetryPoint `json:"telemetry_history"`

	// Configuration
	GlobalConfig map[string]interface{} `json:"global_config"`
}

// ResonanceManagerConfig configures the resonance manager
type ResonanceManagerConfig struct {
	// Core settings
	MaxEngines        int           `json:"max_engines"`
	SyncInterval      time.Duration `json:"sync_interval"`
	TelemetryInterval time.Duration `json:"telemetry_interval"`

	// Performance settings
	CacheSize        int           `json:"cache_size"`
	CacheTTL         time.Duration `json:"cache_ttl"`
	MaxConcurrentOps int           `json:"max_concurrent_ops"`

	// Database settings
	DatabaseURL     string        `json:"database_url"`
	DatabaseTimeout time.Duration `json:"database_timeout"`
	BatchSize       int           `json:"batch_size"`

	// Monitoring settings
	MetricsRetention time.Duration      `json:"metrics_retention"`
	AlertThresholds  map[string]float64 `json:"alert_thresholds"`

	// Debug settings
	EnableDebugLogging bool `json:"enable_debug_logging"`
	EnableProfiler     bool `json:"enable_profiler"`
}

// NewResonanceManager creates a new resonance manager
func NewResonanceManager(config *ResonanceManagerConfig) (*ResonanceManager, error) {
	// Create core resonance engine
	coreConfig := &core.ResonanceEngineConfig{
		Dimension:         1024,
		MaxPrimes:         10000,
		TimeoutSeconds:    30,
		EvolutionSteps:    1000,
		ResonanceStrength: 1.0,
	}

	resonanceEngine, err := core.NewResonanceEngine(coreConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create core resonance engine: %w", err)
	}

	// Initialize global state
	globalState := &GlobalResonanceState{
		GlobalResonance:    1.0,
		SystemCoherence:    1.0,
		UnificationDegree:  0.0,
		EngineStates:       make(map[string]*EngineState),
		SharedPrimes:       resonanceEngine.GetPrimes().GetFirst(100),
		ResonanceOperators: make([]*core.ResonanceOperator, 0),
		AggregatedMetrics:  make(map[string]float64),
		TelemetryHistory:   make([]types.TelemetryPoint, 0),
		GlobalConfig:       make(map[string]interface{}),
		SyncTimestamp:      time.Now(),
	}

	// Create master quantum state
	amplitudes := make([]complex128, coreConfig.Dimension)
	for i := range amplitudes {
		amplitudes[i] = complex(1.0/float64(coreConfig.Dimension), 0)
	}
	masterState, err := resonanceEngine.CreateQuantumState(amplitudes)
	if err != nil {
		return nil, fmt.Errorf("failed to create master quantum state: %w", err)
	}
	globalState.MasterQuantumState = masterState

	// Initialize components
	telemetryCollector := NewTelemetryCollector(config.TelemetryInterval)
	cache := NewResonanceCache(config.CacheSize, config.CacheTTL)
	performanceMonitor := NewPerformanceMonitor()
	eventBus := NewEventBus()

	rm := &ResonanceManager{
		resonanceEngine:    resonanceEngine,
		engines:            make(map[string]ResonanceEngine),
		engineStates:       make(map[string]*EngineState),
		globalState:        globalState,
		telemetryCollector: telemetryCollector,
		telemetryBuffer:    make([]types.TelemetryPoint, 0),
		config:             config,
		cache:              cache,
		performanceMonitor: performanceMonitor,
		eventBus:           eventBus,
		shutdown:           make(chan bool),
		running:            false,
	}

	return rm, nil
}

// Start starts the resonance manager
func (rm *ResonanceManager) Start() error {
	rm.runMutex.Lock()
	defer rm.runMutex.Unlock()

	if rm.running {
		return fmt.Errorf("resonance manager already running")
	}

	// Initialize database connection
	if err := rm.initializeDatabase(); err != nil {
		return fmt.Errorf("database initialization failed: %w", err)
	}

	// Start background processes
	go rm.synchronizationLoop()
	go rm.telemetryLoop()
	go rm.performanceMonitoringLoop()
	go rm.eventProcessingLoop()

	rm.running = true

	// Fire startup event
	rm.eventBus.Publish(Event{
		Type:      "system_startup",
		Timestamp: time.Now(),
		Data:      map[string]interface{}{"status": "started"},
	})

	return nil
}

// Stop stops the resonance manager
func (rm *ResonanceManager) Stop() error {
	rm.runMutex.Lock()
	defer rm.runMutex.Unlock()

	if !rm.running {
		return nil
	}

	// Signal shutdown
	close(rm.shutdown)

	// Cleanup engines
	rm.engineMutex.Lock()
	for _, engine := range rm.engines {
		engine.Cleanup()
	}
	rm.engineMutex.Unlock()

	// Flush telemetry
	if err := rm.flushTelemetry(); err != nil {
		return fmt.Errorf("failed to flush telemetry: %w", err)
	}

	rm.running = false

	// Fire shutdown event
	rm.eventBus.Publish(Event{
		Type:      "system_shutdown",
		Timestamp: time.Now(),
		Data:      map[string]interface{}{"status": "stopped"},
	})

	return nil
}

// RegisterEngine registers a new engine with the manager
func (rm *ResonanceManager) RegisterEngine(engine ResonanceEngine) error {
	rm.engineMutex.Lock()
	defer rm.engineMutex.Unlock()

	id := engine.GetID()
	if _, exists := rm.engines[id]; exists {
		return fmt.Errorf("engine %s already registered", id)
	}

	// Get initial state
	state, err := engine.GetState()
	if err != nil {
		return fmt.Errorf("failed to get engine state: %w", err)
	}

	// Register engine
	rm.engines[id] = engine
	rm.engineStates[id] = state

	// Update global state
	rm.stateMutex.Lock()
	rm.globalState.EngineStates[id] = state
	rm.stateMutex.Unlock()

	// Synchronize with global state
	if err := engine.Synchronize(rm.globalState); err != nil {
		return fmt.Errorf("failed to synchronize engine: %w", err)
	}

	// Fire registration event
	rm.eventBus.Publish(Event{
		Type:      "engine_registered",
		Timestamp: time.Now(),
		Data: map[string]interface{}{
			"engine_id":   id,
			"engine_type": engine.GetType(),
		},
	})

	return nil
}

// UnregisterEngine removes an engine from the manager
func (rm *ResonanceManager) UnregisterEngine(engineID string) error {
	rm.engineMutex.Lock()
	defer rm.engineMutex.Unlock()

	engine, exists := rm.engines[engineID]
	if !exists {
		return fmt.Errorf("engine %s not found", engineID)
	}

	// Cleanup engine
	if err := engine.Cleanup(); err != nil {
		return fmt.Errorf("failed to cleanup engine: %w", err)
	}

	// Remove from registry
	delete(rm.engines, engineID)
	delete(rm.engineStates, engineID)

	// Update global state
	rm.stateMutex.Lock()
	delete(rm.globalState.EngineStates, engineID)
	rm.stateMutex.Unlock()

	// Fire unregistration event
	rm.eventBus.Publish(Event{
		Type:      "engine_unregistered",
		Timestamp: time.Now(),
		Data: map[string]interface{}{
			"engine_id": engineID,
		},
	})

	return nil
}

// GetGlobalState returns the current global state
func (rm *ResonanceManager) GetGlobalState() *GlobalResonanceState {
	rm.stateMutex.RLock()
	defer rm.stateMutex.RUnlock()

	// Create a copy to avoid race conditions
	stateCopy := *rm.globalState
	return &stateCopy
}

// UpdateGlobalResonance updates the global resonance level
func (rm *ResonanceManager) UpdateGlobalResonance(newLevel float64) error {
	rm.stateMutex.Lock()
	defer rm.stateMutex.Unlock()

	oldLevel := rm.globalState.GlobalResonance
	rm.globalState.GlobalResonance = newLevel

	// Fire resonance change event
	rm.eventBus.Publish(Event{
		Type:      "global_resonance_changed",
		Timestamp: time.Now(),
		Data: map[string]interface{}{
			"old_level": oldLevel,
			"new_level": newLevel,
		},
	})

	return nil
}

// synchronizationLoop runs the engine synchronization loop
func (rm *ResonanceManager) synchronizationLoop() {
	ticker := time.NewTicker(rm.config.SyncInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			rm.synchronizeEngines()
		case <-rm.shutdown:
			return
		}
	}
}

// synchronizeEngines synchronizes all engines with global state
func (rm *ResonanceManager) synchronizeEngines() {
	rm.engineMutex.RLock()
	engines := make(map[string]ResonanceEngine)
	for id, engine := range rm.engines {
		engines[id] = engine
	}
	rm.engineMutex.RUnlock()

	// Update global state from engines
	rm.updateGlobalStateFromEngines(engines)

	// Synchronize engines with global state
	for id, engine := range engines {
		if err := engine.Synchronize(rm.globalState); err != nil {
			rm.eventBus.Publish(Event{
				Type:      "synchronization_error",
				Timestamp: time.Now(),
				Data: map[string]interface{}{
					"engine_id": id,
					"error":     err.Error(),
				},
			})
		}
	}

	// Update sync timestamp
	rm.stateMutex.Lock()
	rm.globalState.SyncTimestamp = time.Now()
	rm.stateMutex.Unlock()
}

// updateGlobalStateFromEngines updates global state based on engine states
func (rm *ResonanceManager) updateGlobalStateFromEngines(engines map[string]ResonanceEngine) {
	totalResonance := 0.0
	totalCoherence := 0.0
	totalUnification := 0.0
	engineCount := 0

	aggregatedMetrics := make(map[string]float64)

	for id, engine := range engines {
		state, err := engine.GetState()
		if err != nil {
			continue
		}

		// Update engine state in global state
		rm.stateMutex.Lock()
		rm.globalState.EngineStates[id] = state
		rm.stateMutex.Unlock()

		// Aggregate metrics
		totalResonance += state.ResonanceLevel
		totalCoherence += state.Coherence
		engineCount++

		// Aggregate engine-specific metrics
		for key, value := range state.Metrics {
			if existing, exists := aggregatedMetrics[key]; exists {
				aggregatedMetrics[key] = existing + value
			} else {
				aggregatedMetrics[key] = value
			}
		}

		// Special handling for unification degree
		if unification, exists := state.Metrics["unification_degree"]; exists {
			totalUnification += unification
		}
	}

	// Update global metrics
	if engineCount > 0 {
		rm.stateMutex.Lock()
		rm.globalState.GlobalResonance = totalResonance / float64(engineCount)
		rm.globalState.SystemCoherence = totalCoherence / float64(engineCount)
		rm.globalState.UnificationDegree = totalUnification / float64(engineCount)

		// Average aggregated metrics
		for key, value := range aggregatedMetrics {
			rm.globalState.AggregatedMetrics[key] = value / float64(engineCount)
		}
		rm.stateMutex.Unlock()
	}
}

// telemetryLoop runs the telemetry collection loop
func (rm *ResonanceManager) telemetryLoop() {
	ticker := time.NewTicker(rm.config.TelemetryInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			rm.collectTelemetry()
		case <-rm.shutdown:
			return
		}
	}
}

// collectTelemetry collects telemetry from all engines
func (rm *ResonanceManager) collectTelemetry() {
	rm.engineMutex.RLock()
	engines := make(map[string]ResonanceEngine)
	for id, engine := range rm.engines {
		engines[id] = engine
	}
	rm.engineMutex.RUnlock()

	timestamp := time.Now()
	globalState := rm.GetGlobalState()

	// Create system telemetry point
	systemPoint := types.TelemetryPoint{
		Step:              int(time.Since(time.Unix(0, 0)).Seconds()),
		SymbolicEntropy:   1.0 - globalState.SystemCoherence,
		LyapunovMetric:    globalState.UnificationDegree,
		SatisfactionRate:  globalState.SystemCoherence,
		ResonanceStrength: globalState.GlobalResonance,
		Dominance:         globalState.GlobalResonance,
		Timestamp:         timestamp,
	}

	// Collect from engines
	allTelemetry := []types.TelemetryPoint{systemPoint}
	for _, engine := range engines {
		engineTelemetry := engine.GetTelemetryHistory()
		if len(engineTelemetry) > 0 {
			// Get latest point
			latest := engineTelemetry[len(engineTelemetry)-1]
			allTelemetry = append(allTelemetry, latest)
		}
	}

	// Store in buffer
	rm.telemetryMutex.Lock()
	rm.telemetryBuffer = append(rm.telemetryBuffer, allTelemetry...)

	// Update global telemetry history
	rm.stateMutex.Lock()
	rm.globalState.TelemetryHistory = append(rm.globalState.TelemetryHistory, systemPoint)
	rm.stateMutex.Unlock()

	// Flush if buffer is full
	if len(rm.telemetryBuffer) >= rm.config.BatchSize {
		rm.flushTelemetryLocked()
	}
	rm.telemetryMutex.Unlock()
}

// flushTelemetry flushes telemetry buffer to database
func (rm *ResonanceManager) flushTelemetry() error {
	rm.telemetryMutex.Lock()
	defer rm.telemetryMutex.Unlock()

	return rm.flushTelemetryLocked()
}

// flushTelemetryLocked flushes telemetry buffer (assumes lock held)
func (rm *ResonanceManager) flushTelemetryLocked() error {
	if len(rm.telemetryBuffer) == 0 {
		return nil
	}

	if rm.dbConnection != nil {
		if err := rm.dbConnection.StoreTelemetry(rm.telemetryBuffer); err != nil {
			return fmt.Errorf("failed to store telemetry: %w", err)
		}
	}

	// Clear buffer
	rm.telemetryBuffer = rm.telemetryBuffer[:0]

	return nil
}
