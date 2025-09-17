package shared

import (
	"fmt"
	"time"
)

// Complete the remaining methods for the ResonanceManager

// initializeDatabase initializes the database connection
func (rm *ResonanceManager) initializeDatabase() error {
	if rm.config.DatabaseURL == "" {
		// No database configured, skip
		return nil
	}
	
	// Create PostgreSQL connection
	dbConn := NewPostgreSQLConnection(rm.config.DatabaseURL)
	
	// Connect to database
	if err := dbConn.Connect(); err != nil {
		return fmt.Errorf("failed to connect to database: %w", err)
	}
	
	rm.dbConnection = dbConn
	
	// Load existing global state from database
	globalState, err := dbConn.GetGlobalState()
	if err != nil {
		return fmt.Errorf("failed to load global state: %w", err)
	}
	
	if globalState != nil {
		rm.stateMutex.Lock()
		// Merge with current global state
		rm.globalState.GlobalResonance = globalState.GlobalResonance
		rm.globalState.SystemCoherence = globalState.SystemCoherence
		rm.globalState.UnificationDegree = globalState.UnificationDegree
		rm.globalState.AggregatedMetrics = globalState.AggregatedMetrics
		rm.globalState.GlobalConfig = globalState.GlobalConfig
		rm.stateMutex.Unlock()
	}
	
	return nil
}

// performanceMonitoringLoop runs the performance monitoring loop
func (rm *ResonanceManager) performanceMonitoringLoop() {
	if err := rm.performanceMonitor.Start(); err != nil {
		rm.eventBus.Publish(Event{
			Type: "performance_monitor_error",
			Timestamp: time.Now(),
			Data: map[string]interface{}{
				"error": err.Error(),
			},
		})
		return
	}
	
	ticker := time.NewTicker(30 * time.Second) // Monitor every 30 seconds
	defer ticker.Stop()
	
	for {
		select {
		case <-ticker.C:
			rm.collectPerformanceMetrics()
		case <-rm.shutdown:
			rm.performanceMonitor.Stop()
			return
		}
	}
}

// collectPerformanceMetrics collects and stores performance metrics
func (rm *ResonanceManager) collectPerformanceMetrics() {
	// Get system metrics
	systemMetrics := rm.performanceMonitor.GetSystemMetrics()
	
	// Record metrics
	rm.performanceMonitor.RecordMetric("system_cpu_usage", systemMetrics.CPUUsage)
	rm.performanceMonitor.RecordMetric("system_memory_usage", systemMetrics.MemoryUsage)
	rm.performanceMonitor.RecordMetric("system_goroutine_count", float64(systemMetrics.GoroutineCount))
	rm.performanceMonitor.RecordMetric("system_gc_pause_time", systemMetrics.GCPauseTime)
	rm.performanceMonitor.RecordMetric("system_heap_size", float64(systemMetrics.HeapSize))
	
	// Get engine performance metrics
	rm.engineMutex.RLock()
	engineCount := len(rm.engines)
	rm.engineMutex.RUnlock()
	
	rm.performanceMonitor.RecordMetric("active_engines", float64(engineCount))
	
	// Store in database if available
	if rm.dbConnection != nil {
		allMetrics := rm.performanceMonitor.GetAllMetrics()
		if err := rm.dbConnection.StorePerformanceMetrics(allMetrics); err != nil {
			rm.eventBus.Publish(Event{
				Type: "database_error",
				Timestamp: time.Now(),
				Data: map[string]interface{}{
					"operation": "store_performance_metrics",
					"error": err.Error(),
				},
			})
		}
	}
	
	// Check for performance alerts
	rm.checkPerformanceAlerts(systemMetrics)
}

// checkPerformanceAlerts checks for performance issues and sends alerts
func (rm *ResonanceManager) checkPerformanceAlerts(metrics *SystemMetrics) {
	thresholds := rm.config.AlertThresholds
	
	// Check CPU usage
	if cpuThreshold, exists := thresholds["cpu_usage"]; exists && metrics.CPUUsage > cpuThreshold {
		rm.eventBus.Publish(Event{
			Type: "performance_alert",
			Timestamp: time.Now(),
			Data: map[string]interface{}{
				"metric": "cpu_usage",
				"value": metrics.CPUUsage,
				"threshold": cpuThreshold,
				"severity": "warning",
			},
		})
	}
	
	// Check memory usage
	if memThreshold, exists := thresholds["memory_usage"]; exists && metrics.MemoryUsage > memThreshold {
		rm.eventBus.Publish(Event{
			Type: "performance_alert",
			Timestamp: time.Now(),
			Data: map[string]interface{}{
				"metric": "memory_usage",
				"value": metrics.MemoryUsage,
				"threshold": memThreshold,
				"severity": "warning",
			},
		})
	}
	
	// Check error rate
	if errorThreshold, exists := thresholds["error_rate"]; exists && metrics.ErrorRate > errorThreshold {
		rm.eventBus.Publish(Event{
			Type: "performance_alert",
			Timestamp: time.Now(),
			Data: map[string]interface{}{
				"metric": "error_rate",
				"value": metrics.ErrorRate,
				"threshold": errorThreshold,
				"severity": "critical",
			},
		})
	}
	
	// Check response time
	if responseThreshold, exists := thresholds["response_time"]; exists && metrics.ResponseTime > responseThreshold {
		rm.eventBus.Publish(Event{
			Type: "performance_alert",
			Timestamp: time.Now(),
			Data: map[string]interface{}{
				"metric": "response_time",
				"value": metrics.ResponseTime,
				"threshold": responseThreshold,
				"severity": "warning",
			},
		})
	}
}

// eventProcessingLoop runs the event processing loop
func (rm *ResonanceManager) eventProcessingLoop() {
	// Start event bus
	rm.eventBus.Start()
	
	// Subscribe to system events
	rm.subscribeToSystemEvents()
	
	// Wait for shutdown
	<-rm.shutdown
	
	// Stop event bus
	rm.eventBus.Stop()
}

// subscribeToSystemEvents subscribes to important system events
func (rm *ResonanceManager) subscribeToSystemEvents() {
	// Subscribe to engine events
	rm.eventBus.Subscribe("engine_registered", func(event Event) {
		rm.handleEngineRegistered(event)
	})
	
	rm.eventBus.Subscribe("engine_unregistered", func(event Event) {
		rm.handleEngineUnregistered(event)
	})
	
	rm.eventBus.Subscribe("synchronization_error", func(event Event) {
		rm.handleSynchronizationError(event)
	})
	
	// Subscribe to performance events
	rm.eventBus.Subscribe("performance_alert", func(event Event) {
		rm.handlePerformanceAlert(event)
	})
	
	// Subscribe to database events
	rm.eventBus.Subscribe("database_error", func(event Event) {
		rm.handleDatabaseError(event)
	})
	
	// Subscribe to resonance events
	rm.eventBus.Subscribe("global_resonance_changed", func(event Event) {
		rm.handleGlobalResonanceChanged(event)
	})
}

// handleEngineRegistered handles engine registration events
func (rm *ResonanceManager) handleEngineRegistered(event Event) {
	engineID, _ := event.Data["engine_id"].(string)
	engineType, _ := event.Data["engine_type"].(string)
	
	// Update global metrics
	rm.stateMutex.Lock()
	rm.globalState.AggregatedMetrics["active_engines"] = float64(len(rm.globalState.EngineStates))
	rm.stateMutex.Unlock()
	
	// Log engine registration
	if rm.config.EnableDebugLogging {
		fmt.Printf("Engine registered: %s (type: %s)\n", engineID, engineType)
	}
}

// handleEngineUnregistered handles engine unregistration events
func (rm *ResonanceManager) handleEngineUnregistered(event Event) {
	engineID, _ := event.Data["engine_id"].(string)
	
	// Update global metrics
	rm.stateMutex.Lock()
	rm.globalState.AggregatedMetrics["active_engines"] = float64(len(rm.globalState.EngineStates))
	rm.stateMutex.Unlock()
	
	// Log engine unregistration
	if rm.config.EnableDebugLogging {
		fmt.Printf("Engine unregistered: %s\n", engineID)
	}
}

// handleSynchronizationError handles synchronization errors
func (rm *ResonanceManager) handleSynchronizationError(event Event) {
	engineID, _ := event.Data["engine_id"].(string)
	errorMsg, _ := event.Data["error"].(string)
	
	// Log synchronization error
	if rm.config.EnableDebugLogging {
		fmt.Printf("Synchronization error for engine %s: %s\n", engineID, errorMsg)
	}
	
	// Update error metrics
	rm.performanceMonitor.RecordMetric("synchronization_errors", 1)
}

// handlePerformanceAlert handles performance alerts
func (rm *ResonanceManager) handlePerformanceAlert(event Event) {
	metric, _ := event.Data["metric"].(string)
	value, _ := event.Data["value"].(float64)
	threshold, _ := event.Data["threshold"].(float64)
	severity, _ := event.Data["severity"].(string)
	
	// Log performance alert
	if rm.config.EnableDebugLogging {
		fmt.Printf("Performance alert: %s = %.2f (threshold: %.2f, severity: %s)\n", 
			metric, value, threshold, severity)
	}
	
	// Update alert metrics
	rm.performanceMonitor.RecordMetric("performance_alerts", 1)
	rm.performanceMonitor.RecordMetric(fmt.Sprintf("alert_%s", severity), 1)
}

// handleDatabaseError handles database errors
func (rm *ResonanceManager) handleDatabaseError(event Event) {
	operation, _ := event.Data["operation"].(string)
	errorMsg, _ := event.Data["error"].(string)
	
	// Log database error
	if rm.config.EnableDebugLogging {
		fmt.Printf("Database error during %s: %s\n", operation, errorMsg)
	}
	
	// Update error metrics
	rm.performanceMonitor.RecordMetric("database_errors", 1)
}

// handleGlobalResonanceChanged handles global resonance changes
func (rm *ResonanceManager) handleGlobalResonanceChanged(event Event) {
	oldLevel, _ := event.Data["old_level"].(float64)
	newLevel, _ := event.Data["new_level"].(float64)
	
	// Log resonance change
	if rm.config.EnableDebugLogging {
		fmt.Printf("Global resonance changed: %.6f -> %.6f\n", oldLevel, newLevel)
	}
	
	// Store global state in database
	if rm.dbConnection != nil {
		globalState := rm.GetGlobalState()
		if err := rm.dbConnection.StoreGlobalState(globalState); err != nil {
			rm.eventBus.Publish(Event{
				Type: "database_error",
				Timestamp: time.Now(),
				Data: map[string]interface{}{
					"operation": "store_global_state",
					"error": err.Error(),
				},
			})
		}
	}
}

// GetEngineList returns a list of registered engines
func (rm *ResonanceManager) GetEngineList() []string {
	rm.engineMutex.RLock()
	defer rm.engineMutex.RUnlock()
	
	engines := make([]string, 0, len(rm.engines))
	for id := range rm.engines {
		engines = append(engines, id)
	}
	
	return engines
}

// GetEngineState returns the state of a specific engine
func (rm *ResonanceManager) GetEngineState(engineID string) (*EngineState, error) {
	rm.engineMutex.RLock()
	engine, exists := rm.engines[engineID]
	rm.engineMutex.RUnlock()
	
	if !exists {
		return nil, fmt.Errorf("engine %s not found", engineID)
	}
	
	return engine.GetState()
}

// SetEngineState sets the state of a specific engine
func (rm *ResonanceManager) SetEngineState(engineID string, state *EngineState) error {
	rm.engineMutex.RLock()
	engine, exists := rm.engines[engineID]
	rm.engineMutex.RUnlock()
	
	if !exists {
		return fmt.Errorf("engine %s not found", engineID)
	}
	
	return engine.SetState(state)
}

// GetSystemStats returns system statistics
func (rm *ResonanceManager) GetSystemStats() map[string]interface{} {
	globalState := rm.GetGlobalState()
	performanceStats := rm.performanceMonitor.GetPerformanceSummary()
	eventStats := rm.eventBus.GetStats()
	cacheStats := rm.cache.GetStats()
	
	rm.engineMutex.RLock()
	engineCount := len(rm.engines)
	engineTypes := make(map[string]int)
	for _, engine := range rm.engines {
		engineType := engine.GetType()
		engineTypes[engineType]++
	}
	rm.engineMutex.RUnlock()
	
	return map[string]interface{}{
		"global_state": map[string]interface{}{
			"global_resonance":   globalState.GlobalResonance,
			"system_coherence":   globalState.SystemCoherence,
			"unification_degree": globalState.UnificationDegree,
			"sync_timestamp":     globalState.SyncTimestamp,
		},
		"engines": map[string]interface{}{
			"total_count":  engineCount,
			"types":        engineTypes,
		},
		"performance": performanceStats,
		"events":      eventStats,
		"cache":       cacheStats,
		"telemetry": map[string]interface{}{
			"buffer_size": len(rm.telemetryBuffer),
			"history_size": len(globalState.TelemetryHistory),
		},
	}
}

// GetHealthStatus returns the overall system health status
func (rm *ResonanceManager) GetHealthStatus() map[string]interface{} {
	healthStatus := rm.performanceMonitor.GetHealthStatus()
	globalState := rm.GetGlobalState()
	
	// Add resonance-specific health indicators
	resonanceHealth := "healthy"
	if globalState.SystemCoherence < 0.5 {
		resonanceHealth = "degraded"
	}
	if globalState.SystemCoherence < 0.2 {
		resonanceHealth = "critical"
	}
	
	// Check engine health
	rm.engineMutex.RLock()
	engineHealthy := 0
	engineTotal := len(rm.engines)
	for _, engine := range rm.engines {
		state, err := engine.GetState()
		if err == nil && state.Status == "running" {
			engineHealthy++
		}
	}
	rm.engineMutex.RUnlock()
	
	engineHealth := "healthy"
	if engineTotal > 0 {
		healthyRatio := float64(engineHealthy) / float64(engineTotal)
		if healthyRatio < 0.5 {
			engineHealth = "critical"
		} else if healthyRatio < 0.8 {
			engineHealth = "degraded"
		}
	}
	
	// Overall health
	overallHealth := "healthy"
	if healthStatus["status"] == "unhealthy" || resonanceHealth == "critical" || engineHealth == "critical" {
		overallHealth = "critical"
	} else if healthStatus["status"] == "degraded" || resonanceHealth == "degraded" || engineHealth == "degraded" {
		overallHealth = "degraded"
	}
	
	return map[string]interface{}{
		"overall_status":    overallHealth,
		"system_health":     healthStatus,
		"resonance_health":  resonanceHealth,
		"engine_health":     engineHealth,
		"healthy_engines":   engineHealthy,
		"total_engines":     engineTotal,
		"system_coherence":  globalState.SystemCoherence,
		"global_resonance":  globalState.GlobalResonance,
		"unification_degree": globalState.UnificationDegree,
		"timestamp":         time.Now(),
	}
}

// Cleanup performs cleanup operations
func (rm *ResonanceManager) Cleanup() error {
	// Stop if running
	if err := rm.Stop(); err != nil {
		return fmt.Errorf("failed to stop resonance manager: %w", err)
	}
	
	// Cleanup database connection
	if rm.dbConnection != nil {
		if err := rm.dbConnection.Disconnect(); err != nil {
			return fmt.Errorf("failed to disconnect database: %w", err)
		}
	}
	
	// Cleanup cache
	if rm.cache != nil {
		rm.cache.Stop()
	}
	
	return nil
}