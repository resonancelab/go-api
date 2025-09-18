package tests

import (
	"testing"
	"time"

	"github.com/resonancelab/psizero/engines/qsem"
	"github.com/resonancelab/psizero/engines/srs"
	"github.com/resonancelab/psizero/shared"
	"github.com/resonancelab/psizero/shared/types"
)

// TestResonanceManager tests the shared resonance manager
func TestResonanceManager(t *testing.T) {
	config := &shared.ResonanceManagerConfig{
		MaxEngines:         10,
		SyncInterval:       100 * time.Millisecond,
		TelemetryInterval:  50 * time.Millisecond,
		CacheSize:          1000,
		CacheTTL:           1 * time.Minute,
		MaxConcurrentOps:   5,
		DatabaseURL:        "", // Skip database for tests
		DatabaseTimeout:    5 * time.Second,
		BatchSize:          100,
		MetricsRetention:   1 * time.Hour,
		AlertThresholds:    map[string]float64{"cpu_usage": 80.0, "memory_usage": 85.0},
		EnableDebugLogging: false,
		EnableProfiler:     false,
	}

	manager, err := shared.NewResonanceManager(config)
	if err != nil {
		t.Fatalf("Failed to create resonance manager: %v", err)
	}
	defer manager.Cleanup()

	t.Run("StartStop", func(t *testing.T) {
		if err := manager.Start(); err != nil {
			t.Fatalf("Failed to start manager: %v", err)
		}

		time.Sleep(200 * time.Millisecond) // Let it run briefly

		if err := manager.Stop(); err != nil {
			t.Fatalf("Failed to stop manager: %v", err)
		}
	})

	t.Run("EngineBasicTest", func(t *testing.T) {
		// Start manager
		if err := manager.Start(); err != nil {
			t.Fatalf("Failed to start manager: %v", err)
		}
		defer manager.Stop()

		// Create engines to test they work
		srsEngine, err := srs.NewSRSEngine()
		if err != nil {
			t.Fatalf("Failed to create SRS engine: %v", err)
		}

		// Basic test - verify we can work with the engine
		if srsEngine == nil {
			t.Fatal("SRS engine should not be nil")
		}
	})

	t.Run("GlobalStateManagement", func(t *testing.T) {
		// Start manager
		if err := manager.Start(); err != nil {
			t.Fatalf("Failed to start manager: %v", err)
		}
		defer manager.Stop()

		// Get initial global state
		globalState := manager.GetGlobalState()
		if globalState == nil {
			t.Fatal("Global state should not be nil")
		}

		initialResonance := globalState.GlobalResonance

		// Update global resonance
		newResonance := initialResonance + 0.1
		if err := manager.UpdateGlobalResonance(newResonance); err != nil {
			t.Fatalf("Failed to update global resonance: %v", err)
		}

		// Verify update
		updatedState := manager.GetGlobalState()
		if updatedState.GlobalResonance != newResonance {
			t.Errorf("Expected resonance %f, got %f", newResonance, updatedState.GlobalResonance)
		}
	})

	t.Run("SystemStats", func(t *testing.T) {
		// Start manager
		if err := manager.Start(); err != nil {
			t.Fatalf("Failed to start manager: %v", err)
		}
		defer manager.Stop()

		time.Sleep(100 * time.Millisecond) // Let it collect some data

		stats := manager.GetSystemStats()
		if stats == nil {
			t.Fatal("System stats should not be nil")
		}

		// Check required fields
		if _, exists := stats["global_state"]; !exists {
			t.Error("System stats should include global_state")
		}

		if _, exists := stats["engines"]; !exists {
			t.Error("System stats should include engines")
		}

		if _, exists := stats["performance"]; !exists {
			t.Error("System stats should include performance")
		}
	})

	t.Run("HealthStatus", func(t *testing.T) {
		// Start manager
		if err := manager.Start(); err != nil {
			t.Fatalf("Failed to start manager: %v", err)
		}
		defer manager.Stop()

		time.Sleep(100 * time.Millisecond) // Let it collect some data

		health := manager.GetHealthStatus()
		if health == nil {
			t.Fatal("Health status should not be nil")
		}

		// Check required fields
		if _, exists := health["overall_status"]; !exists {
			t.Error("Health status should include overall_status")
		}

		if _, exists := health["system_coherence"]; !exists {
			t.Error("Health status should include system_coherence")
		}

		if _, exists := health["global_resonance"]; !exists {
			t.Error("Health status should include global_resonance")
		}
	})
}

// TestTelemetryCollector tests the telemetry collection system
func TestTelemetryCollector(t *testing.T) {
	collector := shared.NewTelemetryCollector(10 * time.Millisecond)

	t.Run("TelemetryCollection", func(t *testing.T) {
		// Add test telemetry points
		points := []types.TelemetryPoint{
			{
				Step:              1,
				SymbolicEntropy:   0.5,
				LyapunovMetric:    0.3,
				SatisfactionRate:  0.8,
				ResonanceStrength: 1.2,
				Dominance:         0.6,
				Timestamp:         time.Now(),
			},
			{
				Step:              2,
				SymbolicEntropy:   0.4,
				LyapunovMetric:    0.4,
				SatisfactionRate:  0.9,
				ResonanceStrength: 1.1,
				Dominance:         0.7,
				Timestamp:         time.Now(),
			},
		}

		for _, point := range points {
			collector.AddPoint(point)
		}

		// Get buffer
		buffer := collector.GetBuffer()
		if len(buffer) != 2 {
			t.Errorf("Expected 2 points in buffer, got %d", len(buffer))
		}

		// Check metrics
		metric := collector.GetMetric("symbolic_entropy")
		if metric == nil {
			t.Fatal("Symbolic entropy metric should exist")
		}

		if metric.Count != 2 {
			t.Errorf("Expected 2 values in metric, got %d", metric.Count)
		}

		stats := metric.GetStatistics()
		if stats["mean"] != 0.45 { // (0.5 + 0.4) / 2
			t.Errorf("Expected mean 0.45, got %f", stats["mean"])
		}
	})

	t.Run("MetricAggregation", func(t *testing.T) {
		collector.ClearBuffer()

		// Add multiple points to test aggregation
		for i := 0; i < 10; i++ {
			point := types.TelemetryPoint{
				Step:            i,
				SymbolicEntropy: float64(i) * 0.1,
				Timestamp:       time.Now(),
			}
			collector.AddPoint(point)
		}

		metric := collector.GetMetric("symbolic_entropy")
		if metric == nil {
			t.Fatal("Metric should exist")
		}

		stats := metric.GetStatistics()
		expectedMean := 0.45 // (0 + 0.1 + 0.2 + ... + 0.9) / 10

		if abs(stats["mean"]-expectedMean) > 1e-10 {
			t.Errorf("Expected mean %f, got %f", expectedMean, stats["mean"])
		}

		if stats["min"] != 0.0 {
			t.Errorf("Expected min 0.0, got %f", stats["min"])
		}

		if stats["max"] != 0.9 {
			t.Errorf("Expected max 0.9, got %f", stats["max"])
		}
	})

	t.Run("Callbacks", func(t *testing.T) {
		callbackCalled := false
		collector.AddCallback(func(point types.TelemetryPoint) {
			callbackCalled = true
		})

		point := types.TelemetryPoint{
			Step:      1,
			Timestamp: time.Now(),
		}

		collector.AddPoint(point)
		time.Sleep(10 * time.Millisecond) // Wait for goroutine

		if !callbackCalled {
			t.Error("Callback should have been called")
		}
	})
}

// TestResonanceCache tests the caching system
func TestResonanceCache(t *testing.T) {
	cache := shared.NewResonanceCache(100, 1*time.Second)
	defer cache.Stop()

	t.Run("BasicOperations", func(t *testing.T) {
		// Set and get
		cache.Set("key1", "value1")

		value, exists := cache.Get("key1")
		if !exists {
			t.Error("Key should exist")
		}

		if value != "value1" {
			t.Errorf("Expected 'value1', got '%v'", value)
		}

		// Test non-existent key
		_, exists = cache.Get("nonexistent")
		if exists {
			t.Error("Non-existent key should not exist")
		}
	})

	t.Run("TypedGetters", func(t *testing.T) {
		cache.Set("string_key", "test_string")
		cache.Set("float_key", 3.14)
		cache.Set("int_key", 42)

		// Test string getter
		str, exists := cache.GetString("string_key")
		if !exists || str != "test_string" {
			t.Errorf("Expected 'test_string', got '%s'", str)
		}

		// Test float getter
		f, exists := cache.GetFloat64("float_key")
		if !exists || f != 3.14 {
			t.Errorf("Expected 3.14, got %f", f)
		}

		// Test int getter
		i, exists := cache.GetInt("int_key")
		if !exists || i != 42 {
			t.Errorf("Expected 42, got %d", i)
		}
	})

	t.Run("TTLExpiration", func(t *testing.T) {
		cache.SetWithTTL("ttl_key", "ttl_value", 50*time.Millisecond)

		// Should exist initially
		_, exists := cache.Get("ttl_key")
		if !exists {
			t.Error("Key should exist initially")
		}

		// Wait for expiration
		time.Sleep(100 * time.Millisecond)

		_, exists = cache.Get("ttl_key")
		if exists {
			t.Error("Key should have expired")
		}
	})

	t.Run("Increment", func(t *testing.T) {
		// Start with new key
		result := cache.Increment("counter", 1.0)
		if result != 1.0 {
			t.Errorf("Expected 1.0, got %f", result)
		}

		// Increment existing key
		result = cache.Increment("counter", 2.5)
		if result != 3.5 {
			t.Errorf("Expected 3.5, got %f", result)
		}

		// Decrement
		result = cache.Decrement("counter", 1.0)
		if result != 2.5 {
			t.Errorf("Expected 2.5, got %f", result)
		}
	})

	t.Run("GetOrSet", func(t *testing.T) {
		called := false
		value := cache.GetOrSet("lazy_key", func() interface{} {
			called = true
			return "lazy_value"
		})

		if !called {
			t.Error("Function should have been called")
		}

		if value != "lazy_value" {
			t.Errorf("Expected 'lazy_value', got '%v'", value)
		}

		// Second call should not invoke function
		called = false
		value = cache.GetOrSet("lazy_key", func() interface{} {
			called = true
			return "different_value"
		})

		if called {
			t.Error("Function should not have been called second time")
		}

		if value != "lazy_value" {
			t.Errorf("Expected 'lazy_value', got '%v'", value)
		}
	})

	t.Run("Stats", func(t *testing.T) {
		cache.Clear()
		cache.Set("stat_key1", "value1")
		cache.Set("stat_key2", "value2")

		stats := cache.GetStats()
		if stats["total_items"] != 2 {
			t.Errorf("Expected 2 items, got %v", stats["total_items"])
		}

		if stats["max_size"] != 100 {
			t.Errorf("Expected max size 100, got %v", stats["max_size"])
		}
	})
}

// TestPerformanceMonitor tests the performance monitoring system
func TestPerformanceMonitor(t *testing.T) {
	monitor := shared.NewPerformanceMonitor()

	t.Run("MetricRecording", func(t *testing.T) {
		// Record some metrics
		monitor.RecordMetric("test_metric", 1.5)
		monitor.RecordMetric("test_metric", 2.5)
		monitor.RecordMetric("test_metric", 3.5)

		metric := monitor.GetMetric("test_metric")
		if metric == nil {
			t.Fatal("Metric should exist")
		}

		if metric.CurrentValue != 3.5 {
			t.Errorf("Expected current value 3.5, got %f", metric.CurrentValue)
		}

		if metric.MinValue != 1.5 {
			t.Errorf("Expected min value 1.5, got %f", metric.MinValue)
		}

		if metric.MaxValue != 3.5 {
			t.Errorf("Expected max value 3.5, got %f", metric.MaxValue)
		}

		expectedAverage := 2.5 // (1.5 + 2.5 + 3.5) / 3
		if abs(metric.AverageValue-expectedAverage) > 1e-10 {
			t.Errorf("Expected average %f, got %f", expectedAverage, metric.AverageValue)
		}
	})

	t.Run("SystemMetrics", func(t *testing.T) {
		if err := monitor.Start(); err != nil {
			t.Fatalf("Failed to start monitor: %v", err)
		}
		defer monitor.Stop()

		time.Sleep(100 * time.Millisecond) // Let it collect some data

		metrics := monitor.GetSystemMetrics()
		if metrics == nil {
			t.Fatal("System metrics should not be nil")
		}

		// Check required fields
		if metrics.GoroutineCount <= 0 {
			t.Error("Goroutine count should be positive")
		}

		if metrics.HeapSize <= 0 {
			t.Error("Heap size should be positive")
		}
	})

	t.Run("RequestMetrics", func(t *testing.T) {
		// Record some request metrics
		monitor.RecordRequestMetrics(100.0, false) // 100ms, no error
		monitor.RecordRequestMetrics(200.0, false) // 200ms, no error
		monitor.RecordRequestMetrics(300.0, true)  // 300ms, with error

		errorRate := monitor.GetMetric("error_rate")
		if errorRate == nil {
			t.Fatal("Error rate metric should exist")
		}

		// Should be 33.33% (1 error out of 3 requests)
		expectedErrorRate := 100.0 / 3.0
		if abs(errorRate.CurrentValue-expectedErrorRate) > 1e-10 {
			t.Errorf("Expected error rate %f, got %f", expectedErrorRate, errorRate.CurrentValue)
		}
	})

	t.Run("HealthStatus", func(t *testing.T) {
		health := monitor.GetHealthStatus()
		if health == nil {
			t.Fatal("Health status should not be nil")
		}

		// Check required fields
		if _, exists := health["status"]; !exists {
			t.Error("Health status should include status")
		}

		if _, exists := health["score"]; !exists {
			t.Error("Health status should include score")
		}

		// Score should be between 0 and 100
		score := health["score"].(float64)
		if score < 0 || score > 100 {
			t.Errorf("Score should be between 0 and 100, got %f", score)
		}
	})
}

// TestEventBus tests the event system
func TestEventBus(t *testing.T) {
	eventBus := shared.NewEventBus()
	eventBus.Start()
	defer eventBus.Stop()

	t.Run("PublishSubscribe", func(t *testing.T) {
		received := false
		var receivedEvent shared.Event

		// Subscribe to events
		eventBus.Subscribe("test_event", func(event shared.Event) {
			received = true
			receivedEvent = event
		})

		// Publish event
		testEvent := shared.Event{
			Type:   "test_event",
			Source: "test",
			Data: map[string]interface{}{
				"message": "hello world",
			},
		}

		eventBus.PublishSync(testEvent)

		if !received {
			t.Error("Event should have been received")
		}

		if receivedEvent.Type != "test_event" {
			t.Errorf("Expected event type 'test_event', got '%s'", receivedEvent.Type)
		}

		if receivedEvent.Data["message"] != "hello world" {
			t.Errorf("Expected message 'hello world', got '%v'", receivedEvent.Data["message"])
		}
	})

	t.Run("WildcardSubscription", func(t *testing.T) {
		eventCount := 0

		// Subscribe to all events
		eventBus.Subscribe("*", func(event shared.Event) {
			eventCount++
		})

		// Publish different types of events
		eventBus.PublishSync(shared.Event{Type: "event1"})
		eventBus.PublishSync(shared.Event{Type: "event2"})
		eventBus.PublishSync(shared.Event{Type: "event3"})

		time.Sleep(10 * time.Millisecond) // Wait for processing

		if eventCount != 3 {
			t.Errorf("Expected 3 events, got %d", eventCount)
		}
	})

	t.Run("MultipleSubscribers", func(t *testing.T) {
		count1 := 0
		count2 := 0

		eventBus.Subscribe("multi_event", func(event shared.Event) {
			count1++
		})

		eventBus.Subscribe("multi_event", func(event shared.Event) {
			count2++
		})

		eventBus.PublishSync(shared.Event{Type: "multi_event"})

		time.Sleep(10 * time.Millisecond) // Wait for processing

		if count1 != 1 {
			t.Errorf("First subscriber should receive 1 event, got %d", count1)
		}

		if count2 != 1 {
			t.Errorf("Second subscriber should receive 1 event, got %d", count2)
		}
	})

	t.Run("EventBusStats", func(t *testing.T) {
		stats := eventBus.GetStats()
		if stats == nil {
			t.Fatal("Stats should not be nil")
		}

		// Check required fields
		if _, exists := stats["total_subscribers"]; !exists {
			t.Error("Stats should include total_subscribers")
		}

		if _, exists := stats["event_types"]; !exists {
			t.Error("Stats should include event_types")
		}

		if _, exists := stats["running"]; !exists {
			t.Error("Stats should include running")
		}
	})
}

// TestIntegration tests the integration between multiple components
func TestIntegration(t *testing.T) {
	t.Run("BasicEngineIntegration", func(t *testing.T) {
		// Test basic engine integration without complex manager
		srsEngine, err := srs.NewSRSEngine()
		if err != nil {
			t.Fatalf("Failed to create SRS engine: %v", err)
		}

		qsemEngine, err := qsem.NewQSEMEngine()
		if err != nil {
			t.Fatalf("Failed to create QSEM engine: %v", err)
		}

		// Test engine operations (simplified)
		formula := map[string]interface{}{
			"variables": 3,
			"clauses": []interface{}{
				[]interface{}{
					map[string]interface{}{"var": 1, "neg": false},
					map[string]interface{}{"var": 2, "neg": false},
				},
			},
		}

		result, _, err := srsEngine.SolveProblem("3sat", formula, nil)
		if err != nil {
			t.Fatalf("Failed to solve problem: %v", err)
		}

		if result == nil {
			t.Error("Result should not be nil")
		}

		// Basic test that engines exist
		if qsemEngine == nil {
			t.Fatal("QSEM engine should not be nil")
		}
	})
}

// Helper function for floating point comparison
func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}
