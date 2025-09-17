package shared

import (
	"runtime"
	"sync"
	"time"
)

// PerformanceMonitor tracks system performance metrics
type PerformanceMonitor struct {
	metrics          map[string]*PerformanceMetric
	metricsMutex     sync.RWMutex
	samplingInterval time.Duration
	stopMonitoring   chan bool
	running          bool
	runMutex         sync.Mutex
}

// PerformanceMetric represents a performance metric
type PerformanceMetric struct {
	Name         string
	Values       []float64
	Timestamps   []time.Time
	CurrentValue float64
	MaxValue     float64
	MinValue     float64
	AverageValue float64
	mutex        sync.RWMutex
}

// SystemMetrics represents current system metrics
type SystemMetrics struct {
	CPUUsage         float64   `json:"cpu_usage"`
	MemoryUsage      float64   `json:"memory_usage"`
	GoroutineCount   int       `json:"goroutine_count"`
	GCPauseTime      float64   `json:"gc_pause_time"`
	HeapSize         uint64    `json:"heap_size"`
	AllocatedMemory  uint64    `json:"allocated_memory"`
	RequestsPerSecond float64  `json:"requests_per_second"`
	ResponseTime     float64   `json:"response_time"`
	ErrorRate        float64   `json:"error_rate"`
	Timestamp        time.Time `json:"timestamp"`
}

// NewPerformanceMonitor creates a new performance monitor
func NewPerformanceMonitor() *PerformanceMonitor {
	return &PerformanceMonitor{
		metrics:          make(map[string]*PerformanceMetric),
		samplingInterval: 5 * time.Second,
		stopMonitoring:   make(chan bool),
		running:          false,
	}
}

// Start starts the performance monitoring
func (pm *PerformanceMonitor) Start() error {
	pm.runMutex.Lock()
	defer pm.runMutex.Unlock()
	
	if pm.running {
		return nil
	}
	
	pm.running = true
	go pm.monitoringLoop()
	
	return nil
}

// Stop stops the performance monitoring
func (pm *PerformanceMonitor) Stop() {
	pm.runMutex.Lock()
	defer pm.runMutex.Unlock()
	
	if !pm.running {
		return
	}
	
	pm.running = false
	close(pm.stopMonitoring)
}

// GetSystemMetrics returns current system metrics
func (pm *PerformanceMonitor) GetSystemMetrics() *SystemMetrics {
	var memStats runtime.MemStats
	runtime.ReadMemStats(&memStats)
	
	return &SystemMetrics{
		CPUUsage:        pm.getCurrentCPUUsage(),
		MemoryUsage:     float64(memStats.Alloc) / float64(memStats.Sys) * 100,
		GoroutineCount:  runtime.NumGoroutine(),
		GCPauseTime:     float64(memStats.PauseNs[(memStats.NumGC+255)%256]) / 1e6, // Convert to ms
		HeapSize:        memStats.HeapSys,
		AllocatedMemory: memStats.Alloc,
		RequestsPerSecond: pm.getMetricValue("requests_per_second"),
		ResponseTime:    pm.getMetricValue("response_time"),
		ErrorRate:       pm.getMetricValue("error_rate"),
		Timestamp:       time.Now(),
	}
}

// RecordMetric records a performance metric
func (pm *PerformanceMonitor) RecordMetric(name string, value float64) {
	pm.metricsMutex.Lock()
	defer pm.metricsMutex.Unlock()
	
	metric, exists := pm.metrics[name]
	if !exists {
		metric = &PerformanceMetric{
			Name:     name,
			Values:   make([]float64, 0),
			Timestamps: make([]time.Time, 0),
			MinValue: value,
			MaxValue: value,
		}
		pm.metrics[name] = metric
	}
	
	metric.mutex.Lock()
	defer metric.mutex.Unlock()
	
	// Add new value
	metric.Values = append(metric.Values, value)
	metric.Timestamps = append(metric.Timestamps, time.Now())
	metric.CurrentValue = value
	
	// Update min/max
	if value < metric.MinValue {
		metric.MinValue = value
	}
	if value > metric.MaxValue {
		metric.MaxValue = value
	}
	
	// Calculate average
	sum := 0.0
	for _, v := range metric.Values {
		sum += v
	}
	metric.AverageValue = sum / float64(len(metric.Values))
	
	// Keep only recent values (last 1000)
	if len(metric.Values) > 1000 {
		metric.Values = metric.Values[len(metric.Values)-1000:]
		metric.Timestamps = metric.Timestamps[len(metric.Timestamps)-1000:]
	}
}

// GetMetric returns a specific metric
func (pm *PerformanceMonitor) GetMetric(name string) *PerformanceMetric {
	pm.metricsMutex.RLock()
	defer pm.metricsMutex.RUnlock()
	
	if metric, exists := pm.metrics[name]; exists {
		metric.mutex.RLock()
		defer metric.mutex.RUnlock()
		
		// Return a copy
		copy := *metric
		copy.Values = make([]float64, len(metric.Values))
		copy.Timestamps = make([]time.Time, len(metric.Timestamps))
		copy(copy.Values, metric.Values)
		copy(copy.Timestamps, metric.Timestamps)
		
		return &copy
	}
	
	return nil
}

// GetAllMetrics returns all performance metrics
func (pm *PerformanceMonitor) GetAllMetrics() map[string]*PerformanceMetric {
	pm.metricsMutex.RLock()
	defer pm.metricsMutex.RUnlock()
	
	result := make(map[string]*PerformanceMetric)
	for name, metric := range pm.metrics {
		metric.mutex.RLock()
		
		// Return a copy
		copy := *metric
		copy.Values = make([]float64, len(metric.Values))
		copy.Timestamps = make([]time.Time, len(metric.Timestamps))
		copy(copy.Values, metric.Values)
		copy(copy.Timestamps, metric.Timestamps)
		
		result[name] = &copy
		metric.mutex.RUnlock()
	}
	
	return result
}

// getMetricValue returns the current value of a metric
func (pm *PerformanceMonitor) getMetricValue(name string) float64 {
	pm.metricsMutex.RLock()
	defer pm.metricsMutex.RUnlock()
	
	if metric, exists := pm.metrics[name]; exists {
		metric.mutex.RLock()
		defer metric.mutex.RUnlock()
		return metric.CurrentValue
	}
	
	return 0.0
}

// monitoringLoop runs the continuous monitoring loop
func (pm *PerformanceMonitor) monitoringLoop() {
	ticker := time.NewTicker(pm.samplingInterval)
	defer ticker.Stop()
	
	for {
		select {
		case <-ticker.C:
			pm.collectSystemMetrics()
		case <-pm.stopMonitoring:
			return
		}
	}
}

// collectSystemMetrics collects current system metrics
func (pm *PerformanceMonitor) collectSystemMetrics() {
	var memStats runtime.MemStats
	runtime.ReadMemStats(&memStats)
	
	// Record memory metrics
	pm.RecordMetric("memory_alloc", float64(memStats.Alloc))
	pm.RecordMetric("memory_sys", float64(memStats.Sys))
	pm.RecordMetric("memory_heap_alloc", float64(memStats.HeapAlloc))
	pm.RecordMetric("memory_heap_sys", float64(memStats.HeapSys))
	pm.RecordMetric("memory_heap_idle", float64(memStats.HeapIdle))
	pm.RecordMetric("memory_heap_inuse", float64(memStats.HeapInuse))
	pm.RecordMetric("memory_heap_released", float64(memStats.HeapReleased))
	pm.RecordMetric("memory_heap_objects", float64(memStats.HeapObjects))
	
	// Record GC metrics
	pm.RecordMetric("gc_num", float64(memStats.NumGC))
	pm.RecordMetric("gc_pause_total", float64(memStats.PauseTotalNs)/1e6) // Convert to ms
	if memStats.NumGC > 0 {
		pm.RecordMetric("gc_pause_recent", float64(memStats.PauseNs[(memStats.NumGC+255)%256])/1e6)
	}
	
	// Record goroutine count
	pm.RecordMetric("goroutines", float64(runtime.NumGoroutine()))
	
	// Record CPU count
	pm.RecordMetric("cpu_count", float64(runtime.NumCPU()))
}

// getCurrentCPUUsage estimates current CPU usage (simplified)
func (pm *PerformanceMonitor) getCurrentCPUUsage() float64 {
	// This is a simplified CPU usage calculation
	// In a real implementation, you would use platform-specific APIs
	
	goroutines := float64(runtime.NumGoroutine())
	cpus := float64(runtime.NumCPU())
	
	// Simple estimation based on goroutine count
	usage := (goroutines / (cpus * 10)) * 100
	if usage > 100 {
		usage = 100
	}
	
	return usage
}

// GetPerformanceSummary returns a performance summary
func (pm *PerformanceMonitor) GetPerformanceSummary() map[string]interface{} {
	systemMetrics := pm.GetSystemMetrics()
	
	summary := map[string]interface{}{
		"system_metrics": systemMetrics,
		"uptime":        time.Since(time.Now().Add(-time.Hour)), // Placeholder
		"performance_score": pm.calculatePerformanceScore(systemMetrics),
	}
	
	// Add metric summaries
	allMetrics := pm.GetAllMetrics()
	metricSummaries := make(map[string]map[string]float64)
	
	for name, metric := range allMetrics {
		metricSummaries[name] = map[string]float64{
			"current": metric.CurrentValue,
			"min":     metric.MinValue,
			"max":     metric.MaxValue,
			"average": metric.AverageValue,
		}
	}
	
	summary["metric_summaries"] = metricSummaries
	
	return summary
}

// calculatePerformanceScore calculates an overall performance score
func (pm *PerformanceMonitor) calculatePerformanceScore(metrics *SystemMetrics) float64 {
	// Calculate a score from 0-100 based on various metrics
	score := 100.0
	
	// Penalize high CPU usage
	if metrics.CPUUsage > 80 {
		score -= (metrics.CPUUsage - 80) * 2
	}
	
	// Penalize high memory usage
	if metrics.MemoryUsage > 80 {
		score -= (metrics.MemoryUsage - 80) * 2
	}
	
	// Penalize high error rate
	if metrics.ErrorRate > 1 {
		score -= metrics.ErrorRate * 10
	}
	
	// Penalize slow response times
	if metrics.ResponseTime > 1000 { // 1 second
		score -= (metrics.ResponseTime - 1000) / 100
	}
	
	// Penalize excessive goroutines
	if metrics.GoroutineCount > 1000 {
		score -= float64(metrics.GoroutineCount-1000) / 100
	}
	
	if score < 0 {
		score = 0
	}
	
	return score
}

// RecordRequestMetrics records HTTP request metrics
func (pm *PerformanceMonitor) RecordRequestMetrics(responseTime float64, isError bool) {
	pm.RecordMetric("response_time", responseTime)
	
	// Update requests per second
	current := pm.getMetricValue("requests_total")
	pm.RecordMetric("requests_total", current+1)
	
	// Update error rate
	if isError {
		currentErrors := pm.getMetricValue("errors_total")
		pm.RecordMetric("errors_total", currentErrors+1)
	}
	
	// Calculate rates (simplified)
	requestsTotal := pm.getMetricValue("requests_total")
	errorsTotal := pm.getMetricValue("errors_total")
	
	if requestsTotal > 0 {
		errorRate := (errorsTotal / requestsTotal) * 100
		pm.RecordMetric("error_rate", errorRate)
	}
	
	// Calculate requests per second (over last minute)
	pm.RecordMetric("requests_per_second", pm.calculateRPS())
}

// calculateRPS calculates requests per second over the last minute
func (pm *PerformanceMonitor) calculateRPS() float64 {
	metric := pm.GetMetric("requests_total")
	if metric == nil || len(metric.Timestamps) < 2 {
		return 0
	}
	
	now := time.Now()
	oneMinuteAgo := now.Add(-time.Minute)
	
	// Count requests in the last minute
	requests := 0
	for i, timestamp := range metric.Timestamps {
		if timestamp.After(oneMinuteAgo) && i < len(metric.Values) {
			requests++
		}
	}
	
	return float64(requests) / 60.0 // Requests per second
}

// GetHealthStatus returns the system health status
func (pm *PerformanceMonitor) GetHealthStatus() map[string]interface{} {
	metrics := pm.GetSystemMetrics()
	score := pm.calculatePerformanceScore(metrics)
	
	status := "healthy"
	if score < 50 {
		status = "unhealthy"
	} else if score < 80 {
		status = "degraded"
	}
	
	return map[string]interface{}{
		"status":           status,
		"score":            score,
		"cpu_usage":        metrics.CPUUsage,
		"memory_usage":     metrics.MemoryUsage,
		"response_time":    metrics.ResponseTime,
		"error_rate":       metrics.ErrorRate,
		"goroutine_count":  metrics.GoroutineCount,
		"timestamp":        metrics.Timestamp,
	}
}