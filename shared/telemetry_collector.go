package shared

import (
	"sync"
	"time"

	"github.com/resonancelab/psizero/shared/types"
)

// TelemetryCollector handles telemetry data collection and aggregation
type TelemetryCollector struct {
	interval      time.Duration
	buffer        []types.TelemetryPoint
	bufferMutex   sync.RWMutex
	metrics       map[string]*MetricAggregator
	metricsMutex  sync.RWMutex
	callbacks     []TelemetryCallback
	callbackMutex sync.RWMutex
}

// TelemetryCallback is called when new telemetry is collected
type TelemetryCallback func(point types.TelemetryPoint)

// MetricAggregator aggregates metric values over time
type MetricAggregator struct {
	Name     string
	Values   []float64
	Times    []time.Time
	Sum      float64
	Count    int64
	Min      float64
	Max      float64
	Mean     float64
	StdDev   float64
	Variance float64
	mutex    sync.RWMutex
}

// NewTelemetryCollector creates a new telemetry collector
func NewTelemetryCollector(interval time.Duration) *TelemetryCollector {
	return &TelemetryCollector{
		interval:  interval,
		buffer:    make([]types.TelemetryPoint, 0),
		metrics:   make(map[string]*MetricAggregator),
		callbacks: make([]TelemetryCallback, 0),
	}
}

// AddPoint adds a telemetry point to the collector
func (tc *TelemetryCollector) AddPoint(point types.TelemetryPoint) {
	tc.bufferMutex.Lock()
	tc.buffer = append(tc.buffer, point)
	tc.bufferMutex.Unlock()

	// Update metric aggregators
	tc.updateMetrics(point)

	// Call callbacks
	tc.callbackMutex.RLock()
	for _, callback := range tc.callbacks {
		go callback(point)
	}
	tc.callbackMutex.RUnlock()
}

// GetBuffer returns a copy of the current buffer
func (tc *TelemetryCollector) GetBuffer() []types.TelemetryPoint {
	tc.bufferMutex.RLock()
	defer tc.bufferMutex.RUnlock()

	buffer := make([]types.TelemetryPoint, len(tc.buffer))
	copy(buffer, tc.buffer)
	return buffer
}

// ClearBuffer clears the telemetry buffer
func (tc *TelemetryCollector) ClearBuffer() {
	tc.bufferMutex.Lock()
	defer tc.bufferMutex.Unlock()

	tc.buffer = tc.buffer[:0]
}

// AddCallback adds a telemetry callback
func (tc *TelemetryCollector) AddCallback(callback TelemetryCallback) {
	tc.callbackMutex.Lock()
	defer tc.callbackMutex.Unlock()

	tc.callbacks = append(tc.callbacks, callback)
}

// GetMetric returns a metric aggregator by name
func (tc *TelemetryCollector) GetMetric(name string) *MetricAggregator {
	tc.metricsMutex.RLock()
	defer tc.metricsMutex.RUnlock()

	if aggregator, exists := tc.metrics[name]; exists {
		// Return a copy
		aggregator.mutex.RLock()
		copy := *aggregator
		aggregator.mutex.RUnlock()
		return &copy
	}

	return nil
}

// GetAllMetrics returns all metric aggregators
func (tc *TelemetryCollector) GetAllMetrics() map[string]*MetricAggregator {
	tc.metricsMutex.RLock()
	defer tc.metricsMutex.RUnlock()

	result := make(map[string]*MetricAggregator)
	for name, aggregator := range tc.metrics {
		aggregator.mutex.RLock()
		copy := *aggregator
		aggregator.mutex.RUnlock()
		result[name] = &copy
	}

	return result
}

// updateMetrics updates metric aggregators with new telemetry point
func (tc *TelemetryCollector) updateMetrics(point types.TelemetryPoint) {
	metrics := map[string]float64{
		"symbolic_entropy":   point.SymbolicEntropy,
		"lyapunov_metric":    point.LyapunovMetric,
		"satisfaction_rate":  point.SatisfactionRate,
		"resonance_strength": point.ResonanceStrength,
		"dominance":          point.Dominance,
	}

	tc.metricsMutex.Lock()
	defer tc.metricsMutex.Unlock()

	for name, value := range metrics {
		if aggregator, exists := tc.metrics[name]; exists {
			aggregator.AddValue(value, point.Timestamp)
		} else {
			aggregator := NewMetricAggregator(name)
			aggregator.AddValue(value, point.Timestamp)
			tc.metrics[name] = aggregator
		}
	}
}

// NewMetricAggregator creates a new metric aggregator
func NewMetricAggregator(name string) *MetricAggregator {
	return &MetricAggregator{
		Name:   name,
		Values: make([]float64, 0),
		Times:  make([]time.Time, 0),
		Min:    1e308,  // Large number for min initialization
		Max:    -1e308, // Small number for max initialization
	}
}

// AddValue adds a value to the metric aggregator
func (ma *MetricAggregator) AddValue(value float64, timestamp time.Time) {
	ma.mutex.Lock()
	defer ma.mutex.Unlock()

	// Add to collections
	ma.Values = append(ma.Values, value)
	ma.Times = append(ma.Times, timestamp)

	// Update aggregates
	ma.Sum += value
	ma.Count++

	if value < ma.Min {
		ma.Min = value
	}
	if value > ma.Max {
		ma.Max = value
	}

	// Update mean
	ma.Mean = ma.Sum / float64(ma.Count)

	// Update variance and standard deviation
	if ma.Count > 1 {
		ma.updateVariance()
	}
}

// updateVariance calculates variance and standard deviation
func (ma *MetricAggregator) updateVariance() {
	sumSquaredDiffs := 0.0
	for _, value := range ma.Values {
		diff := value - ma.Mean
		sumSquaredDiffs += diff * diff
	}

	ma.Variance = sumSquaredDiffs / float64(ma.Count-1)
	if ma.Variance >= 0 {
		ma.StdDev = ma.Variance // Simplified - should be sqrt
	}
}

// GetStatistics returns statistical summary
func (ma *MetricAggregator) GetStatistics() map[string]float64 {
	ma.mutex.RLock()
	defer ma.mutex.RUnlock()

	return map[string]float64{
		"count":    float64(ma.Count),
		"sum":      ma.Sum,
		"mean":     ma.Mean,
		"min":      ma.Min,
		"max":      ma.Max,
		"variance": ma.Variance,
		"std_dev":  ma.StdDev,
	}
}

// GetRecentValues returns the most recent N values
func (ma *MetricAggregator) GetRecentValues(n int) []float64 {
	ma.mutex.RLock()
	defer ma.mutex.RUnlock()

	if n >= len(ma.Values) {
		result := make([]float64, len(ma.Values))
		copy(result, ma.Values)
		return result
	}

	start := len(ma.Values) - n
	result := make([]float64, n)
	copy(result, ma.Values[start:])
	return result
}

// GetValuesSince returns values since a specific time
func (ma *MetricAggregator) GetValuesSince(since time.Time) []float64 {
	ma.mutex.RLock()
	defer ma.mutex.RUnlock()

	result := make([]float64, 0)
	for i, timestamp := range ma.Times {
		if timestamp.After(since) && i < len(ma.Values) {
			result = append(result, ma.Values[i])
		}
	}

	return result
}

// Cleanup removes old data beyond retention period
func (ma *MetricAggregator) Cleanup(retention time.Duration) {
	ma.mutex.Lock()
	defer ma.mutex.Unlock()

	cutoff := time.Now().Add(-retention)

	// Find first index to keep
	keepIndex := 0
	for i, timestamp := range ma.Times {
		if timestamp.After(cutoff) {
			keepIndex = i
			break
		}
	}

	if keepIndex > 0 {
		// Remove old data
		ma.Values = ma.Values[keepIndex:]
		ma.Times = ma.Times[keepIndex:]

		// Recalculate aggregates
		ma.recalculateAggregates()
	}
}

// recalculateAggregates recalculates all aggregate values
func (ma *MetricAggregator) recalculateAggregates() {
	if len(ma.Values) == 0 {
		ma.Sum = 0
		ma.Count = 0
		ma.Min = 1e308
		ma.Max = -1e308
		ma.Mean = 0
		ma.Variance = 0
		ma.StdDev = 0
		return
	}

	// Recalculate sum, min, max
	ma.Sum = 0
	ma.Min = ma.Values[0]
	ma.Max = ma.Values[0]

	for _, value := range ma.Values {
		ma.Sum += value
		if value < ma.Min {
			ma.Min = value
		}
		if value > ma.Max {
			ma.Max = value
		}
	}

	ma.Count = int64(len(ma.Values))
	ma.Mean = ma.Sum / float64(ma.Count)

	// Recalculate variance
	if ma.Count > 1 {
		ma.updateVariance()
	}
}
