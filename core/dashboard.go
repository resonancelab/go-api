package core

import (
	"encoding/json"
	"fmt"
	"html/template"
	"net/http"
	"sort"
	"strings"
	"sync"
	"time"
)

// DashboardServer provides a web-based performance monitoring dashboard
type DashboardServer struct {
	telemetryCollector *TelemetryCollector
	latencyMitigator   *LatencyMitigator
	server             *http.Server
	port               int
	templates          *template.Template
	dataCache          *DashboardDataCache
	mu                 sync.RWMutex
}

// DashboardDataCache caches dashboard data to reduce computation overhead
type DashboardDataCache struct {
	GlobalMetrics      *GlobalMetrics                  `json:"global_metrics"`
	NodeMetrics        map[string]TelemetryNodeMetrics `json:"node_metrics"`
	TrendAnalysis      *TrendAnalysis                  `json:"trend_analysis"`
	AlertStatus        *AlertStatus                    `json:"alert_status"`
	PerformanceSummary *PerformanceSummary             `json:"performance_summary"`
	ChartsData         *ChartsData                     `json:"charts_data"`
	LastUpdate         time.Time                       `json:"last_update"`
	CacheDuration      time.Duration                   `json:"cache_duration"`
}

// ChartsData contains data formatted for chart visualization
type ChartsData struct {
	CoherenceOverTime        []TimeSeriesPoint     `json:"coherence_over_time"`
	ResonanceOverTime        []TimeSeriesPoint     `json:"resonance_over_time"`
	PerformanceOverTime      []TimeSeriesPoint     `json:"performance_over_time"`
	NetworkLatencyOverTime   []TimeSeriesPoint     `json:"network_latency_over_time"`
	EconomicActivityOverTime []TimeSeriesPoint     `json:"economic_activity_over_time"`
	NodePerformance          []NodePerformanceData `json:"node_performance"`
	AlertTimeline            []AlertTimelinePoint  `json:"alert_timeline"`
	SystemHealth             *SystemHealthData     `json:"system_health"`
}

// TimeSeriesPoint represents a data point for time series charts
type TimeSeriesPoint struct {
	Timestamp time.Time `json:"timestamp"`
	Value     float64   `json:"value"`
	NodeID    string    `json:"node_id,omitempty"`
	Label     string    `json:"label,omitempty"`
}

// NodePerformanceData contains performance data for a specific node
type NodePerformanceData struct {
	NodeID             string  `json:"node_id"`
	Coherence          float64 `json:"coherence"`
	Resonance          float64 `json:"resonance"`
	PerformanceScore   float64 `json:"performance_score"`
	NetworkHealth      float64 `json:"network_health"`
	EconomicEfficiency float64 `json:"economic_efficiency"`
	Status             string  `json:"status"`
}

// AlertTimelinePoint represents an alert for timeline visualization
type AlertTimelinePoint struct {
	Timestamp time.Time `json:"timestamp"`
	Type      string    `json:"type"`
	Message   string    `json:"message"`
	NodeID    string    `json:"node_id,omitempty"`
	Severity  string    `json:"severity"`
}

// SystemHealthData contains overall system health metrics
type SystemHealthData struct {
	OverallHealth   float64            `json:"overall_health"`
	ComponentHealth map[string]float64 `json:"component_health"`
	CriticalIssues  int                `json:"critical_issues"`
	WarningIssues   int                `json:"warning_issues"`
	ActiveNodes     int                `json:"active_nodes"`
	SystemLoad      float64            `json:"system_load"`
	Uptime          string             `json:"uptime"`
}

// DashboardConfig contains configuration for the dashboard server
type DashboardConfig struct {
	Port            int           `json:"port"`
	CacheDuration   time.Duration `json:"cache_duration"`
	UpdateInterval  time.Duration `json:"update_interval"`
	EnableWebSocket bool          `json:"enable_websocket"`
	MaxConnections  int           `json:"max_connections"`
}

// NewDashboardServer creates a new dashboard server
func NewDashboardServer(telemetryCollector *TelemetryCollector, latencyMitigator *LatencyMitigator, config *DashboardConfig) *DashboardServer {
	if config == nil {
		config = &DashboardConfig{
			Port:            8080,
			CacheDuration:   30 * time.Second,
			UpdateInterval:  10 * time.Second,
			EnableWebSocket: true,
			MaxConnections:  100,
		}
	}

	ds := &DashboardServer{
		telemetryCollector: telemetryCollector,
		latencyMitigator:   latencyMitigator,
		port:               config.Port,
		dataCache: &DashboardDataCache{
			CacheDuration: config.CacheDuration,
		},
	}

	// Initialize templates (simplified for this implementation)
	ds.templates = template.New("dashboard")

	// Set up HTTP routes
	mux := http.NewServeMux()
	mux.HandleFunc("/", ds.handleDashboard)
	mux.HandleFunc("/api/metrics", ds.handleMetricsAPI)
	mux.HandleFunc("/api/charts", ds.handleChartsAPI)
	mux.HandleFunc("/api/alerts", ds.handleAlertsAPI)
	mux.HandleFunc("/api/nodes", ds.handleNodesAPI)
	mux.HandleFunc("/api/health", ds.handleHealthAPI)
	mux.HandleFunc("/static/", ds.handleStatic)

	ds.server = &http.Server{
		Addr:    fmt.Sprintf(":%d", config.Port),
		Handler: mux,
	}

	return ds
}

// Start starts the dashboard server
func (ds *DashboardServer) Start() error {
	fmt.Printf("Starting Reson.net Dashboard Server on port %d\n", ds.port)
	fmt.Printf("Dashboard available at: http://localhost:%d\n", ds.port)

	// Start background update routine
	go ds.backgroundUpdateRoutine()

	return ds.server.ListenAndServe()
}

// Stop stops the dashboard server
func (ds *DashboardServer) Stop() error {
	if ds.server != nil {
		return ds.server.Close()
	}
	return nil
}

// backgroundUpdateRoutine periodically updates the dashboard cache
func (ds *DashboardServer) backgroundUpdateRoutine() {
	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			ds.updateCache()
		}
	}
}

// updateCache updates the dashboard data cache
func (ds *DashboardServer) updateCache() {
	ds.mu.Lock()
	defer ds.mu.Unlock()

	// Generate fresh telemetry report
	report := ds.telemetryCollector.GenerateTelemetryReport()

	// Update cache
	ds.dataCache.GlobalMetrics = &report.GlobalMetrics
	ds.dataCache.NodeMetrics = report.NodeMetrics
	ds.dataCache.TrendAnalysis = &report.TrendAnalysis
	ds.dataCache.AlertStatus = &report.AlertStatus
	ds.dataCache.PerformanceSummary = &report.PerformanceSummary
	ds.dataCache.ChartsData = ds.generateChartsData(report)
	ds.dataCache.LastUpdate = time.Now()
}

// generateChartsData generates chart data from telemetry report
func (ds *DashboardServer) generateChartsData(report *TelemetryReport) *ChartsData {
	chartsData := &ChartsData{
		CoherenceOverTime:        ds.generateCoherenceTimeSeries(),
		ResonanceOverTime:        ds.generateResonanceTimeSeries(),
		PerformanceOverTime:      ds.generatePerformanceTimeSeries(),
		NetworkLatencyOverTime:   ds.generateNetworkLatencyTimeSeries(),
		EconomicActivityOverTime: ds.generateEconomicTimeSeries(),
		NodePerformance:          ds.generateNodePerformanceData(report),
		AlertTimeline:            ds.generateAlertTimeline(report),
		SystemHealth:             ds.generateSystemHealthData(report),
	}

	return chartsData
}

// generateCoherenceTimeSeries generates coherence time series data
func (ds *DashboardServer) generateCoherenceTimeSeries() []TimeSeriesPoint {
	points := make([]TimeSeriesPoint, 0)

	// Get recent coherence metrics
	for _, metric := range ds.telemetryCollector.CoherenceMetrics {
		if time.Since(metric.Timestamp) < time.Hour {
			points = append(points, TimeSeriesPoint{
				Timestamp: metric.Timestamp,
				Value:     metric.GlobalCoherence,
				NodeID:    metric.NodeID,
				Label:     "Global Coherence",
			})
		}
	}

	// Sort by timestamp
	sort.Slice(points, func(i, j int) bool {
		return points[i].Timestamp.Before(points[j].Timestamp)
	})

	return points
}

// generateResonanceTimeSeries generates resonance time series data
func (ds *DashboardServer) generateResonanceTimeSeries() []TimeSeriesPoint {
	points := make([]TimeSeriesPoint, 0)

	// Get recent resonance metrics
	for _, metric := range ds.telemetryCollector.ResonanceMetrics {
		if time.Since(metric.Timestamp) < time.Hour {
			points = append(points, TimeSeriesPoint{
				Timestamp: metric.Timestamp,
				Value:     metric.PrimeResonance,
				NodeID:    metric.NodeID,
				Label:     "Prime Resonance",
			})
		}
	}

	sort.Slice(points, func(i, j int) bool {
		return points[i].Timestamp.Before(points[j].Timestamp)
	})

	return points
}

// generatePerformanceTimeSeries generates performance time series data
func (ds *DashboardServer) generatePerformanceTimeSeries() []TimeSeriesPoint {
	points := make([]TimeSeriesPoint, 0)

	// Get recent performance metrics
	for _, metric := range ds.telemetryCollector.PerformanceMetrics {
		if time.Since(metric.Timestamp) < time.Hour {
			points = append(points, TimeSeriesPoint{
				Timestamp: metric.Timestamp,
				Value:     metric.OperationsPerSec,
				NodeID:    metric.NodeID,
				Label:     "Operations/sec",
			})
		}
	}

	sort.Slice(points, func(i, j int) bool {
		return points[i].Timestamp.Before(points[j].Timestamp)
	})

	return points
}

// generateNetworkLatencyTimeSeries generates network latency time series data
func (ds *DashboardServer) generateNetworkLatencyTimeSeries() []TimeSeriesPoint {
	points := make([]TimeSeriesPoint, 0)

	// Get recent network metrics
	for _, metric := range ds.telemetryCollector.NetworkMetrics {
		if time.Since(metric.Timestamp) < time.Hour {
			points = append(points, TimeSeriesPoint{
				Timestamp: metric.Timestamp,
				Value:     float64(metric.NetworkLatency.Nanoseconds()) / 1e6, // Convert to milliseconds
				NodeID:    metric.NodeID,
				Label:     "Network Latency (ms)",
			})
		}
	}

	sort.Slice(points, func(i, j int) bool {
		return points[i].Timestamp.Before(points[j].Timestamp)
	})

	return points
}

// generateEconomicTimeSeries generates economic activity time series data
func (ds *DashboardServer) generateEconomicTimeSeries() []TimeSeriesPoint {
	points := make([]TimeSeriesPoint, 0)

	// Get recent economic metrics
	for _, metric := range ds.telemetryCollector.EconomicMetrics {
		if time.Since(metric.Timestamp) < time.Hour {
			points = append(points, TimeSeriesPoint{
				Timestamp: metric.Timestamp,
				Value:     metric.TransactionRate,
				NodeID:    metric.NodeID,
				Label:     "Transaction Rate",
			})
		}
	}

	sort.Slice(points, func(i, j int) bool {
		return points[i].Timestamp.Before(points[j].Timestamp)
	})

	return points
}

// generateNodePerformanceData generates node performance data
func (ds *DashboardServer) generateNodePerformanceData(report *TelemetryReport) []NodePerformanceData {
	nodeData := make([]NodePerformanceData, 0, len(report.NodeMetrics))

	for nodeID, metrics := range report.NodeMetrics {
		status := "healthy"
		if metrics.PerformanceScore < 0.5 {
			status = "warning"
		}
		if metrics.PerformanceScore < 0.3 {
			status = "critical"
		}

		nodeData = append(nodeData, NodePerformanceData{
			NodeID:             nodeID,
			Coherence:          metrics.AverageCoherence,
			Resonance:          metrics.AverageResonance,
			PerformanceScore:   metrics.PerformanceScore,
			NetworkHealth:      metrics.NetworkHealth,
			EconomicEfficiency: metrics.EconomicEfficiency,
			Status:             status,
		})
	}

	return nodeData
}

// generateAlertTimeline generates alert timeline data
func (ds *DashboardServer) generateAlertTimeline(report *TelemetryReport) []AlertTimelinePoint {
	alerts := make([]AlertTimelinePoint, 0, len(report.AlertStatus.ActiveAlerts))

	for _, alert := range report.AlertStatus.ActiveAlerts {
		severity := "info"
		if alert.Type == "critical" {
			severity = "critical"
		} else if alert.Type == "warning" {
			severity = "warning"
		}

		alerts = append(alerts, AlertTimelinePoint{
			Timestamp: alert.Timestamp,
			Type:      alert.Type,
			Message:   alert.Message,
			NodeID:    alert.NodeID,
			Severity:  severity,
		})
	}

	// Sort by timestamp (most recent first)
	sort.Slice(alerts, func(i, j int) bool {
		return alerts[i].Timestamp.After(alerts[j].Timestamp)
	})

	return alerts
}

// generateSystemHealthData generates system health data
func (ds *DashboardServer) generateSystemHealthData(report *TelemetryReport) *SystemHealthData {
	health := &SystemHealthData{
		ComponentHealth: make(map[string]float64),
		CriticalIssues:  report.AlertStatus.CriticalCount,
		WarningIssues:   report.AlertStatus.WarningCount,
		ActiveNodes:     report.GlobalMetrics.ActiveNodes,
		Uptime:          "99.9%", // Simplified
	}

	// Calculate component health scores
	health.ComponentHealth["coherence"] = report.GlobalMetrics.AverageCoherence
	health.ComponentHealth["performance"] = report.GlobalMetrics.SystemPerformance / 1000000.0 // Normalize
	health.ComponentHealth["network"] = report.GlobalMetrics.NetworkHealth
	health.ComponentHealth["economic"] = report.GlobalMetrics.EconomicActivity / 100.0 // Normalize

	// Calculate overall health as weighted average
	weights := map[string]float64{
		"coherence":   0.3,
		"performance": 0.3,
		"network":     0.2,
		"economic":    0.2,
	}

	overallHealth := 0.0
	totalWeight := 0.0
	for component, score := range health.ComponentHealth {
		weight := weights[component]
		overallHealth += score * weight
		totalWeight += weight
	}

	if totalWeight > 0 {
		health.OverallHealth = overallHealth / totalWeight
	}

	// Calculate system load (simplified)
	health.SystemLoad = report.GlobalMetrics.SystemPerformance / 1000000.0

	return health
}

// HTTP Handlers

// handleDashboard serves the main dashboard page
func (ds *DashboardServer) handleDashboard(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/html")

	html := ds.generateDashboardHTML()
	w.Write([]byte(html))
}

// handleMetricsAPI serves metrics data as JSON
func (ds *DashboardServer) handleMetricsAPI(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Access-Control-Allow-Origin", "*")

	ds.mu.RLock()
	defer ds.mu.RUnlock()

	if ds.dataCache.GlobalMetrics == nil {
		http.Error(w, "No data available", http.StatusServiceUnavailable)
		return
	}

	json.NewEncoder(w).Encode(ds.dataCache)
}

// handleChartsAPI serves chart data as JSON
func (ds *DashboardServer) handleChartsAPI(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Access-Control-Allow-Origin", "*")

	ds.mu.RLock()
	defer ds.mu.RUnlock()

	if ds.dataCache.ChartsData == nil {
		http.Error(w, "No chart data available", http.StatusServiceUnavailable)
		return
	}

	json.NewEncoder(w).Encode(ds.dataCache.ChartsData)
}

// handleAlertsAPI serves alert data as JSON
func (ds *DashboardServer) handleAlertsAPI(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Access-Control-Allow-Origin", "*")

	ds.mu.RLock()
	defer ds.mu.RUnlock()

	if ds.dataCache.AlertStatus == nil {
		http.Error(w, "No alert data available", http.StatusServiceUnavailable)
		return
	}

	json.NewEncoder(w).Encode(ds.dataCache.AlertStatus)
}

// handleNodesAPI serves node-specific data as JSON
func (ds *DashboardServer) handleNodesAPI(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Access-Control-Allow-Origin", "*")

	nodeID := r.URL.Query().Get("node_id")
	if nodeID == "" {
		// Return all nodes
		ds.mu.RLock()
		defer ds.mu.RUnlock()

		if ds.dataCache.NodeMetrics == nil {
			http.Error(w, "No node data available", http.StatusServiceUnavailable)
			return
		}

		json.NewEncoder(w).Encode(ds.dataCache.NodeMetrics)
		return
	}

	// Return specific node
	metrics, exists := ds.telemetryCollector.GetNodeMetrics(nodeID)
	if !exists {
		http.Error(w, "Node not found", http.StatusNotFound)
		return
	}

	json.NewEncoder(w).Encode(metrics)
}

// handleHealthAPI serves system health data as JSON
func (ds *DashboardServer) handleHealthAPI(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Access-Control-Allow-Origin", "*")

	ds.mu.RLock()
	defer ds.mu.RUnlock()

	if ds.dataCache.ChartsData == nil || ds.dataCache.ChartsData.SystemHealth == nil {
		http.Error(w, "No health data available", http.StatusServiceUnavailable)
		return
	}

	json.NewEncoder(w).Encode(ds.dataCache.ChartsData.SystemHealth)
}

// handleStatic serves static files (CSS, JS, etc.)
func (ds *DashboardServer) handleStatic(w http.ResponseWriter, r *http.Request) {
	// Simplified static file serving
	path := strings.TrimPrefix(r.URL.Path, "/static/")

	switch path {
	case "style.css":
		w.Header().Set("Content-Type", "text/css")
		w.Write([]byte(ds.generateCSS()))
	case "script.js":
		w.Header().Set("Content-Type", "application/javascript")
		w.Write([]byte(ds.generateJavaScript()))
	default:
		http.NotFound(w, r)
	}
}

// generateDashboardHTML generates the main dashboard HTML
func (ds *DashboardServer) generateDashboardHTML() string {
	return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Reson.net Performance Dashboard</title>
    <link rel="stylesheet" href="/static/style.css">
</head>
<body>
    <div class="dashboard">
        <header class="dashboard-header">
            <h1>🔬 Reson.net Performance Dashboard</h1>
            <div class="system-status">
                <span id="system-health" class="status-indicator">Loading...</span>
                <span id="active-nodes" class="node-count">Nodes: --</span>
                <span id="last-update" class="last-update">Last update: --</span>
            </div>
        </header>

        <div class="dashboard-grid">
            <!-- Global Metrics -->
            <div class="metric-card">
                <h3>Global Coherence</h3>
                <div class="metric-value" id="global-coherence">--</div>
                <div class="metric-trend" id="coherence-trend">↗️ +0.00%</div>
            </div>

            <div class="metric-card">
                <h3>System Performance</h3>
                <div class="metric-value" id="system-performance">--</div>
                <div class="metric-trend" id="performance-trend">↗️ +0.00%</div>
            </div>

            <div class="metric-card">
                <h3>Network Health</h3>
                <div class="metric-value" id="network-health">--</div>
                <div class="metric-trend" id="network-trend">↗️ +0.00%</div>
            </div>

            <div class="metric-card">
                <h3>Economic Activity</h3>
                <div class="metric-value" id="economic-activity">--</div>
                <div class="metric-trend" id="economic-trend">↗️ +0.00%</div>
            </div>
        </div>

        <!-- Charts Section -->
        <div class="charts-section">
            <div class="chart-container">
                <h3>Coherence Over Time</h3>
                <canvas id="coherence-chart" width="400" height="200"></canvas>
            </div>

            <div class="chart-container">
                <h3>Performance Over Time</h3>
                <canvas id="performance-chart" width="400" height="200"></canvas>
            </div>

            <div class="chart-container">
                <h3>Network Latency</h3>
                <canvas id="network-chart" width="400" height="200"></canvas>
            </div>
        </div>

        <!-- Node Status -->
        <div class="node-status-section">
            <h3>Node Status</h3>
            <div id="node-list" class="node-list">
                <!-- Node items will be populated by JavaScript -->
            </div>
        </div>

        <!-- Alerts -->
        <div class="alerts-section">
            <h3>Active Alerts</h3>
            <div id="alert-list" class="alert-list">
                <!-- Alert items will be populated by JavaScript -->
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="/static/script.js"></script>
</body>
</html>`
}

// generateCSS generates dashboard CSS
func (ds *DashboardServer) generateCSS() string {
	return `
.dashboard {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    margin: 0;
    padding: 20px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    min-height: 100vh;
    color: #333;
}

.dashboard-header {
    background: rgba(255, 255, 255, 0.95);
    padding: 20px;
    border-radius: 10px;
    margin-bottom: 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.dashboard-header h1 {
    margin: 0;
    color: #2c3e50;
}

.system-status {
    display: flex;
    gap: 20px;
    align-items: center;
}

.status-indicator {
    padding: 5px 15px;
    border-radius: 20px;
    font-weight: bold;
}

.status-indicator.healthy { background: #27ae60; color: white; }
.status-indicator.warning { background: #f39c12; color: white; }
.status-indicator.critical { background: #e74c3c; color: white; }

.dashboard-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 20px;
    margin-bottom: 30px;
}

.metric-card {
    background: rgba(255, 255, 255, 0.95);
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    text-align: center;
}

.metric-card h3 {
    margin: 0 0 15px 0;
    color: #2c3e50;
    font-size: 1.1em;
}

.metric-value {
    font-size: 2.5em;
    font-weight: bold;
    color: #3498db;
    margin-bottom: 10px;
}

.metric-trend {
    font-size: 0.9em;
    color: #27ae60;
}

.charts-section {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
    gap: 20px;
    margin-bottom: 30px;
}

.chart-container {
    background: rgba(255, 255, 255, 0.95);
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.chart-container h3 {
    margin: 0 0 15px 0;
    color: #2c3e50;
    text-align: center;
}

.node-status-section, .alerts-section {
    background: rgba(255, 255, 255, 0.95);
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    margin-bottom: 20px;
}

.node-status-section h3, .alerts-section h3 {
    margin: 0 0 15px 0;
    color: #2c3e50;
}

.node-list {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 10px;
}

.node-item {
    padding: 10px;
    border-radius: 5px;
    background: #f8f9fa;
    border-left: 4px solid #3498db;
}

.node-item.healthy { border-left-color: #27ae60; }
.node-item.warning { border-left-color: #f39c12; }
.node-item.critical { border-left-color: #e74c3c; }

.alert-list {
    max-height: 200px;
    overflow-y: auto;
}

.alert-item {
    padding: 10px;
    margin-bottom: 5px;
    border-radius: 5px;
    border-left: 4px solid;
}

.alert-item.critical { border-left-color: #e74c3c; background: #fdf2f2; }
.alert-item.warning { border-left-color: #f39c12; background: #fdf9f2; }
.alert-item.info { border-left-color: #3498db; background: #f2f9fd; }

@media (max-width: 768px) {
    .dashboard-grid, .charts-section {
        grid-template-columns: 1fr;
    }

    .dashboard-header {
        flex-direction: column;
        gap: 15px;
        text-align: center;
    }
}
`
}

// generateJavaScript generates dashboard JavaScript
func (ds *DashboardServer) generateJavaScript() string {
	return `
// Dashboard JavaScript for real-time updates
let coherenceChart, performanceChart, networkChart;
let lastUpdate = new Date();

function initDashboard() {
    initCharts();
    updateDashboard();
    setInterval(updateDashboard, 10000); // Update every 10 seconds
}

function initCharts() {
    const coherenceCtx = document.getElementById('coherence-chart').getContext('2d');
    coherenceChart = new Chart(coherenceCtx, {
        type: 'line',
        data: {
            datasets: [{
                label: 'Global Coherence',
                borderColor: '#3498db',
                backgroundColor: 'rgba(52, 152, 219, 0.1)',
                data: []
            }]
        },
        options: {
            responsive: true,
            scales: {
                x: { type: 'time', time: { unit: 'minute' } },
                y: { min: 0, max: 1 }
            }
        }
    });

    const performanceCtx = document.getElementById('performance-chart').getContext('2d');
    performanceChart = new Chart(performanceCtx, {
        type: 'line',
        data: {
            datasets: [{
                label: 'Operations/sec',
                borderColor: '#27ae60',
                backgroundColor: 'rgba(39, 174, 96, 0.1)',
                data: []
            }]
        },
        options: {
            responsive: true,
            scales: {
                x: { type: 'time', time: { unit: 'minute' } },
                y: { beginAtZero: true }
            }
        }
    });

    const networkCtx = document.getElementById('network-chart').getContext('2d');
    networkChart = new Chart(networkCtx, {
        type: 'line',
        data: {
            datasets: [{
                label: 'Network Latency (ms)',
                borderColor: '#e74c3c',
                backgroundColor: 'rgba(231, 76, 60, 0.1)',
                data: []
            }]
        },
        options: {
            responsive: true,
            scales: {
                x: { type: 'time', time: { unit: 'minute' } },
                y: { beginAtZero: true }
            }
        }
    });
}

async function updateDashboard() {
    try {
        const [metricsResponse, chartsResponse] = await Promise.all([
            fetch('/api/metrics'),
            fetch('/api/charts')
        ]);

        if (!metricsResponse.ok || !chartsResponse.ok) {
            console.error('Failed to fetch dashboard data');
            return;
        }

        const metrics = await metricsResponse.json();
        const charts = await chartsResponse.json();

        updateGlobalMetrics(metrics.global_metrics);
        updateCharts(charts);
        updateNodeList(metrics.node_metrics);
        updateAlerts(metrics.alert_status);
        updateLastUpdate(metrics.last_update);

    } catch (error) {
        console.error('Error updating dashboard:', error);
    }
}

function updateGlobalMetrics(globalMetrics) {
    document.getElementById('global-coherence').textContent =
        globalMetrics.average_coherence ? globalMetrics.average_coherence.toFixed(3) : '--';

    document.getElementById('system-performance').textContent =
        globalMetrics.system_performance ? (globalMetrics.system_performance / 1000000).toFixed(1) + 'M' : '--';

    document.getElementById('network-health').textContent =
        globalMetrics.network_health ? globalMetrics.network_health.toFixed(3) : '--';

    document.getElementById('economic-activity').textContent =
        globalMetrics.economic_activity ? globalMetrics.economic_activity.toFixed(1) : '--';

    document.getElementById('active-nodes').textContent =
        'Nodes: ' + (globalMetrics.active_nodes || 0);

    // Update system health indicator
    const healthElement = document.getElementById('system-health');
    const coherence = globalMetrics.average_coherence || 0;
    if (coherence > 0.8) {
        healthElement.textContent = 'Healthy';
        healthElement.className = 'status-indicator healthy';
    } else if (coherence > 0.6) {
        healthElement.textContent = 'Warning';
        healthElement.className = 'status-indicator warning';
    } else {
        healthElement.textContent = 'Critical';
        healthElement.className = 'status-indicator critical';
    }
}

function updateCharts(chartsData) {
    // Update coherence chart
    if (chartsData.coherence_over_time) {
        coherenceChart.data.datasets[0].data = chartsData.coherence_over_time.map(point => ({
            x: new Date(point.timestamp),
            y: point.value
        }));
        coherenceChart.update();
    }

    // Update performance chart
    if (chartsData.performance_over_time) {
        performanceChart.data.datasets[0].data = chartsData.performance_over_time.map(point => ({
            x: new Date(point.timestamp),
            y: point.value
        }));
        performanceChart.update();
    }

    // Update network chart
    if (chartsData.network_latency_over_time) {
        networkChart.data.datasets[0].data = chartsData.network_latency_over_time.map(point => ({
            x: new Date(point.timestamp),
            y: point.value
        }));
        networkChart.update();
    }
}

function updateNodeList(nodeMetrics) {
    const nodeList = document.getElementById('node-list');
    nodeList.innerHTML = '';

    Object.entries(nodeMetrics).forEach(([nodeId, metrics]) => {
        const nodeItem = document.createElement('div');
        nodeItem.className = 'node-item ' + metrics.status;
        nodeItem.innerHTML = "<strong>" + nodeId + "</strong><br>" +
            "Coherence: " + strconv.FormatFloat(metrics.average_coherence, 'f', 3, 64) + "<br>" +
            "Performance: " + strconv.FormatFloat(metrics.performance_score, 'f', 3, 64);
        nodeList.appendChild(nodeItem);
    });
}

function updateAlerts(alertStatus) {
    const alertList = document.getElementById('alert-list');
    alertList.innerHTML = '';

    if (!alertStatus.active_alerts || alertStatus.active_alerts.length === 0) {
        alertList.innerHTML = '<div class="alert-item info">No active alerts</div>';
        return;
    }

    alertStatus.active_alerts.slice(0, 10).forEach(alert => {
        const alertItem = document.createElement('div');
        alertItem.className = 'alert-item ' + alert.type;
        alertItem.innerHTML = "<strong>" + alert.category.toUpperCase() + "</strong>: " + alert.message + "<br>" +
            "<small>" + new Date(alert.timestamp).toLocaleString() + "</small>";
        alertList.appendChild(alertItem);
    });
}

function updateLastUpdate(timestamp) {
    const lastUpdateElement = document.getElementById('last-update');
    if (timestamp) {
        lastUpdateElement.textContent = 'Last update: ' + new Date(timestamp).toLocaleTimeString();
    }
}

// Initialize dashboard when page loads
document.addEventListener('DOMContentLoaded', initDashboard);
`
}

// GetDashboardServer returns the dashboard server instance for external access
func (ds *DashboardServer) GetDashboardServer() *http.Server {
	return ds.server
}

// GetPort returns the dashboard server port
func (ds *DashboardServer) GetPort() int {
	return ds.port
}

// IsRunning returns whether the dashboard server is running
func (ds *DashboardServer) IsRunning() bool {
	return ds.server != nil
}
