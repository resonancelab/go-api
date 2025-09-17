package integration

import (
	"fmt"
	"time"

	"github.com/resonancelab/psizero/engines/hqe"
	"github.com/resonancelab/psizero/engines/iching"
	"github.com/resonancelab/psizero/engines/nlc"
	"github.com/resonancelab/psizero/engines/qcr"
	"github.com/resonancelab/psizero/engines/qsem"
	"github.com/resonancelab/psizero/engines/srs"
	"github.com/resonancelab/psizero/engines/unified"
	"github.com/resonancelab/psizero/gateway/services"
)

// HealthCheckResult represents the result of a health check
type HealthCheckResult struct {
	Service  string                 `json:"service"`
	Status   string                 `json:"status"`
	Details  map[string]interface{} `json:"details"`
	Duration time.Duration          `json:"duration"`
	Error    string                 `json:"error,omitempty"`
}

// IntegrationHealthChecker performs comprehensive health checks
type IntegrationHealthChecker struct {
	container *services.ServiceContainer
}

// NewIntegrationHealthChecker creates a new health checker
func NewIntegrationHealthChecker(container *services.ServiceContainer) *IntegrationHealthChecker {
	return &IntegrationHealthChecker{
		container: container,
	}
}

// CheckAllServices performs health checks on all services
func (hc *IntegrationHealthChecker) CheckAllServices() ([]HealthCheckResult, error) {
	results := make([]HealthCheckResult, 0)

	// Check service container
	containerResult := hc.checkServiceContainer()
	results = append(results, containerResult)

	// Check all engines
	engineChecks := []struct {
		name string
		fn   func() HealthCheckResult
	}{
		{"SRS", hc.checkSRSEngine},
		{"HQE", hc.checkHQEEngine},
		{"QSEM", hc.checkQSEMEngine},
		{"NLC", hc.checkNLCEngine},
		{"QCR", hc.checkQCREngine},
		{"I-Ching", hc.checkIChingEngine},
		{"Unified", hc.checkUnifiedEngine},
	}

	for _, check := range engineChecks {
		result := check.fn()
		results = append(results, result)
	}

	return results, nil
}

// checkServiceContainer verifies the service container health
func (hc *IntegrationHealthChecker) checkServiceContainer() HealthCheckResult {
	start := time.Now()

	result := HealthCheckResult{
		Service:  "ServiceContainer",
		Details:  make(map[string]interface{}),
		Duration: time.Since(start),
	}

	if !hc.container.IsInitialized() {
		result.Status = "unhealthy"
		result.Error = "service container not initialized"
		return result
	}

	// Check all engine availability
	healthStatus := hc.container.HealthCheck()
	allHealthy := true
	for service, status := range healthStatus {
		result.Details[service] = status
		if !status {
			allHealthy = false
		}
	}

	if allHealthy {
		result.Status = "healthy"
	} else {
		result.Status = "degraded"
		result.Error = "some engines are not healthy"
	}

	result.Duration = time.Since(start)
	return result
}

// checkSRSEngine verifies SRS engine functionality
func (hc *IntegrationHealthChecker) checkSRSEngine() HealthCheckResult {
	start := time.Now()

	result := HealthCheckResult{
		Service:  "SRS",
		Details:  make(map[string]interface{}),
		Duration: time.Since(start),
	}

	engine := hc.container.GetSRSEngine()
	if engine == nil {
		result.Status = "unhealthy"
		result.Error = "SRS engine not available"
		result.Duration = time.Since(start)
		return result
	}

	// Test basic SRS functionality
	testSpec := map[string]interface{}{
		"variables": 3,
		"clauses": []interface{}{
			[]interface{}{
				map[string]interface{}{"variable": 1, "negated": false},
				map[string]interface{}{"variable": 2, "negated": true},
			},
		},
	}

	config := srs.DefaultSRSConfig()
	config.MaxIterations = 10 // Quick test

	solution, _, err := engine.SolveProblem("3sat", testSpec, config)
	if err != nil {
		result.Status = "unhealthy"
		result.Error = fmt.Sprintf("SRS test failed: %v", err)
	} else {
		result.Status = "healthy"
		result.Details["test_solution"] = solution.Success
		result.Details["feasible"] = solution.Feasible
	}

	result.Duration = time.Since(start)
	return result
}

// checkHQEEngine verifies HQE engine functionality
func (hc *IntegrationHealthChecker) checkHQEEngine() HealthCheckResult {
	start := time.Now()

	result := HealthCheckResult{
		Service:  "HQE",
		Details:  make(map[string]interface{}),
		Duration: time.Since(start),
	}

	engine := hc.container.GetHQEEngine()
	if engine == nil {
		result.Status = "unhealthy"
		result.Error = "HQE engine not available"
		result.Duration = time.Since(start)
		return result
	}

	// Test basic HQE functionality
	params := map[string]interface{}{
		"primes": []int{2, 3, 5},
		"steps":  10,
		"lambda": 0.02,
	}

	config := hqe.DefaultHQEConfig()
	config.MaxIterations = 10

	simulation, _, err := engine.SimulateHolographicDuality("test", params, config)
	if err != nil {
		result.Status = "unhealthy"
		result.Error = fmt.Sprintf("HQE test failed: %v", err)
	} else {
		result.Status = "healthy"
		result.Details["simulation_success"] = simulation.Success
		result.Details["converged"] = simulation.Converged
	}

	result.Duration = time.Since(start)
	return result
}

// checkQSEMEngine verifies QSEM engine functionality
func (hc *IntegrationHealthChecker) checkQSEMEngine() HealthCheckResult {
	start := time.Now()

	result := HealthCheckResult{
		Service:  "QSEM",
		Details:  make(map[string]interface{}),
		Duration: time.Since(start),
	}

	engine := hc.container.GetQSEMEngine()
	if engine == nil {
		result.Status = "unhealthy"
		result.Error = "QSEM engine not available"
		result.Duration = time.Since(start)
		return result
	}

	// Test basic QSEM functionality
	input := map[string]interface{}{
		"type":     "concepts",
		"concepts": []interface{}{"test", "concept"},
	}

	config := qsem.DefaultQSEMConfig()
	config.MaxIterations = 10

	analysis, _, err := engine.AnalyzeSemantics(input, config)
	if err != nil {
		result.Status = "unhealthy"
		result.Error = fmt.Sprintf("QSEM test failed: %v", err)
	} else {
		result.Status = "healthy"
		result.Details["analysis_success"] = analysis.Success
		result.Details["concepts_processed"] = len(analysis.ConceptMap)
	}

	result.Duration = time.Since(start)
	return result
}

// checkNLCEngine verifies NLC engine functionality
func (hc *IntegrationHealthChecker) checkNLCEngine() HealthCheckResult {
	start := time.Now()

	result := HealthCheckResult{
		Service:  "NLC",
		Details:  make(map[string]interface{}),
		Duration: time.Since(start),
	}

	engine := hc.container.GetNLCEngine()
	if engine == nil {
		result.Status = "unhealthy"
		result.Error = "NLC engine not available"
		result.Duration = time.Since(start)
		return result
	}

	// Test basic NLC functionality
	participants := []string{"alice", "bob"}
	config := nlc.DefaultNLCConfig()

	communication, _, err := engine.EstablishNonLocalCommunication("test", participants, config)
	if err != nil {
		result.Status = "unhealthy"
		result.Error = fmt.Sprintf("NLC test failed: %v", err)
	} else {
		result.Status = "healthy"
		result.Details["communication_established"] = communication.Success
		result.Details["average_fidelity"] = communication.AverageFidelity
	}

	result.Duration = time.Since(start)
	return result
}

// checkQCREngine verifies QCR engine functionality
func (hc *IntegrationHealthChecker) checkQCREngine() HealthCheckResult {
	start := time.Now()

	result := HealthCheckResult{
		Service:  "QCR",
		Details:  make(map[string]interface{}),
		Duration: time.Since(start),
	}

	engine := hc.container.GetQCREngine()
	if engine == nil {
		result.Status = "unhealthy"
		result.Error = "QCR engine not available"
		result.Duration = time.Since(start)
		return result
	}

	// Test basic QCR functionality
	params := map[string]interface{}{
		"modes":          []string{"analytical", "creative"},
		"max_iterations": 5,
	}

	config := qcr.DefaultQCRConfig()
	config.MaxSimulationCycles = 5

	consciousness, _, err := engine.SimulateConsciousness("test", params, config)
	if err != nil {
		result.Status = "unhealthy"
		result.Error = fmt.Sprintf("QCR test failed: %v", err)
	} else {
		result.Status = "healthy"
		result.Details["simulation_success"] = consciousness.Success
		result.Details["consciousness_level"] = consciousness.FinalConsciousnessLevel
	}

	result.Duration = time.Since(start)
	return result
}

// checkIChingEngine verifies I-Ching engine functionality
func (hc *IntegrationHealthChecker) checkIChingEngine() HealthCheckResult {
	start := time.Now()

	result := HealthCheckResult{
		Service:  "I-Ching",
		Details:  make(map[string]interface{}),
		Duration: time.Since(start),
	}

	engine := hc.container.GetIChingEngine()
	if engine == nil {
		result.Status = "unhealthy"
		result.Error = "I-Ching engine not available"
		result.Duration = time.Since(start)
		return result
	}

	// Test basic I-Ching functionality
	config := iching.DefaultIChingConfig()

	oracle, _, err := engine.ConsultOracle("test question", "test", "tester", config)
	if err != nil {
		result.Status = "unhealthy"
		result.Error = fmt.Sprintf("I-Ching test failed: %v", err)
	} else {
		result.Status = "healthy"
		result.Details["consultation_success"] = oracle.Success
		result.Details["primary_hexagram"] = oracle.PrimaryHexagram.Number
	}

	result.Duration = time.Since(start)
	return result
}

// checkUnifiedEngine verifies Unified Physics engine functionality
func (hc *IntegrationHealthChecker) checkUnifiedEngine() HealthCheckResult {
	start := time.Now()

	result := HealthCheckResult{
		Service:  "Unified",
		Details:  make(map[string]interface{}),
		Duration: time.Since(start),
	}

	engine := hc.container.GetUnifiedEngine()
	if engine == nil {
		result.Status = "unhealthy"
		result.Error = "Unified engine not available"
		result.Duration = time.Since(start)
		return result
	}

	// Test basic Unified Physics functionality
	conditions := map[string]interface{}{
		"test": "gravity",
	}

	config := unified.DefaultUnifiedConfig()
	config.EvolutionSteps = 5

	physics, _, err := engine.SimulateUnifiedPhysics("test", conditions, config)
	if err != nil {
		result.Status = "unhealthy"
		result.Error = fmt.Sprintf("Unified test failed: %v", err)
	} else {
		result.Status = "healthy"
		result.Details["simulation_success"] = physics.Success
		result.Details["converged"] = physics.Converged
	}

	result.Duration = time.Since(start)
	return result
}

// GetOverallHealth returns overall system health status
func (hc *IntegrationHealthChecker) GetOverallHealth() (string, error) {
	results, err := hc.CheckAllServices()
	if err != nil {
		return "error", err
	}

	healthyCount := 0
	unhealthyCount := 0
	degradedCount := 0

	for _, result := range results {
		switch result.Status {
		case "healthy":
			healthyCount++
		case "unhealthy":
			unhealthyCount++
		case "degraded":
			degradedCount++
		}
	}

	if unhealthyCount > 0 {
		return "unhealthy", nil
	}
	if degradedCount > 0 {
		return "degraded", nil
	}
	return "healthy", nil
}
