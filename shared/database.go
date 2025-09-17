package shared

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"time"

	_ "github.com/lib/pq" // PostgreSQL driver
	"github.com/resonancelab/psizero/shared/types"
)

// DatabaseConnection interface for database operations
type DatabaseConnection interface {
	Connect() error
	Disconnect() error
	StoreTelemetry(points []types.TelemetryPoint) error
	GetTelemetry(engineID string, since time.Time) ([]types.TelemetryPoint, error)
	StoreEngineState(engineID string, state *EngineState) error
	GetEngineState(engineID string) (*EngineState, error)
	StoreGlobalState(state *GlobalResonanceState) error
	GetGlobalState() (*GlobalResonanceState, error)
	StorePerformanceMetrics(metrics map[string]*PerformanceMetric) error
	GetPerformanceMetrics(since time.Time) (map[string]*PerformanceMetric, error)
	Cleanup(retention time.Duration) error
}

// PostgreSQLConnection implements DatabaseConnection for PostgreSQL
type PostgreSQLConnection struct {
	connectionString string
	db               *sql.DB
	connected        bool
}

// NewPostgreSQLConnection creates a new PostgreSQL connection
func NewPostgreSQLConnection(connectionString string) *PostgreSQLConnection {
	return &PostgreSQLConnection{
		connectionString: connectionString,
		connected:        false,
	}
}

// Connect establishes database connection
func (pg *PostgreSQLConnection) Connect() error {
	if pg.connected {
		return nil
	}

	db, err := sql.Open("postgres", pg.connectionString)
	if err != nil {
		return fmt.Errorf("failed to open database: %w", err)
	}

	if err := db.Ping(); err != nil {
		return fmt.Errorf("failed to ping database: %w", err)
	}

	pg.db = db
	pg.connected = true

	// Initialize schema
	if err := pg.initializeSchema(); err != nil {
		return fmt.Errorf("failed to initialize schema: %w", err)
	}

	return nil
}

// Disconnect closes database connection
func (pg *PostgreSQLConnection) Disconnect() error {
	if !pg.connected || pg.db == nil {
		return nil
	}

	err := pg.db.Close()
	pg.connected = false
	pg.db = nil

	return err
}

// initializeSchema creates database tables if they don't exist
func (pg *PostgreSQLConnection) initializeSchema() error {
	schemas := []string{
		`CREATE TABLE IF NOT EXISTS telemetry_points (
			id SERIAL PRIMARY KEY,
			engine_id VARCHAR(255),
			step INTEGER,
			symbolic_entropy DOUBLE PRECISION,
			lyapunov_metric DOUBLE PRECISION,
			satisfaction_rate DOUBLE PRECISION,
			resonance_strength DOUBLE PRECISION,
			dominance DOUBLE PRECISION,
			timestamp TIMESTAMP WITH TIME ZONE,
			created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
		)`,
		`CREATE TABLE IF NOT EXISTS engine_states (
			id VARCHAR(255) PRIMARY KEY,
			type VARCHAR(100),
			status VARCHAR(50),
			last_update TIMESTAMP WITH TIME ZONE,
			metrics JSONB,
			configuration JSONB,
			resonance_level DOUBLE PRECISION,
			coherence DOUBLE PRECISION,
			entanglement_map JSONB,
			created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
			updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
		)`,
		`CREATE TABLE IF NOT EXISTS global_states (
			id SERIAL PRIMARY KEY,
			global_resonance DOUBLE PRECISION,
			system_coherence DOUBLE PRECISION,
			unification_degree DOUBLE PRECISION,
			sync_timestamp TIMESTAMP WITH TIME ZONE,
			aggregated_metrics JSONB,
			global_config JSONB,
			created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
		)`,
		`CREATE TABLE IF NOT EXISTS performance_metrics (
			id SERIAL PRIMARY KEY,
			metric_name VARCHAR(255),
			current_value DOUBLE PRECISION,
			min_value DOUBLE PRECISION,
			max_value DOUBLE PRECISION,
			average_value DOUBLE PRECISION,
			values JSONB,
			timestamps JSONB,
			created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
		)`,
		`CREATE INDEX IF NOT EXISTS idx_telemetry_engine_timestamp ON telemetry_points(engine_id, timestamp)`,
		`CREATE INDEX IF NOT EXISTS idx_telemetry_timestamp ON telemetry_points(timestamp)`,
		`CREATE INDEX IF NOT EXISTS idx_engine_states_type ON engine_states(type)`,
		`CREATE INDEX IF NOT EXISTS idx_performance_metrics_name ON performance_metrics(metric_name)`,
	}

	for _, schema := range schemas {
		if _, err := pg.db.Exec(schema); err != nil {
			return fmt.Errorf("failed to create schema: %w", err)
		}
	}

	return nil
}

// StoreTelemetry stores telemetry points in the database
func (pg *PostgreSQLConnection) StoreTelemetry(points []types.TelemetryPoint) error {
	if !pg.connected {
		return fmt.Errorf("database not connected")
	}

	tx, err := pg.db.Begin()
	if err != nil {
		return fmt.Errorf("failed to begin transaction: %w", err)
	}
	defer tx.Rollback()

	stmt, err := tx.Prepare(`
		INSERT INTO telemetry_points 
		(engine_id, step, symbolic_entropy, lyapunov_metric, satisfaction_rate, resonance_strength, dominance, timestamp)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
	`)
	if err != nil {
		return fmt.Errorf("failed to prepare statement: %w", err)
	}
	defer stmt.Close()

	for _, point := range points {
		_, err := stmt.Exec(
			"system", // Default engine_id for system telemetry
			point.Step,
			point.SymbolicEntropy,
			point.LyapunovMetric,
			point.SatisfactionRate,
			point.ResonanceStrength,
			point.Dominance,
			point.Timestamp,
		)
		if err != nil {
			return fmt.Errorf("failed to insert telemetry point: %w", err)
		}
	}

	return tx.Commit()
}

// GetTelemetry retrieves telemetry points from the database
func (pg *PostgreSQLConnection) GetTelemetry(engineID string, since time.Time) ([]types.TelemetryPoint, error) {
	if !pg.connected {
		return nil, fmt.Errorf("database not connected")
	}

	query := `
		SELECT step, symbolic_entropy, lyapunov_metric, satisfaction_rate, resonance_strength, dominance, timestamp
		FROM telemetry_points
		WHERE engine_id = $1 AND timestamp >= $2
		ORDER BY timestamp ASC
	`

	rows, err := pg.db.Query(query, engineID, since)
	if err != nil {
		return nil, fmt.Errorf("failed to query telemetry: %w", err)
	}
	defer rows.Close()

	var points []types.TelemetryPoint
	for rows.Next() {
		var point types.TelemetryPoint
		err := rows.Scan(
			&point.Step,
			&point.SymbolicEntropy,
			&point.LyapunovMetric,
			&point.SatisfactionRate,
			&point.ResonanceStrength,
			&point.Dominance,
			&point.Timestamp,
		)
		if err != nil {
			return nil, fmt.Errorf("failed to scan telemetry row: %w", err)
		}
		points = append(points, point)
	}

	return points, nil
}

// StoreEngineState stores engine state in the database
func (pg *PostgreSQLConnection) StoreEngineState(engineID string, state *EngineState) error {
	if !pg.connected {
		return fmt.Errorf("database not connected")
	}

	metricsJSON, err := json.Marshal(state.Metrics)
	if err != nil {
		return fmt.Errorf("failed to marshal metrics: %w", err)
	}

	configJSON, err := json.Marshal(state.Configuration)
	if err != nil {
		return fmt.Errorf("failed to marshal configuration: %w", err)
	}

	entanglementJSON, err := json.Marshal(state.EntanglementMap)
	if err != nil {
		return fmt.Errorf("failed to marshal entanglement map: %w", err)
	}

	query := `
		INSERT INTO engine_states 
		(id, type, status, last_update, metrics, configuration, resonance_level, coherence, entanglement_map, updated_at)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, NOW())
		ON CONFLICT (id) DO UPDATE SET
			type = EXCLUDED.type,
			status = EXCLUDED.status,
			last_update = EXCLUDED.last_update,
			metrics = EXCLUDED.metrics,
			configuration = EXCLUDED.configuration,
			resonance_level = EXCLUDED.resonance_level,
			coherence = EXCLUDED.coherence,
			entanglement_map = EXCLUDED.entanglement_map,
			updated_at = NOW()
	`

	_, err = pg.db.Exec(query,
		state.ID,
		state.Type,
		state.Status,
		state.LastUpdate,
		metricsJSON,
		configJSON,
		state.ResonanceLevel,
		state.Coherence,
		entanglementJSON,
	)

	if err != nil {
		return fmt.Errorf("failed to store engine state: %w", err)
	}

	return nil
}

// GetEngineState retrieves engine state from the database
func (pg *PostgreSQLConnection) GetEngineState(engineID string) (*EngineState, error) {
	if !pg.connected {
		return nil, fmt.Errorf("database not connected")
	}

	query := `
		SELECT id, type, status, last_update, metrics, configuration, resonance_level, coherence, entanglement_map
		FROM engine_states
		WHERE id = $1
	`

	row := pg.db.QueryRow(query, engineID)

	var state EngineState
	var metricsJSON, configJSON, entanglementJSON []byte

	err := row.Scan(
		&state.ID,
		&state.Type,
		&state.Status,
		&state.LastUpdate,
		&metricsJSON,
		&configJSON,
		&state.ResonanceLevel,
		&state.Coherence,
		&entanglementJSON,
	)

	if err != nil {
		if err == sql.ErrNoRows {
			return nil, nil // Not found
		}
		return nil, fmt.Errorf("failed to scan engine state: %w", err)
	}

	// Unmarshal JSON fields
	if err := json.Unmarshal(metricsJSON, &state.Metrics); err != nil {
		return nil, fmt.Errorf("failed to unmarshal metrics: %w", err)
	}

	if err := json.Unmarshal(configJSON, &state.Configuration); err != nil {
		return nil, fmt.Errorf("failed to unmarshal configuration: %w", err)
	}

	if err := json.Unmarshal(entanglementJSON, &state.EntanglementMap); err != nil {
		return nil, fmt.Errorf("failed to unmarshal entanglement map: %w", err)
	}

	return &state, nil
}

// StoreGlobalState stores global state in the database
func (pg *PostgreSQLConnection) StoreGlobalState(state *GlobalResonanceState) error {
	if !pg.connected {
		return fmt.Errorf("database not connected")
	}

	metricsJSON, err := json.Marshal(state.AggregatedMetrics)
	if err != nil {
		return fmt.Errorf("failed to marshal aggregated metrics: %w", err)
	}

	configJSON, err := json.Marshal(state.GlobalConfig)
	if err != nil {
		return fmt.Errorf("failed to marshal global config: %w", err)
	}

	query := `
		INSERT INTO global_states 
		(global_resonance, system_coherence, unification_degree, sync_timestamp, aggregated_metrics, global_config)
		VALUES ($1, $2, $3, $4, $5, $6)
	`

	_, err = pg.db.Exec(query,
		state.GlobalResonance,
		state.SystemCoherence,
		state.UnificationDegree,
		state.SyncTimestamp,
		metricsJSON,
		configJSON,
	)

	if err != nil {
		return fmt.Errorf("failed to store global state: %w", err)
	}

	return nil
}

// GetGlobalState retrieves the latest global state from the database
func (pg *PostgreSQLConnection) GetGlobalState() (*GlobalResonanceState, error) {
	if !pg.connected {
		return nil, fmt.Errorf("database not connected")
	}

	query := `
		SELECT global_resonance, system_coherence, unification_degree, sync_timestamp, aggregated_metrics, global_config
		FROM global_states
		ORDER BY created_at DESC
		LIMIT 1
	`

	row := pg.db.QueryRow(query)

	var state GlobalResonanceState
	var metricsJSON, configJSON []byte

	err := row.Scan(
		&state.GlobalResonance,
		&state.SystemCoherence,
		&state.UnificationDegree,
		&state.SyncTimestamp,
		&metricsJSON,
		&configJSON,
	)

	if err != nil {
		if err == sql.ErrNoRows {
			return nil, nil // Not found
		}
		return nil, fmt.Errorf("failed to scan global state: %w", err)
	}

	// Unmarshal JSON fields
	if err := json.Unmarshal(metricsJSON, &state.AggregatedMetrics); err != nil {
		return nil, fmt.Errorf("failed to unmarshal aggregated metrics: %w", err)
	}

	if err := json.Unmarshal(configJSON, &state.GlobalConfig); err != nil {
		return nil, fmt.Errorf("failed to unmarshal global config: %w", err)
	}

	// Initialize other fields
	state.EngineStates = make(map[string]*EngineState)
	state.TelemetryHistory = make([]types.TelemetryPoint, 0)
	state.SharedPrimes = make([]int, 0)

	return &state, nil
}

// StorePerformanceMetrics stores performance metrics in the database
func (pg *PostgreSQLConnection) StorePerformanceMetrics(metrics map[string]*PerformanceMetric) error {
	if !pg.connected {
		return fmt.Errorf("database not connected")
	}

	tx, err := pg.db.Begin()
	if err != nil {
		return fmt.Errorf("failed to begin transaction: %w", err)
	}
	defer tx.Rollback()

	// Clear existing metrics
	_, err = tx.Exec("DELETE FROM performance_metrics")
	if err != nil {
		return fmt.Errorf("failed to clear existing metrics: %w", err)
	}

	stmt, err := tx.Prepare(`
		INSERT INTO performance_metrics 
		(metric_name, current_value, min_value, max_value, average_value, values, timestamps)
		VALUES ($1, $2, $3, $4, $5, $6, $7)
	`)
	if err != nil {
		return fmt.Errorf("failed to prepare statement: %w", err)
	}
	defer stmt.Close()

	for name, metric := range metrics {
		valuesJSON, err := json.Marshal(metric.Values)
		if err != nil {
			return fmt.Errorf("failed to marshal values: %w", err)
		}

		timestampsJSON, err := json.Marshal(metric.Timestamps)
		if err != nil {
			return fmt.Errorf("failed to marshal timestamps: %w", err)
		}

		_, err = stmt.Exec(
			name,
			metric.CurrentValue,
			metric.MinValue,
			metric.MaxValue,
			metric.AverageValue,
			valuesJSON,
			timestampsJSON,
		)
		if err != nil {
			return fmt.Errorf("failed to insert performance metric: %w", err)
		}
	}

	return tx.Commit()
}

// GetPerformanceMetrics retrieves performance metrics from the database
func (pg *PostgreSQLConnection) GetPerformanceMetrics(since time.Time) (map[string]*PerformanceMetric, error) {
	if !pg.connected {
		return nil, fmt.Errorf("database not connected")
	}

	query := `
		SELECT metric_name, current_value, min_value, max_value, average_value, values, timestamps
		FROM performance_metrics
		WHERE created_at >= $1
	`

	rows, err := pg.db.Query(query, since)
	if err != nil {
		return nil, fmt.Errorf("failed to query performance metrics: %w", err)
	}
	defer rows.Close()

	metrics := make(map[string]*PerformanceMetric)

	for rows.Next() {
		var metric PerformanceMetric
		var valuesJSON, timestampsJSON []byte

		err := rows.Scan(
			&metric.Name,
			&metric.CurrentValue,
			&metric.MinValue,
			&metric.MaxValue,
			&metric.AverageValue,
			&valuesJSON,
			&timestampsJSON,
		)
		if err != nil {
			return nil, fmt.Errorf("failed to scan performance metric: %w", err)
		}

		// Unmarshal JSON fields
		if err := json.Unmarshal(valuesJSON, &metric.Values); err != nil {
			return nil, fmt.Errorf("failed to unmarshal values: %w", err)
		}

		if err := json.Unmarshal(timestampsJSON, &metric.Timestamps); err != nil {
			return nil, fmt.Errorf("failed to unmarshal timestamps: %w", err)
		}

		metrics[metric.Name] = &metric
	}

	return metrics, nil
}

// Cleanup removes old data beyond retention period
func (pg *PostgreSQLConnection) Cleanup(retention time.Duration) error {
	if !pg.connected {
		return fmt.Errorf("database not connected")
	}

	cutoff := time.Now().Add(-retention)

	tables := []string{"telemetry_points", "global_states", "performance_metrics"}

	for _, table := range tables {
		query := fmt.Sprintf("DELETE FROM %s WHERE created_at < $1", table)
		_, err := pg.db.Exec(query, cutoff)
		if err != nil {
			return fmt.Errorf("failed to cleanup table %s: %w", table, err)
		}
	}

	return nil
}
