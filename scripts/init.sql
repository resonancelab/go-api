-- PsiZero Resonance Platform Database Initialization
-- This script sets up the initial database schema for the platform

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "vector" WITH SCHEMA public;

-- Users table
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    api_key VARCHAR(255) UNIQUE NOT NULL DEFAULT encode(gen_random_bytes(32), 'hex'),
    plan VARCHAR(50) DEFAULT 'free',
    quota_limit INTEGER DEFAULT 1000,
    quota_used INTEGER DEFAULT 0,
    quota_reset_at TIMESTAMP DEFAULT NOW() + INTERVAL '1 month',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    last_active TIMESTAMP DEFAULT NOW(),
    status VARCHAR(20) DEFAULT 'active'
);

-- API Usage tracking
CREATE TABLE IF NOT EXISTS api_usage (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    service VARCHAR(50) NOT NULL,
    endpoint VARCHAR(255) NOT NULL,
    method VARCHAR(10) NOT NULL,
    status_code INTEGER NOT NULL,
    request_size INTEGER DEFAULT 0,
    response_size INTEGER DEFAULT 0,
    duration_ms FLOAT DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW()
);

-- SRS Problem tracking
CREATE TABLE IF NOT EXISTS srs_problems (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    problem_type VARCHAR(50) NOT NULL,
    input_data JSONB NOT NULL,
    status VARCHAR(20) DEFAULT 'pending',
    solution JSONB,
    iterations INTEGER DEFAULT 0,
    duration_ms FLOAT DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW(),
    completed_at TIMESTAMP
);

-- HQE Simulations
CREATE TABLE IF NOT EXISTS hqe_simulations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    simulation_type VARCHAR(50) NOT NULL,
    parameters JSONB NOT NULL,
    status VARCHAR(20) DEFAULT 'pending',
    results JSONB,
    energy_levels JSONB,
    convergence_achieved BOOLEAN DEFAULT FALSE,
    iterations INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW(),
    completed_at TIMESTAMP
);

-- QSEM Concept vectors
CREATE TABLE IF NOT EXISTS qsem_concepts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    concept_text TEXT NOT NULL,
    embedding VECTOR(1536),
    resonance_frequency FLOAT,
    coherence_score FLOAT DEFAULT 0,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- NLC Communication sessions
CREATE TABLE IF NOT EXISTS nlc_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    session_name VARCHAR(255) NOT NULL,
    participants INTEGER DEFAULT 2,
    entanglement_strength FLOAT DEFAULT 0,
    coherence_level FLOAT DEFAULT 0,
    status VARCHAR(20) DEFAULT 'active',
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    closed_at TIMESTAMP
);

-- QCR Consciousness states
CREATE TABLE IF NOT EXISTS qcr_states (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    session_id UUID NOT NULL,
    observer_effect FLOAT DEFAULT 0,
    consciousness_coherence FLOAT DEFAULT 0,
    entanglement_density FLOAT DEFAULT 0,
    state_vector JSONB,
    observations JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- I-Ching Evolution tracking
CREATE TABLE IF NOT EXISTS iching_evolutions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    question TEXT NOT NULL,
    initial_hexagram JSONB NOT NULL,
    evolved_hexagram JSONB,
    evolution_steps JSONB,
    guidance_text TEXT,
    resonance_score FLOAT DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Unified Physics computations
CREATE TABLE IF NOT EXISTS unified_computations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    computation_type VARCHAR(50) NOT NULL,
    field_parameters JSONB NOT NULL,
    results JSONB,
    gravitational_field JSONB,
    field_equations JSONB,
    constants JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Webhooks configuration
CREATE TABLE IF NOT EXISTS webhooks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    url VARCHAR(500) NOT NULL,
    events TEXT[] NOT NULL,
    secret VARCHAR(255),
    active BOOLEAN DEFAULT TRUE,
    config JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    last_ping TIMESTAMP
);

-- Webhook deliveries
CREATE TABLE IF NOT EXISTS webhook_deliveries (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    webhook_id UUID NOT NULL REFERENCES webhooks(id) ON DELETE CASCADE,
    event_type VARCHAR(100) NOT NULL,
    event_data JSONB NOT NULL,
    success BOOLEAN DEFAULT FALSE,
    status_code INTEGER,
    response_body TEXT,
    error_message TEXT,
    duration_ms FLOAT DEFAULT 0,
    attempt_count INTEGER DEFAULT 1,
    delivered_at TIMESTAMP DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_users_api_key ON users(api_key);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_api_usage_user_id ON api_usage(user_id);
CREATE INDEX IF NOT EXISTS idx_api_usage_created_at ON api_usage(created_at);
CREATE INDEX IF NOT EXISTS idx_srs_problems_user_id ON srs_problems(user_id);
CREATE INDEX IF NOT EXISTS idx_hqe_simulations_user_id ON hqe_simulations(user_id);
CREATE INDEX IF NOT EXISTS idx_qsem_concepts_user_id ON qsem_concepts(user_id);
CREATE INDEX IF NOT EXISTS idx_nlc_sessions_user_id ON nlc_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_qcr_states_user_id ON qcr_states(user_id);
CREATE INDEX IF NOT EXISTS idx_iching_evolutions_user_id ON iching_evolutions(user_id);
CREATE INDEX IF NOT EXISTS idx_unified_computations_user_id ON unified_computations(user_id);
CREATE INDEX IF NOT EXISTS idx_webhooks_user_id ON webhooks(user_id);
CREATE INDEX IF NOT EXISTS idx_webhook_deliveries_webhook_id ON webhook_deliveries(webhook_id);

-- Insert sample users for testing
INSERT INTO users (email, password_hash, plan, quota_limit) VALUES
    ('demo@psizero.ai', crypt('demo123', gen_salt('bf')), 'enterprise', 100000),
    ('test@psizero.ai', crypt('test123', gen_salt('bf')), 'pro', 10000),
    ('free@psizero.ai', crypt('free123', gen_salt('bf')), 'free', 1000)
ON CONFLICT (email) DO NOTHING;

-- Create a function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for updated_at
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_webhooks_updated_at BEFORE UPDATE ON webhooks FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO psizero;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO psizero;