# PsiZero Resonance Platform API

Enterprise-grade backend API for the PsiZero Resonance Platform, featuring 7 specialized API services for advanced resonance computing, quantum simulation, and consciousness modeling.

## Architecture Overview

The PsiZero Resonance Platform API is built as a microservices-based architecture using Go, featuring:

- **API Gateway**: Central routing and authentication
- **7 Core Services**: Specialized resonance computing APIs
- **Shared Infrastructure**: Common types, middleware, and utilities
- **Database Integration**: PostgreSQL with vector extensions
- **Caching Layer**: Redis for performance optimization
- **Real-time Events**: Webhook system for notifications

## Services

### 1. SRS - Stochastic Resonance Solver
Advanced problem-solving through quantum resonance techniques
- 3-SAT problem solving
- Constraint satisfaction
- Optimization algorithms
- Real-time progress tracking

### 2. HQE - Hypercomplex Quantum Engine  
Quantum simulation and computation platform
- Quantum state simulation
- Energy level optimization
- Convergence analysis
- Multi-dimensional quantum systems

### 3. QSEM - Quantum Semantic Encoding Machine
Advanced concept encoding and semantic analysis
- Vector embeddings with quantum enhancement
- Semantic search and clustering
- Resonance frequency computation
- Contextual understanding

### 4. NLC - Non-Local Communication
Quantum-entangled communication protocols
- Entangled communication sessions
- Message transmission via quantum channels
- Channel quality monitoring
- Quantum coherence tracking

### 5. QCR - Quantum Consciousness Resonator
Consciousness simulation and evolution platform
- Observer effect modeling
- Consciousness coherence measurement
- Quantum state evolution
- Awareness pattern analysis

### 6. I-Ching Quantum Oracle
Evolutionary guidance through ancient wisdom and quantum mechanics
- Dynamic hexagram evolution
- Quantum state progression
- Wisdom pattern recognition
- Guidance generation

### 7. Unified Physics Engine
Fundamental physics computation and analysis
- Gravitational field calculations
- Field equation analysis
- Physical constant computation
- Spacetime simulation

## Quick Start

### Prerequisites

- Go 1.21 or later
- Docker and Docker Compose
- PostgreSQL 15+ with vector extension
- Redis 7+

### Local Development

1. **Clone and setup:**
```bash
cd api/
go mod download
```

2. **Start dependencies:**
```bash
docker-compose up -d postgres redis
```

3. **Run the API:**
```bash
go run gateway/main.go
```

4. **Test the API:**
```bash
./scripts/test_api.sh
```

The API will be available at `http://localhost:8080` with documentation at `http://localhost:8080/docs/`

### Production Deployment

1. **Build and deploy with Docker:**
```bash
docker-compose up -d
```

2. **With monitoring stack:**
```bash
docker-compose --profile monitoring up -d
```

3. **With load balancer:**
```bash
docker-compose --profile production up -d
```

## API Documentation

### Authentication

All API endpoints require authentication via API key or JWT token:

```bash
# API Key authentication
curl -H "X-API-Key: your_api_key" https://api.psizero.ai/v1/status

# JWT Bearer token
curl -H "Authorization: Bearer your_jwt_token" https://api.psizero.ai/v1/status
```

### Example Usage

#### SRS Problem Solving
```bash
curl -X POST https://api.psizero.ai/v1/srs/solve \
  -H "X-API-Key: your_api_key" \
  -H "Content-Type: application/json" \
  -d '{
    "problem_type": "3sat",
    "clauses": [[1, -2, 3], [-1, 2, -3]],
    "variables": 3,
    "max_iterations": 1000
  }'
```

#### Quantum Simulation
```bash
curl -X POST https://api.psizero.ai/v1/hqe/simulate \
  -H "X-API-Key: your_api_key" \
  -H "Content-Type: application/json" \
  -d '{
    "system_type": "harmonic_oscillator",
    "dimensions": 3,
    "particles": 2,
    "time_steps": 100
  }'
```

#### Concept Encoding
```bash
curl -X POST https://api.psizero.ai/v1/qsem/encode \
  -H "X-API-Key: your_api_key" \
  -H "Content-Type: application/json" \
  -d '{
    "concepts": ["quantum", "consciousness", "resonance"],
    "encoding_type": "contextual",
    "dimension": 1536
  }'
```

### API Response Format

All API responses follow a consistent format:

```json
{
  "success": true,
  "data": {
    "result": "API-specific response data"
  },
  "metadata": {
    "request_id": "req_abc123",
    "timestamp": "2024-01-01T00:00:00Z",
    "processing_time_ms": 123.45
  }
}
```

Error responses:
```json
{
  "success": false,
  "error": {
    "code": "API_ERROR_001",
    "message": "Human-readable error message",
    "details": "Technical error details",
    "request_id": "req_abc123"
  }
}
```

## Configuration

### Environment Variables

```bash
# Server Configuration
PORT=8080
ENVIRONMENT=production

# Database
DATABASE_URL=postgres://user:pass@localhost/psizero_resonance?sslmode=disable

# Cache
REDIS_URL=redis://localhost:6379

# Security
JWT_SECRET=your-secret-key-here

# Logging
LOG_LEVEL=info

# Metrics
METRICS_PORT=9090
HEALTH_PORT=8081
```

### Database Setup

The API uses PostgreSQL with vector extensions for advanced semantic operations:

```sql
-- Required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "vector";
```

Run the initialization script:
```bash
psql -f scripts/init.sql psizero_resonance
```

## Testing

### Automated Testing
```bash
# Test all endpoints
./scripts/test_api.sh

# Test specific service
./scripts/test_api.sh srs
./scripts/test_api.sh hqe
./scripts/test_api.sh qsem
```

### Manual Testing
```bash
# Health check
curl http://localhost:8080/health

# Authenticated status
curl -H "X-API-Key: demo_api_key_12345" http://localhost:8080/v1/status
```

## Monitoring

### Metrics
- Prometheus metrics available on port 9090
- Custom metrics for each service
- Request/response tracking
- Performance monitoring

### Health Checks
- `/health` - Public health endpoint
- `/v1/status` - Authenticated status with service info
- Docker health checks included

### Logging
- Structured JSON logging
- Request ID tracking
- Performance metrics
- Error tracking

## Development

### Project Structure
```
api/
├── gateway/           # API Gateway and main entry point
│   ├── main.go       # Server initialization
│   └── router/       # Service route handlers
├── shared/           # Shared components
│   ├── types/        # Common types and structures
│   └── middleware/   # Authentication, CORS, rate limiting
├── scripts/          # Deployment and testing scripts
├── Dockerfile        # Container definition
├── docker-compose.yml # Multi-service deployment
└── README.md        # This file
```

### Adding New Endpoints

1. **Define types in** [`shared/types/`](shared/types/)
2. **Create router in** [`gateway/router/`](gateway/router/)
3. **Add route setup in** [`gateway/main.go`](gateway/main.go)
4. **Add tests in** [`scripts/test_api.sh`](scripts/test_api.sh)

### Code Standards
- Go modules for dependency management
- Swagger/OpenAPI documentation
- Comprehensive error handling
- Request ID tracking
- Structured logging

## Security

### Authentication Methods
- API Key authentication via `X-API-Key` header
- JWT Bearer token authentication
- Rate limiting per user/IP
- CORS protection

### Data Protection
- Request/response encryption in transit
- API key hashing in database
- Input validation and sanitization
- SQL injection prevention

## Production Considerations

### Scaling
- Horizontal scaling via Docker Compose
- Database connection pooling
- Redis caching for performance
- Load balancing with Nginx

### Monitoring
- Prometheus metrics collection
- Grafana dashboard visualization
- Health check endpoints
- Error rate tracking

### Backup
- Automated database backups
- Redis persistence configuration
- Configuration backup procedures
- Disaster recovery planning

## Support

### Documentation
- API documentation: `http://localhost:8080/docs/`
- Interactive Swagger UI
- Request/response examples
- Authentication guides

### Troubleshooting
- Check service logs: `docker-compose logs psizero-api`
- Verify database connection: `docker-compose logs postgres`
- Test Redis cache: `docker-compose logs redis`
- Run health checks: `curl http://localhost:8080/health`

For technical support and questions, please visit our documentation portal or contact the development team.