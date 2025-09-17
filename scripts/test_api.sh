#!/bin/bash

# PsiZero Resonance Platform API Testing Script
# This script validates all API endpoints with comprehensive test cases

set -e

# Configuration
API_BASE_URL="${API_BASE_URL:-http://localhost:8080}"
API_KEY="${API_KEY:-demo_api_key_12345}"
TEST_USER_ID="${TEST_USER_ID:-user_demo_123}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Counters
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Test results array
declare -a FAILED_ENDPOINTS

# Function to print colored output
print_status() {
    local status=$1
    local message=$2
    case $status in
        "PASS")
            echo -e "${GREEN}✓ PASS${NC}: $message"
            ((PASSED_TESTS++))
            ;;
        "FAIL")
            echo -e "${RED}✗ FAIL${NC}: $message"
            ((FAILED_TESTS++))
            FAILED_ENDPOINTS+=("$message")
            ;;
        "INFO")
            echo -e "${BLUE}ℹ INFO${NC}: $message"
            ;;
        "WARN")
            echo -e "${YELLOW}⚠ WARN${NC}: $message"
            ;;
    esac
}

# Function to test an endpoint
test_endpoint() {
    local method=$1
    local endpoint=$2
    local expected_status=$3
    local description=$4
    local data=$5
    
    ((TOTAL_TESTS++))
    
    print_status "INFO" "Testing $method $endpoint - $description"
    
    local curl_cmd="curl -s -w '%{http_code}' -X $method"
    curl_cmd="$curl_cmd -H 'X-API-Key: $API_KEY'"
    curl_cmd="$curl_cmd -H 'Content-Type: application/json'"
    
    if [ -n "$data" ]; then
        curl_cmd="$curl_cmd -d '$data'"
    fi
    
    curl_cmd="$curl_cmd '$API_BASE_URL$endpoint'"
    
    local response=$(eval $curl_cmd)
    local status_code=${response: -3}
    local body=${response%???}
    
    if [ "$status_code" = "$expected_status" ]; then
        print_status "PASS" "$description"
        if [ -n "$body" ] && [ "$body" != "null" ]; then
            echo "    Response preview: $(echo "$body" | head -c 100)..."
        fi
    else
        print_status "FAIL" "$description (Expected: $expected_status, Got: $status_code)"
        if [ -n "$body" ]; then
            echo "    Response: $body"
        fi
    fi
    echo
}

# Function to test health endpoints
test_health() {
    echo -e "${BLUE}=== Testing Health Endpoints ===${NC}"
    
    test_endpoint "GET" "/health" "200" "Health check (public)"
    test_endpoint "GET" "/v1/status" "200" "Authenticated status check"
}

# Function to test SRS endpoints
test_srs() {
    echo -e "${BLUE}=== Testing SRS (Stochastic Resonance Solver) ===${NC}"
    
    # Test solve endpoint
    local srs_problem='{
        "problem_type": "3sat",
        "clauses": [[1, -2, 3], [-1, 2, -3], [1, 2, 3]],
        "variables": 3,
        "max_iterations": 1000
    }'
    
    test_endpoint "POST" "/v1/srs/solve" "200" "Submit SRS problem" "$srs_problem"
    test_endpoint "GET" "/v1/srs/status/job_abc123" "200" "Get SRS job status"
    test_endpoint "DELETE" "/v1/srs/cancel/job_abc123" "200" "Cancel SRS job"
    test_endpoint "GET" "/v1/srs/history" "200" "Get SRS history"
}

# Function to test HQE endpoints
test_hqe() {
    echo -e "${BLUE}=== Testing HQE (Hypercomplex Quantum Engine) ===${NC}"
    
    local simulation_params='{
        "system_type": "harmonic_oscillator",
        "dimensions": 3,
        "particles": 2,
        "time_steps": 100,
        "precision": "high"
    }'
    
    test_endpoint "POST" "/v1/hqe/simulate" "200" "Start HQE simulation" "$simulation_params"
    test_endpoint "GET" "/v1/hqe/status/sim_xyz789" "200" "Get simulation status"
    
    local optimization_params='{
        "target_energy": -1.5,
        "tolerance": 0.001,
        "method": "gradient_descent"
    }'
    
    test_endpoint "POST" "/v1/hqe/optimize" "200" "Start optimization" "$optimization_params"
    test_endpoint "GET" "/v1/hqe/states/sim_xyz789" "200" "Get quantum states"
}

# Function to test QSEM endpoints
test_qsem() {
    echo -e "${BLUE}=== Testing QSEM (Quantum Semantic Encoding Machine) ===${NC}"
    
    local encoding_request='{
        "concepts": ["quantum", "resonance", "consciousness"],
        "encoding_type": "contextual",
        "dimension": 1536
    }'
    
    test_endpoint "POST" "/v1/qsem/encode" "200" "Encode concepts" "$encoding_request"
    
    local search_request='{
        "query": "quantum consciousness",
        "limit": 10,
        "threshold": 0.8
    }'
    
    test_endpoint "POST" "/v1/qsem/search" "200" "Search concepts" "$search_request"
    test_endpoint "GET" "/v1/qsem/clusters" "200" "Get concept clusters"
    
    local resonance_request='{
        "concept_ids": ["concept_123", "concept_456"],
        "method": "harmonic"
    }'
    
    test_endpoint "POST" "/v1/qsem/resonance" "200" "Compute resonance" "$resonance_request"
}

# Function to test NLC endpoints
test_nlc() {
    echo -e "${BLUE}=== Testing NLC (Non-Local Communication) ===${NC}"
    
    local session_request='{
        "participants": 2,
        "entanglement_type": "bell_state",
        "session_name": "test_session"
    }'
    
    test_endpoint "POST" "/v1/nlc/sessions" "201" "Create NLC session" "$session_request"
    test_endpoint "GET" "/v1/nlc/sessions" "200" "List NLC sessions"
    test_endpoint "GET" "/v1/nlc/sessions/session_abc" "200" "Get session details"
    
    local message_request='{
        "message": "Hello quantum world!",
        "encoding": "quantum_state"
    }'
    
    test_endpoint "POST" "/v1/nlc/sessions/session_abc/send" "200" "Send message" "$message_request"
    test_endpoint "GET" "/v1/nlc/sessions/session_abc/receive" "200" "Receive messages"
    test_endpoint "GET" "/v1/nlc/sessions/session_abc/quality" "200" "Check channel quality"
}

# Function to test QCR endpoints  
test_qcr() {
    echo -e "${BLUE}=== Testing QCR (Quantum Consciousness Resonator) ===${NC}"
    
    local session_request='{
        "consciousness_type": "observer_effect",
        "entanglement_depth": 5,
        "observation_frequency": 10
    }'
    
    test_endpoint "POST" "/v1/qcr/sessions" "201" "Create QCR session" "$session_request"
    test_endpoint "GET" "/v1/qcr/sessions" "200" "List QCR sessions"
    test_endpoint "GET" "/v1/qcr/sessions/qcr_session_123" "200" "Get session details"
    
    local observation_request='{
        "observation_type": "state_collapse",
        "measurement_basis": "computational"
    }'
    
    test_endpoint "POST" "/v1/qcr/sessions/qcr_session_123/observe" "200" "Make observation" "$observation_request"
    
    local evolution_request='{
        "evolution_steps": 100,
        "consciousness_coherence": 0.85
    }'
    
    test_endpoint "POST" "/v1/qcr/sessions/qcr_session_123/evolve" "200" "Evolve consciousness" "$evolution_request"
    test_endpoint "GET" "/v1/qcr/sessions/qcr_session_123/insights" "200" "Get insights"
}

# Function to test I-Ching endpoints
test_iching() {
    echo -e "${BLUE}=== Testing I-Ching Quantum Oracle ===${NC}"
    
    local evolution_request='{
        "question": "What is the nature of quantum consciousness?",
        "initial_state": "random",
        "evolution_steps": 64
    }'
    
    test_endpoint "POST" "/v1/iching/evolve" "200" "Start I-Ching evolution" "$evolution_request"
    test_endpoint "GET" "/v1/iching/hexagrams" "200" "Get hexagram library"
    test_endpoint "GET" "/v1/iching/evolution/evo_123/guidance" "200" "Get guidance"
    test_endpoint "GET" "/v1/iching/evolution/evo_123/insights" "200" "Get insights"
}

# Function to test Unified Physics endpoints
test_unified() {
    echo -e "${BLUE}=== Testing Unified Physics Engine ===${NC}"
    
    local gravity_request='{
        "field_type": "gravitational",
        "coordinates": {"x": 0, "y": 0, "z": 0},
        "masses": [{"mass": 1.0, "position": [1, 0, 0]}],
        "precision": "high"
    }'
    
    test_endpoint "POST" "/v1/unified/gravity" "200" "Compute gravitational field" "$gravity_request"
    
    local field_request='{
        "field_equations": ["einstein_field"],
        "boundary_conditions": {"type": "vacuum"},
        "spacetime_metric": "minkowski"
    }'
    
    test_endpoint "POST" "/v1/unified/fields" "200" "Analyze field equations" "$field_request"
    test_endpoint "GET" "/v1/unified/constants" "200" "Get physical constants"
    
    local simulation_request='{
        "simulation_type": "gravitational_waves",
        "parameters": {"frequency": 100, "amplitude": 1e-21},
        "duration": 10.0
    }'
    
    test_endpoint "POST" "/v1/unified/simulate" "200" "Run physics simulation" "$simulation_request"
}

# Function to test Webhook endpoints
test_webhooks() {
    echo -e "${BLUE}=== Testing Webhook Management ===${NC}"
    
    local webhook_request='{
        "url": "https://api.example.com/webhooks/psizero",
        "events": ["srs.solution.found", "hqe.simulation.complete"],
        "secret": "webhook_secret_123",
        "active": true
    }'
    
    test_endpoint "POST" "/v1/webhooks" "201" "Create webhook" "$webhook_request"
    test_endpoint "GET" "/v1/webhooks" "200" "List webhooks"
    test_endpoint "GET" "/v1/webhooks/wh_abc123" "200" "Get webhook details"
    
    local update_request='{
        "active": false,
        "events": ["qsem.encoding.complete"]
    }'
    
    test_endpoint "PUT" "/v1/webhooks/wh_abc123" "200" "Update webhook" "$update_request"
    
    local test_request='{
        "event_type": "test.webhook",
        "data": {"test": true}
    }'
    
    test_endpoint "POST" "/v1/webhooks/wh_abc123/test" "200" "Test webhook" "$test_request"
    test_endpoint "GET" "/v1/webhooks/wh_abc123/deliveries" "200" "Get deliveries"
    test_endpoint "GET" "/v1/webhooks/events" "200" "Get available events"
    test_endpoint "DELETE" "/v1/webhooks/wh_abc123" "200" "Delete webhook"
}

# Function to run all tests
run_all_tests() {
    echo -e "${BLUE}=========================================${NC}"
    echo -e "${BLUE}  PsiZero Resonance Platform API Tests    ${NC}"
    echo -e "${BLUE}=========================================${NC}"
    echo -e "API Base URL: $API_BASE_URL"
    echo -e "API Key: ${API_KEY:0:10}..."
    echo -e ""
    
    # Wait for API to be ready
    print_status "INFO" "Waiting for API to be ready..."
    for i in {1..30}; do
        if curl -s "$API_BASE_URL/health" > /dev/null 2>&1; then
            print_status "PASS" "API is ready!"
            break
        fi
        if [ $i -eq 30 ]; then
            print_status "FAIL" "API not ready after 30 attempts"
            exit 1
        fi
        sleep 2
    done
    echo
    
    # Run all test suites
    test_health
    test_srs
    test_hqe
    test_qsem
    test_nlc
    test_qcr
    test_iching
    test_unified
    test_webhooks
    
    # Print summary
    echo -e "${BLUE}=========================================${NC}"
    echo -e "${BLUE}           Test Summary                  ${NC}"
    echo -e "${BLUE}=========================================${NC}"
    echo -e "Total Tests:  $TOTAL_TESTS"
    echo -e "${GREEN}Passed:       $PASSED_TESTS${NC}"
    echo -e "${RED}Failed:       $FAILED_TESTS${NC}"
    
    if [ $FAILED_TESTS -gt 0 ]; then
        echo -e "\n${RED}Failed Endpoints:${NC}"
        for endpoint in "${FAILED_ENDPOINTS[@]}"; do
            echo -e "  ${RED}✗${NC} $endpoint"
        done
        exit 1
    else
        echo -e "\n${GREEN}🎉 All tests passed! The PsiZero Resonance Platform API is fully functional.${NC}"
        exit 0
    fi
}

# Main execution
case "${1:-all}" in
    "health")
        test_health
        ;;
    "srs")
        test_srs
        ;;
    "hqe")
        test_hqe
        ;;
    "qsem")
        test_qsem
        ;;
    "nlc")
        test_nlc
        ;;
    "qcr")
        test_qcr
        ;;
    "iching")
        test_iching
        ;;
    "unified")
        test_unified
        ;;
    "webhooks")
        test_webhooks
        ;;
    "all")
        run_all_tests
        ;;
    *)
        echo "Usage: $0 [health|srs|hqe|qsem|nlc|qcr|iching|unified|webhooks|all]"
        echo "  health    - Test health endpoints only"
        echo "  srs       - Test SRS endpoints only"
        echo "  hqe       - Test HQE endpoints only"
        echo "  qsem      - Test QSEM endpoints only"
        echo "  nlc       - Test NLC endpoints only"
        echo "  qcr       - Test QCR endpoints only"
        echo "  iching    - Test I-Ching endpoints only"
        echo "  unified   - Test Unified Physics endpoints only"
        echo "  webhooks  - Test Webhook endpoints only"
        echo "  all       - Run all tests (default)"
        exit 1
        ;;
esac