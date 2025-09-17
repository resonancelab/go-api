package services

import (
	"fmt"
	"sync"

	"github.com/resonancelab/psizero/engines/hqe"
	"github.com/resonancelab/psizero/engines/iching"
	"github.com/resonancelab/psizero/engines/nlc"
	"github.com/resonancelab/psizero/engines/qcr"
	"github.com/resonancelab/psizero/engines/qsem"
	"github.com/resonancelab/psizero/engines/srs"
	"github.com/resonancelab/psizero/engines/unified"
	"github.com/resonancelab/psizero/shared/types"
)

// ServiceContainer holds all initialized services and engines
type ServiceContainer struct {
	// Core engines
	SRSEngine     *srs.SRSEngine
	HQEEngine     *hqe.HQEEngine
	QSEMEngine    *qsem.QSEMEngine
	NLCEngine     *nlc.NLCEngine
	QCREngine     *qcr.QCREngine
	IChingEngine  *iching.IChingEngine
	UnifiedEngine *unified.UnifiedEngine

	// Configuration
	Config *types.Config

	// Initialization state
	initialized bool
	mu          sync.RWMutex
}

// NewServiceContainer creates and initializes a new service container
func NewServiceContainer(config *types.Config) (*ServiceContainer, error) {
	container := &ServiceContainer{
		Config: config,
	}

	if err := container.InitializeServices(); err != nil {
		return nil, fmt.Errorf("failed to initialize services: %w", err)
	}

	return container, nil
}

// InitializeServices initializes all engines and services
func (sc *ServiceContainer) InitializeServices() error {
	sc.mu.Lock()
	defer sc.mu.Unlock()

	if sc.initialized {
		return nil // Already initialized
	}

	// Initialize SRS Engine
	var err error
	sc.SRSEngine, err = srs.NewSRSEngine()
	if err != nil {
		return fmt.Errorf("failed to initialize SRS engine: %w", err)
	}

	// Initialize HQE Engine
	sc.HQEEngine, err = hqe.NewHQEEngine()
	if err != nil {
		return fmt.Errorf("failed to initialize HQE engine: %w", err)
	}

	// Initialize QSEM Engine
	sc.QSEMEngine, err = qsem.NewQSEMEngine()
	if err != nil {
		return fmt.Errorf("failed to initialize QSEM engine: %w", err)
	}

	// Initialize NLC Engine
	sc.NLCEngine, err = nlc.NewNLCEngine()
	if err != nil {
		return fmt.Errorf("failed to initialize NLC engine: %w", err)
	}

	// Initialize QCR Engine
	sc.QCREngine, err = qcr.NewQCREngine()
	if err != nil {
		return fmt.Errorf("failed to initialize QCR engine: %w", err)
	}

	// Initialize I-Ching Engine
	sc.IChingEngine, err = iching.NewIChingEngine()
	if err != nil {
		return fmt.Errorf("failed to initialize I-Ching engine: %w", err)
	}

	// Initialize Unified Physics Engine
	sc.UnifiedEngine, err = unified.NewUnifiedEngine()
	if err != nil {
		return fmt.Errorf("failed to initialize Unified Physics engine: %w", err)
	}

	sc.initialized = true
	return nil
}

// GetSRSEngine returns the SRS engine instance
func (sc *ServiceContainer) GetSRSEngine() *srs.SRSEngine {
	sc.mu.RLock()
	defer sc.mu.RUnlock()
	return sc.SRSEngine
}

// GetHQEEngine returns the HQE engine instance
func (sc *ServiceContainer) GetHQEEngine() *hqe.HQEEngine {
	sc.mu.RLock()
	defer sc.mu.RUnlock()
	return sc.HQEEngine
}

// GetQSEMEngine returns the QSEM engine instance
func (sc *ServiceContainer) GetQSEMEngine() *qsem.QSEMEngine {
	sc.mu.RLock()
	defer sc.mu.RUnlock()
	return sc.QSEMEngine
}

// GetNLCEngine returns the NLC engine instance
func (sc *ServiceContainer) GetNLCEngine() *nlc.NLCEngine {
	sc.mu.RLock()
	defer sc.mu.RUnlock()
	return sc.NLCEngine
}

// GetQCREngine returns the QCR engine instance
func (sc *ServiceContainer) GetQCREngine() *qcr.QCREngine {
	sc.mu.RLock()
	defer sc.mu.RUnlock()
	return sc.QCREngine
}

// GetIChingEngine returns the I-Ching engine instance
func (sc *ServiceContainer) GetIChingEngine() *iching.IChingEngine {
	sc.mu.RLock()
	defer sc.mu.RUnlock()
	return sc.IChingEngine
}

// GetUnifiedEngine returns the Unified Physics engine instance
func (sc *ServiceContainer) GetUnifiedEngine() *unified.UnifiedEngine {
	sc.mu.RLock()
	defer sc.mu.RUnlock()
	return sc.UnifiedEngine
}

// IsInitialized returns whether the container has been fully initialized
func (sc *ServiceContainer) IsInitialized() bool {
	sc.mu.RLock()
	defer sc.mu.RUnlock()
	return sc.initialized
}

// Shutdown gracefully shuts down all services
func (sc *ServiceContainer) Shutdown() error {
	sc.mu.Lock()
	defer sc.mu.Unlock()

	// Add cleanup logic here if engines need graceful shutdown
	// For now, just mark as uninitialized
	sc.initialized = false

	return nil
}

// HealthCheck performs health checks on all engines
func (sc *ServiceContainer) HealthCheck() map[string]bool {
	sc.mu.RLock()
	defer sc.mu.RUnlock()

	health := make(map[string]bool)

	// Check each engine
	health["srs"] = sc.SRSEngine != nil
	health["hqe"] = sc.HQEEngine != nil
	health["qsem"] = sc.QSEMEngine != nil
	health["nlc"] = sc.NLCEngine != nil
	health["qcr"] = sc.QCREngine != nil
	health["iching"] = sc.IChingEngine != nil
	health["unified"] = sc.UnifiedEngine != nil
	health["container"] = sc.initialized

	return health
}
