package qcr

import (
	"time"

	"github.com/resonancelab/psizero/core/hilbert"
)

// Missing type definitions for QCR Engine

type ConsolidationProcess struct {
	Type       string        `json:"type"`
	Strength   float64       `json:"strength"`
	Duration   time.Duration `json:"duration"`
	Active     bool          `json:"active"`
	Efficiency float64       `json:"efficiency"`
}

type AttentionalControl struct {
	ControlStrength float64 `json:"control_strength"`
	Flexibility     float64 `json:"flexibility"`
	Inhibition      float64 `json:"inhibition"`
	Updating        float64 `json:"updating"`
}

type MaintenanceMechanism struct {
	Type            string  `json:"type"`
	Strength        float64 `json:"strength"`
	RehearsalRate   float64 `json:"rehearsal_rate"`
	DecayPrevention float64 `json:"decay_prevention"`
}

type CompetitionMechanism struct {
	CompetitionStrength float64 `json:"competition_strength"`
	WinnerTakeAll       bool    `json:"winner_take_all"`
	Threshold           float64 `json:"threshold"`
}

type AccessConsciousness struct {
	AccessibleContent  *hilbert.QuantumState `json:"-"`
	ReportabilityLevel float64               `json:"reportability_level"`
	GlobalAccess       bool                  `json:"global_access"`
}

type PhenomenalConsciousness struct {
	SubjectiveExperience *hilbert.QuantumState `json:"-"`
	QualiaIntensity      float64               `json:"qualia_intensity"`
	PhenomenalBinding    float64               `json:"phenomenal_binding"`
}

type AttentionFilter struct {
	FilterType     string  `json:"filter_type"`
	Selectivity    float64 `json:"selectivity"`
	FilterStrength float64 `json:"filter_strength"`
	Threshold      float64 `json:"threshold"`
}

type AttentionNetwork struct {
	NetworkType  string             `json:"network_type"`
	Connectivity map[string]float64 `json:"connectivity"`
	Efficiency   float64            `json:"efficiency"`
	Alerting     float64            `json:"alerting"`
	Orienting    float64            `json:"orienting"`
	Executive    float64            `json:"executive"`
}

type Coalition struct {
	Members   []string `json:"members"`
	Strength  float64  `json:"strength"`
	Purpose   string   `json:"purpose"`
	Stability float64  `json:"stability"`
}

type MetaCognition struct {
	SelfAwareness     float64 `json:"self_awareness"`
	Introspection     float64 `json:"introspection"`
	CognitiveControl  float64 `json:"cognitive_control"`
	MetaMemory        float64 `json:"meta_memory"`
	MetaComprehension float64 `json:"meta_comprehension"`
}

type SelfReflection struct {
	ReflectionDepth      int     `json:"reflection_depth"`
	SelfModel            string  `json:"self_model"`
	Accuracy             float64 `json:"accuracy"`
	IntrospectiveAbility float64 `json:"introspective_ability"`
}

type PredictiveModel struct {
	ModelType       string             `json:"model_type"`
	Accuracy        float64            `json:"accuracy"`
	Predictions     map[string]float64 `json:"predictions"`
	Confidence      float64            `json:"confidence"`
	UpdateFrequency float64            `json:"update_frequency"`
}

type CausalModel struct {
	CausalChains []CausalChain `json:"causal_chains"`
	Strength     float64       `json:"strength"`
	Reliability  float64       `json:"reliability"`
	Completeness float64       `json:"completeness"`
}

type CausalChain struct {
	Cause    string  `json:"cause"`
	Effect   string  `json:"effect"`
	Strength float64 `json:"strength"`
	Delay    float64 `json:"delay"`
}

type RegulationMechanism struct {
	Type          string  `json:"type"`
	Strength      float64 `json:"strength"`
	Target        string  `json:"target"`
	Effectiveness float64 `json:"effectiveness"`
	Strategy      string  `json:"strategy"`
}

type DecisionProcess struct {
	ProcessID  string             `json:"process_id"`
	Options    []string           `json:"options"`
	Criteria   map[string]float64 `json:"criteria"`
	Selected   string             `json:"selected"`
	Confidence float64            `json:"confidence"`
	Timestamp  time.Time          `json:"timestamp"`
	Duration   time.Duration      `json:"duration"`
}

type ValueSystem struct {
	Values     map[string]float64 `json:"values"`
	Priorities []string           `json:"priorities"`
	Stability  float64            `json:"stability"`
	Coherence  float64            `json:"coherence"`
}

type Decision struct {
	ID         string    `json:"id"`
	Choice     string    `json:"choice"`
	Rationale  string    `json:"rationale"`
	Confidence float64   `json:"confidence"`
	Timestamp  time.Time `json:"timestamp"`
	Outcome    string    `json:"outcome"`
	Quality    float64   `json:"quality"`
}

type ProcessingBuffer struct {
	Capacity    int                     `json:"capacity"`
	CurrentLoad int                     `json:"current_load"`
	Items       []*hilbert.QuantumState `json:"-"`
	Efficiency  float64                 `json:"efficiency"`
	Throughput  float64                 `json:"throughput"`
}

type AdaptationMechanism struct {
	Type          string  `json:"type"`
	Rate          float64 `json:"rate"`
	Threshold     float64 `json:"threshold"`
	Active        bool    `json:"active"`
	Effectiveness float64 `json:"effectiveness"`
}
