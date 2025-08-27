from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum

@dataclass
class QualityMetrics:
    structure_score: float
    coherence_score: float
    instruction_adherence: float
    completeness_score: float
    confidence_score: float

class TaskStatus(Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    ESCALATED = "escalated"

class InterventionLevel(Enum):
    WARNING = "warning"
    CORRECTION = "correction"
    ESCALATION = "escalation"

@dataclass
class MonitoringRules:
    quality_threshold: float = 0.7
    escalation_threshold: float = 0.4
    max_token_threshold: int = 16000
    enable_learning: bool = True

@dataclass
class ResourceUsage:
    token_count: int = 0
    api_calls: int = 0

@dataclass
class AgentTask:
    task_id: str
    agent_name: str
    framework: str
    original_input: str
    instructions: List[str]
    status: TaskStatus
    created_at: datetime
    updated_at: datetime
    outputs: List[Dict[str, Any]] = field(default_factory=list)
    interventions: List[Dict[str, Any]] = field(default_factory=list)
    quality_metrics: Optional[QualityMetrics] = None
    resource_usage: ResourceUsage = field(default_factory=ResourceUsage)
    last_known_good_state: Optional[Dict[str, Any]] = None

@dataclass
class SupervisionReport:
    report_id: str
    generated_at: datetime
    tasks_monitored: int
    total_interventions: int
    interventions_by_level: Dict[InterventionLevel, int]
    quality_trends: Dict[str, float]
    common_failures: List[Any]
    recommendations: List[str]
    confidence_distribution: Dict[Any, int]

@dataclass
class EscalationConfig:
    auto_pause_on_escalation: bool = True

@dataclass
class KnowledgeBaseEntry:
    pattern_id: str
    pattern_description: str
    failure_type: str
    common_causes: List[str]
    suggested_fixes: List[str]
    confidence_score: float
    occurrences: int
    last_seen: datetime

class ConfidenceLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

@dataclass
class InterventionRequired:
    intervention_required: bool
    level: Optional[InterventionLevel]
    reason: str
    confidence: float
    pattern_match: Optional[Dict[str, Any]]
