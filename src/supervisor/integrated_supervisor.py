#!/usr/bin/env python3
"""
Integrated Supervisor System

Unifies all supervisor components into a single, production-ready system:
- Monitoring capabilities
- Error handling and recovery
- Reporting and analytics
- Framework integration hooks
- Configuration management
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, asdict
from contextlib import asynccontextmanager
import threading
from concurrent.futures import ThreadPoolExecutor

# Import all subsystems
from .monitor_engine import MonitoringEngine
from .task_monitor import TaskCompletionMonitor
from .instruction_monitor import InstructionAdherenceMonitor
from .quality_monitor import OutputQualityMonitor
from .error_tracker import ErrorTracker
from .resource_monitor import ResourceUsageMonitor
from .confidence_scorer import ConfidenceScorer

from .error_handling_system import SupervisorErrorHandlingSystem
from .error_types import SupervisorError, ErrorType

from reporting.integrated_system import (
    IntegratedReportingSystem,
    IntegratedReportingConfig,
    SystemIntegrationEvent,
    EventRouter,
    BackgroundProcessor
)
from reporting.alert_system import RealTimeAlertSystem as ComprehensiveAlertSystem
from reporting.report_generator import PeriodicReportGenerator
from reporting.audit_system import ComprehensiveAuditSystem, AuditEventType, AuditLevel
from reporting.confidence_system import ConfidenceReportingSystem
from reporting.pattern_system import ComprehensivePatternSystem
from reporting.dashboard_system import ComprehensiveDashboardSystem

from .core import SupervisorCore

@dataclass
class SupervisorConfig:
    """Comprehensive configuration for integrated supervisor"""
    # Storage paths
    data_dir: str = "supervisor_data"
    
    # Monitoring settings
    monitoring_enabled: bool = True
    quality_threshold: float = 0.7
    confidence_threshold: float = 0.8
    max_token_threshold: int = 10000
    max_loop_cycles: int = 50
    max_runtime_minutes: float = 30.0
    
    # Error handling settings
    error_handling_enabled: bool = True
    max_retries: int = 3
    rollback_enabled: bool = True
    escalation_enabled: bool = True
    
    # Reporting settings
    reporting_enabled: bool = True
    real_time_alerts: bool = True
    auto_reports: bool = True
    report_frequency_hours: int = 24
    
    # Framework integration
    framework_hooks_enabled: bool = True
    
    # Performance settings
    background_processing: bool = True
    max_workers: int = 4
    log_level: str = "INFO"

class FrameworkIntegrationHooks:
    """Hooks for integrating with different AI agent frameworks"""
    
    def __init__(self, supervisor: 'IntegratedSupervisor'):
        self.supervisor = supervisor
        self.logger = logging.getLogger(__name__)
        
        # Framework-specific adapters
        self.mcp_adapter = MCPFrameworkAdapter(supervisor)
        self.langchain_adapter = LangChainFrameworkAdapter(supervisor)
        self.autogen_adapter = AutoGenFrameworkAdapter(supervisor)
        self.custom_adapter = CustomFrameworkAdapter(supervisor)
    
    def get_adapter(self, framework: str):
        """Get framework-specific adapter"""
        adapters = {
            'mcp': self.mcp_adapter,
            'langchain': self.langchain_adapter,
            'autogen': self.autogen_adapter,
            'custom': self.custom_adapter
        }
        return adapters.get(framework.lower(), self.custom_adapter)

class MCPFrameworkAdapter:
    """MCP-specific integration adapter"""
    
    def __init__(self, supervisor: 'IntegratedSupervisor'):
        self.supervisor = supervisor
    
    async def wrap_tool_call(self, tool_name: str, args: Dict[str, Any], task_id: str):
        """Wrap MCP tool calls with supervision"""
        await self.supervisor.monitoring_engine.start_task_monitoring(
            task_id=task_id,
            task_type="mcp_tool_call",
            metadata={"tool_name": tool_name, "args": args}
        )
        
        return await self.supervisor.execute_supervised_task(
            task_id=task_id,
            task_callable=lambda: self._execute_tool(tool_name, args),
            framework="mcp"
        )
    
    async def _execute_tool(self, tool_name: str, args: Dict[str, Any]):
        """Execute MCP tool with error handling"""
        # Tool execution logic would be implemented here
        # This is a placeholder for demonstration
        return {"result": f"Executed {tool_name} with {args}"}

class LangChainFrameworkAdapter:
    """LangChain-specific integration adapter"""
    
    def __init__(self, supervisor: 'IntegratedSupervisor'):
        self.supervisor = supervisor
    
    async def wrap_chain_execution(self, chain, inputs: Dict[str, Any], task_id: str):
        """Wrap LangChain execution with supervision"""
        await self.supervisor.monitoring_engine.start_task_monitoring(
            task_id=task_id,
            task_type="langchain_execution",
            metadata={"chain_type": str(type(chain)), "inputs": inputs}
        )
        
        return await self.supervisor.execute_supervised_task(
            task_id=task_id,
            task_callable=lambda: chain.arun(inputs),
            framework="langchain"
        )

class AutoGenFrameworkAdapter:
    """AutoGen-specific integration adapter"""
    
    def __init__(self, supervisor: 'IntegratedSupervisor'):
        self.supervisor = supervisor
    
    async def wrap_agent_execution(self, agent, message: str, task_id: str):
        """Wrap AutoGen agent execution with supervision"""
        await self.supervisor.monitoring_engine.start_task_monitoring(
            task_id=task_id,
            task_type="autogen_execution",
            metadata={"agent_name": agent.name, "message": message}
        )
        
        return await self.supervisor.execute_supervised_task(
            task_id=task_id,
            task_callable=lambda: agent.generate_reply(message),
            framework="autogen"
        )

class CustomFrameworkAdapter:
    """Adapter for custom agent frameworks"""
    
    def __init__(self, supervisor: 'IntegratedSupervisor'):
        self.supervisor = supervisor
    
    async def wrap_execution(self, task_callable: Callable, task_id: str, metadata: Dict[str, Any] = None):
        """Wrap custom framework execution with supervision"""
        await self.supervisor.monitoring_engine.start_task_monitoring(
            task_id=task_id,
            task_type="custom_execution",
            metadata=metadata or {}
        )
        
        return await self.supervisor.execute_supervised_task(
            task_id=task_id,
            task_callable=task_callable,
            framework="custom"
        )

class IntegratedSupervisor:
    """Main integrated supervisor system that coordinates all components"""
    
    def __init__(self, reporting_system: IntegratedReportingSystem, config: Optional[SupervisorConfig] = None):
        self.config = config or SupervisorConfig()
        self.reporting_system = reporting_system
        self.data_dir = Path(self.config.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing Integrated Supervisor System")
        
        # Initialize all subsystems
        self._initialize_subsystems()
        
        # Setup framework integration hooks
        self.framework_hooks = FrameworkIntegrationHooks(self) if self.config.framework_hooks_enabled else None
        
        # System state
        self.is_running = False
        self.tasks = {}
        self.system_stats = {
            "start_time": datetime.utcnow(),
            "tasks_supervised": 0,
            "errors_handled": 0,
            "reports_generated": 0
        }
        self.metrics_emitter_thread = None
        
        self.logger.info("Integrated Supervisor System initialized successfully")
    
    def _setup_logging(self):
        """Setup comprehensive logging"""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.data_dir / 'supervisor.log'),
                logging.StreamHandler()
            ]
        )
    
    def _initialize_subsystems(self):
        """Initialize all supervisor subsystems"""
        # Core supervisor
        self.supervisor_core = SupervisorCore()
        
        # Monitoring system
        if self.config.monitoring_enabled:
            self.monitoring_engine = MonitoringEngine()
            self.task_monitor = TaskCompletionMonitor()
            self.instruction_monitor = InstructionAdherenceMonitor()
            self.quality_monitor = OutputQualityMonitor()
            self.error_tracker = ErrorTracker()
            self.resource_monitor = ResourceUsageMonitor(event_router=self.reporting_system.event_router if self.reporting_system else None)
            self.confidence_scorer = ConfidenceScorer()
        
        # Error handling system
        if self.config.error_handling_enabled:
            alert_system = self.reporting_system.systems.get('alert_system') if self.reporting_system else None
            self.error_handling_system = SupervisorErrorHandlingSystem(
                storage_path=self.data_dir / "error_handling",
                max_retries=self.config.max_retries,
                escalation_enabled=self.config.escalation_enabled,
                alert_system=alert_system
            )
        
        # Reporting system is now passed in
    
    def _metrics_emitter_loop(self):
        """Periodically emits system metrics."""
        while self.is_running:
            try:
                # Since this is running in a separate thread, we need to run the async func
                status = asyncio.run(self.get_system_status())

                if self.reporting_system:
                    metrics_event = SystemIntegrationEvent(
                        event_id=f"metrics_{datetime.utcnow().timestamp()}",
                        timestamp=datetime.utcnow().isoformat(),
                        event_type="system_metrics_update",
                        source_system="integrated_supervisor",
                        data=status
                    )
                    self.reporting_system.event_router.route_event(metrics_event)
            except Exception as e:
                self.logger.error(f"Error in metrics emitter loop: {e}")

            time.sleep(5) # Emit metrics every 5 seconds
    
    async def start(self):
        """Start the integrated supervisor system"""
        if self.is_running:
            return
        
        self.logger.info("Starting Integrated Supervisor System")
        
        # Start reporting system
        if self.reporting_system:
            self.reporting_system.start()
        
        self.is_running = True

        # Start metrics emitter
        if self.config.reporting_enabled:
            self.metrics_emitter_thread = threading.Thread(
                target=self._metrics_emitter_loop,
                daemon=True
            )
            self.metrics_emitter_thread.start()

        self.logger.info("Integrated Supervisor System started successfully")
    
    async def stop(self):
        """Stop the integrated supervisor system"""
        if not self.is_running:
            return
        
        self.logger.info("Stopping Integrated Supervisor System")
        
        # Stop reporting system
        if self.reporting_system:
            self.reporting_system.stop()
        
        self.is_running = False
        self.logger.info("Integrated Supervisor System stopped")
    
    async def execute_supervised_task(
        self,
        task_id: str,
        task_callable: Callable,
        framework: str = "unknown",
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute a task under full supervision"""
        
        self.system_stats["tasks_supervised"] += 1
        start_time = time.time()

        try:
            # Start monitoring
            if self.config.monitoring_enabled:
                await self.monitoring_engine.start_task_monitoring(
                    task_id=task_id,
                    task_type=f"{framework}_task",
                    metadata=context or {}
                )
            
            # Execute task with error handling
            if self.config.error_handling_enabled:
                try:
                    result = await asyncio.get_event_loop().run_in_executor(
                        None, task_callable
                    )
                except Exception as e:
                    # Handle error through error handling system
                    error_result = await self.error_handling_system.handle_error(
                        error=e,
                        agent_id=f"{framework}_agent",
                        task_id=task_id,
                        context=context,
                        recovery_callback=task_callable
                    )
                    
                    self.system_stats["errors_handled"] += 1
                    
                    # Route error event
                    if self.reporting_system:
                        error_event = SystemIntegrationEvent(
                            event_id=f"error_{task_id}_{datetime.utcnow().timestamp()}",
                            timestamp=datetime.utcnow().isoformat(),
                            event_type="error_occurred",
                            source_system="integrated_supervisor",
                            data={
                                "task_id": task_id,
                                "error": str(e),
                                "framework": framework,
                                "error_handling_result": error_result
                            }
                        )
                        self.reporting_system.event_router.route_event(error_event)
                    
                    if not error_result.get("success", False):
                        raise e
                    
                    # If error handling succeeded, try to get result
                    result = error_result.get("recovered_result")
            else:
                result = await asyncio.get_event_loop().run_in_executor(
                    None, task_callable
                )

            # Extract token usage if available
            token_data = {}
            if isinstance(result, dict) and 'usage' in result:
                usage = result.get('usage', {})
                token_data = {
                    'input_tokens': usage.get('prompt_tokens', 0),
                    'output_tokens': usage.get('completion_tokens', 0),
                    'total_tokens': usage.get('total_tokens', 0),
                    'api_call': True
                }

            # Update resource monitor
            if self.config.monitoring_enabled and hasattr(self, 'resource_monitor'):
                resource_data = {
                    'token_data': token_data,
                    'execution_time': time.time() - start_time
                }
                self.resource_monitor.evaluate_usage(resource_data)

            # Validate output if monitoring enabled
            if self.config.monitoring_enabled and result:
                # --- Tiered Response System: Warning Tier ---

                # 1. Check for instruction drift
                if hasattr(self, 'instruction_monitor'):
                    adherence_result = self.instruction_monitor.evaluate_adherence(
                        instructions=context.get("instructions", []),
                        agent_steps=[str(result)], # Assuming result is a single step
                        constraints=context.get("constraints", {})
                    )
                    if adherence_result['score'] < 0.8:
                        warning_error = SupervisorError(
                            message=f"Instruction adherence score is low: {adherence_result['score']:.2f}",
                            error_type=ErrorType.INSTRUCTION_DRIFT_WARNING,
                            context={**context, 'adherence_result': adherence_result}
                        )
                        await self.error_handling_system.handle_error(
                            error=warning_error,
                            agent_id=f"{framework}_agent",
                            task_id=task_id,
                            context=context,
                            recovery_callback=task_callable
                        )

                # 2. Check for quality degradation
                if hasattr(self, 'quality_monitor'):
                    quality_result = self.quality_monitor.evaluate_output_quality(
                        outputs=[str(result)],
                        expected_format=context.get("expected_format", {})
                    )
                    if quality_result['score'] < 0.7:
                        warning_error = SupervisorError(
                            message=f"Output quality score is low: {quality_result['score']:.2f}",
                            error_type=ErrorType.QUALITY_DEGRADATION_WARNING,
                            context={**context, 'quality_result': quality_result}
                        )
                        await self.error_handling_system.handle_error(
                            error=warning_error,
                            agent_id=f"{framework}_agent",
                            task_id=task_id,
                            context=context,
                            recovery_callback=task_callable
                        )

                # Note: The original quality/confidence score logic is now replaced by the above checks.
                # The following is left here for reference but should be removed in a future refactoring.
                quality_score = await self.quality_monitor.analyze_output(
                    output=str(result),
                    expected_format="json" if isinstance(result, dict) else "text",
                    task_instructions=context.get("instructions", []) if context else []
                )
                
                confidence_score = self.confidence_scorer.calculate_confidence(
                    task_type=f"{framework}_task",
                    output_quality=quality_score,
                    error_count=0,  # Would be tracked properly in real implementation
                    completion_time=time.time() - start_time
                )
            
            # Route success event
            if self.reporting_system:
                success_event = SystemIntegrationEvent(
                    event_id=f"success_{task_id}_{datetime.utcnow().timestamp()}",
                    timestamp=datetime.utcnow().isoformat(),
                    event_type="task_completed",
                    source_system="integrated_supervisor",
                    data={
                        "task_id": task_id,
                        "framework": framework,
                        "success": True,
                        "result_type": type(result).__name__
                    }
                )
                self.reporting_system.event_router.route_event(success_event)
            
            return {
                "success": True,
                "result": result,
                "task_id": task_id,
                "framework": framework,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Task execution failed: {e}")
            raise
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        status = {
            "system_info": {
                "is_running": self.is_running,
                "uptime": str(datetime.utcnow() - self.system_stats["start_time"]),
                "config": asdict(self.config)
            },
            "statistics": self.system_stats,
            "subsystems": {
                "monitoring": self.config.monitoring_enabled,
                "error_handling": self.config.error_handling_enabled,
                "reporting": self.config.reporting_enabled,
                "framework_hooks": self.config.framework_hooks_enabled
            }
        }
        
        # Add detailed subsystem status
        if hasattr(self, 'error_handling_system'):
            status["error_handling_stats"] = self.error_handling_system.system_stats

        if hasattr(self, 'resource_monitor'):
            usage_summary = self.resource_monitor.get_usage_summary()
            status['statistics']['token_usage'] = usage_summary.get('token_usage', {})

            # Calculate cost
            token_usage = usage_summary.get('token_usage', {})
            input_tokens = token_usage.get('input_tokens', 0)
            output_tokens = token_usage.get('output_tokens', 0)

            # TODO: Centralize cost configuration
            input_cost_per_1k = 0.001
            output_cost_per_1k = 0.003

            input_cost = (input_tokens / 1000) * input_cost_per_1k
            output_cost = (output_tokens / 1000) * output_cost_per_1k
            total_cost = input_cost + output_cost

            status['statistics']['estimated_cost'] = total_cost
        
        return status
    
    @asynccontextmanager
    async def supervision_context(self, task_id: str, framework: str = "unknown"):
        """Context manager for supervised task execution"""
        try:
            # Setup supervision context
            if self.config.monitoring_enabled:
                await self.monitoring_engine.start_task_monitoring(
                    task_id=task_id,
                    task_type=f"{framework}_context"
                )
            
            yield self
            
        finally:
            # Cleanup supervision context
            if self.config.monitoring_enabled:
                await self.monitoring_engine.stop_task_monitoring(task_id)
