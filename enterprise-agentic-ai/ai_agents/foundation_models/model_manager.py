"""
Model Manager - Coordinates multiple AI models and implements XBOW-style alloying
Manages model selection, load balancing, and cost optimization for the pentesting platform
"""

import asyncio
import json
import logging
import os
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import random
import statistics
from collections import defaultdict

from .llm_interface import LLMInterface, BaseLLMAgent, ModelCapability, TaskType, LLMRequest, LLMResponse
from .deepseek_r1_agent import DeepSeekR1Agent, DeepSeekConfig
from .Gemini_2_5_pro import GeminiAgent, GeminiConfig # type: ignore

logger = logging.getLogger(__name__)

class AlloyStrategy(Enum):
    """Different strategies for model alloying"""
    RANDOM = "random"
    TASK_BASED = "task_based"
    PERFORMANCE_BASED = "performance_based"
    COST_OPTIMIZED = "cost_optimized"
    ADAPTIVE = "adaptive"

@dataclass
class AlloyConfig:
    """Configuration for model alloying"""
    strategy: AlloyStrategy = AlloyStrategy.TASK_BASED
    models: List[str] = field(default_factory=list)
    task_distribution: Dict[str, float] = field(default_factory=dict)
    cost_threshold: float = 1000.0  # Max cost per session in rupees
    performance_weight: float = 0.7
    cost_weight: float = 0.3

@dataclass
class ModelPerformance:
    """Track model performance metrics"""
    success_rate: float = 0.0
    average_latency: float = 0.0
    average_cost: float = 0.0
    total_requests: int = 0
    recent_failures: int = 0
    last_updated: datetime = field(default_factory=datetime.now)

class ModelManager:
    """Manages multiple AI models with XBOW-style alloying capabilities"""
    
    def __init__(self, llm_interface: Optional[LLMInterface] = None):
        self.llm_interface = llm_interface or LLMInterface()
        self.alloy_config = AlloyConfig()
        self.model_performance: Dict[str, ModelPerformance] = {}
        self.conversation_contexts: Dict[str, List[Dict]] = {}  # session_id -> context
        self.task_history: Dict[str, List[Tuple[str, TaskType, bool]]] = {}  # session_id -> [(model, task, success)]
        self.cost_tracking: Dict[str, float] = {}  # session_id -> total_cost
        
        # Model specialization mapping (like XBOW's approach)
        self.task_specialization = {
            TaskType.EXPLOIT_GENERATION: ["deepseek_r1_agent"],
            TaskType.VULNERABILITY_ANALYSIS: ["deepseek_r1_agent"],
            TaskType.RECONNAISSANCE_PLANNING: ["deepseek_r1_agent"],
            TaskType.PAYLOAD_GENERATION: ["deepseek_r1_agent"],
            TaskType.CODE_GENERATION: ["deepseek_r1_agent"],
            TaskType.REASONING: ["deepseek_r1_agent"],
            TaskType.REPORT_GENERATION: ["gemini_2_5_pro_agent"]
        }
        
        # Cost optimization thresholds (in rupees) - targeting ₹1000/session
        self.cost_thresholds = {
            "session_limit": 1000,  # Max cost per pentest session
            "warning_level": 750,   # Warning threshold
            "emergency_fallback": 950  # Emergency fallback to cheaper models
        }
    
    async def initialize(self):
        """Initialize the model manager"""
        logger.info("Initializing Model Manager with alloying capabilities")
        
        # Initialize LLM interface
        await self.llm_interface.initialize()
        
        # Initialize DeepSeek R1 agent
        deepseek_config = DeepSeekConfig(
            api_key=os.getenv("DEEPSEEK_API_KEY", ""),
            temperature=0.1,
            max_tokens=4000
        )
        deepseek_agent = DeepSeekR1AgentWrapper(deepseek_config)
        await deepseek_agent.initialize()
        
        # Initialize Gemini agent
        gemini_config = GeminiConfig(
            api_key=os.getenv("GEMINI_API_KEY", ""),
            temperature=0.1
        )
        gemini_agent = GeminiAgentWrapper(gemini_config)
        await gemini_agent.initialize()
        
        # Register agents with LLM interface
        self.llm_interface.register_agent(deepseek_agent, priority=9)
        self.llm_interface.register_agent(gemini_agent, priority=8)
        
        # Initialize performance tracking
        for agent_id in ["deepseek_r1_agent", "gemini_2_5_pro_agent"]:
            self.model_performance[agent_id] = ModelPerformance()
        
        logger.info("Model Manager initialized successfully")
    
    async def process_alloyed_request(self, 
                                    session_id: str,
                                    task_type: TaskType,
                                    prompt: str,
                                    context: Optional[Dict[str, Any]] = None) -> LLMResponse:
        """Process request using alloyed model selection"""
        
        # Check cost limits
        session_cost = self.cost_tracking.get(session_id, 0.0)
        if session_cost >= self.cost_thresholds["session_limit"]:
            logger.warning(f"Session {session_id} hit cost limit: ₹{session_cost}")
            return self._create_error_response("Session cost limit exceeded")
        
        # Select model using alloying strategy
        selected_model = self._select_alloyed_model(session_id, task_type, session_cost)
        
        # Build request with conversation context
        request = self._build_contextual_request(session_id, task_type, prompt, context, selected_model)
        
        # Process request
        response = await self.llm_interface.process_request(request)
        
        # Update tracking
        self._update_performance_metrics(selected_model, response)
        self._update_conversation_context(session_id, request.prompt, response.content)
        self._update_cost_tracking(session_id, response.cost)
        
        # Log alloying decision
        self._log_alloying_decision(session_id, task_type, selected_model, response)
        
        return response
    
    def _select_alloyed_model(self, session_id: str, task_type: TaskType, current_cost: float) -> str:
        """Select model using XBOW-style alloying strategy"""
        
        if self.alloy_config.strategy == AlloyStrategy.RANDOM:
            return self._random_selection(task_type)
        
        elif self.alloy_config.strategy == AlloyStrategy.TASK_BASED:
            return self._task_based_selection(session_id, task_type)
        
        elif self.alloy_config.strategy == AlloyStrategy.PERFORMANCE_BASED:
            return self._performance_based_selection(task_type)
        
        elif self.alloy_config.strategy == AlloyStrategy.COST_OPTIMIZED:
            return self._cost_optimized_selection(task_type, current_cost)
        
        elif self.alloy_config.strategy == AlloyStrategy.ADAPTIVE:
            return self._adaptive_selection(session_id, task_type, current_cost)
        
        else:
            return self._default_selection(task_type)
    
    def _random_selection(self, task_type: TaskType) -> str:
        """XBOW-style random model selection"""
        available_models = self.task_specialization.get(task_type, ["deepseek_r1_agent"])
        return random.choice(available_models)
    
    def _task_based_selection(self, session_id: str, task_type: TaskType) -> str:
        """Select model based on task type and session history"""
        
        # Get specialized models for task
        specialized_models = self.task_specialization.get(task_type, ["deepseek_r1_agent"])
        
        # Check session history for diversity (XBOW alloying concept)
        history = self.task_history.get(session_id, [])
        recent_models = [entry[0] for entry in history[-5:]]  # Last 5 model uses
        
        # Prefer models not recently used (encourage diversity)
        unused_models = [model for model in specialized_models if model not in recent_models]
        
        if unused_models:
            return random.choice(unused_models)
        else:
            return specialized_models[0]  # Fall back to primary model
    
    def _performance_based_selection(self, task_type: TaskType) -> str:
        """Select based on performance metrics"""
        available_models = self.task_specialization.get(task_type, ["deepseek_r1_agent"])
        
        best_model = available_models[0]
        best_score = 0.0
        
        for model_id in available_models:
            perf = self.model_performance.get(model_id, ModelPerformance())
            
            # Calculate composite score
            score = (perf.success_rate * 0.6 + 
                    (1.0 / (1.0 + perf.average_latency)) * 0.4)
            
            if score > best_score:
                best_score = score
                best_model = model_id
        
        return best_model
    
    def _cost_optimized_selection(self, task_type: TaskType, current_cost: float) -> str:
        """Select model optimized for cost"""
        available_models = self.task_specialization.get(task_type, ["deepseek_r1_agent"])
        
        # If approaching cost limit, prefer cheaper models
        if current_cost > self.cost_thresholds["warning_level"]:
            # Prefer Gemini for report generation (free tier)
            if task_type == TaskType.REPORT_GENERATION:
                return "gemini_2_5_pro_agent"
            
            # For other tasks, calculate cost per token
            cheapest_model = available_models[0]
            lowest_cost = float('inf')
            
            for model_id in available_models:
                perf = self.model_performance.get(model_id, ModelPerformance())
                if perf.average_cost < lowest_cost:
                    lowest_cost = perf.average_cost
                    cheapest_model = model_id
            
            return cheapest_model
        
        # Normal cost situation, use performance-based selection
        return self._performance_based_selection(task_type)
    
    def _adaptive_selection(self, session_id: str, task_type: TaskType, current_cost: float) -> str:
        """Adaptive selection based on multiple factors"""
        available_models = self.task_specialization.get(task_type, ["deepseek_r1_agent"])
        
        scores = {}
        
        for model_id in available_models:
            perf = self.model_performance.get(model_id, ModelPerformance())
            
            # Performance score
            perf_score = perf.success_rate * (1.0 / (1.0 + perf.average_latency))
            
            # Cost score (lower cost is better)
            cost_score = 1.0 / (1.0 + perf.average_cost)
            
            # Diversity score (prefer recently unused models)
            history = self.task_history.get(session_id, [])
            recent_uses = sum(1 for entry in history[-10:] if entry[0] == model_id)
            diversity_score = 1.0 / (1.0 + recent_uses)
            
            # Combined score
            total_score = (perf_score * 0.5 + 
                          cost_score * 0.3 + 
                          diversity_score * 0.2)
            
            scores[model_id] = total_score
        
        # Select model with highest score
        best_model = max(scores.items(), key=lambda x: x[1])[0]
        return best_model
    
    def _default_selection(self, task_type: TaskType) -> str:
        """Default model selection"""
        return self.task_specialization.get(task_type, ["deepseek_r1_agent"])[0]
    
    def _build_contextual_request(self, 
                                session_id: str,
                                task_type: TaskType,
                                prompt: str,
                                context: Optional[Dict[str, Any]],
                                model_id: str) -> LLMRequest:
        """Build request with conversation context"""
        
        # Get conversation context
        conversation = self.conversation_contexts.get(session_id, [])
        
        # Build enhanced context
        enhanced_context = context or {}
        enhanced_context["conversation_history"] = conversation[-5:]  # Last 5 exchanges
        enhanced_context["session_id"] = session_id
        enhanced_context["model_sequence"] = [entry[0] for entry in self.task_history.get(session_id, [])][-3:]
        
        # Adjust prompt for model switching (XBOW alloying style)
        if len(conversation) > 0:
            contextual_prompt = self._build_alloyed_prompt(prompt, conversation, model_id)
        else:
            contextual_prompt = prompt
        
        return LLMRequest(
            task_type=task_type,
            prompt=contextual_prompt,
            context=enhanced_context,
            model_preference=model_id,
            temperature=0.1 if task_type in [TaskType.EXPLOIT_GENERATION, TaskType.CODE_GENERATION] else 0.2
        )
    
    def _build_alloyed_prompt(self, prompt: str, conversation: List[Dict], model_id: str) -> str:
        """Build prompt for model alloying (XBOW style)"""
        
        # Add context from previous exchanges without revealing model switches
        context_summary = ""
        if len(conversation) >= 2:
            last_exchange = conversation[-2:]  # Last question and answer
            context_summary = f"Previous context:\nQ: {last_exchange[0]['content']}\nA: {last_exchange[1]['content']}\n\n"
        
        # Build alloyed prompt that maintains conversation continuity
        alloyed_prompt = f"{context_summary}Current request: {prompt}"
        
        return alloyed_prompt
    
    def _update_performance_metrics(self, model_id: str, response: LLMResponse):
        """Update model performance metrics"""
        if model_id not in self.model_performance:
            self.model_performance[model_id] = ModelPerformance()
        
        perf = self.model_performance[model_id]
        
        # Update counters
        perf.total_requests += 1
        
        # Update success rate (rolling average)
        alpha = 0.1  # Learning rate
        if response.success:
            perf.success_rate = alpha * 1.0 + (1 - alpha) * perf.success_rate
            perf.recent_failures = max(0, perf.recent_failures - 1)
        else:
            perf.success_rate = alpha * 0.0 + (1 - alpha) * perf.success_rate
            perf.recent_failures += 1
        
        # Update latency (rolling average)
        perf.average_latency = alpha * response.latency + (1 - alpha) * perf.average_latency
        
        # Update cost (rolling average)
        perf.average_cost = alpha * response.cost + (1 - alpha) * perf.average_cost
        
        perf.last_updated = datetime.now()
    
    def _update_conversation_context(self, session_id: str, prompt: str, response: str):
        """Update conversation context"""
        if session_id not in self.conversation_contexts:
            self.conversation_contexts[session_id] = []
        
        context = self.conversation_contexts[session_id]
        context.append({"role": "user", "content": prompt})
        context.append({"role": "assistant", "content": response})
        
        # Limit context size
        if len(context) > 20:
            context = context[-20:]  # Keep last 20 messages
            self.conversation_contexts[session_id] = context
    
    def _update_cost_tracking(self, session_id: str, cost: float):
        """Update cost tracking for session"""
        if session_id not in self.cost_tracking:
            self.cost_tracking[session_id] = 0.0
        
        self.cost_tracking[session_id] += cost
        
        # Log cost warnings
        total_cost = self.cost_tracking[session_id]
        if total_cost > self.cost_thresholds["warning_level"]:
            logger.warning(f"Session {session_id} cost: ₹{total_cost:.2f}")
    
    def _log_alloying_decision(self, session_id: str, task_type: TaskType, model_id: str, response: LLMResponse):
        """Log alloying decision for analysis"""
        if session_id not in self.task_history:
            self.task_history[session_id] = []
        
        self.task_history[session_id].append((model_id, task_type, response.success))
        
        logger.debug(f"Alloying: {session_id} -> {task_type.value} -> {model_id} (Success: {response.success})")
    
    def _create_error_response(self, error_message: str) -> LLMResponse:
        """Create error response"""
        return LLMResponse(
            success=False,
            content="",
            model_used="none",
            task_type=TaskType.REASONING,
            token_count=0,
            latency=0,
            cost=0,
            error=error_message
        )
    
    async def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get summary of session activity"""
        return {
            "session_id": session_id,
            "total_cost": self.cost_tracking.get(session_id, 0.0),
            "total_requests": len(self.task_history.get(session_id, [])),
            "model_usage": self._get_model_usage_stats(session_id),
            "task_distribution": self._get_task_distribution(session_id),
            "success_rate": self._get_session_success_rate(session_id),
            "conversation_length": len(self.conversation_contexts.get(session_id, []))
        }
    
    def _get_model_usage_stats(self, session_id: str) -> Dict[str, int]:
        """Get model usage statistics for session"""
        history = self.task_history.get(session_id, [])
        usage = defaultdict(int)
        
        for model_id, _, _ in history:
            usage[model_id] += 1
        
        return dict(usage)
    
    def _get_task_distribution(self, session_id: str) -> Dict[str, int]:
        """Get task type distribution for session"""
        history = self.task_history.get(session_id, [])
        distribution = defaultdict(int)
        
        for _, task_type, _ in history:
            distribution[task_type.value] += 1
        
        return dict(distribution)
    
    def _get_session_success_rate(self, session_id: str) -> float:
        """Get success rate for session"""
        history = self.task_history.get(session_id, [])
        if not history:
            return 0.0
        
        successes = sum(1 for _, _, success in history if success)
        return successes / len(history)
    
    async def optimize_costs(self):
        """Optimize model selection based on cost analysis"""
        logger.info("Running cost optimization analysis")
        
        # Analyze cost patterns
        total_sessions = len(self.cost_tracking)
        total_cost = sum(self.cost_tracking.values())
        
        if total_sessions > 0:
            avg_cost_per_session = total_cost / total_sessions
            
            # Adjust strategy based on cost performance
            if avg_cost_per_session > self.cost_thresholds["warning_level"]:
                logger.warning(f"High average cost per session: ₹{avg_cost_per_session:.2f}")
                self.alloy_config.strategy = AlloyStrategy.COST_OPTIMIZED
                self.alloy_config.cost_weight = 0.6
                self.alloy_config.performance_weight = 0.4
            else:
                # Normal cost performance, optimize for performance
                self.alloy_config.strategy = AlloyStrategy.ADAPTIVE
                self.alloy_config.cost_weight = 0.3
                self.alloy_config.performance_weight = 0.7
    
    async def cleanup_old_sessions(self, max_age_hours: int = 24):
        """Clean up old session data"""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        # Clean up conversation contexts
        old_sessions = []
        for session_id in self.conversation_contexts:
            # Simple heuristic: if session_id looks like timestamp
            try:
                session_time = datetime.fromtimestamp(float(session_id.split('_')[-1]))
                if session_time < cutoff_time:
                    old_sessions.append(session_id)
            except:
                continue
        
        for session_id in old_sessions:
            self.conversation_contexts.pop(session_id, None)
            self.task_history.pop(session_id, None)
            self.cost_tracking.pop(session_id, None)
        
        logger.info(f"Cleaned up {len(old_sessions)} old sessions")
    
    async def get_cost_analysis(self) -> Dict[str, Any]:
        """Get detailed cost analysis"""
        total_cost = sum(self.cost_tracking.values())
        total_sessions = len(self.cost_tracking)
        
        analysis = {
            "total_cost": total_cost,
            "total_sessions": total_sessions,
            "average_cost_per_session": total_cost / max(total_sessions, 1),
            "session_breakdown": self.cost_tracking.copy(),
            "cost_by_model": {},
            "target_metrics": {
                "target_cost_per_session": 1000,  # ₹1000 target
                "astra_comparison": {
                    "astra_min_cost": 70 * 82.5,  # $70 in INR
                    "astra_max_cost": 700 * 82.5,  # $700 in INR
                    "our_target": 1000
                }
            }
        }
        
        # Calculate cost by model
        for model_id, perf in self.model_performance.items():
            analysis["cost_by_model"][model_id] = {
                "total_requests": perf.total_requests,
                "average_cost": perf.average_cost,
                "estimated_total_cost": perf.total_requests * perf.average_cost
            }
        
        return analysis
    
    async def set_strategy(self, strategy: AlloyStrategy, **kwargs):
        """Set alloying strategy with optional parameters"""
        self.alloy_config.strategy = strategy
        
        # Update configuration based on kwargs
        if "cost_threshold" in kwargs:
            self.alloy_config.cost_threshold = kwargs["cost_threshold"]
        
        if "performance_weight" in kwargs:
            self.alloy_config.performance_weight = kwargs["performance_weight"]
        
        if "cost_weight" in kwargs:
            self.alloy_config.cost_weight = kwargs["cost_weight"]
        
        logger.info(f"Updated alloying strategy to {strategy.value}")
    
    async def get_model_performance_stats(self) -> Dict[str, Any]:
        """Get detailed model performance statistics"""
        stats = {}
        
        for model_id, perf in self.model_performance.items():
            stats[model_id] = {
                "success_rate": perf.success_rate,
                "average_latency": perf.average_latency,
                "average_cost": perf.average_cost,
                "total_requests": perf.total_requests,
                "recent_failures": perf.recent_failures,
                "last_updated": perf.last_updated.isoformat(),
                "cost_efficiency": perf.success_rate / max(perf.average_cost, 0.001),  # Success per rupee
                "performance_score": perf.success_rate * (1.0 / (1.0 + perf.average_latency))
            }
        
        return stats


class DeepSeekR1AgentWrapper(BaseLLMAgent):
    """Wrapper for DeepSeek R1 Agent"""
    
    def __init__(self, config: DeepSeekConfig):
        capabilities = ModelCapability(
            task_types=[
                TaskType.EXPLOIT_GENERATION,
                TaskType.VULNERABILITY_ANALYSIS,
                TaskType.RECONNAISSANCE_PLANNING,
                TaskType.PAYLOAD_GENERATION,
                TaskType.CODE_GENERATION,
                TaskType.REASONING
            ],
            max_context_length=128000,
            cost_per_token=0.00014,  # ₹0.14 per million tokens / 1M
            average_latency=2.0,
            reliability_score=0.9,
            specialization_score=0.95
        )
        super().__init__("deepseek_r1_agent", capabilities)
        self.deepseek = DeepSeekR1Agent(config)
    
    async def initialize(self):
        await self.deepseek.initialize()
    
    async def process_request(self, request: LLMRequest) -> LLMResponse:
        start_time = datetime.now()
        
        try:
            if request.task_type == TaskType.EXPLOIT_GENERATION:
                result = await self.deepseek.generate_exploit(
                    request.context or {},
                    request.context or {}
                )
            elif request.task_type == TaskType.VULNERABILITY_ANALYSIS:
                result = await self.deepseek.analyze_vulnerability(request.context or {})
            elif request.task_type == TaskType.PAYLOAD_GENERATION:
                result = await self.deepseek.generate_payload("generic", request.context or {})
            elif request.task_type == TaskType.RECONNAISSANCE_PLANNING:
                result = await self.deepseek.generate_reconnaissance_plan(request.context or {})
            elif request.task_type == TaskType.REASONING:
                result = await self.deepseek.reason_about_attack_chain(
                    request.context or {}, 
                    request.context.get("objectives", []) if request.context else []
                )
            else:
                # Generic content generation for other tasks
                result = {
                    "success": True, 
                    "content": f"DeepSeek R1 processing: {request.prompt}",
                    "analysis": "Generic processing completed"
                }
            
            latency = (datetime.now() - start_time).total_seconds()
            token_count = len(request.prompt.split()) * 1.3  # Rough estimate
            cost = token_count * self.capabilities.cost_per_token
            
            return LLMResponse(
                success=result.get("success", True),
                content=result.get("content", result.get("analysis", str(result))),
                model_used=self.agent_id,
                task_type=request.task_type,
                token_count=int(token_count),
                latency=latency,
                cost=cost,
                metadata=result
            )
        
        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds()
            return LLMResponse(
                success=False,
                content="",
                model_used=self.agent_id,
                task_type=request.task_type,
                token_count=0,
                latency=latency,
                cost=0,
                error=str(e)
            )
    
    async def is_healthy(self) -> bool:
        try:
            # Simple health check
            result = await self.deepseek.analyze_vulnerability({"test": "health_check"})
            return result.get("success", False)
        except:
            return False


class GeminiAgentWrapper(BaseLLMAgent):
    """Wrapper for Gemini 2.5 Pro Agent"""
    
    def __init__(self, config: GeminiConfig):
        capabilities = ModelCapability(
            task_types=[TaskType.REPORT_GENERATION],
            max_context_length=1048576,  # 1M tokens
            cost_per_token=0.0,  # Free tier for now
            average_latency=3.0,
            reliability_score=0.85,
            specialization_score=0.9
        )
        super().__init__("gemini_2_5_pro_agent", capabilities)
        self.gemini = GeminiAgent(config)
    
    async def initialize(self):
        await self.gemini.initialize()
    
    async def process_request(self, request: LLMRequest) -> LLMResponse:
        start_time = datetime.now()
        
        try:
            # Default to executive report if no specific type provided
            report_type = request.context.get("report_type", "executive") if request.context else "executive"
            
            if report_type == "executive":
                result = await self.gemini.generate_executive_report(
                    request.context or {},
                    request.context.get("client_info", {}) if request.context else {}
                )
            elif report_type == "technical":
                result = await self.gemini.generate_technical_report(
                    request.context.get("vulnerabilities", []) if request.context else [],
                    request.context or {}
                )
            elif report_type == "compliance":
                result = await self.gemini.generate_compliance_report(
                    request.context or {},
                    request.context.get("framework", "iso27001") if request.context else "iso27001"
                )
            elif report_type == "hindi":
                result = await self.gemini.generate_hindi_report(
                    request.context or {},
                    request.context.get("client_info", {}) if request.context else {}
                )
            elif report_type == "remediation":
                result = await self.gemini.generate_remediation_guide(
                    request.context.get("vulnerabilities", []) if request.context else [],
                    request.context.get("client_context", {}) if request.context else {}
                )
            else:
                result = await self.gemini.generate_custom_report(
                    report_type,
                    request.context or {},
                    request.context.get("requirements", {}) if request.context else {}
                )
            
            latency = (datetime.now() - start_time).total_seconds()
            token_count = len(request.prompt.split()) * 1.3  # Rough estimate
            cost = token_count * self.capabilities.cost_per_token  # Currently 0 for free tier
            
            return LLMResponse(
                success=result.get("success", True),
                content=result.get("content", str(result)),
                model_used=self.agent_id,
                task_type=request.task_type,
                token_count=int(token_count),
                latency=latency,
                cost=cost,
                metadata=result
            )
        
        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds()
            return LLMResponse(
                success=False,
                content="",
                model_used=self.agent_id,
                task_type=request.task_type,
                token_count=0,
                latency=latency,
                cost=0,
                error=str(e)
            )
    
    async def is_healthy(self) -> bool:
        try:
            # Simple health check
            result = await self.gemini.generate_executive_report({}, {"company_name": "Health Check"})
            return result.get("success", False)
        except:
            return False


# Global model manager instance
model_manager = ModelManager()

# Convenience functions for external use
async def initialize_model_manager():
    """Initialize the global model manager"""
    await model_manager.initialize()

async def process_pentest_request(session_id: str, task_type: TaskType, prompt: str, context: Dict[str, Any] = None) -> LLMResponse:
    """Process a pentesting request with cost-optimized model selection"""
    return await model_manager.process_alloyed_request(session_id, task_type, prompt, context)

async def get_session_cost(session_id: str) -> float:
    """Get current session cost"""
    return model_manager.cost_tracking.get(session_id, 0.0)

async def get_cost_analysis() -> Dict[str, Any]:
    """Get cost analysis comparing to Astra Security pricing"""
    return await model_manager.get_cost_analysis()

async def optimize_for_cost():
    """Switch to cost-optimized mode"""
    await model_manager.set_strategy(AlloyStrategy.COST_OPTIMIZED)

async def optimize_for_performance():
    """Switch to performance-optimized mode"""
    await model_manager.set_strategy(AlloyStrategy.PERFORMANCE_BASED)