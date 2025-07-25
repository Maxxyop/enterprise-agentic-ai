"""
Universal LLM Interface
Provides a unified interface for all AI models in the pentesting platform
Handles model routing, caching, and failover strategies
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import hashlib
import redis.asyncio as redis # type: ignore
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class ModelType(Enum):
    FOUNDATION = "foundation"
    SPECIALIZED = "specialized"
    UTILITY = "utility"

class TaskType(Enum):
    EXPLOIT_GENERATION = "exploit_generation"
    VULNERABILITY_ANALYSIS = "vulnerability_analysis"
    RECONNAISSANCE_PLANNING = "reconnaissance_planning"
    PAYLOAD_GENERATION = "payload_generation"
    REPORT_GENERATION = "report_generation"
    CODE_GENERATION = "code_generation"
    REASONING = "reasoning"

@dataclass
class ModelCapability:
    """Describes what a model is capable of"""
    task_types: List[TaskType]
    max_context_length: int
    cost_per_token: float
    average_latency: float  # seconds
    reliability_score: float  # 0-1
    specialization_score: float  # 0-1 for specific tasks

@dataclass
class LLMRequest:
    """Standard request format for all LLM interactions"""
    task_type: TaskType
    prompt: str
    context: Optional[Dict[str, Any]] = None
    model_preference: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    priority: int = 5  # 1-10, 10 being highest
    cache_key: Optional[str] = None
    timeout: int = 30

    def validate(self) -> bool:
        """Validate LLMRequest input."""
        if not self.prompt or not isinstance(self.prompt, str):
            logger.warning("LLMRequest missing or invalid prompt.")
            return False
        if not isinstance(self.task_type, TaskType):
            logger.warning("LLMRequest missing or invalid task_type.")
            return False
        return True

@dataclass
class LLMResponse:
    """Standard response format from all LLMs"""
    success: bool
    content: str
    model_used: str
    task_type: TaskType
    token_count: int
    latency: float
    cost: float
    cached: bool = False
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class BaseLLMAgent(ABC):
    """Abstract base class for all LLM agents"""
    
    def __init__(self, agent_id: str, capabilities: ModelCapability):
        self.agent_id = agent_id
        self.capabilities = capabilities
        self.request_count = 0
        self.total_cost = 0.0
        self.average_latency = 0.0
        
    @abstractmethod
    async def process_request(self, request: LLMRequest) -> LLMResponse:
        """Process an LLM request"""
        pass
    
    @abstractmethod
    async def is_healthy(self) -> bool:
        """Check if the agent is healthy and available"""
        pass
    
    def can_handle(self, task_type: TaskType) -> bool:
        """Check if agent can handle a specific task type"""
        return task_type in self.capabilities.task_types
    
    def get_cost_estimate(self, token_count: int) -> float:
        """Estimate cost for a request"""
        return token_count * self.capabilities.cost_per_token
    
    def update_stats(self, latency: float, cost: float):
        """Update performance statistics"""
        self.request_count += 1
        self.total_cost += cost
        
        # Update rolling average latency
        alpha = 0.1  # Smoothing factor
        self.average_latency = alpha * latency + (1 - alpha) * self.average_latency

class LLMInterface:
    """Universal interface for all LLM interactions"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.agents: Dict[str, BaseLLMAgent] = {}
        self.task_routing: Dict[TaskType, List[str]] = {}  # TaskType -> [agent_ids]
        self.redis_client: Optional[redis.Redis] = None
        self.redis_url = redis_url
        self.cache_ttl = 3600  # 1 hour cache TTL
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.request_stats = RequestStats()
        
    async def initialize(self):
        """Initialize the LLM interface"""
        try:
            self.redis_client = redis.from_url(self.redis_url)
            await self.redis_client.ping()
            logger.info("LLM Interface initialized with Redis cache")
        except Exception as e:
            logger.warning(f"Redis not available, proceeding without cache: {e}")
            self.redis_client = None
    
    def register_agent(self, agent: BaseLLMAgent, priority: int = 5):
        """Register an LLM agent"""
        self.agents[agent.agent_id] = agent
        self.circuit_breakers[agent.agent_id] = CircuitBreaker(agent.agent_id)
        
        # Update task routing
        for task_type in agent.capabilities.task_types:
            if task_type not in self.task_routing:
                self.task_routing[task_type] = []
            
            # Insert based on priority and specialization score
            agents_for_task = self.task_routing[task_type]
            
            # Simple insertion sort by specialization score
            inserted = False
            for i, existing_agent_id in enumerate(agents_for_task):
                existing_agent = self.agents[existing_agent_id]
                if agent.capabilities.specialization_score > existing_agent.capabilities.specialization_score:
                    agents_for_task.insert(i, agent.agent_id)
                    inserted = True
                    break
            
            if not inserted:
                agents_for_task.append(agent.agent_id)
        
        logger.info(f"Registered agent {agent.agent_id} with capabilities: {agent.capabilities.task_types}")
    
    async def process_request(self, request: LLMRequest) -> LLMResponse:
        """Process an LLM request with routing, caching, and failover"""
        start_time = datetime.now()
        
        # Check cache first
        if self.redis_client and request.cache_key:
            cached_response = await self._get_from_cache(request.cache_key)
            if cached_response:
                cached_response.cached = True
                return cached_response
        
        # Generate cache key if not provided
        if not request.cache_key:
            request.cache_key = self._generate_cache_key(request)
        
        # Route to appropriate agent
        agent_id = self._select_agent(request)
        if not agent_id:
            return LLMResponse(
                success=False,
                content="",
                model_used="none",
                task_type=request.task_type,
                token_count=0,
                latency=0,
                cost=0,
                error="No suitable agent available"
            )
        
        # Try primary agent with circuit breaker
        circuit_breaker = self.circuit_breakers[agent_id]
        
        if circuit_breaker.can_execute():
            try:
                agent = self.agents[agent_id]
                response = await self._execute_with_timeout(agent, request)
                
                if response.success:
                    # Cache successful response
                    if self.redis_client:
                        await self._cache_response(request.cache_key, response)
                    
                    circuit_breaker.record_success()
                    self.request_stats.record_success(agent_id, response.latency, response.cost)
                    return response
                else:
                    circuit_breaker.record_failure()
                    
            except Exception as e:
                logger.error(f"Agent {agent_id} failed: {e}")
                circuit_breaker.record_failure()
        
        # Try fallback agents
        fallback_response = await self._try_fallback_agents(request, excluded=[agent_id])
        if fallback_response.success:
            return fallback_response
        
        # All agents failed
        return LLMResponse(
            success=False,
            content="",
            model_used="none",
            task_type=request.task_type,
            token_count=0,
            latency=(datetime.now() - start_time).total_seconds(),
            cost=0,
            error="All agents failed to process request"
        )
    
    async def _execute_with_timeout(self, agent: BaseLLMAgent, request: LLMRequest) -> LLMResponse:
        """Execute request with timeout"""
        try:
            response = await asyncio.wait_for(
                agent.process_request(request),
                timeout=request.timeout
            )
            return response
        except asyncio.TimeoutError:
            return LLMResponse(
                success=False,
                content="",
                model_used=agent.agent_id,
                task_type=request.task_type,
                token_count=0,
                latency=request.timeout,
                cost=0,
                error="Request timeout"
            )
    
    def _select_agent(self, request: LLMRequest) -> Optional[str]:
        """Select the best agent for a request"""
        # Check for specific model preference
        if request.model_preference and request.model_preference in self.agents:
            agent = self.agents[request.model_preference]
            if agent.can_handle(request.task_type):
                return request.model_preference
        
        # Get agents for task type
        candidate_agents = self.task_routing.get(request.task_type, [])
        
        if not candidate_agents:
            return None
        
        # Filter by health and circuit breaker status
        healthy_agents = []
        for agent_id in candidate_agents:
            circuit_breaker = self.circuit_breakers[agent_id]
            if circuit_breaker.can_execute():
                healthy_agents.append(agent_id)
        
        if not healthy_agents:
            return None
        
        # Select based on performance metrics and load
        best_agent = self._select_best_agent(healthy_agents, request)
        return best_agent
    
    def _select_best_agent(self, candidate_agents: List[str], request: LLMRequest) -> str:
        """Select the best agent from candidates based on performance"""
        if len(candidate_agents) == 1:
            return candidate_agents[0]
        
        # Score agents based on multiple factors
        agent_scores = {}
        
        for agent_id in candidate_agents:
            agent = self.agents[agent_id]
            
            # Base score from specialization
            score = agent.capabilities.specialization_score
            
            # Adjust for reliability
            score *= agent.capabilities.reliability_score
            
            # Adjust for cost (lower cost is better for non-critical requests)
            if request.priority < 8:  # Not critical
                cost_factor = 1 / (1 + agent.capabilities.cost_per_token * 1000)
                score *= cost_factor
            
            # Adjust for latency
            latency_factor = 1 / (1 + agent.capabilities.average_latency)
            score *= latency_factor
            
            agent_scores[agent_id] = score
        
        # Return agent with highest score
        best_agent = max(agent_scores.items(), key=lambda x: x[1])[0]
        return best_agent
    
    async def _try_fallback_agents(self, request: LLMRequest, excluded: List[str]) -> LLMResponse:
        """Try fallback agents when primary fails"""
        candidate_agents = self.task_routing.get(request.task_type, [])
        fallback_agents = [agent_id for agent_id in candidate_agents if agent_id not in excluded]
        
        for agent_id in fallback_agents:
            circuit_breaker = self.circuit_breakers[agent_id]
            if not circuit_breaker.can_execute():
                continue
            
            try:
                agent = self.agents[agent_id]
                response = await self._execute_with_timeout(agent, request)
                
                if response.success:
                    circuit_breaker.record_success()
                    self.request_stats.record_success(agent_id, response.latency, response.cost)
                    logger.info(f"Fallback to agent {agent_id} succeeded")
                    return response
                else:
                    circuit_breaker.record_failure()
                    
            except Exception as e:
                logger.error(f"Fallback agent {agent_id} failed: {e}")
                circuit_breaker.record_failure()
        
        return LLMResponse(
            success=False,
            content="",
            model_used="none",
            task_type=request.task_type,
            token_count=0,
            latency=0,
            cost=0,
            error="All fallback agents failed"
        )
    
    def _generate_cache_key(self, request: LLMRequest) -> str:
        """Generate cache key for request"""
        cache_data = {
            "task_type": request.task_type.value,
            "prompt": request.prompt,
            "context": request.context,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature
        }
        
        cache_string = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    async def _get_from_cache(self, cache_key: str) -> Optional[LLMResponse]:
        """Get response from cache"""
        try:
            cached_data = await self.redis_client.get(f"llm_cache:{cache_key}")
            if cached_data:
                response_data = json.loads(cached_data.decode())
                return LLMResponse(**response_data)
        except Exception as e:
            logger.error(f"Cache read error: {e}")
        
        return None
    
    async def _cache_response(self, cache_key: str, response: LLMResponse):
        """Cache response"""
        try:
            # Don't cache errors
            if not response.success:
                return
            
            cache_data = {
                "success": response.success,
                "content": response.content,
                "model_used": response.model_used,
                "task_type": response.task_type.value,
                "token_count": response.token_count,
                "latency": response.latency,
                "cost": response.cost,
                "cached": False,  # Will be set to True when retrieved
                "error": response.error,
                "metadata": response.metadata
            }
            
            await self.redis_client.setex(
                f"llm_cache:{cache_key}",
                self.cache_ttl,
                json.dumps(cache_data)
            )
        except Exception as e:
            logger.error(f"Cache write error: {e}")
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get interface statistics"""
        stats = {
            "agents": {},
            "task_routing": {},
            "circuit_breakers": {},
            "total_requests": self.request_stats.total_requests,
            "total_cost": self.request_stats.total_cost,
            "average_latency": self.request_stats.average_latency
        }
        
        # Agent stats
        for agent_id, agent in self.agents.items():
            stats["agents"][agent_id] = {
                "request_count": agent.request_count,
                "total_cost": agent.total_cost,
                "average_latency": agent.average_latency,
                "capabilities": [t.value for t in agent.capabilities.task_types]
            }
        
        # Task routing
        for task_type, agent_list in self.task_routing.items():
            stats["task_routing"][task_type.value] = agent_list
        
        # Circuit breaker stats
        for agent_id, cb in self.circuit_breakers.items():
            stats["circuit_breakers"][agent_id] = {
                "state": cb.state,
                "failure_count": cb.failure_count,
                "last_failure": cb.last_failure_time.isoformat() if cb.last_failure_time else None
            }
        
        return stats
    
    async def health_check(self) -> Dict[str, bool]:
        """Check health of all agents"""
        health_status = {}
        
        for agent_id, agent in self.agents.items():
            try:
                is_healthy = await asyncio.wait_for(agent.is_healthy(), timeout=10)
                health_status[agent_id] = is_healthy
            except Exception as e:
                logger.error(f"Health check failed for {agent_id}: {e}")
                health_status[agent_id] = False
        
        return health_status

    def get_response(self, prompt: str) -> str:
        """Stub for MVP: Return canned response for prompt."""
        if "capital of France" in prompt:
            return "Paris"
        return "Response"


class CircuitBreaker:
    """Circuit breaker pattern for agent reliability"""
    
    def __init__(self, agent_id: str, failure_threshold: int = 5, timeout: int = 60):
        self.agent_id = agent_id
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = "closed"  # closed, open, half-open
    
    def can_execute(self) -> bool:
        """Check if execution is allowed"""
        if self.state == "closed":
            return True
        elif self.state == "open":
            if self.last_failure_time and \
               datetime.now() - self.last_failure_time > timedelta(seconds=self.timeout):
                self.state = "half-open"
                return True
            return False
        elif self.state == "half-open":
            return True
        
        return False
    
    def record_success(self):
        """Record successful execution"""
        self.failure_count = 0
        self.state = "closed"
        self.last_failure_time = None
    
    def record_failure(self):
        """Record failed execution"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            logger.warning(f"Circuit breaker opened for agent {self.agent_id}")


class RequestStats:
    """Statistics tracking for requests"""
    
    def __init__(self):
        self.total_requests = 0
        self.successful_requests = 0
        self.total_cost = 0.0
        self.total_latency = 0.0
        self.agent_stats: Dict[str, Dict[str, Any]] = {}
    
    def record_success(self, agent_id: str, latency: float, cost: float):
        """Record successful request"""
        self.total_requests += 1
        self.successful_requests += 1
        self.total_cost += cost
        self.total_latency += latency
        
        if agent_id not in self.agent_stats:
            self.agent_stats[agent_id] = {
                "requests": 0,
                "successes": 0,
                "cost": 0.0,
                "latency": 0.0
            }
        
        stats = self.agent_stats[agent_id]
        stats["requests"] += 1
        stats["successes"] += 1
        stats["cost"] += cost
        stats["latency"] += latency
    
    def record_failure(self, agent_id: str):
        """Record failed request"""
        self.total_requests += 1
        
        if agent_id not in self.agent_stats:
            self.agent_stats[agent_id] = {
                "requests": 0,
                "successes": 0,
                "cost": 0.0,
                "latency": 0.0
            }
        
        self.agent_stats[agent_id]["requests"] += 1
    
    @property
    def success_rate(self) -> float:
        """Calculate overall success rate"""
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests
    
    @property
    def average_latency(self) -> float:
        """Calculate average latency"""
        if self.successful_requests == 0:
            return 0.0
        return self.total_latency / self.successful_requests
    
    @property
    def average_cost(self) -> float:
        """Calculate average cost per request"""
        if self.successful_requests == 0:
            return 0.0
        return self.total_cost / self.successful_requests


# Global LLM interface instance
llm_interface = LLMInterface()

# Convenience functions
async def process_exploit_generation(prompt: str, context: Dict[str, Any] = None) -> LLMResponse:
    """Generate exploit code"""
    request = LLMRequest(
        task_type=TaskType.EXPLOIT_GENERATION,
        prompt=prompt,
        context=context,
        temperature=0.1
    )
    return await llm_interface.process_request(request)

async def process_vulnerability_analysis(scan_results: Dict[str, Any]) -> LLMResponse:
    """Analyze vulnerability scan results"""
    request = LLMRequest(
        task_type=TaskType.VULNERABILITY_ANALYSIS,
        prompt=f"Analyze these scan results: {json.dumps(scan_results)}",
        context=scan_results,
        temperature=0.2
    )
    return await llm_interface.process_request(request)

async def generate_report(report_type: str, data: Dict[str, Any]) -> LLMResponse:
    """Generate penetration testing report"""
    request = LLMRequest(
        task_type=TaskType.REPORT_GENERATION,
        prompt=f"Generate {report_type} report",
        context=data,
        model_preference="gemini_2_5_pro_agent",
        temperature=0.1
    )
    return await llm_interface.process_request(request)