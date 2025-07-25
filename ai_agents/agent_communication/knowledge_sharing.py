"""
AI Agent Knowledge Sharing System
Enables agents to share vulnerability data, attack patterns, and learned techniques
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import redis # type: ignore
from pydantic import BaseModel # type: ignore

logger = logging.getLogger(__name__)

@dataclass
class KnowledgeItem:
    """Represents a piece of shared knowledge between agents"""
    id: str
    type: str  # 'vulnerability', 'technique', 'pattern', 'payload'
    content: Dict[str, Any]
    source_agent: str
    confidence_score: float
    timestamp: datetime
    tags: List[str]

class SharedKnowledge(BaseModel):
    """Model for knowledge items shared between agents"""
    vulnerability_patterns: Dict[str, Any] = {}
    exploit_techniques: Dict[str, Any] = {}
    successful_payloads: List[Dict] = []
    failed_attempts: List[Dict] = []
    target_fingerprints: Dict[str, Any] = {}

class KnowledgeShareManager:
    """Manages knowledge sharing between AI agents"""
    
    def __init__(self, redis_host: str = "localhost", redis_port: int = 6379):
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.knowledge_cache = SharedKnowledge()
        
    async def share_vulnerability_pattern(self, agent_id: str, pattern: Dict[str, Any], confidence: float = 0.8):
        """Share a discovered vulnerability pattern with other agents"""
        knowledge_item = KnowledgeItem(
            id=f"vuln_{agent_id}_{datetime.now().timestamp()}",
            type="vulnerability",
            content=pattern,
            source_agent=agent_id,
            confidence_score=confidence,
            timestamp=datetime.now(),
            tags=pattern.get('tags', [])
        )
        
        # Store in Redis for real-time sharing
        await self._store_knowledge_item(knowledge_item)
        
        # Update local cache
        self.knowledge_cache.vulnerability_patterns[knowledge_item.id] = pattern
        
        logger.info(f"Agent {agent_id} shared vulnerability pattern: {pattern.get('name', 'Unknown')}")
        return knowledge_item.id
    
    async def share_exploit_technique(self, agent_id: str, technique: Dict[str, Any], success_rate: float):
        """Share a successful exploit technique"""
        knowledge_item = KnowledgeItem(
            id=f"technique_{agent_id}_{datetime.now().timestamp()}",
            type="technique",
            content={**technique, "success_rate": success_rate},
            source_agent=agent_id,
            confidence_score=success_rate,
            timestamp=datetime.now(),
            tags=technique.get('mitre_techniques', [])
        )
        
        await self._store_knowledge_item(knowledge_item)
        self.knowledge_cache.exploit_techniques[knowledge_item.id] = technique
        
        logger.info(f"Agent {agent_id} shared technique: {technique.get('name')} (Success: {success_rate:.2%})")
        return knowledge_item.id
    
    async def share_successful_payload(self, agent_id: str, payload_data: Dict[str, Any]):
        """Share a payload that successfully exploited a vulnerability"""
        payload_item = {
            "payload": payload_data,
            "agent_id": agent_id,
            "timestamp": datetime.now().isoformat(),
            "target_info": payload_data.get("target_info", {}),
            "vulnerability_type": payload_data.get("vuln_type")
        }
        
        self.knowledge_cache.successful_payloads.append(payload_item)
        
        # Store in Redis with expiration (7 days)
        redis_key = f"payload:successful:{agent_id}:{datetime.now().timestamp()}"
        self.redis_client.setex(redis_key, 604800, json.dumps(payload_item))
        
        logger.info(f"Agent {agent_id} shared successful payload for {payload_data.get('vuln_type')}")
    
    async def get_relevant_knowledge(self, agent_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve knowledge relevant to current agent context"""
        target_type = context.get("target_type", "web")
        vulnerability_types = context.get("vuln_types", [])
        
        relevant_knowledge = {
            "vulnerability_patterns": [],
            "exploit_techniques": [],
            "successful_payloads": [],
            "recommendations": []
        }
        
        # Get vulnerability patterns
        for pattern_id, pattern in self.knowledge_cache.vulnerability_patterns.items():
            if any(tag in vulnerability_types for tag in pattern.get('tags', [])):
                relevant_knowledge["vulnerability_patterns"].append(pattern)
        
        # Get exploit techniques
        for tech_id, technique in self.knowledge_cache.exploit_techniques.items():
            if technique.get("target_type") == target_type:
                relevant_knowledge["exploit_techniques"].append(technique)
        
        # Get successful payloads
        for payload in self.knowledge_cache.successful_payloads[-50:]:  # Last 50 successful payloads
            if payload.get("vulnerability_type") in vulnerability_types:
                relevant_knowledge["successful_payloads"].append(payload)
        
        logger.info(f"Retrieved {len(relevant_knowledge['vulnerability_patterns'])} patterns for agent {agent_id}")
        return relevant_knowledge
    
    async def record_failed_attempt(self, agent_id: str, attempt_data: Dict[str, Any]):
        """Record a failed exploitation attempt to prevent repeated failures"""
        failure_item = {
            "agent_id": agent_id,
            "timestamp": datetime.now().isoformat(),
            "target_info": attempt_data.get("target_info", {}),
            "technique_used": attempt_data.get("technique"),
            "failure_reason": attempt_data.get("error", "Unknown"),
            "payload": attempt_data.get("payload")
        }
        
        self.knowledge_cache.failed_attempts.append(failure_item)
        
        # Limit failed attempts cache to prevent memory bloat
        if len(self.knowledge_cache.failed_attempts) > 1000:
            self.knowledge_cache.failed_attempts = self.knowledge_cache.failed_attempts[-500:]
        
        logger.warning(f"Agent {agent_id} failed attempt recorded: {attempt_data.get('technique')}")
    
    async def get_learning_insights(self, agent_id: str) -> Dict[str, Any]:
        """Generate learning insights based on shared knowledge"""
        insights = {
            "top_techniques": [],
            "common_failures": [],
            "emerging_patterns": [],
            "recommendations": []
        }
        
        # Analyze successful techniques
        technique_success = {}
        for technique in self.knowledge_cache.exploit_techniques.values():
            tech_name = technique.get("name", "Unknown")
            success_rate = technique.get("success_rate", 0)
            
            if tech_name not in technique_success:
                technique_success[tech_name] = []
            technique_success[tech_name].append(success_rate)
        
        # Calculate average success rates
        for tech_name, rates in technique_success.items():
            avg_success = sum(rates) / len(rates)
            insights["top_techniques"].append({
                "technique": tech_name,
                "avg_success_rate": avg_success,
                "usage_count": len(rates)
            })
        
        # Sort by success rate
        insights["top_techniques"].sort(key=lambda x: x["avg_success_rate"], reverse=True)
        
        # Analyze common failures
        failure_reasons = {}
        for failure in self.knowledge_cache.failed_attempts[-100:]:  # Last 100 failures
            reason = failure.get("failure_reason", "Unknown")
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
        
        insights["common_failures"] = [
            {"reason": reason, "count": count}
            for reason, count in sorted(failure_reasons.items(), key=lambda x: x[1], reverse=True)
        ][:10]
        
        # Generate recommendations
        if insights["top_techniques"]:
            best_technique = insights["top_techniques"][0]
            insights["recommendations"].append(
                f"Consider using '{best_technique['technique']}' - {best_technique['avg_success_rate']:.1%} success rate"
            )
        
        if insights["common_failures"]:
            top_failure = insights["common_failures"][0]
            insights["recommendations"].append(
                f"Watch out for '{top_failure['reason']}' - common failure pattern"
            )
        
        return insights
    
    async def _store_knowledge_item(self, item: KnowledgeItem):
        """Store knowledge item in Redis"""
        redis_key = f"knowledge:{item.type}:{item.id}"
        item_data = {
            "id": item.id,
            "type": item.type,
            "content": item.content,
            "source_agent": item.source_agent,
            "confidence_score": item.confidence_score,
            "timestamp": item.timestamp.isoformat(),
            "tags": item.tags
        }
        
        # Store with 24 hour expiration
        self.redis_client.setex(redis_key, 86400, json.dumps(item_data))
    
    async def sync_knowledge_from_redis(self):
        """Sync knowledge from Redis to local cache"""
        try:
            # Get all knowledge keys
            knowledge_keys = self.redis_client.keys("knowledge:*")
            
            for key in knowledge_keys:
                item_data = self.redis_client.get(key)
                if item_data:
                    item = json.loads(item_data)
                    
                    if item["type"] == "vulnerability":
                        self.knowledge_cache.vulnerability_patterns[item["id"]] = item["content"]
                    elif item["type"] == "technique":
                        self.knowledge_cache.exploit_techniques[item["id"]] = item["content"]
            
            logger.info(f"Synced {len(knowledge_keys)} knowledge items from Redis")
            
        except Exception as e:
            logger.error(f"Failed to sync knowledge from Redis: {e}")

# Global knowledge manager instance
knowledge_manager = KnowledgeShareManager()

async def share_discovery(agent_id: str, discovery_type: str, data: Dict[str, Any]):
    """Convenience function for agents to share discoveries"""
    if discovery_type == "vulnerability":
        return await knowledge_manager.share_vulnerability_pattern(agent_id, data)
    elif discovery_type == "technique":
        return await knowledge_manager.share_exploit_technique(agent_id, data, data.get("success_rate", 0.5))
    elif discovery_type == "payload":
        return await knowledge_manager.share_successful_payload(agent_id, data)
    else:
        logger.warning(f"Unknown discovery type: {discovery_type}")
        return None