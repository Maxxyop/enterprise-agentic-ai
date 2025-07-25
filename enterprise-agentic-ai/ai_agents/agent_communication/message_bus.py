"""
Inter-Agent Message Bus System
Handles communication between different AI agents in the pentesting platform

"""
import asyncio
import json
import logging
from typing import Dict, List, Any, Callable, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import uuid
import redis.asyncio as redis # type: ignore
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class MessagePriority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

class MessageType(Enum):
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    STATUS_UPDATE = "status_update"
    DISCOVERY_ALERT = "discovery_alert"
    ERROR_REPORT = "error_report"
    COORDINATION = "coordination"
    SHUTDOWN = "shutdown"

@dataclass
class Message:
    """Represents a message between agents"""
    id: str
    sender_id: str
    recipient_id: str  # Can be "all" for broadcast
    type: MessageType
    priority: MessagePriority
    payload: Dict[str, Any]
    timestamp: datetime
    correlation_id: Optional[str] = None
    ttl: int = 300  # Time to live in seconds

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "sender_id": self.sender_id,
            "recipient_id": self.recipient_id,
            "type": self.type.value,
            "priority": self.priority.value,
            "payload": self.payload,
            "timestamp": self.timestamp.isoformat(),
            "correlation_id": self.correlation_id,
            "ttl": self.ttl
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Message':
        return cls(
            id=data["id"],
            sender_id=data["sender_id"],
            recipient_id=data["recipient_id"],
            type=MessageType(data["type"]),
            priority=MessagePriority(data["priority"]),
            payload=data["payload"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            correlation_id=data.get("correlation_id"),
            ttl=data.get("ttl", 300)
        )

class MessageBus:
    """Central message bus for agent communication"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis_client: Optional[redis.Redis] = None
        self.subscribers: Dict[str, List[Callable]] = {}  # agent_id -> [callback functions]
        self.message_handlers: Dict[str, Dict[MessageType, Callable]] = {}  # agent_id -> {type -> handler}
        self.running = False
        self.executor = ThreadPoolExecutor(max_workers=10)
        
    async def initialize(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = redis.from_url(self.redis_url)
            await self.redis_client.ping()
            logger.info("MessageBus initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize MessageBus: {e}")
            raise
    
    async def start(self):
        """Start the message bus"""
        if not self.redis_client:
            await self.initialize()
        
        self.running = True
        # Start message processing loop
        asyncio.create_task(self._process_messages())
        logger.info("MessageBus started")
    
    async def stop(self):
        """Stop the message bus"""
        self.running = False
        if self.redis_client:
            await self.redis_client.close()
        self.executor.shutdown(wait=True)
        logger.info("MessageBus stopped")
    
    async def register_agent(self, agent_id: str, message_handlers: Dict[MessageType, Callable]):
        """Register an agent with its message handlers"""
        self.message_handlers[agent_id] = message_handlers
        self.subscribers[agent_id] = []
        
        # Subscribe to agent-specific and broadcast channels
        await self._subscribe_to_channels([agent_id, "all"])
        
        logger.info(f"Agent {agent_id} registered with {len(message_handlers)} handlers")
    
    async def unregister_agent(self, agent_id: str):
        """Unregister an agent"""
        if agent_id in self.message_handlers:
            del self.message_handlers[agent_id]
        if agent_id in self.subscribers:
            del self.subscribers[agent_id]
        
        logger.info(f"Agent {agent_id} unregistered")
    
    async def send_message(self, message: Message) -> str:
        """Send a message to an agent or broadcast"""
        try:
            # Set message ID if not provided
            if not message.id:
                message.id = str(uuid.uuid4())
            
            # Serialize message
            message_data = json.dumps(message.to_dict())
            
            # Determine Redis channel based on recipient
            channel = f"agent:{message.recipient_id}" if message.recipient_id != "all" else "agent:all"
            
            # Priority queue handling
            queue_name = f"messages:{message.priority.name.lower()}"
            
            # Store message in Redis with TTL
            await self.redis_client.setex(
                f"message:{message.id}", 
                message.ttl, 
                message_data
            )
            
            # Publish to channel
            await self.redis_client.publish(channel, message.id)
            
            # Also add to priority queue
            await self.redis_client.lpush(queue_name, message.id)
            await self.redis_client.expire(queue_name, message.ttl)
            
            logger.debug(f"Message {message.id} sent from {message.sender_id} to {message.recipient_id}")
            return message.id
            
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            raise
    
    async def send_task_request(self, sender_id: str, recipient_id: str, task_type: str, 
                               task_params: Dict[str, Any], correlation_id: str = None) -> str:
        """Send a task request message"""
        message = Message(
            id=str(uuid.uuid4()),
            sender_id=sender_id,
            recipient_id=recipient_id,
            type=MessageType.TASK_REQUEST,
            priority=MessagePriority.NORMAL,
            payload={
                "task_type": task_type,
                "parameters": task_params
            },
            timestamp=datetime.now(),
            correlation_id=correlation_id or str(uuid.uuid4())
        )
        
        return await self.send_message(message)
    
    async def send_task_response(self, sender_id: str, recipient_id: str, 
                                task_result: Dict[str, Any], correlation_id: str) -> str:
        """Send a task response message"""
        message = Message(
            id=str(uuid.uuid4()),
            sender_id=sender_id,
            recipient_id=recipient_id,
            type=MessageType.TASK_RESPONSE,
            priority=MessagePriority.NORMAL,
            payload=task_result,
            timestamp=datetime.now(),
            correlation_id=correlation_id
        )
        
        return await self.send_message(message)
    
    async def send_discovery_alert(self, sender_id: str, discovery_data: Dict[str, Any]) -> str:
        """Broadcast a discovery alert to all agents"""
        message = Message(
            id=str(uuid.uuid4()),
            sender_id=sender_id,
            recipient_id="all",
            type=MessageType.DISCOVERY_ALERT,
            priority=MessagePriority.HIGH,
            payload=discovery_data,
            timestamp=datetime.now()
        )
        
        return await self.send_message(message)
    
    async def send_status_update(self, sender_id: str, status_data: Dict[str, Any]) -> str:
        """Send status update to orchestrator"""
        message = Message(
            id=str(uuid.uuid4()),
            sender_id=sender_id,
            recipient_id="orchestrator",
            type=MessageType.STATUS_UPDATE,
            priority=MessagePriority.LOW,
            payload=status_data,
            timestamp=datetime.now()
        )
        
        return await self.send_message(message)
    
    async def send_error_report(self, sender_id: str, error_data: Dict[str, Any]) -> str:
        """Send error report with high priority"""
        message = Message(
            id=str(uuid.uuid4()),
            sender_id=sender_id,
            recipient_id="orchestrator",
            type=MessageType.ERROR_REPORT,
            priority=MessagePriority.CRITICAL,
            payload=error_data,
            timestamp=datetime.now()
        )
        
        return await self.send_message(message)
    
    async def request_coordination(self, sender_id: str, coordination_request: Dict[str, Any]) -> str:
        """Request coordination with other agents"""
        message = Message(
            id=str(uuid.uuid4()),
            sender_id=sender_id,
            recipient_id="all",
            type=MessageType.COORDINATION,
            priority=MessagePriority.HIGH,
            payload=coordination_request,
            timestamp=datetime.now()
        )
        
        return await self.send_message(message)
    
    async def _subscribe_to_channels(self, channels: List[str]):
        """Subscribe to Redis pub/sub channels"""
        try:
            pubsub = self.redis_client.pubsub()
            for channel in channels:
                await pubsub.subscribe(f"agent:{channel}")
            
            logger.debug(f"Subscribed to channels: {channels}")
            
        except Exception as e:
            logger.error(f"Failed to subscribe to channels: {e}")
    
    async def _process_messages(self):
        """Main message processing loop"""
        try:
            pubsub = self.redis_client.pubsub()
            await pubsub.subscribe("agent:all")  # Subscribe to broadcast channel
            
            while self.running:
                try:
                    # Process priority queues
                    await self._process_priority_queues()
                    
                    # Process pub/sub messages
                    message = await pubsub.get_message(timeout=1.0)
                    if message and message['type'] == 'message':
                        message_id = message['data'].decode('utf-8')
                        await self._handle_message_id(message_id)
                    
                    await asyncio.sleep(0.1)  # Prevent busy waiting
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in message processing loop: {e}")
                    await asyncio.sleep(1)
        
        finally:
            await pubsub.close()
    
    async def _process_priority_queues(self):
        """Process messages from priority queues"""
        priorities = ["critical", "high", "normal", "low"]
        
        for priority in priorities:
            queue_name = f"messages:{priority}"
            message_id = await self.redis_client.rpop(queue_name)
            
            if message_id:
                await self._handle_message_id(message_id.decode('utf-8'))
    
    async def _handle_message_id(self, message_id: str):
        """Handle a message by ID"""
        try:
            # Retrieve message from Redis
            message_data = await self.redis_client.get(f"message:{message_id}")
            if not message_data:
                logger.warning(f"Message {message_id} not found or expired")
                return
            
            # Deserialize message
            message_dict = json.loads(message_data.decode('utf-8'))
            message = Message.from_dict(message_dict)
            
            # Route to appropriate handlers
            await self._route_message(message)
            
        except Exception as e:
            logger.error(f"Failed to handle message {message_id}: {e}")
    
    async def _route_message(self, message: Message):
        """Route message to appropriate agent handlers"""
        try:
            # Handle broadcast messages
            if message.recipient_id == "all":
                for agent_id, handlers in self.message_handlers.items():
                    if message.sender_id != agent_id:  # Don't send back to sender
                        await self._deliver_to_agent(agent_id, message, handlers)
            else:
                # Handle direct messages
                if message.recipient_id in self.message_handlers:
                    handlers = self.message_handlers[message.recipient_id]
                    await self._deliver_to_agent(message.recipient_id, message, handlers)
                else:
                    logger.warning(f"No handler found for agent {message.recipient_id}")
        
        except Exception as e:
            logger.error(f"Failed to route message {message.id}: {e}")
    
    async def _deliver_to_agent(self, agent_id: str, message: Message, handlers: Dict[MessageType, Callable]):
        """Deliver message to specific agent"""
        try:
            if message.type in handlers:
                handler = handlers[message.type]
                
                # Execute handler in thread pool to prevent blocking
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(self.executor, handler, message)
                
                logger.debug(f"Message {message.id} delivered to agent {agent_id}")
            else:
                logger.debug(f"No handler for message type {message.type} in agent {agent_id}")
        
        except Exception as e:
            logger.error(f"Failed to deliver message to agent {agent_id}: {e}")
    
    async def get_message_stats(self) -> Dict[str, Any]:
        """Get message bus statistics"""
        stats = {
            "registered_agents": len(self.message_handlers),
            "queue_lengths": {},
            "total_messages": 0
        }
        
        priorities = ["critical", "high", "normal", "low"]
        for priority in priorities:
            queue_name = f"messages:{priority}"
            length = await self.redis_client.llen(queue_name)
            stats["queue_lengths"][priority] = length
            stats["total_messages"] += length
        
        return stats

# Global message bus instance
message_bus = MessageBus()

async def initialize_message_bus():
    """Initialize the global message bus"""
    await message_bus.initialize()
    await message_bus.start()

async def shutdown_message_bus():
    """Shutdown the global message bus"""
    await message_bus.stop()

# Convenience functions for agents
async def send_to_orchestrator(sender_id: str, message_type: str, data: Dict[str, Any]):
    """Send message to orchestrator"""
    if message_type == "status":
        return await message_bus.send_status_update(sender_id, data)
    elif message_type == "error":
        return await message_bus.send_error_report(sender_id, data)
    else:
        # Generic task request
        return await message_bus.send_task_request(sender_id, "orchestrator", message_type, data)

async def broadcast_discovery(sender_id: str, discovery_data: Dict[str, Any]):
    """Broadcast discovery to all agents"""
    return await message_bus.send_discovery_alert(sender_id, discovery_data)