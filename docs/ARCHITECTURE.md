# Architecture Overview

## Introduction
The Enterprise Agentic AI project is designed to automate and enhance the process of penetration testing through the integration of various AI agents, orchestration strategies, and execution tools. This document outlines the architecture of the system, detailing its components, their interactions, and the overall workflow.

## System Components

### 1. Orchestrator
The orchestrator is responsible for managing the flow of tasks and coordinating between different AI agents. It consists of several core modules:
- **OODA Loop**: Implements the Observe, Orient, Decide, and Act loop for task orchestration.
- **Decision Engine**: Routes tasks to the appropriate AI agents (DeepSeek R1 and Qwen-7B).
- **Task Planner**: Plans DAST tasks such as reconnaissance, scanning, and exploitation.
- **Agent Coordinator**: Manages interactions between the AI agents.
- **Goal Manager**: Defines and manages the objectives for penetration testing.
- **Context Manager**: Handles the context for DeepSeek R1, ensuring efficient processing.

### 2. AI Agents
The AI agents are specialized models that perform various tasks:
- **Foundation Models**: Includes DeepSeek R1 for reasoning and Qwen-7B for command execution.
- **Specialized Models**: Fine-tuned models for specific tasks such as vulnerability classification and command generation.
- **Agent Communication**: Facilitates communication and knowledge sharing between agents.

### 3. Knowledge Base
The knowledge base stores and retrieves information relevant to vulnerabilities:
- **Vector Database**: Implements semantic search and stores embeddings for vulnerabilities.
- **Traditional Database**: Manages engagement models and findings from assessments.
- **External Integrations**: Integrates with external databases for CVE and exploit information.

### 4. Execution Engine
The execution engine is responsible for running the actual penetration testing tools:
- **Core**: Manages command execution, sandbox environments, and result parsing.
- **Tools**: Includes wrappers for tools like Nmap, SQLMap, and OWASP ZAP.
- **Dynamic Tooling**: Generates scripts and validates code for execution.

### 5. Observation System
The observation system monitors the execution of tasks and analyzes outputs:
- **Monitors**: Tracks web applications for vulnerabilities.
- **Parsers**: Processes outputs from tools and analyzes errors.
- **Feedback Loops**: Adapts strategies based on feedback from the execution results.

### 6. Training Pipeline
The training pipeline is responsible for collecting data and fine-tuning models:
- **Data Collection**: Scrapes data from various sources for training purposes.
- **Fine Tuning**: Implements training logic for specialized models and evaluates metrics.

### 7. Backend
The backend provides the necessary services and API for the application:
- **Core Services**: Manages configuration, database connections, and authentication.
- **API**: Exposes endpoints for engagement management, AI agents, and execution.

### 8. Frontend
The frontend provides a user interface for interacting with the system:
- **Components**: Includes dashboard, engagement overview, and tool output viewer.
- **Pages**: Implements navigation and displays relevant information to users.

## Workflow
1. **Task Initiation**: The orchestrator receives a task request and initiates the OODA loop.
2. **Task Planning**: The task planner defines the steps required for the task.
3. **Agent Coordination**: The agent coordinator assigns tasks to the appropriate AI agents.
4. **Execution**: The execution engine runs the necessary tools and collects results.
5. **Monitoring and Feedback**: The observation system monitors the execution and provides feedback for strategy adaptation.
6. **Training and Improvement**: The training pipeline collects data and fine-tunes models based on performance.

## Conclusion
The architecture of the Enterprise Agentic AI project is designed to be modular and scalable, allowing for the integration of new tools and models as they become available. This structure ensures that the system can adapt to the evolving landscape of cybersecurity threats and provide effective penetration testing solutions.