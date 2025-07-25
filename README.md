
## Introduction

Welcome to **enterprise-agentic-ai**, an open-source, AI-powered penetration testing platform engineered to redefine offensive security. Inspired by the trailblazing XBOW.ai, this project is your ultimate weapon for autonomous, enterprise-grade vulnerability assessments. Whether you're a red teamer hunting zero-days, a startup securing Indian SMEs, or a DevSecOps pro integrating cutting-edge tools, this platform delivers unmatched automation, precision, and scalability.

Built on a foundation of advanced machine learning, XBOW.ai Enterprise Clone orchestrates the entire pentesting lifecycle—reconnaissance, enumeration, vulnerability detection, controlled exploitation, and compliance reporting—with the finesse of an OSCP pro. Its agentic AI, driven by OODA loops, blends large language models (like GPT-4 for strategic reasoning) with fine-tuned specialists (trained on Hack The Box and TryHackMe datasets) to deliver tactical exploits. With a focus on web vulnerabilities, it analyzes source code line by line, covering 100+ attack vectors (XSS, SQLi, SSRF, IDOR, and more), all while maintaining enterprise-grade security and compliance (GDPR, PCI DSS, RBI, ISO 27001).

Designed to outshine commercial tools like Astra’s $70–$700/target pricing, this platform offers a cost-efficient ₹1000/session (~$12) DAST solution with 97% margins, perfect for Indian SMEs. Deploy it on Docker, Kubernetes, or cloud platforms, and watch it pwn networks like a digital ninja. Join us, contribute, and let’s build the future of autonomous cybersecurity!

## Key Features

- **AI-Driven Automation**: Leverages large LLMs (e.g., GPT-4 via API for high-level reasoning) and fine-tuned models (e.g., DeepSeek R1 for exploits, Gemini 2.5 Pro for reports) to automate pentesting workflows.
- **Agentic OODA Loops**: Implements Observe-Orient-Decide-Act cycles for dynamic task planning, execution, and adaptation, mimicking elite pentester workflows.
- **Comprehensive Web Vuln Coverage**: Scans source code line by line for 100+ attack vectors, including OWASP Top 10 (XSS, SQLi, CSRF), API flaws, and logic bugs, with autonomous task chaining.
- **Safe Exploitation**: Executes AI-generated PoCs in sandboxed Docker environments, ensuring compliance and safety.
- **Customizable Reporting**: Produces OSCP-style technical reports and SME-friendly summaries in Hindi/English, mapped to compliance standards.
- **Modular Architecture**: Microservices-based backend with independent modules for recon, enumeration, vuln scanning, exploitation, and reporting.
- **Scalable Deployment**: Containerized with Docker, orchestrated via Kubernetes/Helm, and provisioned with Terraform for AWS/GCP/Azure.
- **Observability**: Integrates ELK Stack for logging, Prometheus/Grafana for metrics, and Sentry for error tracking.

## System Architecture

Enterpirse-agentic-ai is a cloud-native, modular platform:

- **Backend**: FastAPI for APIs, Celery for task queues, Redis for caching, and Go for performance-critical modules.
- **Frontend**: React/Next.js dashboard with TailwindCSS, featuring goal-setting, real-time monitoring, and report visualization via WebSockets.
- **AI Agents**: Multi-model core with `foundation_models/` (DeepSeek R1 for exploits, Gemini for reports) and `specialized_models/` (fine-tuned for tool commands, vuln classification). Prompt engineering drives autonomy with JSON templates for 100+ vulns.
- **Execution Engine**: Sandboxed tool wrappers (Nmap, ZAP, Metasploit) ensure safe execution.
- **Databases**: PostgreSQL for structured data, Neo4j for attack graphs, Pinecone for vector search/memory.
- **Monitoring**: ELK Stack, Prometheus, Jaeger, and Sentry for robust observability.

Detailed structure in [docs/README.md](docs/README.md).

```bash
└───enterprise-agentic-ai
    │   .env.template
    │   LICENSE
    │   pyproject.toml
    │   
    ├───ai_agents
    │   ├───agent_communication
    │   │       knowledge_sharing.py
    │   │       message_bus.py
    │   │
    │   ├───foundation_models
    │   │   │   deepseek_r1_agent.py
    │   │   │   Gemini_2.5_pro.py
    │   │   │   llm_interface.py
    │   │   │   model_manager.py
    │   │   │
    │   │   └───prompt_engineering
    │   │       │   code_generation.py
    │   │       │   dast_prompts.py
    │   │       │   exploit_generation.py
    │   │       │   vulnerability_analysis.py
    │   │       │
    │   │       └───templates
    │   │               csrf_exploit.json
    │   │               exploit.json
    │   │               nmap_analysis.json
    │   │               nmap_script.json
    │   │               recon.json
    │   │               scan.json
    │   │               sqli_exploit.json
    │   │               sqlmap_analysis.json
    │   │               sqlmap_script.json
    │   │               xss_exploit.json
    │   │               zap_analysis.json
    │   │               zap_script.json
    │   │
    │   └───specialized_models
    │       ├───command_generator
    │       │       command_templates.json
    │       │       inference.py
    │       │
    │       ├───pentest_gpt
    │       │       inference.py
    │       │       knowledge_distillation.py
    │       │       training.py
    │       │
    │       └───vulnerability_classifier
    │               classification_rules.yaml
    │               inference.py
    │
    ├───backend
    │   ├───api
    │   │   │   main.py
    │   │   │
    │   │   └───endpoints
    │   │           ai_agents.py
    │   │           engagements.py
    │   │           execution.py
    │   │
    │   ├───core
    │   │       agentic_core.py
    │   │       auth.py
    │   │       config.py
    │   │       database.py
    │   │
    │   └───services
    │           ai_orchestration_service.py
    │           engagement_service.py
    │           reporting_service.py
    │
    ├───config
    │   ├───ai_models
    │   │       deepseek_r1.yaml
    │   │       gemini_2.5_pro.yaml
    │   │       model_routing.yaml
    │   │
    │   ├───execution
    │   │       sandbox_settings.yaml
    │   │       tool_configs.yaml
    │   │
    │   └───orchestrator
    │           decision_rules.yaml
    │           strategy_configs.yaml
    │
    ├───docker
    │       docker-compose.agentic.yml
    │       Dockerfile.ai-agents
    │       Dockerfile.execution-engine
    │
    ├───docs
    │       ARCHITECTURE.md
    │       DEPLOYMENT.md
    │       README.md
    │
    ├───execution_engine
    │   ├───core
    │   │       command_validator.py
    │   │       executor.py
    │   │       result_parser.py
    │   │       sandbox_manager.py
    │   │
    │   ├───dynamic_tooling
    │   │       code_validator.py
    │   │       tool_generator.py
    │   │       tool_templates
    │   │
    │   ├───environments
    │   │       docker_sandbox.py
    │   │
    │   └───tools
    │       ├───exploitation
    │       │       burp_integration.py
    │       │       sqlmap_wrapper.py
    │       │       zap_wrapper.py
    │       │
    │       └───reconnaissance
    │               nmap_wrapper.py
    │               subfinder_wrapper.py
    │
    ├───frontend
    │   └───src
    │       ├───components
    │       │   └───dashboard
    │       │           Dashboard.jsx
    │       │           EngagementOverview.jsx
    │       │           HindiReport.jsx
    │       │           ToolOutputViewer.jsx
    │       │
    │       ├───hooks
    │       │       useApi.js
    │       │       useAuth.js
    │       │
    │       ├───pages
    │       │       Engagements.jsx
    │       │       Home.jsx
    │       │
    │       └───services
    │               api.js
    │               auth.js
    │
    ├───future
    │       ai_platform
    │       backup_disaster_recovery
    │       cicd
    │       compliance
    │       data_management
    │       graph_db
    │       infrastructure
    │       integration
    │       ml_specialists
    │       monitoring
    │       multitenancy
    │       performance
    │       plugins
    │       reinforcement_learning
    │       security
    │
    ├───knowledge_base
    │   ├───external_integrations
    │   │       cve_database.py
    │   │       exploitdb_integration.py
    │   │
    │   ├───traditional_db
    │   │       engagement_models.py
    │   │       finding_repository.py
    │   │       sqlite_vuln_db.py
    │   │
    │   └───vector_db
    │       │   semantic_retrieval.py
    │       │   similarity_search.py
    │       │
    │       └───embeddings
    │               vulnerability_embeddings.py
    │
    ├───localization
    │   ├───formatters
    │   │       date_formatter.py
    │   │
    │   └───translations
    │           en
    │           hi
    │
    ├───observation_system
    │   ├───feedback_loops
    │   │       learning_feedback.py
    │   │       strategy_adaptation.py
    │   │
    │   ├───monitors
    │   │       application_monitor.py
    │   │
    │   └───parsers
    │           error_analyzer.py
    │           output_interpreter.py
    │           success_detector.py
    │
    ├───orchestrator
    │   ├───config
    │   │       orchestrator_config.yaml
    │   │       strategy_templates.yaml
    │   │
    │   ├───core
    │   │       agent_coordinator.py
    │   │       context_manager.py
    │   │       decision_engine.py
    │   │       goal_manager.py
    │   │       ooda_loop.py
    │   │       task_planner.py
    │   │
    │   ├───memory
    │   │       prompt_cache.py
    │   │       semantic_memory.py
    │   │       working_memory.py
    │   │
    │   └───strategies
    │           dast_strategy.py
    │
    ├───requirements
    │       ai.txt
    │       base.txt
    │
    ├───scripts
    │   ├───deployment
    │   │       deploy_ai_models.sh
    │   │
    │   └───training
    │           train_specialized_models.py
    │
    ├───tests
    │   ├───integration
    │   │       ai_integration_tests.py
    │   │       execution_integration_tests.py
    │   │
    │   └───unit
    │           ai_agents
    │           execution_engine
    │           orchestrator
    │
    └───training_pipeline
        ├───data_collection
        │       dataset_builder.py
        │       htb_scraper.py
        │
        └───fine_tuning
                evaluation_metrics.py
                lora_training.py
