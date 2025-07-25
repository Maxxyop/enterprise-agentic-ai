"""
DeepSeek R1 Agent - Specialized for complex reasoning, attack chains, code and exploit generation
Uses DeepSeek's model for pentesting logic, vulnerability analysis, and exploit development
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import os

logger = logging.getLogger(__name__)

@dataclass
class DeepSeekConfig:
    """Configuration for DeepSeek R1"""
    api_key: str
    model: str = "deepseek-coder"
    base_url: str = "https://api.deepseek.com"
    max_tokens: int = 8192
    temperature: float = 0.1
    top_p: float = 0.8

class DeepSeekAgent:
    """AI Agent using DeepSeek R1 for complex reasoning and exploit generation"""
    def __init__(self, config: Optional[DeepSeekConfig] = None):
        """Initialize DeepSeekAgent with config and client."""
        self.config = config or DeepSeekConfig(
            api_key=os.getenv("DEEPSEEK_API_KEY", "") # TODO: Set your API key in environment
        )
        self.agent_id = "deepseek_r1_agent"
        self.client = self._setup_client()
        self.templates = self._load_templates()

    def _setup_client(self):
        """Setup DeepSeek client (modular for future extension)."""
        try:
            import openai # type: ignore
            return openai.OpenAI(
                base_url=self.config.base_url,
                api_key=self.config.api_key
            )
        except Exception as e:
            logger.error(f"Failed to setup DeepSeek client: {e}")
            return None

    def _load_templates(self) -> Dict[str, str]:
        """Load all prompt templates."""
        return {
            "attack_chain": self._load_template("attack_chain"),
            "exploit_generation": self._load_template("exploit_generation"),
            "vulnerability_analysis": self._load_template("vulnerability_analysis"),
            "hindi_analysis": self._load_template("hindi_analysis")
        }

    async def initialize(self) -> None:
        """Initialize the DeepSeek agent and validate API connection."""
        logger.info("Initializing DeepSeek R1 Agent for complex reasoning and exploit generation")
        await self._validate_api_connection()

    async def _validate_api_connection(self) -> None:
        """Validate API connection with DeepSeek."""
        try:
            response = await self._generate_content("Test connection", max_tokens=10)
            logger.info("DeepSeek API connection validated")
        except Exception as e:
            logger.error(f"Failed to validate DeepSeek API: {e}")
            raise
    
    async def _generate_content(self, prompt: str, **kwargs) -> str:
        """Generate content using DeepSeek API"""
        try:
            # Wrap synchronous call in async thread
            def sync_generate():
                return self.client.chat.completions.create(
                    model=self.config.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
                    temperature=self.config.temperature,
                    top_p=self.config.top_p
                ).choices[0].message.content

            response = await asyncio.to_thread(sync_generate)
            return response
        except Exception as e:
            logger.error(f"DeepSeek content generation failed: {e}")
            raise
    
    async def generate_attack_chain(self, 
                                    target_info: Dict[str, Any], 
                                    vuln_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate attack chain for penetration testing"""
        
        template = self.templates["attack_chain"]
        
        prompt = f"""
        {template}
        
        Target Information:
        {json.dumps(target_info, indent=2)}
        
        Vulnerability Data:
        {json.dumps(vuln_data, indent=2)}
        
        Create a detailed attack chain that:
        1. Outlines multi-step exploitation paths
        2. Identifies pivot points and escalation methods
        3. Quantifies success probabilities
        4. Includes alternative branches
        5. Specifies required tools and prerequisites
        6. Uses logical reasoning for each step
        
        Focus on realistic attack vectors and defense bypass techniques.
        """
        
        try:
            chain_content = await self._generate_content(prompt)
            
            return {
                "success": True,
                "type": "attack_chain",
                "content": chain_content,
                "target": target_info.get('host', 'Unknown'),
                "generated_at": datetime.now().isoformat(),
                "risk_score": self._calculate_attack_risk(vuln_data),
                "key_steps": self._extract_key_steps(chain_content)
            }
        
        except Exception as e:
            logger.error(f"Failed to generate attack chain: {e}")
            return {"success": False, "error": str(e)}
    
    async def generate_exploit(self, 
                               vuln_details: Dict[str, Any], 
                               target_context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate exploit code and procedures"""
        
        template = self.templates["exploit_generation"]
        
        prompt = f"""
        {template}
        
        Vulnerability Details:
        {json.dumps(vuln_details, indent=2)}
        
        Target Context:
        {json.dumps(target_context, indent=2)}
        
        Generate a comprehensive exploit including:
        1. Detailed code in appropriate language (Python, etc.)
        2. Step-by-step execution instructions
        3. Payload variations
        4. Detection evasion techniques
        5. Cleanup procedures
        6. Success verification methods
        7. Potential side effects
        
        Use secure coding practices where possible and include comments.
        """
        
        try:
            exploit_content = await self._generate_content(prompt)
            
            return {
                "success": True,
                "type": "exploit_generation",
                "content": exploit_content,
                "vulnerability": vuln_details.get('name', 'Unknown'),
                "generated_at": datetime.now().isoformat(),
                "language": self._detect_code_language(exploit_content),
                "tools_required": self._extract_tools(exploit_content)
            }
        
        except Exception as e:
            logger.error(f"Failed to generate exploit: {e}")
            return {"success": False, "error": str(e)}
    
    async def generate_vulnerability_analysis(self, 
                                              scan_data: Dict[str, Any], 
                                              framework: str = "cve") -> Dict[str, Any]:
        """Generate in-depth vulnerability analysis"""
        
        template = self.templates["vulnerability_analysis"]
        
        framework_mapping = {
            "cve": "CVE Database",
            "owasp": "OWASP Top 10",
            "mitre": "MITRE ATT&CK",
            "nist": "NIST Vulnerability Database"
        }
        
        framework_name = framework_mapping.get(framework, framework.upper())
        
        prompt = f"""
        {template}
        
        Analysis Framework: {framework_name}
        
        Scan Data:
        {json.dumps(scan_data, indent=2)}
        
        Create a detailed vulnerability analysis that:
        1. Maps vulnerabilities to {framework_name} entries
        2. Assesses exploitation feasibility
        3. Provides root cause analysis
        4. Includes impact assessment
        5. Suggests detection signatures
        6. Estimates exploitation timeline
        7. Documents analysis methodology
        
        Structure for technical review and further research.
        Include specific references and scoring metrics.
        """
        
        try:
            analysis_content = await self._generate_content(prompt)
            
            return {
                "success": True,
                "type": "vulnerability_analysis",
                "framework": framework,
                "content": analysis_content,
                "generated_at": datetime.now().isoformat(),
                "exploitability_score": self._calculate_exploitability_score(scan_data),
                "impact_areas": self._extract_impact_areas(analysis_content),
                "mitigation_priority": self._extract_mitigation_priorities(analysis_content)
            }
        
        except Exception as e:
            logger.error(f"Failed to generate vulnerability analysis: {e}")
            return {"success": False, "error": str(e)}
    
    async def generate_hindi_analysis(self, 
                                      vuln_summary: Dict[str, Any], 
                                      target_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generate Hindi language analysis for Indian SME clients"""
        
        template = self.templates["hindi_analysis"]
        
        prompt = f"""
        {template}
        
        भेद्यता सारांश:
        {json.dumps(vuln_summary, indent=2)}
        
        लक्ष्य जानकारी:
        - होस्ट: {target_info.get('host', 'अज्ञात')}
        - उद्योग: {target_info.get('industry', 'अज्ञात')}
        - मुख्य संपत्ति: {target_info.get('primary_assets', [])}
        
        Create a comprehensive analysis in Hindi that includes:
        1. कार्यकारी सारांश (Executive Summary)
        2. मुख्य भेद्यताएं (Key Vulnerabilities)
        3. जोखिम मूल्यांकन (Risk Assessment)
        4. हमला चेन (Attack Chain)
        5. व्यावसायिक प्रभाव (Business Impact)
        6. कोड उदाहरण (Code Examples)
        
        Use professional Hindi terminology for cybersecurity concepts.
        Make it accessible for Indian SME business owners and IT staff.
        Include both Hindi and English technical terms where appropriate.
        """
        
        try:
            analysis_content = await self._generate_content(prompt)
            
            return {
                "success": True,
                "type": "hindi_analysis",
                "language": "hindi",
                "content": analysis_content,
                "target": target_info.get('host', 'अज्ञात'),
                "generated_at": datetime.now().isoformat(),
                "summary_english": self._generate_english_summary(vuln_summary)
            }
        
        except Exception as e:
            logger.error(f"Failed to generate Hindi analysis: {e}")
            return {"success": False, "error": str(e)}
    
    async def generate_reasoning_chain(self, 
                                       problem_data: List[Dict[str, Any]], 
                                       context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate step-by-step reasoning chain"""
        
        prompt = f"""
        Create a detailed reasoning chain for the following problems:
        
        Problem Data:
        {json.dumps(problem_data, indent=2)}
        
        Context:
        - Environment: {context.get('environment', 'web')}
        - Constraints: {context.get('constraints', 'none')}
        - Goals: {context.get('goals', 'exploitation')}
        - Known Facts: {context.get('facts', [])}
        
        For each problem, provide:
        1. Logical step-by-step reasoning
        2. Hypothesis testing
        3. Evidence evaluation
        4. Alternative considerations
        5. Conclusion and next actions
        6. Potential pitfalls
        
        Prioritize logical rigor and evidence-based conclusions.
        Include diagrams or pseudocode where helpful.
        """
        
        try:
            reasoning_content = await self._generate_content(prompt)
            
            return {
                "success": True,
                "type": "reasoning_chain",
                "content": reasoning_content,
                "problems_count": len(problem_data),
                "generated_at": datetime.now().isoformat(),
                "confidence_level": self._estimate_confidence(problem_data),
                "next_steps": self._extract_next_steps(reasoning_content)
            }
        
        except Exception as e:
            logger.error(f"Failed to generate reasoning chain: {e}")
            return {"success": False, "error": str(e)}
    
    async def generate_custom_analysis(self, 
                                       template_type: str, 
                                       data: Dict[str, Any], 
                                       custom_requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Generate custom analysis based on specific requirements"""
        
        prompt = f"""
        Create a custom penetration testing analysis with the following requirements:
        
        Template Type: {template_type}
        Data: {json.dumps(data, indent=2)}
        
        Custom Requirements:
        - Focus: {custom_requirements.get('focus', 'exploitation')}
        - Depth: {custom_requirements.get('depth', 'detailed')}
        - Areas: {custom_requirements.get('areas', [])}
        - Special Instructions: {custom_requirements.get('instructions', 'None')}
        - Style: {custom_requirements.get('style', 'technical')}
        
        Generate an analysis that meets these specific requirements while maintaining
        rigorous reasoning and accurate technical details.
        """
        
        try:
            analysis_content = await self._generate_content(prompt)
            
            return {
                "success": True,
                "type": "custom_analysis",
                "template_type": template_type,
                "content": analysis_content,
                "requirements_met": custom_requirements,
                "generated_at": datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Failed to generate custom analysis: {e}")
            return {"success": False, "error": str(e)}
    
    def _load_template(self, template_name: str) -> str:
        """Load analysis template"""
        templates = {
            "attack_chain": """
            You are generating an attack chain for penetration testing.
            Focus on multi-step logic, risk assessment, and tactical recommendations.
            Use clear, technical language that pentesters can understand and implement.
            """,
            "exploit_generation": """
            You are generating exploits for security research.
            Include detailed code, procedures, and safety considerations.
            Use appropriate technical terminology and provide actionable guidance.
            """,
            "vulnerability_analysis": """
            You are generating vulnerability analysis for security assessments.
            Map to frameworks and provide in-depth technical insights.
            Include evidence, procedures, and analysis details.
            """,
            "hindi_analysis": """
            Generate in a mix of hindi+english for Indian SME clients. more like hinglish.
            Use easy and understable terminology for this.
            """
        }
        return templates.get(template_name, "Generate a professional penetration testing analysis.")
    
    def _calculate_attack_risk(self, data: List[Dict[str, Any]]) -> str:
        """Calculate attack risk level"""
        critical_count = sum(1 for d in data if d.get('severity', '').lower() == 'critical')
        high_count = sum(1 for d in data if d.get('severity', '').lower() == 'high')
        
        if critical_count > 0:
            return "Critical"
        elif high_count > 2:
            return "High"
        elif high_count > 0:
            return "Medium"
        else:
            return "Low"
    
    def _extract_key_steps(self, content: str) -> List[str]:
        """Extract key steps from content"""
        steps = []
        lines = content.split('\n')
        
        for line in lines:
            if any(keyword in line.lower() for keyword in ['step', 'phase', 'action', 'exploit']):
                if len(line.strip()) > 20:
                    steps.append(line.strip())
        
        return steps[:5]
    
    def _detect_code_language(self, content: str) -> str:
        """Detect code language in content"""
        if 'python' in content.lower() or 'def ' in content:
            return "Python"
        elif 'bash' in content.lower() or '#!' in content:
            return "Bash"
        else:
            return "Unknown"
    
    def _extract_tools(self, content: str) -> List[str]:
        """Extract required tools"""
        tools = []
        lines = content.split('\n')
        
        for line in lines:
            if any(keyword in line.lower() for keyword in ['tool', 'require', 'use', 'install']):
                if len(line.strip()) > 20:
                    tools.append(line.strip())
        
        return tools[:10]
    
    def _calculate_exploitability_score(self, data: Dict[str, Any]) -> int:
        """Calculate exploitability score (0-100)"""
        total_factors = data.get('total_factors', 100)
        exploitable_factors = data.get('exploitable_factors', 50)
        
        if total_factors == 0:
            return 0
        
        return int((exploitable_factors / total_factors) * 100)
    
    def _extract_impact_areas(self, content: str) -> List[str]:
        """Extract impact areas"""
        areas = []
        lines = content.split('\n')
        
        for line in lines:
            if any(keyword in line.lower() for keyword in ['impact', 'affect', 'consequence', 'damage']):
                if len(line.strip()) > 20:
                    areas.append(line.strip())
        
        return areas[:5]
    
    def _extract_mitigation_priorities(self, content: str) -> List[Dict[str, str]]:
        """Extract mitigation priorities"""
        priorities = []
        lines = content.split('\n')
        
        priority_levels = ['critical', 'high', 'medium', 'low']
        
        for line in lines:
            for level in priority_levels:
                if level in line.lower() and any(action in line.lower() for action in ['priority', 'immediate', 'urgent']):
                    priorities.append({
                        "priority": level,
                        "description": line.strip()
                    })
                    break
        
        return priorities[:10]
    
    def _generate_english_summary(self, summary: Dict[str, Any]) -> str:
        """Generate English summary for Hindi analyses"""
        return f"""
        Executive Summary:
        - Total Vulnerabilities: {summary.get('total_vulnerabilities', 0)}
        - Critical Issues: {summary.get('critical_count', 0)}
        - Risk Level: {summary.get('overall_risk', 'Medium')}
        - Recommended Timeline: {summary.get('timeline', '30 days')}
        """
    
    def _estimate_confidence(self, data: List[Dict]) -> str:
        """Estimate confidence level"""
        evidence_count = sum(1 for d in data if d.get('evidence', False))
        
        if evidence_count / len(data) > 0.8:
            return "High"
        elif evidence_count / len(data) > 0.5:
            return "Medium"
        else:
            return "Low"
    
    def _extract_next_steps(self, content: str) -> List[str]:
        """Extract next steps"""
        steps = []
        lines = content.split('\n')
        
        for line in lines:
            if any(keyword in line.lower() for keyword in ['next', 'action', 'recommend', 'follow']):
                if len(line.strip()) > 20:
                    steps.append(line.strip())
        
        return steps[:5]
    
    def process_task(self, task: dict) -> dict:
        """Stub for MVP: Return success for reasoning task."""
        if task.get('type') == 'reasoning':
            return {'success': True, 'result': 'Reasoning complete'}
        raise ValueError('Invalid task type')
    
    def infer(self, input_data: str) -> dict:
        """Stub for MVP: Return success for inference."""
        return {'success': True, 'output': f'Inference for {input_data}'}

# Alias for test compatibility
DeepSeekR1Agent = DeepSeekAgent