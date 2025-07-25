"""
Gemini 2.5 Pro Agent - Specialized for report generation and professional documentation
Uses Google's Gemini 2.5 Pro for creating SME-friendly penetration testing reports
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import os
import google.generativeai as genai # type: ignore
from google.generativeai.types import HarmCategory, HarmBlockThreshold # type: ignore

logger = logging.getLogger(__name__)

@dataclass
class GeminiConfig:
    """Configuration for Gemini 2.5 Pro"""
    api_key: str
    model: str = "gemini-2.5-pro"
    max_output_tokens: int = 8192
    temperature: float = 0.1
    top_p: float = 0.8
    top_k: int = 40

class GeminiAgent:
    """AI Agent using Gemini 2.5 Pro for report generation and documentation"""
    
    def __init__(self, config: Optional[GeminiConfig] = None):
        self.config = config or GeminiConfig(
            api_key=os.getenv("GEMINI_API_KEY", "")
        )
        self.agent_id = "gemini_2_5_pro_agent"
        
        # Configure Gemini
        genai.configure(api_key=self.config.api_key)
        
        # Initialize model with safety settings for pentesting content
        self.model = genai.GenerativeModel(
            model_name=self.config.model,
            generation_config=genai.GenerationConfig(
                max_output_tokens=self.config.max_output_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
            ),
            safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        )
        
        self.report_templates = {
            "executive_summary": self._load_template("executive_summary"),
            "technical_details": self._load_template("technical_details"),
            "compliance_report": self._load_template("compliance_report"),
            "hindi_report": self._load_template("hindi_report")
        }
    
    async def initialize(self):
        """Initialize the Gemini agent"""
        logger.info("Initializing Gemini 2.5 Pro Agent for report generation")
        await self._validate_api_connection()
    
    async def _validate_api_connection(self):
        """Validate API connection"""
        try:
            response = await self._generate_content("Test connection", max_tokens=10)
            logger.info("Gemini API connection validated")
        except Exception as e:
            logger.error(f"Failed to validate Gemini API: {e}")
            raise
    
    async def _generate_content(self, prompt: str, **kwargs) -> str:
        """Generate content using Gemini API"""
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Gemini content generation failed: {e}")
            raise
    
    async def generate_executive_report(self, 
                                      pentest_results: Dict[str, Any], 
                                      client_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary report for SME leadership"""
        
        template = self.report_templates["executive_summary"]
        
        prompt = f"""
        {template}
        
        Penetration Testing Results:
        {json.dumps(pentest_results, indent=2)}
        
        Client Information:
        - Company: {client_info.get('company_name', 'Unknown')}
        - Industry: {client_info.get('industry', 'Unknown')}
        - Size: {client_info.get('size', 'SME')}
        - Primary Assets: {client_info.get('primary_assets', [])}
        
        Create an executive summary that:
        1. Explains security findings in business terms
        2. Quantifies risk with clear severity levels
        3. Provides actionable recommendations
        4. Includes cost-benefit analysis for fixes
        5. Sets realistic remediation timelines
        6. Uses language appropriate for C-level executives
        
        Focus on business impact rather than technical details.
        """
        
        try:
            report_content = await self._generate_content(prompt)
            
            return {
                "success": True,
                "report_type": "executive_summary",
                "content": report_content,
                "client": client_info.get('company_name', 'Unknown'),
                "generated_at": datetime.now().isoformat(),
                "risk_score": self._calculate_business_risk(pentest_results),
                "key_findings": self._extract_key_findings(report_content)
            }
        
        except Exception as e:
            logger.error(f"Failed to generate executive report: {e}")
            return {"success": False, "error": str(e)}
    
    async def generate_technical_report(self, 
                                      vulnerability_data: List[Dict[str, Any]], 
                                      scan_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detailed technical report for IT teams"""
        
        template = self.report_templates["technical_details"]
        
        prompt = f"""
        {template}
        
        Vulnerability Data:
        {json.dumps(vulnerability_data, indent=2)}
        
        Scan Results:
        {json.dumps(scan_results, indent=2)}
        
        Create a comprehensive technical report including:
        1. Detailed vulnerability descriptions with CVE references
        2. Step-by-step exploitation procedures
        3. Proof-of-concept code examples
        4. Technical remediation steps
        5. Verification procedures
        6. Tool-specific configurations
        7. Network diagrams where relevant
        
        Use technical language appropriate for system administrators and developers.
        Include specific commands, configurations, and code samples.
        """
        
        try:
            report_content = await self._generate_content(prompt)
            
            return {
                "success": True,
                "report_type": "technical_details",
                "content": report_content,
                "vulnerabilities_count": len(vulnerability_data),
                "generated_at": datetime.now().isoformat(),
                "technical_recommendations": self._extract_technical_recommendations(report_content),
                "tools_used": scan_results.get('tools_used', [])
            }
        
        except Exception as e:
            logger.error(f"Failed to generate technical report: {e}")
            return {"success": False, "error": str(e)}
    
    async def generate_compliance_report(self, 
                                       pentest_results: Dict[str, Any], 
                                       compliance_framework: str = "iso27001") -> Dict[str, Any]:
        """Generate compliance-focused report (ISO 27001, RBI, PCI-DSS, etc.)"""
        
        template = self.report_templates["compliance_report"]
        
        framework_mapping = {
            "iso27001": "ISO 27001:2013",
            "rbi": "RBI Cyber Security Framework",
            "pci_dss": "PCI DSS 4.0",
            "nist": "NIST Cybersecurity Framework"
        }
        
        framework_name = framework_mapping.get(compliance_framework, compliance_framework.upper())
        
        prompt = f"""
        {template}
        
        Compliance Framework: {framework_name}
        
        Penetration Testing Results:
        {json.dumps(pentest_results, indent=2)}
        
        Create a compliance-focused report that:
        1. Maps findings to specific {framework_name} controls
        2. Assesses compliance gaps and risks
        3. Provides control implementation guidance
        4. Includes audit trail documentation
        5. Suggests policy and procedure updates
        6. Provides timeline for compliance achievement
        7. Documents testing methodology alignment with standards
        
        Structure the report for regulatory review and audit purposes.
        Include specific control references and remediation priorities.
        """
        
        try:
            report_content = await self._generate_content(prompt)
            
            return {
                "success": True,
                "report_type": "compliance_report",
                "framework": compliance_framework,
                "content": report_content,
                "generated_at": datetime.now().isoformat(),
                "compliance_score": self._calculate_compliance_score(pentest_results),
                "control_gaps": self._extract_control_gaps(report_content),
                "remediation_priority": self._extract_remediation_priorities(report_content)
            }
        
        except Exception as e:
            logger.error(f"Failed to generate compliance report: {e}")
            return {"success": False, "error": str(e)}
    
    async def generate_hindi_report(self, 
                                  pentest_summary: Dict[str, Any], 
                                  client_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generate Hindi language report for Indian SME clients"""
        
        template = self.report_templates["hindi_report"]
        
        prompt = f"""
        {template}
        
        पेनिट्रेशन टेस्टिंग परिणाम:
        {json.dumps(pentest_summary, indent=2)}
        
        क्लाइंट जानकारी:
        - कंपनी: {client_info.get('company_name', 'अज्ञात')}
        - उद्योग: {client_info.get('industry', 'अज्ञात')}
        - मुख्य संपत्ति: {client_info.get('primary_assets', [])}
        
        Create a comprehensive report in Hindi that includes:
        1. कार्यकारी सारांश (Executive Summary)
        2. मुख्य सुरक्षा समस्याएं (Key Security Issues)
        3. जोखिम मूल्यांकन (Risk Assessment)
        4. सुधारात्मक सिफारिशें (Remediation Recommendations)
        5. व्यावसायिक प्रभाव (Business Impact)
        6. समयसीमा और लागत (Timeline and Cost)
        
        Use professional Hindi terminology for cybersecurity concepts.
        Make it accessible for Indian SME business owners and IT staff.
        Include both Hindi and English technical terms where appropriate.
        """
        
        try:
            report_content = await self._generate_content(prompt)
            
            return {
                "success": True,
                "report_type": "hindi_report",
                "language": "hindi",
                "content": report_content,
                "client": client_info.get('company_name', 'अज्ञात'),
                "generated_at": datetime.now().isoformat(),
                "summary_english": self._generate_english_summary(pentest_summary)
            }
        
        except Exception as e:
            logger.error(f"Failed to generate Hindi report: {e}")
            return {"success": False, "error": str(e)}
    
    async def generate_remediation_guide(self, 
                                       vulnerabilities: List[Dict[str, Any]], 
                                       client_context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate step-by-step remediation guide"""
        
        prompt = f"""
        Create a detailed remediation guide for the following vulnerabilities:
        
        Vulnerabilities:
        {json.dumps(vulnerabilities, indent=2)}
        
        Client Context:
        - Technical Expertise: {client_context.get('tech_level', 'intermediate')}
        - Budget: {client_context.get('budget', 'moderate')}
        - Timeline: {client_context.get('timeline', '30 days')}
        - Infrastructure: {client_context.get('infrastructure', 'cloud')}
        
        For each vulnerability, provide:
        1. Detailed step-by-step remediation instructions
        2. Required tools and resources
        3. Estimated time and cost
        4. Risk level if left unfixed
        5. Testing procedures to verify fix
        6. Alternative solutions if primary fix isn't feasible
        
        Prioritize fixes based on risk level and ease of implementation.
        Include screenshots or diagrams where helpful.
        """
        
        try:
            guide_content = await self._generate_content(prompt)
            
            return {
                "success": True,
                "guide_type": "remediation",
                "content": guide_content,
                "vulnerabilities_count": len(vulnerabilities),
                "generated_at": datetime.now().isoformat(),
                "estimated_cost": self._estimate_remediation_cost(vulnerabilities),
                "timeline": self._estimate_remediation_timeline(vulnerabilities),
                "priority_order": self._prioritize_remediations(vulnerabilities)
            }
        
        except Exception as e:
            logger.error(f"Failed to generate remediation guide: {e}")
            return {"success": False, "error": str(e)}
    
    async def generate_custom_report(self, 
                                   template_type: str, 
                                   data: Dict[str, Any], 
                                   custom_requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Generate custom report based on specific requirements"""
        
        prompt = f"""
        Create a custom penetration testing report with the following requirements:
        
        Template Type: {template_type}
        Data: {json.dumps(data, indent=2)}
        
        Custom Requirements:
        - Audience: {custom_requirements.get('audience', 'technical')}
        - Format: {custom_requirements.get('format', 'detailed')}
        - Focus Areas: {custom_requirements.get('focus_areas', [])}
        - Special Instructions: {custom_requirements.get('instructions', 'None')}
        - Brand Guidelines: {custom_requirements.get('branding', 'professional')}
        
        Generate a report that meets these specific requirements while maintaining
        professional standards and clear communication of security findings.
        """
        
        try:
            report_content = await self._generate_content(prompt)
            
            return {
                "success": True,
                "report_type": "custom",
                "template_type": template_type,
                "content": report_content,
                "requirements_met": custom_requirements,
                "generated_at": datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Failed to generate custom report: {e}")
            return {"success": False, "error": str(e)}
    
    def _load_template(self, template_name: str) -> str:
        """Load report template"""
        templates = {
            "executive_summary": """
            You are generating an executive summary report for a penetration testing engagement.
            Focus on business impact, risk quantification, and strategic recommendations.
            Use clear, non-technical language that executives can understand and act upon.
            """,
            "technical_details": """
            You are generating a technical report for IT professionals.
            Include detailed technical information, code examples, and specific remediation steps.
            Use appropriate technical terminology and provide actionable technical guidance.
            """,
            "compliance_report": """
            You are generating a compliance-focused report for regulatory requirements.
            Map all findings to specific compliance controls and provide audit-ready documentation.
            Include evidence, testing procedures, and compliance gap analysis.
            """,
            "hindi_report": """
            Generate in a mix of hindi+english for Indian SME clients. more like hinglish.
            Use easy and understable terminology for this.
            """
        }
        return templates.get(template_name, "Generate a professional penetration testing report.")
    
    def _calculate_business_risk(self, results: Dict[str, Any]) -> str:
        """Calculate business risk level"""
        critical_count = results.get('critical_vulnerabilities', 0)
        high_count = results.get('high_vulnerabilities', 0)
        
        if critical_count > 0:
            return "Critical"
        elif high_count > 2:
            return "High"
        elif high_count > 0:
            return "Medium"
        else:
            return "Low"
    
    def _extract_key_findings(self, content: str) -> List[str]:
        """Extract key findings from report content"""
        findings = []
        lines = content.split('\n')
        
        for line in lines:
            if any(keyword in line.lower() for keyword in ['critical', 'high risk', 'vulnerability', 'finding']):
                if len(line.strip()) > 20:  # Avoid empty or very short lines
                    findings.append(line.strip())
        
        return findings[:5]  # Top 5 findings
    
    def _extract_technical_recommendations(self, content: str) -> List[str]:
        """Extract technical recommendations"""
        recommendations = []
        lines = content.split('\n')
        
        for line in lines:
            if any(keyword in line.lower() for keyword in ['recommend', 'should', 'implement', 'patch', 'update']):
                if len(line.strip()) > 20:
                    recommendations.append(line.strip())
        
        return recommendations[:10]
    
    def _calculate_compliance_score(self, results: Dict[str, Any]) -> int:
        """Calculate compliance score (0-100)"""
        total_checks = results.get('total_checks', 100)
        passed_checks = results.get('passed_checks', 50)
        
        if total_checks == 0:
            return 0
        
        return int((passed_checks / total_checks) * 100)
    
    def _extract_control_gaps(self, content: str) -> List[str]:
        """Extract compliance control gaps"""
        gaps = []
        lines = content.split('\n')
        
        for line in lines:
            if any(keyword in line.lower() for keyword in ['gap', 'missing', 'non-compliant', 'fails']):
                if len(line.strip()) > 20:
                    gaps.append(line.strip())
        
        return gaps[:5]
    
    def _extract_remediation_priorities(self, content: str) -> List[Dict[str, str]]:
        """Extract remediation priorities"""
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
        """Generate English summary for Hindi reports"""
        return f"""
        Executive Summary:
        - Total Vulnerabilities: {summary.get('total_vulnerabilities', 0)}
        - Critical Issues: {summary.get('critical_count', 0)}
        - Risk Level: {summary.get('overall_risk', 'Medium')}
        - Recommended Timeline: {summary.get('timeline', '30 days')}
        """
    
    def _estimate_remediation_cost(self, vulnerabilities: List[Dict]) -> str:
        """Estimate remediation cost"""
        cost_map = {"critical": 50000, "high": 25000, "medium": 10000, "low": 5000}
        total_cost = 0
        
        for vuln in vulnerabilities:
            severity = vuln.get('severity', 'medium').lower()
            total_cost += cost_map.get(severity, 10000)
        
        if total_cost < 20000:
            return "₹10,000 - ₹20,000"
        elif total_cost < 50000:
            return "₹20,000 - ₹50,000"
        elif total_cost < 100000:
            return "₹50,000 - ₹1,00,000"
        else:
            return "₹1,00,000+"
    
    def _estimate_remediation_timeline(self, vulnerabilities: List[Dict]) -> str:
        """Estimate remediation timeline"""
        critical_count = sum(1 for v in vulnerabilities if v.get('severity', '').lower() == 'critical')
        high_count = sum(1 for v in vulnerabilities if v.get('severity', '').lower() == 'high')
        
        if critical_count > 0:
            return "Immediate (1-7 days)"
        elif high_count > 2:
            return "Urgent (1-2 weeks)"
        else:
            return "Standard (2-4 weeks)"
    
    def _prioritize_remediations(self, vulnerabilities: List[Dict]) -> List[Dict]:
        """Prioritize remediation order"""
        priority_map = {"critical": 1, "high": 2, "medium": 3, "low": 4}
        
        sorted_vulns = sorted(
            vulnerabilities, 
            key=lambda x: priority_map.get(x.get('severity', 'medium').lower(), 3)
        )
        
        return [{"name": v.get('name', 'Unknown'), "severity": v.get('severity', 'medium')} for v in sorted_vulns[:5]]