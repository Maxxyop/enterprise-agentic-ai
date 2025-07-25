import json
import logging
from typing import Dict, Any, Optional
from ai_agents.foundation_models.llm_interface import LLMFactory
from pathlib import Path

logger = logging.getLogger(__name__)

class CodeGenerator:
    """Generates scripts for DAST tools using DeepSeek R1."""
    
    def __init__(self, llm_factory: LLMFactory, prompt_dir: str = "prompt_engineering/templates"):
        self.llm_factory = llm_factory
        self.prompt_dir = Path(prompt_dir)
        self.prompt_templates = self._load_templates()
        
    def _load_templates(self) -> Dict[str, str]:
        """Load prompt templates from JSON files."""
        templates = {}
        template_files = ["nmap_script.json", "zap_script.json", "sqlmap_script.json"]
        try:
            for file in template_files:
                file_path = self.prompt_dir / file
                if file_path.exists():
                    with open(file_path, "r") as f:
                        templates[file] = json.load(f)["prompt"]
                else:
                    logger.warning(f"Template file {file} not found")
            return templates
        except Exception as e:
            logger.error(f"Failed to load templates: {e}")
            return {}
    
    async def generate_script(self, tool: str, target: str, params: Dict[str, Any]) -> Optional[str]:
        """Generate a script for a specific DAST tool."""
        try:
            deepseek_client = self.llm_factory.get_client("code_generation")
            template_key = f"{tool.lower()}_script.json"
            prompt_template = self.prompt_templates.get(template_key, "Generate a {tool} script for {target} with parameters: {params}")
            prompt = prompt_template.format(tool=tool, target=target, params=json.dumps(params))
            script = await deepseek_client.generate(prompt, max_tokens=500)
            logger.debug(f"Generated {tool} script for {target}")
            return script
        except Exception as e:
            logger.error(f"Failed to generate {tool} script: {e}")
            return None
    
    async def validate_script(self, script: str, tool: str) -> bool:
        """Validate generated script syntax."""
        try:
            # Basic syntax check (extend for tool-specific validation)
            if not script or "error" in script.lower():
                return False
            return True
        except Exception as e:
            logger.error(f"Script validation failed: {e}")
            return False

async def main():
    """Example usage of CodeGenerator."""
    llm_factory = LLMFactory()
    generator = CodeGenerator(llm_factory)
    script = await generator.generate_script(
        tool="nmap",
        target="shop.example.com",
        params={"ports": "80,443", "scan_type": "syn"}
    )
    print(f"Nmap Script: {script}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())