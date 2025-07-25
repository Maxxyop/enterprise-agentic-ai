import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class DASTPrompts:
    """Manages DAST-specific prompt templates for autonomous scanning."""
    
    def __init__(self, prompt_dir: str = "prompt_engineering/templates"):
        self.prompt_dir = Path(prompt_dir)
        self.prompt_templates = self._load_templates()
        
    def _load_templates(self) -> Dict[str, str]:
        """Load DAST prompt templates from JSON files."""
        templates = {}
        template_files = ["recon.json", "scan.json", "exploit.json"]
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
            logger.error(f"Failed to load DAST templates: {e}")
            return {}
    
    def get_prompt(self, task_type: str, target: str, params: Dict[str, Any]) -> Optional[str]:
        """Generate a DAST prompt for a specific task."""
        try:
            template_key = f"{task_type.lower()}.json"
            prompt_template = self.prompt_templates.get(
                template_key,
                "Perform {task_type} on {target} with parameters: {params}"
            )
            prompt = prompt_template.format(task_type=task_type, target=target, params=json.dumps(params))
            logger.debug(f"Generated prompt for {task_type} on {target}")
            return prompt
        except Exception as e:
            logger.error(f"Failed to generate prompt for {task_type}: {e}")
            return None

async def main():
    """Example usage of DASTPrompts."""
    prompts = DASTPrompts()
    prompt = prompts.get_prompt(
        task_type="recon",
        target="shop.example.com",
        params={"scan_type": "port_scan", "ports": "80,443"}
    )
    print(f"Recon Prompt: {prompt}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())