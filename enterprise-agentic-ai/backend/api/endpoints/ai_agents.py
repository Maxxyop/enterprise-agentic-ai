from fastapi import APIRouter, HTTPException
from ai_agents.foundation_models.model_manager import ModelManager

router = APIRouter()
model_manager = ModelManager()

@router.post("/ai_agents/deepseek_r1")
async def run_deepseek_r1_task(task: dict):
    try:
        result = model_manager.run_deepseek_r1(task)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ai_agents/qwen_7b")
async def run_qwen_7b_task(task: dict):
    try:
        result = model_manager.run_qwen_7b(task)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))