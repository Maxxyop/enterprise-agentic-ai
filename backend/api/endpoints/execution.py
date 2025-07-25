from fastapi import APIRouter, HTTPException
from execution_engine.core.executor import Executor

router = APIRouter()
executor = Executor()

@router.post("/execute")
async def execute_command(command: str):
    try:
        result = executor.run_command(command)
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status")
async def get_status():
    return {"status": "running"}