from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .endpoints import engagements, ai_agents, execution

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this as needed for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(engagements.router, prefix="/engagements", tags=["engagements"])
app.include_router(ai_agents.router, prefix="/ai_agents", tags=["ai_agents"])
app.include_router(execution.router, prefix="/execution", tags=["execution"])

@app.get("/")
def read_root():
    return {"message": "Welcome to the Enterprise Agentic AI API"}