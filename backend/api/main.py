from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .endpoints import engagements, ai_agents, execution
import logging

logger = logging.getLogger(__name__)

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

@app.on_event("startup")
def startup_event():
    logger.info("Enterprise Agentic AI API is starting up.")

@app.on_event("shutdown")
def shutdown_event():
    logger.info("Enterprise Agentic AI API is shutting down.")

@app.get("/")
def read_root():
    """Root endpoint for health check and welcome message."""
    logger.info("Root endpoint accessed.")
    return {"message": "Welcome to the Enterprise Agentic AI API"}