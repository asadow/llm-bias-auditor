"""FastAPI application for LLM bias auditing"""

import os
from typing import Optional, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from app.audit import LLMAuditor


# Load environment variables from .env if present
load_dotenv()

app = FastAPI(
    title="LLM Bias Auditing Service",
    description="API for auditing LLMs against potential demographic bias",
    version="0.1.0"
)


class AuditRequest(BaseModel):
    """Request model for audit endpoint"""
    model: Optional[str] = Field(None, description="Model identifier (uses backend default if not specified)")
    scenario: str = Field(..., description="Audit scenario: 'hiring', 'recommendation', or 'credit'")
    attributes: Optional[List[str]] = Field(None, description="Demographic attributes to test (default: all)")
    temperature: float = Field(0.7, ge=0, le=2, description="Sampling temperature")
    max_tokens: int = Field(300, ge=50, le=2000, description="Maximum response tokens")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint with service info"""
    return {
        "status": "operational",
        "version": "0.1.0"
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "0.1.0"
    }


@app.get("/backend")
async def get_backend():
    """Get backend configuration information"""
    backend = os.getenv("LLM_BACKEND", "ollama")
    kwargs = {"backend": backend}
    if backend == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            kwargs["api_key"] = api_key
    
    auditor = LLMAuditor(**kwargs)
    info = auditor.get_backend_info()
    return {
        "backend": info["backend"],
        "model": info["model"],
        "base_url": info.get("base_url")
    }


@app.get("/scenarios")
async def list_scenarios():
    """List available audit scenarios"""
    backend = os.getenv("LLM_BACKEND", "ollama")
    # Only pass api_key if using OpenAI backend
    kwargs = {"backend": backend}
    if backend == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            kwargs["api_key"] = api_key
    
    auditor = LLMAuditor(**kwargs)
    scenarios = auditor.get_available_scenarios()
    return {
        "scenarios": scenarios,
        "descriptions": {
            "hiring": "Evaluate candidate for a senior software engineering position",
            "recommendation": "Write a recommendation letter for PhD program application",
            "credit": "Assess a small business loan application"
        }
    }


@app.post("/audit")
async def run_audit(request: AuditRequest):
    """
    Run a bias audit on the specified model and scenario.
    
    Returns a complete audit report with disparity metrics and detailed responses.
    """
    # Get backend configuration from environment
    backend = os.getenv("LLM_BACKEND", "ollama")
    
    # Initialize auditor with backend-specific configuration
    kwargs = {"backend": backend}
    if backend == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(
                status_code=400,
                detail="OpenAI backend requires OPENAI_API_KEY environment variable"
            )
        kwargs["api_key"] = api_key
    
    auditor = LLMAuditor(**kwargs)
    
    # Validate scenario
    available_scenarios = auditor.get_available_scenarios()
    if request.scenario not in available_scenarios:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid scenario. Available: {available_scenarios}"
        )
    
    try:
        # Run audit
        report = await auditor.run_audit(
            model=request.model,
            scenario=request.scenario,
            attributes=request.attributes,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        return report
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Audit failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
