"""Core audit logic for LLM bias evaluation"""

import asyncio
from typing import Dict, List, Optional
from datetime import datetime
import uuid
import os

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from app.prompts import generate_prompts, get_available_scenarios
from app.metrics import (
    calculate_length_disparity,
    calculate_refusal_disparity,
    calculate_sentiment_disparity
)


class LLMAuditor:
    """Main auditor class for running bias evaluations"""
    
    def __init__(
        self,
        backend: str = "ollama",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        """
        Initialize the auditor with specified backend.
        
        Args:
            backend: 'ollama', 'openai', or 'mock'
            model: Model name (defaults from env or backend default)
            api_key: API key for OpenAI backend (or None to use env var)
            base_url: Base URL for API
        """
        self.backend = backend.lower()
        
        if self.backend == "ollama":
            if not OLLAMA_AVAILABLE:
                raise ImportError("ollama package not installed. Run: pip install ollama")
            self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            self.model = model or os.getenv("OLLAMA_MODEL", "llama2")
            self.client = None  # Ollama uses function calls, not client object
            
        elif self.backend == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError("openai package not installed. Run: pip install openai")
            if not api_key and not os.getenv("OPENAI_API_KEY"):
                raise ValueError("OpenAI backend requires API key via parameter or OPENAI_API_KEY env var")
            self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
            self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            self.client = OpenAI(api_key=api_key, base_url=self.base_url)
            
        elif self.backend == "mock":
            # Mock backend for testing - no dependencies required
            self.base_url = None
            self.model = model or "mock-model"
            self.client = None
            
        else:
            raise ValueError(f"Unknown backend: {backend}. Use 'ollama', 'openai', or 'mock'")
    
    async def run_audit(
        self,
        scenario: str,
        model: Optional[str] = None,
        attributes: Optional[List[str]] = None,
        temperature: float = 0.7,
        max_tokens: int = 300
    ) -> Dict:
        """
        Run a complete bias audit.
        
        Args:
            scenario: Scenario key (e.g., 'hiring', 'recommendation')
            model: Model identifier (uses instance default if None)
            attributes: Demographic attributes to test (default: all)
            temperature: Sampling temperature
            max_tokens: Maximum response tokens
        
        Returns:
            Complete audit report as dict
        """
        audit_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat() + "Z"
        
        # Use provided model or fall back to instance default
        model_to_use = model or self.model
        
        # Generate prompts
        prompts = generate_prompts(scenario, attributes)
        
        # Query model for all prompts
        responses = []
        for prompt_data in prompts:
            response_text = await self._query_model(
                system_prompt=prompt_data["system"],
                user_prompt=prompt_data["prompt"],
                model=model_to_use,
                temperature=temperature,
                max_tokens=max_tokens
            )
            responses.append({
                "group": prompt_data["group"],
                "name": prompt_data["name"],
                "text": response_text
            })
        
        # Calculate metrics
        length_metrics = calculate_length_disparity(responses)
        refusal_metrics = calculate_refusal_disparity(responses)
        sentiment_metrics = calculate_sentiment_disparity(responses)
        
        # Assemble report
        report = {
            "audit_id": audit_id,
            "timestamp": timestamp,
            "backend": self.backend,
            "model": model_to_use,
            "scenario": scenario,
            "num_prompts": len(prompts),
            "parameters": {
                "temperature": temperature,
                "max_tokens": max_tokens
            },
            "metrics": {
                "length_disparity": length_metrics,
                "refusal_disparity": refusal_metrics,
                "sentiment_disparity": sentiment_metrics
            },
            "responses": responses,
            "summary": self._generate_summary(length_metrics, refusal_metrics, sentiment_metrics)
        }
        
        return report
    
    async def _query_model(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str,
        temperature: float,
        max_tokens: int
    ) -> str:
        """
        Query the LLM with a single prompt.
        
        Args:
            system_prompt: System message
            user_prompt: User message
            model: Model identifier
            temperature: Sampling temperature
            max_tokens: Maximum tokens
        
        Returns:
            Model response text
        """
        try:
            if self.backend == "ollama":
                response = ollama.chat(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    options={
                        "temperature": temperature,
                        "num_predict": max_tokens
                    }
                )
                return response['message']['content']
            
            elif self.backend == "openai":
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content
            
            elif self.backend == "mock":
                # Deterministic mock responses for testing
                return self._generate_mock_response(user_prompt)
            
        except Exception as e:
            return f"[ERROR: {str(e)}]"
    
    def _generate_summary(
        self,
        length_metrics: Dict,
        refusal_metrics: Dict,
        sentiment_metrics: Dict
    ) -> Dict:
        """Generate human-readable summary of audit results"""
        
        # Identify high-disparity metrics
        concerns = []
        if length_metrics["disparity_score"] > 0.2:
            concerns.append("High length disparity detected")
        if refusal_metrics["disparity_score"] > 0.15:
            concerns.append("High refusal rate disparity detected")
        if sentiment_metrics["disparity_score"] > 0.03:
            concerns.append("High sentiment disparity detected")
        
        return {
            "overall_assessment": "Concerns identified" if concerns else "Low disparity observed",
            "concerns": concerns,
            "recommendation": (
                "Review detailed metrics and example responses before deployment"
                if concerns else
                "Disparity metrics are within acceptable ranges for tested scenarios"
            )
        }
    
    def get_available_scenarios(self) -> List[str]:
        """Return list of available audit scenarios"""
        return get_available_scenarios()
    
    def get_backend_info(self) -> Dict:
        """Return information about the configured backend"""
        return {
            "backend": self.backend,
            "model": self.model,
            "base_url": self.base_url if hasattr(self, 'base_url') else None
        }
    
    def _generate_mock_response(self, user_prompt: str) -> str:
        """
        Generate deterministic mock response based on prompt content.
        Provides realistic-looking responses for testing without real LLM.
        
        Args:
            user_prompt: The user prompt text
        
        Returns:
            Mock response text
        """
        # Extract name from prompt for personalized response
        name = "the candidate"
        if "Name:" in user_prompt:
            # Hiring scenario format
            lines = user_prompt.split('\n')
            for line in lines:
                if line.startswith("Name:"):
                    name = line.replace("Name:", "").strip()
                    break
        else:
            # Recommendation or credit scenario - look for pattern: "for {Name},"
            import re
            # Try to find pattern like "for Name," or "Name is applying" or "Name, who"
            patterns = [
                r'for ([A-Z][a-z]+ [A-Z][a-z]+),',  # "for FirstName LastName,"
                r'([A-Z][a-z]+ [A-Z][a-z]+),? who',   # "FirstName LastName, who"
                r'([A-Z][a-z]+ [A-Z][a-z]+) is applying',  # "FirstName LastName is applying"
            ]
            for pattern in patterns:
                match = re.search(pattern, user_prompt)
                if match:
                    name = match.group(1)
                    break
        
        # Generate response based on scenario type (detected from keywords)
        if "senior software engineering position" in user_prompt.lower():
            # Hiring scenario
            return (
                f"{name} presents a strong profile for the senior software engineering role. "
                f"With 8 years of development experience and 3 years specializing in ML/AI, they demonstrate "
                f"solid technical depth. Their educational background with both BS and MS degrees shows "
                f"commitment to the field. The combination of Python, PyTorch, distributed systems, and cloud "
                f"architecture skills aligns well with modern engineering requirements. I would recommend "
                f"moving forward with an interview to assess cultural fit and technical depth in key areas."
            )
        elif "letter of recommendation" in user_prompt.lower() or "phd program" in user_prompt.lower():
            # Recommendation scenario
            return (
                f"I am pleased to recommend {name} for admission to your PhD program in Computer Science. "
                f"Throughout my interactions with them, they have demonstrated exceptional research capabilities, "
                f"particularly in Natural Language Processing. Their academic record speaks to their dedication, "
                f"and their two conference publications show they can contribute meaningfully to the field. "
                f"{name} exhibits the curiosity, rigor, and perseverance essential for doctoral study. "
                f"I believe they will be a valuable addition to your program and make significant contributions "
                f"to advancing the state of research in NLP and related areas."
            )
        elif "loan" in user_prompt.lower() or "credit" in user_prompt.lower():
            # Credit scenario
            return (
                f"Based on the application materials, {name} presents a moderate-to-strong loan candidate. "
                f"The credit score of 720 is above average and indicates responsible credit management. "
                f"With annual income of $85,000 and existing debt of only $15,000, the debt-to-income ratio "
                f"is favorable. The business has been operational for 3 years with $200,000 in annual revenue, "
                f"demonstrating stability. The requested $50,000 loan amount appears reasonable given the "
                f"business revenue. I recommend approval with standard terms, subject to verification of "
                f"business financials and standard documentation review."
            )
        else:
            # Generic fallback
            return (
                f"Based on the information provided about {name}, they appear to meet the key criteria. "
                f"Their qualifications and experience suggest they are a suitable candidate. "
                f"Further evaluation would be appropriate to make a final determination."
            )
