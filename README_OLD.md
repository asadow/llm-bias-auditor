# LLM Bias Auditing Service

A FastAPI-based service for auditing large language models against potential demographic bias through controlled prompt variation and disparity analysis.

**By default, the system runs entirely locally using Ollama. Support for OpenAI-compatible APIs is provided optionally.**

## Purpose

This tool operationalizes ethical AI evaluation by:
- Testing LLM behavior across demographic attributes
- Computing interpretable disparity metrics
- Generating actionable audit reports
- Supporting responsible AI deployment decisions

## Technical Stack

- **FastAPI**: REST API framework
- **Ollama**: Local LLM backend (default, no API key required)
- **OpenAI-compatible endpoints**: Optional cloud LLM integration
- **Python standard libraries**: Core logic
- **Simple NLP metrics**: Length, refusal detection, sentiment proxy

## Quick Start

### Local Setup (Default - No API Key Needed)

Use Python 3.11 for best compatibility with pinned deps.

```bash
# Install Ollama (if not already installed)
# Visit: https://ollama.ai

# Pull a model (pick a small one on 8GB Macs)
# Examples: llama3.2:3b, qwen2.5:1.5b, phi3:mini
ollama pull llama3.2:3b

# Install Python dependencies
pip install -r requirements.txt

# Run the service (uses Ollama by default)
uvicorn app.main:app --reload

# Run an audit
curl -X POST "http://localhost:8000/audit" \
  -H "Content-Type: application/json" \
  -d '{"scenario": "hiring"}'
```

### Optional: Using OpenAI or Compatible APIs

If you prefer to use OpenAI or other compatible endpoints:

```bash
# Copy environment template
cp .env.example .env

# Edit .env and set:
# LLM_BACKEND=openai
# OPENAI_API_KEY=your-key-here
# OPENAI_MODEL=gpt-4o-mini (optional)

# Run the service
uvicorn app.main:app --reload
```

## How It Works

## Running on 8GB Macs (M1/M2 Air)

- Lightweight approach: Use a small Ollama model.
  - Recommended: `llama3.2:3b`, `qwen2.5:1.5b`, or `phi3:mini`.
  - Set your model in `.env` or per-request:
    - `.env`: `OLLAMA_MODEL=llama3.2:3b`
    - Request override: `{"model": "llama3.2:3b"}` in `POST /audit`.
- Reduce output size: Lower `max_tokens` (e.g., 200) in the request.
- Avoid parallel load: The auditor runs prompts sequentially by default to limit memory pressure.

If you need larger models (7B+), prefer offloading inference:

- Use OpenAI-compatible APIs:
  - Set `LLM_BACKEND=openai` and `OPENAI_API_KEY` in `.env`.
  - Start the API locally as usual; compute happens in the cloud.
- Point to a remote Ollama server:
  - Run Ollama on a stronger machine and expose/forward port 11434.
  - Set `OLLAMA_BASE_URL` in `.env` (example: `http://your-remote:11434`).
  - With SSH port forwarding:
    - `ssh -L 11434:localhost:11434 user@remote-host`
    - Keep `OLLAMA_BASE_URL=http://localhost:11434` locally.

Full remote dev (e.g., Codespaces) is optional; generally you only need to remote the model backend, not the FastAPI app.

1. **Controlled Variation**: Generates prompts varying only demographic attributes (gender, ethnicity, age)
2. **Consistent Evaluation**: Queries the same LLM with identical contexts
3. **Disparity Metrics**: Compares response characteristics across groups
4. **Clear Reporting**: Outputs interpretable JSON reports

## API Endpoints

### `POST /audit`

Run a bias audit on a specified model and scenario.

**Request Body:**
```json
{
  "scenario": "hiring",
  "attributes": ["gender", "race"]
}
```

Optional fields:
- `model`: Override default model (e.g., "llama2" for Ollama or "gpt-4o-mini" for OpenAI)
- `temperature`: Sampling temperature (default: 0.7)
- `max_tokens`: Maximum response tokens (default: 300)

**Response:**
```json
{
  "audit_id": "uuid",
  "timestamp": "2026-01-10T...",
  "backend": "ollama",
  "model": "llama2",
  "metrics": {
    "length_disparity": 0.15,
    "refusal_rate_disparity": 0.03,
    "sentiment_disparity": 0.08
  },
  "details": {...}
}
```

## Metrics

### Length Disparity
Measures variance in response length across demographic groups. High disparity may indicate differential treatment.

### Refusal Rate Disparity
Tracks how often the model declines to answer for different groups. Unequal refusal suggests potential bias.

### Sentiment Disparity
Proxy metric using keyword analysis to estimate sentiment variance across groups.

## Responsible AI Considerations

### What This Tool Does
- Surfaces potential disparities in model behavior
- Provides quantitative basis for evaluation
- Supports informed deployment decisions

### What This Tool Does NOT Do
- Prove causation or intent
- Replace human judgment
- Guarantee fairness comprehensively

### Limitations
- Limited to tested scenarios and attributes
- Metrics are proxies, not ground truth
- Results require contextual interpretation
- Not a substitute for comprehensive fairness evaluation

### Recommended Usage
1. Use as one input among many in model evaluation
2. Combine with domain expert review
3. Test scenarios relevant to your deployment context
4. Document results as part of model cards
5. Re-audit when models or use cases change

## Future Enhancements

If time permits, planned extensions include:
- PyTorch-based embedding similarity analysis
- Ollama integration for local model testing
- Web UI for interactive auditing
- Additional statistical fairness metrics
- Confidence intervals and significance testing

## References

- [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework)
- [Fairness and Machine Learning (fairmlbook.org)](https://fairmlbook.org/)
- Mitchell et al., "Model Cards for Model Reporting" (2019)

## License

MIT

## Author

Built to support responsible AI deployment in research and educational contexts.
