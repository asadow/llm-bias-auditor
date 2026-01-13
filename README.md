# llm-bias-auditor

![CI](https://github.com/asadow/llm-bias-auditor/actions/workflows/ci.yml/badge.svg)

Bias auditing service for large language models using controlled prompt variation and disparity metrics.

## Why This Exists

LLMs can exhibit demographic bias in high-stakes scenarios like hiring, lending, and recommendations. This tool provides a systematic way to measure and document potential disparities in model behavior across demographic groups before deployment. It's designed for ML engineers, researchers, and organizations who need quantitative fairness evaluation as part of responsible AI workflows.

## Quick Start (No LLM Required)

The fastest way to explore this tool is with the built-in mock backend—**no LLM installation or API keys needed**:

```bash
# Install dependencies
pip install -r requirements.txt

# Run with mock backend
LLM_BACKEND=mock uvicorn app.main:app --reload

# Try the API
curl http://localhost:8000/health
# {"status":"healthy"}

curl http://localhost:8000/backend
# {"backend":"mock","model":"mock-model","base_url":null}

curl -X POST "http://localhost:8000/audit" \
  -H "Content-Type: application/json" \
  -d '{"scenario": "hiring"}'
# Returns full audit report with metrics (see below)
```

**Why mock mode exists:** It enables instant experimentation, CI/CD testing, and demonstrations without external dependencies. The mock backend generates deterministic, realistic responses for all scenarios, allowing you to understand the audit workflow and API structure before connecting to a real LLM.

**All tests run with no LLM required:**
```bash
pytest  # Uses mock backend automatically
```

### Example Response

```json
{
  "audit_id": "a1b2c3d4...",
  "timestamp": "2026-01-13T10:30:00Z",
  "backend": "mock",
  "model": "mock-model",
  "scenario": "hiring",
  "num_prompts": 24,
  "metrics": {
    "length_disparity": {
      "disparity_score": 0.0,
      "interpretation": "Low disparity - responses are relatively consistent in length",
      "group_means": {...},
      "overall_mean": 89.0
    },
    "refusal_disparity": {
      "disparity_score": 0.0,
      "interpretation": "Low disparity - refusal rates are similar across groups",
      "group_rates": {...}
    },
    "sentiment_disparity": {
      "disparity_score": 0.0,
      "interpretation": "Low disparity - sentiment is consistent across groups",
      "group_means": {...}
    }
  },
  "summary": {
    "overall_assessment": "Low disparity observed",
    "concerns": [],
    "recommendation": "Disparity metrics are within acceptable ranges for tested scenarios"
  },
  "responses": [...]
}
```

## Technical Stack

- **FastAPI**: REST API framework
- **Mock Backend**: Built-in deterministic responses (no dependencies)
- **Ollama**: Local LLM backend (optional, no API key required)
- **OpenAI-compatible endpoints**: Optional cloud LLM integration
- **Python standard libraries**: Core logic with simple NLP metrics

## Using Real LLMs

### Option 1: Local with Ollama (Recommended)

```bash
# Install Ollama: https://ollama.ai

# Pull a model (examples for 8GB Macs)
ollama pull llama3.2:3b    # or qwen2.5:1.5b, phi3:mini

# Run the service
uvicorn app.main:app --reload

# Run an audit
curl -X POST "http://localhost:8000/audit" \
  -H "Content-Type: application/json" \
  -d '{"scenario": "hiring", "model": "llama3.2:3b"}'
```

### Option 2: Cloud with OpenAI

```bash
# Set up environment
cp .env.example .env
# Edit .env:
#   LLM_BACKEND=openai
#   OPENAI_API_KEY=your-key-here
#   OPENAI_MODEL=gpt-4o-mini

# Run the service
uvicorn app.main:app --reload
```

### Option 3: Remote Ollama

```bash
# On remote machine with GPU: ollama serve
# Local SSH port forward:
ssh -L 11434:localhost:11434 user@remote-host

# In .env:
#   OLLAMA_BASE_URL=http://localhost:11434
#   OLLAMA_MODEL=llama3.2:3b
```

## API Endpoints

### `GET /health`
Health check returning service status.

### `GET /backend`
Returns currently configured LLM backend and model:
```json
{
  "backend": "mock",
  "model": "mock-model",
  "base_url": null
}
```

### `GET /scenarios`
Lists available audit scenarios with descriptions.

### `POST /audit`
Run a bias audit on the specified scenario.

**Request:**
```json
{
  "scenario": "hiring",
  "model": "llama3.2:3b",
  "temperature": 0.7,
  "max_tokens": 300
}
```

**Parameters:**
- `scenario` (required): `"hiring"`, `"recommendation"`, or `"credit"`
- `model` (optional): Override default model
- `temperature` (optional): Sampling temperature (0-2, default: 0.7)
- `max_tokens` (optional): Max response tokens (50-2000, default: 300)
- `attributes` (optional): Demographic groups to test (default: all)

## How It Works

1. **Controlled Variation**: Generates prompts varying only demographic attributes (name signals for gender/ethnicity)
2. **Consistent Evaluation**: Queries the same LLM with identical contexts and qualifications
3. **Disparity Metrics**: Compares response characteristics across demographic groups
4. **Clear Reporting**: Outputs interpretable JSON reports with actionable summaries

## Metrics

### Length Disparity
Measures variance in response length across demographic groups. Coefficient of variation (CV) indicates relative disparity—higher scores suggest differential treatment.

### Refusal Rate Disparity
Tracks how often the model declines to respond for different groups. Computed as the range between max and min refusal rates.

### Sentiment Disparity
Keyword-based proxy using positive/negative word counts. Measures tone differences across groups.

## Running on 8GB Macs

- Use lightweight models: `llama3.2:3b`, `qwen2.5:1.5b`, or `phi3:mini`
- Reduce `max_tokens` to 200 in requests
- Prompts run sequentially by default (low memory pressure)
- For larger models, use remote Ollama or cloud APIs

## Testing

All tests run without external LLM dependencies:

```bash
# Run all tests (uses mock backend)
pytest

# Run only mock backend tests
pytest tests/test_mock_backend.py -v

# Verify no-LLM execution
LLM_BACKEND=mock pytest tests/test_mock_backend.py -v
```

The test suite explicitly verifies:
- Mock backend initialization without dependencies
- Deterministic response generation
- Complete audit workflows for all scenarios
- Proper metric calculations

## Responsible AI Considerations

### What This Tool Does
- Surfaces potential disparities in model behavior across demographic groups
- Provides quantitative metrics for evaluation
- Supports informed deployment decisions with documented evidence

### What This Tool Does NOT Do
- Prove causation, intent, or bias in isolation
- Replace human judgment or domain expertise
- Guarantee fairness comprehensively across all dimensions
- Substitute for comprehensive fairness evaluation frameworks

### Limitations
- Limited to tested scenarios and demographic signals (name-based proxies)
- Metrics are statistical proxies, not ground truth measurements
- Results require contextual interpretation by domain experts
- Name-based demographic signaling has known limitations and cultural variance
- Does not test intersectional combinations systematically

### Recommended Usage
1. Use as one input among many in model evaluation pipelines
2. Combine with domain expert review and qualitative analysis
3. Test scenarios directly relevant to your deployment context
4. Document audit results in model cards and deployment documentation
5. Re-audit when models, prompts, or use cases change
6. Supplement with additional fairness testing tools and frameworks

## Future Enhancements

If time permits:
- PyTorch-based embedding similarity analysis
- Web UI for interactive auditing
- Additional statistical fairness metrics (demographic parity, equalized odds)
- Confidence intervals and significance testing
- Support for custom scenarios and demographic attributes

## References

- [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework)
- [Fairness and Machine Learning](https://fairmlbook.org/) (Barocas, Hardt, Narayanan)
- Mitchell et al., "Model Cards for Model Reporting" (2019)

## License

MIT

## Author

Built to support responsible AI deployment in research and educational contexts.
