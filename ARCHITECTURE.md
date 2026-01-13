# Architecture Overview

## Design Philosophy: Local-First, Cloud-Optional

This project follows best practices for privacy-conscious, developer-friendly LLM applications.

### Backend Architecture

**Default: Ollama (Local)**
- No API keys required
- Runs entirely on your machine
- Zero external dependencies
- Full data privacy

**Optional: OpenAI-Compatible APIs**
- Requires API key configuration
- Supports OpenAI and compatible endpoints
- Opt-in via environment variables

### Configuration

The system uses environment variables for backend selection:

```bash
# Default configuration (no setup needed)
LLM_BACKEND=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama2

# Optional OpenAI configuration
LLM_BACKEND=openai
OPENAI_API_KEY=your-key-here
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
```

### Key Files

- **[.env.example](.env.example)**: Template showing Ollama as default with OpenAI commented out
- **[app/audit.py](app/audit.py)**: Backend abstraction layer supporting both Ollama and OpenAI
- **[app/main.py](app/main.py)**: FastAPI endpoints that read backend config from environment
- **[requirements.txt](requirements.txt)**: Both backends listed (ollama first, openai marked optional)

### API Design

Endpoints automatically use the configured backend:

```bash
# Works immediately with Ollama (no keys needed)
POST /audit
{
  "scenario": "hiring"
}

# Backend determined by LLM_BACKEND env var
# Model uses backend default unless overridden
```

### Benefits

1. **Privacy**: Local execution by default
2. **Simplicity**: Works out of the box without configuration
3. **Flexibility**: Easy cloud integration when needed
4. **Transparency**: Clear documentation about what requires external services
5. **Cost**: Free local testing before paid API usage
