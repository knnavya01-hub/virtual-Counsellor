# 🕉️ Ancient Wisdom Guidance App - Backend

A spiritual guidance application that delivers personalized wisdom from Sanskrit scriptures using a multi-LLM translation pipeline, Retrieval-Augmented Generation (RAG), and crisis detection & intervention.

## ✨ Features

- **Multi-LLM Translation Pipeline**: 6-stage Sanskrit → English translation (Gemini → Perplexity → Claude → Llama → Perplexity → Claude) designed to improve faithfulness and reduce hallucination.
- **RAG System**: Hybrid semantic + keyword search backed by a Weaviate vector database for accurate provenance and context retrieval.
- **Crisis Detection**: Multi-layer detection for suicidal ideation and other high-risk signals with immediate intervention procedures and safe fallback responses.
- **Personalized Wisdom**: User journey-stage-aware responses tailored to the user's current spiritual growth and prior interactions.
- **Memory System**: Short-term caching (Redis) and long-term persistence (PostgreSQL) with auto-consolidation of important interactions into long-term memory.
- **Feedback Loop**: User feedback ingestion and automated fine-tuning signals to adapt responses and preferences over time.
- **Visual Generation**: Support for 6-scene cosmic story creation (integration-ready for image-generation backends).
- **Evaluation Metrics**: Comprehensive quality scoring (BLEU, ROUGE), faithfulness checks, hallucination detection, and human-review hooks.

## 🏗️ Architecture

Backend (FastAPI + Python)

Core components:
- API: FastAPI endpoints for conversation, session management, feedback, and admin.
- Translation Pipeline: Orchestration layer that forwards source Sanskrit text through multiple LLM translators and aggregates/filters results.
- RAG Layer: Weaviate vector index + keyword fallback; semantic retriever + context scoring.
- Crisis Engine: Fast, deterministic classifiers + LLM-based verifier and intervention dispatcher.
- Memory: Redis for short-term session state; PostgreSQL for long-term memory and user profiles.
- Auth & Security: Token-based auth, role-based access for admin tools, encrypted secrets for LLM/DB keys.
- Observability: Logging, metrics, and tracing for quality control and incident response.

## ⚙️ Deployment & Setup (high-level)

1. Provision infrastructure:
   - PostgreSQL for long-term storage
   - Redis for session cache
   - Weaviate (or managed alternative) for vector store
   - Object storage for any media assets
2. Configure environment:
   - Set LLM API keys (Gemini, Perplexity, Claude, Llama endpoints)
   - DB connection strings, Redis URL, Weaviate endpoint
   - Secrets management (vault or environment variables)
3. Run migrations and start services:
   - Run DB migrations
   - Boot FastAPI with Uvicorn/Gunicorn behind a reverse proxy
4. Optional: set up a CI/CD pipeline, monitoring, and auto-scaling.

## 📚 Data, Safety & Ethics

- All scripture translations and generated guidance are filtered for safety and flagged where content could be sensitive or actionable.
- Crisis detection has clearly defined escalation paths. Human-in-the-loop review is recommended for flagged cases.
- Users should be informed about limitations, and a prominent disclaimer must be displayed regarding clinical or medical advice.

## 🧭 Recommended Next Steps

- Add example env file and a setup script for local development.
- Provide a Postman collection or OpenAPI spec for quick testing.
- Add unit and integration tests for the translation pipeline and crisis detection flow.
- Harden privacy & consent flows, and add telemetry dashboards.

## Contributing

Contributions, issue reports, and feature requests are welcome — please follow repository guidelines and include reproducible steps or minimal repros for bugs.