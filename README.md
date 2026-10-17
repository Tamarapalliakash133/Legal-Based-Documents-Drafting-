# Chambers — Law Research & Drafting

Chambers is an AI-powered legal research and drafting platform built on a **hybrid RAG architecture** that combines BM25 keyword search and dense vector retrieval to deliver relevant legal information with grounded answers.

### Architecture Overview

- **Frontend:** React-based UI with workspaces for chat history and legal drafting.
- **Backend:** Python API serving the frontend and handling RAG requests.
- **Retrieval Pipeline:** Parallel BM25 and dense retrieval, HyDE query expansion, Reciprocal Rank Fusion (RRF), and cross-encoder reranking.
- **Generation:** LLM-based legal research responses and document drafting using retrieved context.
- **Guardrails:** Input and output validation to enforce legal-domain restrictions and reduce unsafe or irrelevant responses.
- **Caching:** Multi-layer caching for answers, embeddings, HyDE outputs, retrieval results, and LLM responses to reduce latency.
- **Storage:** Optional Redis for caching and MongoDB for logs, with in-memory fallbacks.
- **Observability:** LangSmith tracing for retrieval, generation, guardrails, and drafting pipelines.

### Architecture Flow

User → React Frontend → Python API → Input Guardrails → Hybrid Retrieval (BM25 + Dense + HyDE) → RRF Fusion → Cross-Encoder Reranking → LLM Generation → Output Guardrails → Response

**Drafting flow:** User Request → Reference Retrieval → LLM Draft Generation → Final Draft

**Key Features:** Streaming responses, parallel retrieval, multi-layer caching, workspace-based chat history, legal document drafting, and end-to-end tracing with LangSmith.