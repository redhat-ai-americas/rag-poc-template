# RAG POC Template

A template repository for deploying RAG (Retrieval Augmented Generation) applications on OpenShift using Helm and ArgoCD.

## Features

- **Hybrid Retrieval** - Combines vector similarity search (60%) with BM25 lexical search (40%) for better keyword matching
- **YAML Frontmatter Support** - Extract metadata from markdown files for enhanced filtering
- **Configurable Chunking** - Enable/disable document chunking based on your use case
- **Flexible Build Scripts** - Selective image builds and custom commit messages
- **Container Optimization** - `.containerignore` to exclude large data files from images

## Components

- **Streamlit App** - Web UI for the RAG application
- **Chroma DB** - Vector database for document embeddings
- **InferenceService** - KServe-based LLM serving (vLLM)

## Quick Start

### 1. Configure Values

Edit `deploy/helm/values.yaml` with your settings:

```yaml
chroma:
  name: chroma-db
  image: quay.io/YOUR_ORG/chroma-db
  tag: "1"
  claimName: chroma-data

streamlit:
  name: streamlit-app
  image: quay.io/YOUR_ORG/streamlit-app
  tag: "1"
  secretRef: app-env

inferenceService:
  name: llm-model
  runtime: llm-model
  storageKey: model-storage
  storagePath: model
```

### 2. Create Secret (Required)

**The secret must be created before deploying the Helm chart.** The Streamlit app will fail to start without it.

Copy the example environment file and fill in your values:

```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

Create the namespace and secret:

```bash
oc new-project rag-poc
oc create secret generic app-env --from-env-file=.env -n rag-poc
```

See `.env.example` for required environment variables.

### 3. Deploy with Helm

```bash
helm install rag-poc ./deploy/helm -n rag-poc
```

Or deploy with ArgoCD:

```bash
oc apply -f argocd/argo-app.yaml
```

## Building Images

```bash
export REGISTRY=quay.io/your-org
./scripts/build-push.sh
```

### Build Script Options

| Flag | Description |
|------|-------------|
| `--app-only` | Build only the Streamlit app image |
| `--chroma-only` | Build only the Chroma DB image |
| `-m "message"` | Custom git commit message |
| `--no-commit` | Skip git commit and push |

Examples:

```bash
# Build only the app image with a feature commit
./scripts/build-push.sh --app-only -m "feat: add new search feature"

# Build both images without committing
./scripts/build-push.sh --no-commit

# Build only chroma
./scripts/build-push.sh --chroma-only
```

## Configuration

Key environment variables (set in `.env`):

| Variable | Description | Default |
|----------|-------------|---------|
| `ENABLE_CHUNKING` | Enable document chunking | `true` |
| `CHUNK_SIZE` | Characters per chunk | `2000` |
| `CHUNK_OVERLAP` | Overlap between chunks | `200` |
| `CONTEXT_MAX_CHARS` | Max context sent to LLM | `30000` |
| `RETRIEVAL_K` | Number of documents to retrieve | - |
| `WIKI_RETRIEVAL_SIMILARITY_THRESHOLD` | Minimum similarity score | - |

## Document Format

Place markdown files in `assets/kb_markdown/`. Optional YAML frontmatter is extracted as metadata:

```markdown
---
title: Document Title
category: Technical
tags: api,reference
---

# Document Content

Your markdown content here...
```

Frontmatter fields are stored in ChromaDB and can be used for filtering.

## Project Structure

```
├── agents/              # LangGraph agent code
│   ├── agent_nodes.py   # Retrieval and LLM logic (hybrid search)
│   └── workflow.py      # LangGraph workflow
├── app.py               # Streamlit application
├── argocd/              # ArgoCD Application manifests
├── assets/
│   └── kb_markdown/     # Knowledge base documents
├── config.py            # Application configuration
├── containers/          # Container build files
├── data_processors/     # Document processing code
│   └── wiki_processor.py
├── deploy/
│   └── helm/            # Helm chart
│       ├── Chart.yaml
│       ├── values.yaml
│       └── templates/
├── scripts/             # Build and utility scripts
├── .containerignore     # Excludes large files from container builds
└── requirements.txt     # Python dependencies (pinned versions)
```

## Requirements

- OpenShift cluster with:
  - OpenShift GitOps (ArgoCD)
  - OpenShift AI / KServe
  - GPU nodes (for InferenceService)
- Container registry access (Quay.io, etc.)

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Streamlit     │────▶│   LangGraph     │────▶│  InferenceService│
│   Web UI        │     │   Workflow      │     │  (vLLM/LLM)     │
└─────────────────┘     └────────┬────────┘     └─────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    ▼                         ▼
           ┌─────────────────┐       ┌─────────────────┐
           │  Vector Search  │       │  BM25 Search    │
           │  (ChromaDB)     │       │  (Lexical)      │
           └─────────────────┘       └─────────────────┘
                    │                         │
                    └──────────┬──────────────┘
                               ▼
                    ┌─────────────────┐
                    │ Ensemble (60/40)│
                    └─────────────────┘
```
