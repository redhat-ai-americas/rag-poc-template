# RAG POC Template

A template repository for deploying RAG (Retrieval Augmented Generation) applications on OpenShift using Helm and ArgoCD.

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

## Project Structure

```
├── agents/              # LangGraph agent code
├── app.py               # Streamlit application
├── argocd/              # ArgoCD Application manifests
├── config.py            # Application configuration
├── containers/          # Container build files
├── data_processors/     # Document processing code
├── deploy/
│   └── helm/            # Helm chart
│       ├── Chart.yaml
│       ├── values.yaml
│       └── templates/
└── scripts/             # Build and utility scripts
```

## Requirements

- OpenShift cluster with:
  - OpenShift GitOps (ArgoCD)
  - OpenShift AI / KServe
  - GPU nodes (for InferenceService)
- Container registry access (Quay.io, etc.)
