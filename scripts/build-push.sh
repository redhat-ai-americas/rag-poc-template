#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# RAG POC - Build and Push Script
# =============================================================================
#
# Usage:
#   ./scripts/build-push.sh                        # Build both images
#   ./scripts/build-push.sh --app-only             # Build only streamlit app
#   ./scripts/build-push.sh --chroma-only          # Build only chroma
#   ./scripts/build-push.sh -m "feat: new feature" # Custom commit message
#   ./scripts/build-push.sh --no-commit            # Skip commit/push
#

# Configure these for your environment
REGISTRY=${REGISTRY:-"quay.io/CHANGE_ME"}
CHROMA_IMAGE=${CHROMA_IMAGE:-"chroma-db"}
STREAMLIT_IMAGE=${STREAMLIT_IMAGE:-"streamlit-app"}

# Parse arguments
COMMIT_MSG=""
NO_COMMIT=false
BUILD_CHROMA=true
BUILD_APP=true

while [[ $# -gt 0 ]]; do
  case $1 in
    -m|--message)
      COMMIT_MSG="$2"
      shift 2
      ;;
    --no-commit)
      NO_COMMIT=true
      shift
      ;;
    --app-only)
      BUILD_CHROMA=false
      shift
      ;;
    --chroma-only)
      BUILD_APP=false
      shift
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [-m \"commit message\"] [--no-commit] [--app-only] [--chroma-only]"
      exit 1
      ;;
  esac
done

# Use git SHA or "latest" if not in a git repo
SHA=$(git rev-parse --short HEAD 2>/dev/null || echo "latest")

# Default commit message if not provided
if [[ -z "$COMMIT_MSG" ]]; then
  COMMIT_MSG="chore: update image tags to $SHA"
fi

echo "=============================================="
echo "Building RAG POC Images"
echo "=============================================="
echo "Registry: $REGISTRY"
echo "Tag: $SHA"
echo ""

# Optional: login to Quay if credentials are provided
if [[ -n "${QUAY_USERNAME:-}" && -n "${QUAY_PASSWORD:-}" ]]; then
  echo "Logging into Quay.io..."
  podman login -u "$QUAY_USERNAME" -p "$QUAY_PASSWORD" quay.io
fi

# Build Chroma image
if [[ "$BUILD_CHROMA" == "true" ]]; then
  echo ""
  echo "Building Chroma DB image..."
  podman build -f containers/chroma/Containerfile.chroma -t $REGISTRY/$CHROMA_IMAGE:$SHA .
else
  echo ""
  echo "Skipping Chroma build (--app-only)"
fi

# Build Streamlit app image
if [[ "$BUILD_APP" == "true" ]]; then
  echo ""
  echo "Building Streamlit app image..."
  podman build -f containers/app/Containerfile.streamlit -t $REGISTRY/$STREAMLIT_IMAGE:$SHA .
else
  echo ""
  echo "Skipping Streamlit build (--chroma-only)"
fi

# Push images
echo ""
echo "Pushing images to registry..."
if [[ "$BUILD_CHROMA" == "true" ]]; then
  podman push $REGISTRY/$CHROMA_IMAGE:$SHA
fi
if [[ "$BUILD_APP" == "true" ]]; then
  podman push $REGISTRY/$STREAMLIT_IMAGE:$SHA
fi

# Update Helm values with new image tags (only for images that were built)
echo ""
if [[ "$BUILD_CHROMA" == "true" && "$BUILD_APP" == "true" ]]; then
  echo "Updating Helm values.yaml - both images to tag: $SHA"
  sed -i "s|^\(\s*tag:\s*\).*|\\1\"$SHA\"|" deploy/helm/values.yaml
elif [[ "$BUILD_CHROMA" == "true" ]]; then
  echo "Updating Helm values.yaml - chroma only to tag: $SHA"
  # Update only chroma tag (line after chroma.image)
  sed -i "/^chroma:/,/^[a-z]/ s|^\(\s*tag:\s*\).*|\\1\"$SHA\"|" deploy/helm/values.yaml
elif [[ "$BUILD_APP" == "true" ]]; then
  echo "Updating Helm values.yaml - streamlit only to tag: $SHA"
  # Update only streamlit tag (line after streamlit.image)
  sed -i "/^streamlit:/,/^[a-z]/ s|^\(\s*tag:\s*\).*|\\1\"$SHA\"|" deploy/helm/values.yaml
fi

# Commit and push so Argo CD sees the change
if [[ "$NO_COMMIT" == "true" ]]; then
  echo ""
  echo "Skipping commit/push (--no-commit flag)"
elif ! git diff --quiet || ! git diff --cached --quiet; then
  echo ""
  echo "Committing all changes..."
  git add -A
  git commit -m "$COMMIT_MSG"
  git push
  echo "Pushed to git - ArgoCD will sync automatically"
else
  echo "No changes to commit"
fi

echo ""
echo "=============================================="
echo "Build complete!"
echo "=============================================="
echo "Images pushed:"
if [[ "$BUILD_CHROMA" == "true" ]]; then
  echo "  - $REGISTRY/$CHROMA_IMAGE:$SHA"
fi
if [[ "$BUILD_APP" == "true" ]]; then
  echo "  - $REGISTRY/$STREAMLIT_IMAGE:$SHA"
fi
echo ""
