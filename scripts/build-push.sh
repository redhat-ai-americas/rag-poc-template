#!/usr/bin/env bash
set -euo pipefail

# Configure these for your environment
REGISTRY=${REGISTRY:-"quay.io/CHANGE_ME"}
CHROMA_IMAGE=${CHROMA_IMAGE:-"chroma-db"}
STREAMLIT_IMAGE=${STREAMLIT_IMAGE:-"streamlit-app"}

SHA=$(git rev-parse --short HEAD)

# Optional: login to Quay if credentials are provided
if [[ -n "${QUAY_USERNAME:-}" && -n "${QUAY_PASSWORD:-}" ]]; then
  buildah login -u "$QUAY_USERNAME" -p "$QUAY_PASSWORD" quay.io
fi

# Build images with Buildah
buildah bud -f containers/chroma/Containerfile.chroma -t $REGISTRY/$CHROMA_IMAGE:$SHA .
buildah bud -f containers/app/Containerfile.streamlit -t $REGISTRY/$STREAMLIT_IMAGE:$SHA .

# Push images
buildah push $REGISTRY/$CHROMA_IMAGE:$SHA
buildah push $REGISTRY/$STREAMLIT_IMAGE:$SHA

# Update Helm values with new image tags
sed -i "s|^\(\s*tag:\s*\).*|\\1\"$SHA\"|" deploy/helm/values.yaml

# Commit and push so Argo CD sees the change
if ! git diff --quiet -- deploy/helm/values.yaml; then
  git add deploy/helm/values.yaml
  git commit -m "Update image tags to $SHA"
  git push
fi
