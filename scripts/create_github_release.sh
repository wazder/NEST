#!/bin/bash
# Create a GitHub release for NEST with model weights and evaluation results.
#
# Usage:
#   bash scripts/create_github_release.sh [MODEL_PATH]
#
# Requirements:
#   - gh CLI installed and authenticated (gh auth login)
#   - RELEASE_NOTES.md present in repo root
#   - Git tag must not already exist

set -euo pipefail

VERSION="v1.0.0"
MODEL_PATH="${1:-results/best_model.pt}"
EVAL_PATH="${2:-results/evaluation_results.json}"
NOTES_FILE="RELEASE_NOTES.md"

# Verify prerequisites
if ! command -v gh &>/dev/null; then
    echo "ERROR: gh CLI not found. Install from https://cli.github.com/"
    exit 1
fi

if ! gh auth status &>/dev/null; then
    echo "ERROR: Not authenticated with gh. Run: gh auth login"
    exit 1
fi

if [ ! -f "$NOTES_FILE" ]; then
    echo "ERROR: $NOTES_FILE not found."
    exit 1
fi

if [ ! -f "$MODEL_PATH" ]; then
    echo "WARNING: Model file not found at $MODEL_PATH"
    echo "Continuing without model asset..."
    MODEL_PATH=""
fi

if [ ! -f "$EVAL_PATH" ]; then
    echo "WARNING: Evaluation results not found at $EVAL_PATH"
    EVAL_PATH=""
fi

# Check if tag already exists
if git tag -l "$VERSION" | grep -q "$VERSION"; then
    echo "ERROR: Tag $VERSION already exists."
    echo "Delete it first with: git tag -d $VERSION && git push origin :refs/tags/$VERSION"
    exit 1
fi

echo "Creating git tag $VERSION ..."
git tag -a "$VERSION" -m "NEST $VERSION: State-of-the-art EEG-to-text decoding"

echo "Pushing tag to remote ..."
git push origin "$VERSION"

# Build asset arguments
ASSETS=()
if [ -n "$MODEL_PATH" ]; then
    ASSETS+=("${MODEL_PATH}#NEST-model-weights.pt")
fi
if [ -n "$EVAL_PATH" ]; then
    ASSETS+=("${EVAL_PATH}#evaluation-results.json")
fi

echo "Creating GitHub release $VERSION ..."
gh release create "$VERSION" \
    --title "NEST $VERSION - EEG-to-Text Decoding" \
    --notes-file "$NOTES_FILE" \
    --latest \
    "${ASSETS[@]}"

echo ""
echo "Release created: $(gh release view "$VERSION" --json url -q .url)"
