#!/bin/bash
# NEST Automated Pipeline Starter Script

set -e

echo "🚀 NEST Automated Pipeline Starter"
echo "===================================="
echo ""

# Check if gh CLI is installed
if ! command -v gh &> /dev/null; then
    echo "❌ Error: GitHub CLI (gh) is not installed"
    echo "Please install it from: https://cli.github.com/"
    exit 1
fi

# Check if user is authenticated
if ! gh auth status &> /dev/null; then
    echo "❌ Error: Not authenticated with GitHub CLI"
    echo "Please run: gh auth login"
    exit 1
fi

echo "✅ GitHub CLI is ready"
echo ""

# Display phase options
echo "Available Phases:"
echo "  1 - Literature Review & Foundation"
echo "  2 - Data Acquisition & Preprocessing"
echo "  3 - Model Architecture Development"
echo "  4 - Advanced Model Features & Robustness"
echo "  5 - Evaluation & Optimization"
echo "  6 - Documentation & Dissemination"
echo ""

# Get phase number
if [ -z "$1" ]; then
    echo "Usage: ./start-pipeline.sh <phase-number>"
    echo "Example: ./start-pipeline.sh 1"
    exit 1
fi

PHASE=$1

# Validate phase number
if [[ ! "$PHASE" =~ ^[1-6]$ ]]; then
    echo "❌ Error: Invalid phase number. Must be 1-6"
    exit 1
fi

# Confirm with user
echo "📋 You are about to start Phase $PHASE"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

echo ""
echo "🎯 Starting Phase $PHASE..."
echo ""

# Trigger the workflow
gh workflow run phase-execution.yml -f phase=$PHASE

echo "✅ Workflow triggered successfully!"
echo ""
echo "📊 Monitor progress:"
echo "  - View workflow: gh run list --workflow=phase-execution.yml"
echo "  - View PRs: gh pr list --label automated"
echo "  - Web UI: gh repo view --web"
echo ""
echo "The pipeline will:"
echo "  1. Create a branch for Phase $PHASE"
echo "  2. Create a PR with phase instructions"
echo "  3. Let Copilot Workspace implement the phase"
echo "  4. Auto-review and auto-merge"
echo "  5. Trigger the next phase automatically"
echo ""
echo "🎉 Happy coding!"
