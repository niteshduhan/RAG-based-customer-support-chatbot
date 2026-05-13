#!/usr/bin/env bash
# build.sh — runs once per deploy on Render
# 1. Install Python dependencies
# 2. Build the FAISS vector store from DATA/ documents
# Render sets the working directory to the repo root automatically.

set -e          # exit on any error
set -o pipefail

echo "════════════════════════════════════════════════════"
echo "  RAG Agent — Render Build Script"
echo "════════════════════════════════════════════════════"

# ── 1. Dependencies ───────────────────────────────────────────
echo ""
echo "▶ Installing Python dependencies..."
pip install --upgrade pip --quiet
pip install -r requirements.txt

# ── 2. Vector store ───────────────────────────────────────────
# Build the FAISS index from DATA/ so retriever.py can load it at startup.
# The sentence-transformer model is downloaded here too (cached by HuggingFace).
echo ""
echo "▶ Building FAISS vector store from DATA/..."
python vector_store.py

echo ""
echo "════════════════════════════════════════════════════"
echo "  Build complete ✅"
echo "════════════════════════════════════════════════════"