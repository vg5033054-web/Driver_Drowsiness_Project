#!/usr/bin/env bash
# Quick launcher for Driver Drowsiness Detection System
cd "$(dirname "$0")"

if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

echo "🚀 Starting Driver Drowsiness Detection System on local server..."
streamlit run app.py
