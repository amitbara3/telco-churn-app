#!/usr/bin/env bash
# Starts the FastAPI backend in the background on port 8000, then runs the
# Streamlit UI in the foreground on port 7860 (the port Hugging Face Spaces
# Docker SDK expects). Streamlit calls the API over localhost.
set -e

uvicorn app.main:app --host 0.0.0.0 --port 8000 &
API_PID=$!

cleanup() {
  kill "$API_PID" 2>/dev/null || true
}
trap cleanup EXIT

# Wait for the API to be ready before starting the UI.
for _ in $(seq 1 30); do
  if curl -sf http://localhost:8000/health > /dev/null; then
    break
  fi
  sleep 1
done

exec streamlit run app/streamlit_app.py \
  --server.address 0.0.0.0 \
  --server.port 7860 \
  --server.headless true
