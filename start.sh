#!/usr/bin/env bash
# Starts the FastAPI backend in the background on port 8000, then runs the
# Streamlit UI in the foreground on port 7860 (the port Hugging Face Spaces
# Docker SDK expects). Streamlit calls the API over localhost.
set -euo pipefail

API_PORT="${API_PORT:-8000}"
UI_PORT="${PORT:-7860}"          # Render injects $PORT; HF Spaces uses 7860.
API_STARTUP_TIMEOUT="${API_STARTUP_TIMEOUT:-60}"

uvicorn app.main:app --host 0.0.0.0 --port "$API_PORT" &
API_PID=$!

cleanup() {
  kill "$API_PID" 2>/dev/null || true
}
trap cleanup EXIT

# Wait for the API to become healthy, and abort if it never does. Starting
# the UI regardless would produce a container that looks up but errors on
# every prediction — a silent-degradation failure that's harder to diagnose
# than an outright crash.
for _ in $(seq 1 "$API_STARTUP_TIMEOUT"); do
  if curl -sf "http://localhost:${API_PORT}/health" > /dev/null; then
    echo "API healthy on port ${API_PORT}."
    break
  fi
  if ! kill -0 "$API_PID" 2>/dev/null; then
    echo "FATAL: the API process exited during startup." >&2
    wait "$API_PID" || true
    exit 1
  fi
  sleep 1
done

if ! curl -sf "http://localhost:${API_PORT}/health" > /dev/null; then
  echo "FATAL: API did not become healthy within ${API_STARTUP_TIMEOUT}s." >&2
  exit 1
fi

exec streamlit run app/streamlit_app.py \
  --server.address 0.0.0.0 \
  --server.port "$UI_PORT" \
  --server.headless true
