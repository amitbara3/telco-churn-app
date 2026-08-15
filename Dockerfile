FROM python:3.11-slim

# curl is used by start.sh's readiness probe and by HEALTHCHECK below.
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Run as a non-root user with a real, writable home directory. Hugging
# Face Spaces runs Docker containers as UID 1000, and Streamlit needs a
# writable HOME for its config/cache — as root-owned /root, it fails to
# start there even though it works fine locally.
RUN useradd --create-home --uid 1000 appuser
USER appuser
ENV HOME=/home/appuser \
    PATH=/home/appuser/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
WORKDIR $HOME/app

# requirements.lock pins all 68 packages (direct + transitive) with hashes,
# resolved for linux/py3.11 — this image, not a developer's laptop.
# requirements.txt alone pins only the 10 direct dependencies, so a rebuild
# months from now could silently resolve different transitive versions than
# the ones CI verified.
COPY --chown=appuser:appuser requirements.lock .
RUN pip install --no-cache-dir --user --require-hashes -r requirements.lock

COPY --chown=appuser:appuser app ./app
COPY --chown=appuser:appuser model ./model
COPY --chown=appuser:appuser start.sh .

# Hugging Face Spaces (Docker SDK) routes traffic to this port.
EXPOSE 7860
# The FastAPI backend runs alongside it on 8000, reachable inside the
# container / via `docker run -p 8000:8000 ...` if you want to hit it directly.
EXPOSE 8000

# Probes the API rather than the UI: Streamlit serving a page while the
# backend is down is exactly the failure this should catch.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -sf http://localhost:8000/health || exit 1

CMD ["bash", "start.sh"]
