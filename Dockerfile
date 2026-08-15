FROM python:3.11-slim

WORKDIR /code

# curl is used by start.sh to wait for the API to come up.
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY model ./model
COPY start.sh .
RUN chmod +x start.sh

# Hugging Face Spaces (Docker SDK) routes traffic to this port.
EXPOSE 7860
# The FastAPI backend runs alongside it on 8000, reachable inside the
# container / via `docker run -p 8000:8000 ...` if you want to hit it directly.
EXPOSE 8000

CMD ["./start.sh"]
