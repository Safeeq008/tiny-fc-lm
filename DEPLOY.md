# Production Deployment Runbook: Tiny Function-Calling LM

This guide covers deployment patterns, operational monitoring, resource sizing, and security hardening for running TinyLM-FC in production environments.

---

## 1. Architecture Overview & Resource Footprint

TinyLM-FC is designed as an ultra-low-footprint gateway router. It sits between incoming client traffic and your downstream agents or tool APIs.

```text
                  ┌────────────────────────┐
                  │ Client / API Gateway   │
                  └───────────┬────────────┘
                              │ HTTP POST /predict
                              ▼
                  ┌────────────────────────┐
                  │ TinyLM Router Service  │  Memory: ~35 MB RSS
                  │ (FastAPI + Uvicorn)    │  CPU: < 5% of 1 vCPU
                  └─────┬────────────┬─────┘  P99 Latency: ~4.5 ms
                        │            │
            Tool Needed │            │ Abstain ('none')
                        ▼            ▼
      ┌─────────────────────┐    ┌─────────────────────┐
      │ DuckDuckGo / Search │    │ Conversational LLM  │
      │ Execution Engine    │    │ (GPT-4 / Claude)    │
      └─────────────────────┘    └─────────────────────┘
```

### Resource Sizing Guidelines

| Metric | Measured Footprint | Recommended Container Allocation |
| :--- | :--- | :--- |
| **RAM (Process RSS)** | ~35 MB | 128 MB |
| **Model Weights (RAM)**| ~1.9 MB | Included in process RSS |
| **CPU Usage (Idle)** | 0% | 0.1 vCPU |
| **CPU Usage (100 RPS)**| ~40% of 1 core | 0.5–1.0 vCPU |
| **P50 Latency (Routing)**| 2.8 ms | Target SLA: < 10 ms |
| **P99 Latency (Routing)**| 4.6 ms | Target SLA: < 20 ms |

---

## 2. Local & Server Deployment

### Prerequisites
Verify that the model weights and tokenizer have been trained:
```bash
# Verify artifacts exist:
python -c "from config import CKPT, TOK_PATH; assert CKPT.exists() and TOK_PATH.exists(), 'Model not trained!'"
```
If missing, execute the pipeline:
```bash
python gen_data.py && python train_tok.py && python train.py
```

### Starting the Server
```bash
# Option A: Direct script invocation
python serve.py 8000

# Option B: Multi-worker Uvicorn invocation (for multi-core concurrency)
python -m uvicorn serve:app --host 0.0.0.0 --port 8000 --workers 4
```

---

## 3. Docker Containerization

### Build Image
```bash
docker build -t tiny-fc-lm:latest .
```

### Run Container
```bash
docker run -d \
  --name tiny-router \
  --restart unless-stopped \
  -p 8000:8000 \
  --memory="256m" \
  --cpus="1.0" \
  tiny-fc-lm:latest
```

### Verify Container Health
```bash
docker inspect --format='{{json .State.Health}}' tiny-router
```

---

## 4. API Specification & Integration Examples

### Endpoint: `POST /predict`

#### Request Payload
```json
{
  "text": "weather in chennai today"
}
```

#### Response (Tool Call Generated)
```json
{
  "call": "web_search|query=chennai weather today|recency=day",
  "tool": "web_search",
  "args": {
    "query": "chennai weather today",
    "recency": "day"
  },
  "result": [
    "Chennai Weather - Regional Meteorological Centre - https://...",
    "AccuWeather Chennai 10-Day Weather - https://..."
  ],
  "latency_ms": 3.84,
  "status": "success"
}
```

#### Response (Abstention on Chit-Chat)
```json
{
  "call": "none",
  "tool": null,
  "args": null,
  "result": "(no tool needed)",
  "latency_ms": 2.65,
  "status": "success"
}
```

### Endpoint: `GET /health`
Returns process readiness and loaded architecture specs:
```json
{
  "status": "ok",
  "model_loaded": true,
  "model_details": {
    "architecture": "TinyLM-decoder-only",
    "d_model": 80,
    "n_layers": 4,
    "n_heads": 4,
    "max_len": 80,
    "parameter_count": 471760,
    "vocab_size": 2048
  }
}
```

---

## 5. Security & Reverse Proxy Configuration

When exposing TinyLM to public networks, run it behind an edge proxy (such as Nginx, Traefik, or Cloudflare) to provide TLS termination, rate limiting, and API key verification.

### Example Nginx Configuration (`/etc/nginx/conf.d/tiny-router.conf`)
```nginx
limit_req_zone $binary_remote_addr zone=router_limit:10m rate=50r/s;

server {
    listen 443 ssl http2;
    server_name router.yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/router.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/router.yourdomain.com/privkey.pem;

    location / {
        limit_req zone=router_limit burst=20 nodelay;

        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_connect_timeout 2s;
        proxy_read_timeout 5s;
    }
}
```

---

## 6. Operational Monitoring & Health Checks

### Liveness Probe
- Path: `GET /health`
- Interval: 10s
- Timeout: 2s
- Failure threshold: 3

### Monitoring Metrics to Track
1. **P99 Inference Latency**: Normal range 3ms–8ms. Spikes above 50ms indicate CPU throttling or thread contention.
2. **Search Dispatch Latency**: Network queries to DuckDuckGo typically take 150ms–600ms. If DuckDuckGo rate-limits the IP, the service falls back cleanly to placeholder stubs without crashing.
3. **Abstention Ratio**: Percentage of incoming requests classified as `none`. Useful for detecting domain shift if user distributions change.

---

## 7. Production Checklist & Trade-Offs

- [ ] Model weights pre-trained and mounted into container image or volume.
- [ ] Non-root container execution enabled (configured in `Dockerfile`).
- [ ] Reverse proxy with TLS and rate-limiting configured in front of Uvicorn.
- [ ] Healthcheck probe configured in orchestration layer (Kubernetes / Docker Compose).
- [ ] Client applications configured with fallback logic if live search backend fails.

### When to Use TinyLM vs Alternatives

| Requirement | Recommended Choice |
| :--- | :--- |
| Single-tool intent routing with <5ms latency on CPU | **TinyLM-FC (This Model)** |
| Static keyword match or exact regex pattern | **Regular Expression / Deterministic Trie** |
| Multi-tool routing across 50+ diverse APIs | **Small Language Model (Phi-3 / Qwen-2.5-7B) or Flagship LLM** |
| Generating full natural-language synthesis answers | **Downstream Conversational LLM** |
