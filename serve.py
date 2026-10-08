"""Production-grade FastAPI service for TinyLM Function-Calling Model.

Features
--------
- Non-blocking async lifespan model loading.
- Input validation and schema conformance via Pydantic v2.
- CORS middleware enabled for cross-origin web app integration.
- Detailed health check endpoint `/health` returning model status, device, and specs.
- Interactive API documentation at `/docs` (Swagger UI) and `/redoc`.
- Clean static web interface served at `/`.

Quickstart:
  python serve.py
  # or:
  uvicorn serve:app --host 0.0.0.0 --port 8000
"""

import sys
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config import CKPT, ROOT, D_MODEL, N_LAYERS, N_HEADS, MAX_LEN
from infer import load, run

STATIC_DIR = ROOT / "static"
_model, _tok = None, None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model once at startup into memory and release cleanly on shutdown."""
    global _model, _tok
    try:
        _model, _tok = load()
        print(f"[Lifespan] TinyLM model loaded successfully into memory.")
    except Exception as e:
        print(f"[Lifespan Error] Could not load checkpoint: {e}", file=sys.stderr)
        _model, _tok = None, None
    yield
    _model, _tok = None, None
    print("[Lifespan] Model unloaded on shutdown.")


app = FastAPI(
    title="Tiny Function-Calling LM API",
    description=(
        "Production-ready routing API powered by a lightweight ~0.47M parameter transformer. "
        "Routes user input to structured web_search queries or abstains on chit-chat."
    ),
    version="1.1.0",
    lifespan=lifespan,
)

# Enable CORS for cross-origin browser dashboards or microfrontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ---------------------------------------------------------------------------
# Request & Response Schemas
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="The natural language user input string to classify and route.",
        examples=["weather in chennai today", "who is elon musk", "tell me a joke"]
    )


class PredictionResponse(BaseModel):
    call: str = Field(..., description="Raw generated tool call or 'none'.")
    tool: str | None = Field(None, description="Extracted tool name, e.g. 'web_search'.")
    args: dict[str, Any] | None = Field(None, description="Parsed arguments matching schema.")
    result: list[str] | str | None = Field(None, description="Dispatched execution results or status message.")
    latency_ms: float = Field(..., description="Inference & dispatch latency in milliseconds.")
    status: str = Field("success", description="Execution status: 'success' or 'error'.")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_details: dict[str, Any]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
def index():
    """Serve the interactive web demo interface."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "TinyLM API is running. Visit /docs for the Swagger specification."}


@app.get("/health", response_model=HealthResponse)
def health():
    """System health check and loaded architecture metadata."""
    is_loaded = (_model is not None and _tok is not None)
    return HealthResponse(
        status="ok" if is_loaded else "unhealthy",
        model_loaded=is_loaded,
        model_details={
            "architecture": "TinyLM-decoder-only",
            "d_model": D_MODEL,
            "n_layers": N_LAYERS,
            "n_heads": N_HEADS,
            "max_len": MAX_LEN,
            "parameter_count": _model.n_params() if is_loaded else None,
            "vocab_size": _tok.get_vocab_size() if is_loaded else None,
        }
    )


@app.post("/predict", response_model=PredictionResponse)
def predict_endpoint(req: QueryRequest):
    """Predict tool routing and execute search tool if invoked."""
    if _model is None or _tok is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Train the model or verify checkpoint files exist."
        )

    res_dict = run(_model, _tok, req.text)

    # Determine status flag
    is_err = isinstance(res_dict.get("result"), str) and res_dict["result"].startswith("Invalid tool call")

    return PredictionResponse(
        call=res_dict["raw"],
        tool=res_dict["tool"],
        args=res_dict["args"],
        result=res_dict["result"],
        latency_ms=res_dict["latency_ms"],
        status="error" if is_err else "success"
    )


def main():
    if not CKPT.exists():
        sys.exit(
            f"Missing model checkpoint: {CKPT}\n"
            "Please train the model before serving:\n"
            "  python gen_data.py\n"
            "  python train_tok.py\n"
            "  python train.py\n"
        )
    port = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8000
    print(f"Starting TinyLM Server on http://localhost:{port} (Swagger docs at /docs)")
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
