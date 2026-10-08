"""Constrained decoding, parsing, and tool execution for TinyLM.

Inference Pipeline & Technical Rationale
---------------------------------------
A common problem with generative function calling in tiny models (<1M parameters)
is format drift: the model may hallucinate invalid punctuation, omit required
parameters, or produce invalid tool names. We solve this with a hybrid
**constrained decoding + semantic parsing** pipeline:

1. **First-Token Tool Routing & Margin Check**:
   The model checks the logits at the first output position.
   If logit("none") > logit("web_search") + NONE_MARGIN:
       return "none" (abstain from tool execution)
   Otherwise:
       Force prefix: ``web_search|query=``

2. **Greedy Generation with Context Window Sliding**:
   Tokens are generated greedily until the special ``<end>`` token or ``max_new``
   limit is reached. A sliding context window (last MAX_LEN tokens) guarantees
   RoPE positional encoding stability.

3. **Schema Validation & Fallback Guardrails**:
   ``parse_call()`` validates the output against the defined tool schema in ``config.py``.
   If the extracted query collapsed to a degenerate stopword (e.g. bare "day",
   "news", ""), we fall back to extracting the user's raw prompt rather than failing.

4. **Live Tool Execution**:
   Valid tool calls dispatch to ``web_search()`` via `ddgs` (DuckDuckGo search)
   with timeout and graceful stub fallback if offline.

Limitations (Honesty)
---------------------
- **Single Tool Only**: Only routes to `web_search` or `none`.
- **Greedy Decoding**: Does not perform beam search or sampling; deterministic.
- **Entity Transduction**: If an input contains named entities absent from the
  training set, the model may synthesize an imperfectly matched query.
"""

import argparse
import sys
import time
from typing import Any
import torch
from tokenizers import Tokenizer

from config import (
    CKPT, END, MAX_LEN, NONE_MARGIN, TOK_PATH, TOOLS, USER, CALL
)
from model import TinyLM


def load(ckpt_path=CKPT, tok_path=TOK_PATH) -> tuple[TinyLM, Tokenizer]:
    """Load model weights and tokenizer from disk.

    Raises:
        FileNotFoundError: If checkpoint or tokenizer file is missing.
    """
    if not tok_path.exists():
        raise FileNotFoundError(
            f"Tokenizer file not found at {tok_path}. "
            f"Run `python train_tok.py` before inferring."
        )
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint file not found at {ckpt_path}. "
            f"Run `python train.py` before inferring."
        )

    tok = Tokenizer.from_file(str(tok_path))
    ck = torch.load(ckpt_path, map_location="cpu")
    model = TinyLM(**ck["cfg"])
    model.load_state_dict(ck["model"])
    model.eval()
    return model, tok


@torch.no_grad()
def predict(model: TinyLM, tok: Tokenizer, text: str, max_new: int = 40) -> str:
    """Generate the tool call string using constrained decoding.

    Args:
        model: Loaded TinyLM instance in eval mode.
        tok: Trained Tokenizer instance.
        text: Raw natural language input from the user.
        max_new: Maximum new tokens to generate after the forced prefix.

    Returns:
        Structured call string, e.g. "web_search|query=tokyo weather today|recency=day"
        or "none".
    """
    prompt = [USER] + tok.encode(text).ids + [CALL]
    prompt = prompt[-MAX_LEN:]

    tool_ids = tok.encode("web_search|query=").ids
    none_first = tok.encode("none").ids[0]

    # Evaluate logits at prompt boundary
    logits = model(torch.tensor([prompt]))[0][0, -1]

    # Decision rule: Abstain if 'none' logit beats 'web_search' by NONE_MARGIN
    if logits[none_first] > logits[tool_ids[0]] + NONE_MARGIN:
        return "none"

    # Constrained generation: force valid tool name & parameter prefix
    ids = prompt + tool_ids
    out = []
    for _ in range(max_new):
        nxt = model(torch.tensor([ids[-MAX_LEN:]]))[0][0, -1].argmax().item()
        if nxt == END:
            break
        ids.append(nxt)
        out.append(nxt)

    return "web_search|query=" + tok.decode(out)


DEGENERATE_QUERIES = {"day", "week", "any", "none", "news", "", "today"}


def parse_call(call: str, raw_user_text: str | None = None) -> dict[str, Any] | None:
    """Parse raw call string into structured tool invocation dictionary.

    Args:
        call: Raw string from predict(), e.g. "web_search|query=foo|recency=day"
        raw_user_text: Optional user query string used as a fallback if the generated
                       query is degenerate.

    Returns:
        None if call == "none", or a dictionary {"name": str, "args": dict}.

    Raises:
        ValueError: If tool is unknown or query cannot be resolved.
    """
    call = call.strip()
    if call == "none":
        return None

    parts = call.split("|")
    tool_name = parts[0]
    if tool_name not in TOOLS:
        raise ValueError(f"Unknown tool: '{tool_name}'. Allowed: {list(TOOLS.keys())}")

    args = {}
    for p in parts[1:]:
        if "=" in p:
            k, v = p.split("=", 1)
            args[k.strip()] = v.strip()

    query = args.get("query", "").strip()

    # Guard against generation collapse on degenerate tokens
    if query.lower() in DEGENERATE_QUERIES or len(query) < 3:
        if raw_user_text:
            cleaned = raw_user_text.strip().rstrip("?").strip()
            args["query"] = cleaned if len(cleaned) >= 2 else "latest news"
        else:
            raise ValueError(f"Degenerate or empty query extracted: '{query}'")

    # Validate or default recency argument
    allowed_recencies = TOOLS[tool_name].get("recency", ["any"])
    if args.get("recency") not in allowed_recencies:
        args["recency"] = "any"

    return {"name": tool_name, "args": args}


def web_search(query: str, recency: str = "any", k: int = 3) -> list[str]:
    """Execute live web search via DuckDuckGo with fallback stub.

    Args:
        query: Search term string.
        recency: Temporal filter ('day', 'week', or 'any').
        k: Maximum number of search results to return.

    Returns:
        List of formatted search result strings.
    """
    try:
        from ddgs import DDGS
        timelimit = {"day": "d", "week": "w"}.get(recency)
        with DDGS(timeout=5) as ddgs:
            res = list(ddgs.text(query, max_results=k, timelimit=timelimit))
        if res:
            return [f"{r.get('title', 'No Title')} - {r.get('href', '')}" for r in res]
        return [f"No live results found for '{query}' (recency={recency})."]
    except Exception as e:
        # Graceful fallback: return clear stub message without crashing
        return [
            f"[stub result] Executed search for '{query}' (recency={recency}). "
            f"[Live backend note: {type(e).__name__}]"
        ]


DISPATCH = {
    "web_search": web_search,
}


def run(model: TinyLM, tok: Tokenizer, text: str) -> dict[str, Any]:
    """Complete inference and tool execution flow.

    Returns structured dict containing:
      - raw: Raw model output string
      - tool: Tool name or None
      - args: Extracted arguments dict or None
      - result: Execution result or '(no tool needed)'
      - latency_ms: Execution elapsed time in milliseconds
    """
    t0 = time.perf_counter()
    raw = predict(model, tok, text)

    try:
        call = parse_call(raw, raw_user_text=text)
    except ValueError as e:
        latency = (time.perf_counter() - t0) * 1000.0
        return {
            "raw": raw,
            "tool": None,
            "args": None,
            "result": f"Invalid tool call: {e}",
            "latency_ms": round(latency, 2),
        }

    if call is None:
        latency = (time.perf_counter() - t0) * 1000.0
        return {
            "raw": raw,
            "tool": None,
            "args": None,
            "result": "(no tool needed)",
            "latency_ms": round(latency, 2),
        }

    tool_fn = DISPATCH.get(call["name"])
    if not tool_fn:
        result = f"Error: No implementation handler registered for tool '{call['name']}'"
    else:
        result = tool_fn(**call["args"])

    latency = (time.perf_counter() - t0) * 1000.0
    return {
        "raw": raw,
        "tool": call["name"],
        "args": call["args"],
        "result": result,
        "latency_ms": round(latency, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Run TinyLM Function Calling Inference")
    parser.add_argument("--query", "-q", type=str, default=None,
                        help="Single query to test directly from CLI")
    parser.add_argument("--no-dispatch", action="store_true",
                        help="Only generate tool call, do not execute search backend")
    args = parser.parse_args()

    model, tok = load()

    # One-shot CLI query mode
    if args.query:
        if args.no_dispatch:
            raw = predict(model, tok, args.query)
            print(f"Predicted Call: {raw}")
        else:
            out = run(model, tok, args.query)
            print(f"Query   : {args.query}")
            print(f"Call    : {out['raw']}")
            print(f"Latency : {out['latency_ms']} ms")
            print(f"Result  :")
            res = out["result"]
            if isinstance(res, list):
                for item in res:
                    print(f"  • {item}")
            else:
                print(f"  • {res}")
        return

    # Interactive REPL mode
    print("=" * 64)
    print("TinyLM Function-Calling Interactive Console (~0.47M parameters)")
    print("Commands: Type your query, or 'quit'/'exit' to stop.")
    print("=" * 64)

    while True:
        try:
            q = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not q:
            continue
        if q.lower() in {"quit", "exit"}:
            break

        out = run(model, tok, q)
        print(f"call   : {out['raw']}")
        print(f"latency: {out['latency_ms']} ms")
        print("result :")
        res = out["result"]
        if isinstance(res, list):
            for item in res:
                print(f"  • {item}")
        else:
            print(f"  • {res}")


if __name__ == "__main__":
    main()
