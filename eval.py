"""Rigorous evaluation suite for TinyLM function-calling model.

Evaluation Dimensions & Metrics
-------------------------------
1. **Tool Routing (Binary Classification)**:
   - Tool vs Abstain (`none`): Precision, Recall, F1, and Accuracy.
   - Measures whether the model correctly decides when to invoke `web_search`
     and when to abstain on chit-chat or generic banter.

2. **Schema & Argument Validity**:
   - Syntactic Validity Rate: Percentage of outputs conforming strictly to schema
     (`web_search|query=<str>|recency=<day|week|any>` or `none`).
   - Query Extraction Accuracy: Exact string match of extracted search query.
   - Recency Slot Accuracy: Exact match of temporal constraint ("day", "week", "any").

3. **Per-Intent Breakdown**:
   - Evaluates performance across specific intent categories (weather, news, stock,
     how-to, sports, definitions, people, price, and chit-chat).
   - Identifies specific failure modes and blind spots.

4. **Robustness & Generalization**:
   - Compares in-distribution validation (`val.jsonl`) with held-out phrasing
     out-of-distribution (`test_ood.jsonl`).

5. **Inference Latency & Throughput**:
   - Measures average latency (ms per query) and generation speed.

Limitations & Honesty
---------------------
- Metrics rely on exact match against synthetic gold labels.
- In real life, multiple different search queries could be valid for the same user request
  (e.g., "chennai weather" vs "weather in chennai today"). Exact match is an intentionally
  strict lower-bound metric.
- OOD evaluates novel template phrasings, but uses the same underlying entity pool; it does
  not evaluate completely unseen named entities outside the training vocabulary.
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
import torch

from config import DATA_DIR, TOOLS
from infer import load, predict, parse_call


def classify_intent(user_text: str, gold_call: str) -> str:
    """Heuristically tag intent category for per-intent metric breakdown."""
    if gold_call.strip() == "none":
        return "chit-chat (none)"
    if "weather" in gold_call or "temperature" in user_text:
        return "weather"
    if "stock price" in gold_call or "share" in user_text or "trading" in user_text:
        return "stock"
    if "score" in gold_call or "win" in user_text or "match" in user_text or "game" in user_text:
        return "sports score"
    if "how to" in gold_call or "guide" in user_text or "tutorial" in user_text:
        return "how-to"
    if "price" in gold_call or "cost" in user_text or "buy" in user_text:
        return "product price"
    if "this week" in gold_call or "weekly" in user_text:
        return "weekly recap"
    if "news" in gold_call:
        return "news"
    if "who is" in user_text or "biography" in user_text or "who was" in user_text:
        return "person bio"
    return "general concept"


def safe_parse(call: str):
    """Safely parse a tool call string; returns None, parsed dict, or 'INVALID'."""
    try:
        return parse_call(call)
    except ValueError:
        return "INVALID"


def run_evaluation(model, tok, filename: str, limit: int | None = None, verbose: bool = True):
    """Run full evaluation suite on a dataset split."""
    filepath = DATA_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset split not found: {filepath}. Run gen_data.py first.")

    with open(filepath, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]

    if limit is not None and limit > 0:
        rows = rows[:limit]

    total = len(rows)
    if total == 0:
        print(f"Empty dataset in {filename}")
        return {}

    # Global counters
    tp = fp = tn = fn = 0
    exact_matches = 0
    valid_syntax_count = 0
    query_matches = 0
    recency_matches = 0
    total_tool_calls_gold = 0

    # Per-intent stats: intent -> {total, exact, tool_ok, query_ok, recency_ok}
    intent_stats = defaultdict(lambda: defaultdict(int))
    error_samples = []
    latencies_ms = []

    for r in rows:
        user_text = r["user"]
        gold_raw = r["call"].strip()
        intent = classify_intent(user_text, gold_raw)
        intent_stats[intent]["total"] += 1

        t0 = time.perf_counter()
        pred_raw = predict(model, tok, user_text).strip()
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        p_parsed = safe_parse(pred_raw)
        g_parsed = safe_parse(gold_raw)

        is_valid_syntax = (p_parsed != "INVALID")
        if is_valid_syntax:
            valid_syntax_count += 1

        is_gold_tool = (gold_raw != "none")
        is_pred_tool = (pred_raw != "none" and is_valid_syntax)

        if is_gold_tool:
            total_tool_calls_gold += 1

        # Binary routing classification matrix
        if is_pred_tool and is_gold_tool:
            tp += 1
        elif is_pred_tool and not is_gold_tool:
            fp += 1
        elif not is_pred_tool and not is_gold_tool:
            tn += 1
        elif not is_pred_tool and is_gold_tool:
            fn += 1

        # Slot matching
        q_match = False
        r_match = False
        if is_gold_tool:
            if is_pred_tool and isinstance(p_parsed, dict) and isinstance(g_parsed, dict):
                p_args = p_parsed.get("args", {})
                g_args = g_parsed.get("args", {})
                q_match = (p_args.get("query") == g_args.get("query"))
                r_match = (p_args.get("recency") == g_args.get("recency"))
        else:
            q_match = (pred_raw == "none")
            r_match = (pred_raw == "none")

        if q_match:
            query_matches += 1
            intent_stats[intent]["query_ok"] += 1
        if r_match:
            recency_matches += 1
            intent_stats[intent]["recency_ok"] += 1

        tool_decision_ok = (is_pred_tool == is_gold_tool)
        if tool_decision_ok:
            intent_stats[intent]["tool_ok"] += 1

        is_exact = (pred_raw == gold_raw)
        if is_exact:
            exact_matches += 1
            intent_stats[intent]["exact"] += 1
        else:
            if len(error_samples) < 10:
                error_samples.append({
                    "user": user_text,
                    "gold": gold_raw,
                    "pred": pred_raw,
                    "intent": intent,
                    "syntax_valid": is_valid_syntax
                })

    # Metric calculations
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    routing_acc = (tp + tn) / total
    exact_acc = exact_matches / total
    syntax_rate = valid_syntax_count / total
    query_acc = query_matches / total
    recency_acc = recency_matches / total
    avg_latency = sum(latencies_ms) / len(latencies_ms)

    metrics = {
        "split": filename,
        "n_samples": total,
        "exact_match_acc": exact_acc,
        "syntax_validity_rate": syntax_rate,
        "routing_accuracy": routing_acc,
        "routing_precision": precision,
        "routing_recall": recall,
        "routing_f1": f1,
        "query_accuracy": query_acc,
        "recency_accuracy": recency_acc,
        "avg_latency_ms": avg_latency,
        "per_intent": dict(intent_stats),
        "error_samples": error_samples,
    }

    if verbose:
        print("=" * 72)
        print(f"EVALUATION RESULTS: {filename} (N = {total:,})")
        print("=" * 72)
        print(f"Global Metrics:")
        print(f"  Exact Match Accuracy   : {exact_acc * 100:6.2f}% ({exact_matches}/{total})")
        print(f"  Syntax Validity Rate   : {syntax_rate * 100:6.2f}% ({valid_syntax_count}/{total})")
        print(f"  Routing Accuracy       : {routing_acc * 100:6.2f}% (TP={tp}, TN={tn}, FP={fp}, FN={fn})")
        print(f"  Routing Precision      : {precision * 100:6.2f}%")
        print(f"  Routing Recall         : {recall * 100:6.2f}%")
        print(f"  Routing F1-Score       : {f1 * 100:6.2f}%")
        print(f"  Query Slot Match       : {query_acc * 100:6.2f}%")
        print(f"  Recency Slot Match     : {recency_acc * 100:6.2f}%")
        print(f"  Avg Latency per query  : {avg_latency:6.2f} ms")
        print("-" * 72)
        print(f"{'Intent Category':<20} | {'Count':<6} | {'Exact %':<8} | {'Routing %':<10} | {'Query %':<8}")
        print("-" * 72)
        for cat, s in sorted(intent_stats.items()):
            n = s["total"]
            ex_pct = (s["exact"] / n) * 100
            rt_pct = (s["tool_ok"] / n) * 100
            qy_pct = (s["query_ok"] / n) * 100
            print(f"{cat:<20} | {n:<6} | {ex_pct:6.2f}%  | {rt_pct:8.2f}%  | {qy_pct:6.2f}%")
        print("-" * 72)

        if error_samples:
            print("\nRepresentative Failure Cases:")
            for i, err in enumerate(error_samples[:6], 1):
                print(f"  [{i}] Input  : \"{err['user']}\"")
                print(f"      Intent : {err['intent']}")
                print(f"      Gold   : {err['gold']}")
                print(f"      Pred   : {err['pred']}")
                if not err["syntax_valid"]:
                    print(f"      Note   : SYNTAX ERROR / UNPARSEABLE")
            print()

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate TinyLM Function Calling Model")
    parser.add_argument("--split", choices=["val", "ood", "all"], default="all",
                        help="Dataset split to evaluate on (default: all)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of test samples (default: full split)")
    parser.add_argument("--json", action="store_true",
                        help="Output evaluation results as raw JSON")
    args = parser.parse_args()

    model, tok = load()

    splits_to_run = []
    if args.split in ("val", "all"):
        splits_to_run.append("val.jsonl")
    if args.split in ("ood", "all"):
        splits_to_run.append("test_ood.jsonl")

    all_metrics = {}
    for s in splits_to_run:
        all_metrics[s] = run_evaluation(model, tok, s, limit=args.limit, verbose=not args.json)

    if args.json:
        # Convert non-serializable elements
        print(json.dumps(all_metrics, indent=2, default=str))


if __name__ == "__main__":
    main()
