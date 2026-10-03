"""Print BCB + LCB union scores for every scored DLM_CodeEdit run.

Same formula as scripts/simple_debug_eval.sh: per-task mean over both datasets.
A MODEL/ARM is printed only when both score files exist.

Usage (from /workspace/PDB):
  .venv/bin/python union_scores.py                 # PDB-Single-1k
  .venv/bin/python union_scores.py pdb_multi       # another eval_set prefix
"""
import json
import os
import sys

EVAL_SET = sys.argv[1] if len(sys.argv) > 1 else "pdb_single_1k"
ARMS = ["armA", "armB", "armC", "armC_delta+2", "armC_delta-2", "armC_delta+25pct", "armC_delta-25pct", "armC_delta+50pct", "armC_delta-50pct", "ar"]
MODELS = ["llada", "dream", "dream-coder", "qwen2.5-coder", "qwen2.5", "llama3.1"]
DATASETS = [("bigcodebench", "bcb"), ("livecodebench", "lcb")]

for arm in ARMS:
    for model in MODELS:
        files = [
            f"results/{ds}/eval_results/{model}_{arm}_on_{EVAL_SET}_{tag}_round_1_scores.json"
            for ds, tag in DATASETS
        ]
        if not all(os.path.exists(f) for f in files):
            continue
        unit = prec = rec = f1 = 0.0
        n = 0
        for f in files:
            scores = json.load(open(f))
            sym = scores["Symbolic block scores"]
            unit += sum(scores["Unit score"].values())
            prec += sum(v["precision"] for v in sym.values())
            rec += sum(v["recall"] for v in sym.values())
            f1 += sum(v["f1"] for v in sym.values())
            n += len(scores["Unit score"])
        print(f"{model:12s} {arm}  union  unit={unit/n:.3f} prec={prec/n:.3f} rec={rec/n:.3f} f1={f1/n:.3f} (n={n})")
