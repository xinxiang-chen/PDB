# Scoring DLM_CodeEdit outputs with PDB

Run everything from this directory (`/workspace/PDB`) with `.venv/bin/python`.

- `MODEL`: `llada`, `dream`, `dream-coder`
- `ARM`: `armA`, `armB`, `armC` (the name used in the score file names)
- `RUN_DIR`: the runner's output root, e.g. `/workspace/DLM_CodeEdit/outputs/armA`, `.../outputs/armB` or `.../outputs/armC`

Score files are written to
`results/{bigcodebench,livecodebench}/eval_results/<MODEL>_<ARM>_on_pdb_single_1k_{bcb,lcb}_round_1_scores.json`
(re-scoring the same MODEL/ARM overwrites them). `src/evaluator.py` scores one
dataset per call and only prints that dataset's `[summary]` line; the BCB+LCB
total (`union`) is computed separately below, with the same formula as
`scripts/simple_debug_eval.sh` (per-task mean over both datasets).

## Setup (this fork, pinned upstream commit 22b21ed)

Environments: main `.venv` and LiveCodeBench sandbox on Python 3.12, BigCodeBench
sandbox on Python 3.10 (uv 0.9.0 was used). Two deviations from the upstream README:

- BigCodeBench sandbox needs `--python 3.10` (its tensorflow 2.11 pin has only cp310 wheels).
- LiveCodeBench's `lcb_runner/prompts/few_shot_examples/generation/{func,stdin}.json`
  are ignored by upstream's `.gitignore`; this fork commits them (import-time only, not used in scoring).

```bash
uv sync
```

```bash
cd dataset/bigcodebench/install && uv sync --extra eval --python 3.10 && cd -
```

```bash
cd dataset/livecodebench/install && uv sync && cd -
```

Sanity check: LLaDA Arm A on PDB-Single-1k BCB should give `unit=0.253 prec=0.480 rec=0.462 f1=0.427 (n=487)`.

### PDB-Single-1k: score one model/arm (BCB + LCB), then print the union

```bash
MODEL=dream-coder ARM=armC RUN_DIR=/workspace/DLM_CodeEdit/outputs/armC; for pair in bigcodebench:bcb livecodebench:lcb; do ds=${pair%%:*}; s=${pair##*:}; .venv/bin/python src/evaluator.py --dataset_name $ds --input_file $RUN_DIR/$MODEL/pdb_single/${MODEL}_on_pdb_single_${s}.json --eval_model_name ${MODEL}_${ARM} --eval_set_name pdb_single_1k_${s} --mode single --max_iter 1 2>&1 | grep "\[summary\]"; done
```

### Union (BCB + LCB) for already-scored runs

Prints one line per MODEL/ARM whose BCB and LCB score files both exist
(script: `union_scores.py`; pass another eval_set prefix, e.g. `pdb_multi`, as the argument).

```bash
.venv/bin/python union_scores.py
```

### PDB-Multi

Same commands with `pdb_multi` in the input file name, `--eval_set_name pdb_multi_{bcb,lcb}`,
and `--mode multi` (ε = 1).
