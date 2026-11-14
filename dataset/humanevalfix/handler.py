import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

from dataset.base import DatasetHandler, test_workers


class HumanEvalFixHandler(DatasetHandler):
    """
    Handler for HumanEvalFix (bigcode/humanevalpack, Python split) in PDB format.

    The records are built outside PDB (DLM_CodeEdit's dlm.pdb.humanevalpack): buggy_code and
    gt_solution are HumanEval's declaration + buggy / canonical body, and each record carries
    its `test` (HumanEvalPack's import + test_setup + test, which defines check() and calls it
    on the entry point). A solution passes when `solution + test` exits with status 0 within
    the per-task timeout, as in HumanEvalPack's own harness. The tests use the standard library
    only, so they run in the evaluator's own interpreter; no install tree is needed.
    """

    def preprocess(self, raw_data):
        return [{"task_id": str(d["task_id"]), "gt_solution": d["gt_solution"], "task_prompt": d["task_prompt"]}
                for d in raw_data]

    def build_verify_unit_test(self, log_file_prefix, results, sol_field="solution"):
        entries = [{"task_id": e["task_id"], "solution": e[sol_field]} for e in results if e.get(sol_field) is not None]
        if not entries:
            print("No submissions to evaluate.")
            return None
        verify_file = log_file_prefix + ".json"
        with open(verify_file, "w") as f:
            json.dump(entries, f, indent=2)
        return verify_file

    def __init__(self):
        self.tests = {}             # task id -> test, from the full records the unit score passes first

    def test_of(self, entry):
        """A record's test; the evaluator's semantic-equivalence and redundancy checks pass
        candidates without one, as `<task_id>_<block>[_<...>]`, so those take their task's test
        (HumanEvalFix ids, `Python/<n>`, have no `_`)."""
        if "test" in entry:
            self.tests[entry["task_id"]] = entry["test"]
            return entry["test"]
        task_id = entry["task_id"].split("_", 1)[0]
        if task_id not in self.tests:
            raise KeyError(f"no test known for {entry['task_id']}")
        return self.tests[task_id]

    def save_formatted_gt(self, log_file_prefix, data):
        gt_file = log_file_prefix + ".json"
        with open(gt_file, "w") as f:
            json.dump({d["task_id"]: self.test_of(d) for d in data}, f, indent=2)
        return gt_file

    def test_command(self, path):
        """The interpreter command a program runs with (subclasses: another venv)."""
        return [sys.executable, path]

    def test_env(self):
        return None

    def _run(self, program, timeout_per_task):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "program.py")
            with open(path, "w") as f:
                f.write(program)
            try:
                proc = subprocess.run(self.test_command(path), cwd=tmp, capture_output=True, text=True,
                                      timeout=timeout_per_task, env=self.test_env())
            except subprocess.TimeoutExpired:
                return False, "timeout"
            return proc.returncode == 0, proc.stderr[-2000:]

    def verify_unit_test(self, verify_file, gt_file=None, timeout_per_task=20, timeout=1800):
        if verify_file is None:
            return [], [], []
        with open(verify_file) as f:
            entries = json.load(f)
        with open(gt_file) as f:
            tests = json.load(f)
        programs = [e["solution"] + "\n\n" + tests[e["task_id"]] + "\n" for e in entries]
        with ThreadPoolExecutor(int(test_workers())) as pool:
            outcomes = list(pool.map(lambda p: self._run(p, timeout_per_task), programs))
        fail_ids, correct_ids, feedback = [], [], []
        for entry, (passed, error) in zip(entries, outcomes):
            if passed:
                correct_ids.append(entry["task_id"])
            else:
                fail_ids.append(entry["task_id"])
                feedback.append(error)
        return fail_ids, correct_ids, feedback
