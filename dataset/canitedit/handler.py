import os

from dataset.humanevalfix.handler import HumanEvalFixHandler


class CanItEditHandler(HumanEvalFixHandler):
    """
    Handler for CanItEdit (nuprl/CanItEdit) in PDB format: its "corrective" (bug-fix) tasks, built
    outside PDB by DLM_CodeEdit's dlm.pdb.canitedit (buggy_code = before, gt_solution = after,
    task id `CanItEdit/<id>`, `test` = the task's tests). As in CanItEdit's own harness, a solution
    passes when `solution + tests` exits with status 0.

    The tests need numpy, pandas, scipy, scikit-learn, z3, flask and autograd, so they run in this
    dataset's uv venv (install/: `uv venv --python 3.11 .venv && uv pip install -r pyproject.toml`).
    vllm (78_llm_inference, whose tests mock its `LLM`) is a two-name stub in install/stubs/,
    put on PYTHONPATH.
    """

    def test_command(self, path):
        if not self.venv_python.exists():
            raise RuntimeError(f"uv venv missing at {self.venv_python}; see this handler's docstring")
        return [str(self.venv_python), path]

    def test_env(self):
        env = dict(os.environ)
        stubs = str(self.install_dir / "stubs")
        env["PYTHONPATH"] = stubs + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        return env
