"""mutation_check.py: change the service code in five ways. Each change must make a test fail.

Run from the repository root:
    python partA/mutation_check.py

For each change, the script copies the folder partA to a temporary folder, changes one rule in the copy,
and runs test_eta_app.py there. A change is "caught" when at least one test fails.
The script does not change the real files. It exits with code 1 if a change is not caught.
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PART_A = Path(__file__).resolve().parent
ROOT = PART_A.parent
MODEL = ROOT / "data" / "eta_v1.joblib"

# Each change: a name, the text to find, and the new text.
CHANGES = [
    ("no cross-field rule", "if self.distance_km > INFEASIBLE_DISTANCE_KM and", "if False and"),
    ("batch limit 1000", "MAX_BATCH = 500", "MAX_BATCH = 1000"),
    ("no request-ID echo", "request.state.request_id = client_id if client_id.strip() else str(uuid.uuid4())",
     "request.state.request_id = str(uuid.uuid4())"),
    ("count /metrics", "if request.url.path != METRICS_PATH:", "if True:"),
    ("leak the exception", "internal_error(request.state.request_id)", "RAISE"),
]


def run_tests(folder):
    """Run the tests in the folder. Give the last line of the pytest output."""
    env = {**os.environ, "ETA_MODEL": str(MODEL)}
    result = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                             "test_eta_app.py"], cwd=folder, env=env, capture_output=True, text=True)
    return result.stdout.strip().splitlines()[-1] if result.stdout.strip() else result.stderr[-200:]


def main():
    missed = 0
    with tempfile.TemporaryDirectory() as tmp:
        print(f"unchanged code: {run_tests(PART_A)}")
        for name, old, new in CHANGES:
            folder = Path(tmp) / name.replace(" ", "_").replace("/", "")
            shutil.copytree(PART_A, folder, ignore=shutil.ignore_patterns("__pycache__", "*.json", "*.txt"))
            text = (folder / "eta_app.py").read_text()
            if old not in text:
                sys.exit(f"cannot find the text to change for: {name}")
            if new == "RAISE":
                # Replace the two lines that return the generic 500 answer.
                text = re.sub(r"(return|response =) internal_error\(request\.state\.request_id\)",
                              lambda m: "raise" if m.group(1) == "return" else "raise", text)
            else:
                text = text.replace(old, new)
            (folder / "eta_app.py").write_text(text)
            last_line = run_tests(folder)
            caught = " failed" in last_line
            missed += not caught
            print(f"change '{name}': {last_line} -> {'caught' if caught else 'NOT CAUGHT'}")
    print(f"result: {len(CHANGES) - missed} of {len(CHANGES)} changes caught")
    sys.exit(1 if missed else 0)


if __name__ == "__main__":
    main()
