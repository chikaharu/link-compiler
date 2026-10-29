"""Version forgery by trial-and-error.

Idea: some libraries gate on the *version string* of their dependency
(np.__version__ checks). That gate can be forged: present the real
numpy module with a rewritten __version__, and repeat the trial over
candidate versions until the target library accepts one.

Each candidate is tried in a FRESH subprocess (a failed import can
poison interpreter state). Stops at the first accepted candidate.
Known limit (tested separately): gates that are not strings but the
binary ABI (dtype layout etc.) cannot be forged this way.
"""
import subprocess, sys, os, json

RUNNER = r'''
import sys, types, importlib
numpy_path, candidate, target, probe = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
if numpy_path != "-":
    sys.path.insert(0, numpy_path)
import numpy as real
shim = types.ModuleType("numpy")
shim.__dict__.update(real.__dict__)
shim.__version__ = candidate
sys.modules["numpy"] = shim
try:
    mod = importlib.import_module(target)
    out = getattr(mod, probe)() if probe != "-" else "import-ok"
    print("FORGE-OK", out)
except Exception as e:
    print("FORGE-FAIL", type(e).__name__ + ":", str(e).replace("\n", " ")[:180])
'''

def trial(candidates, target, probe="-", numpy_path="-"):
    print(f"target={target} probe={probe} numpy_path={numpy_path}")
    for cand in candidates:
        r = subprocess.run([sys.executable, "-c", RUNNER, numpy_path, cand, target, probe],
                           capture_output=True, text=True, timeout=120,
                           cwd=os.path.dirname(os.path.abspath(__file__)))
        line = (r.stdout.strip().splitlines() or ["<no output> " + r.stderr.strip()[-160:]])[-1]
        print(f"  try __version__={cand:<9} -> {line}")
        if line.startswith("FORGE-OK"):
            print(f"  ==> accepted: {cand}")
            return cand
    print("  ==> all candidates rejected")
    return None

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    if which in ("all", "strict"):
        print("[1] synthetic version-gated library (gate: numpy>=2.0 string check, real numpy is 1.26.4)")
        trial(["1.26.4", "1.99.9", "2.0.0", "2.2.6"], "strict_lib", "probe")
    if which in ("all", "pandas"):
        print("[2] real library: pandas 2.1.4 (real numpy 1.26.4 underneath, forged strings)")
        trial(["1.19.5", "1.26.4", "2.2.6"], "pandas", "-")
    if which in ("all", "scipy-abi"):
        print("[3] real ABI case: scipy 1.11.4 under REAL numpy 2.2.6, forged strings (expect: unforgeable)")
        trial(["1.26.4", "1.25.2", "1.22.0", "2.2.6"], "scipy.linalg", "-", numpy_path="/tmp/np2")
