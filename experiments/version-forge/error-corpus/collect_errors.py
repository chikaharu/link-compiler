"""Run every specimen in a disposable subprocess and collect the errors
as data: exit status, signal, exception type, final message line.
Errors are not failures here -- they are the harvest."""
import json, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
# per-specimen environment: err04 needs the numpy 2.x overlay
ENVS = {"err04_dtype_size.py": {"PYTHONPATH": "/tmp/np2"}}

def harvest(path):
    env = dict(os.environ)
    env.update(ENVS.get(os.path.basename(path), {}))
    r = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=60, env=env)
    tb = [l for l in r.stderr.splitlines() if re.match(r"^[A-Za-z_.]+(Error|Exit|Interrupt|Warning)", l)]
    last_out = (r.stdout.strip().splitlines() or [""])[-1]
    return {
        "script": os.path.basename(path),
        "returncode": r.returncode,
        "signal": -r.returncode if r.returncode < 0 else None,
        "exception": tb[-1].split(":")[0] if tb else None,
        "message": (tb[-1].split(":", 1)[1].strip()[:110] if tb and ":" in tb[-1] else last_out[:110]),
        "stdout_last": last_out[:80],
    }

if __name__ == "__main__":
    files = sorted(f for f in os.listdir(HERE) if (f.startswith("err") or f.startswith("ok")) and f.endswith(".py"))
    catalog = [harvest(os.path.join(HERE, f)) for f in files]
    with open(os.path.join(HERE, "errors.json"), "w") as fh:
        json.dump(catalog, fh, indent=1, ensure_ascii=False)
    for c in catalog:
        sig = f"signal {c['signal']}" if c["signal"] else f"rc={c['returncode']}"
        print(f"{c['script']:<28} {sig:<9} {str(c['exception']):<22} {c['message']}")
    print(f"\n{len(catalog)} specimens -> errors.json")
