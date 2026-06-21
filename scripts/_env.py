"""Load environment variables from .env file (no external deps)."""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = REPO_ROOT / ".env"

def load_env(path: Path = ENV_PATH) -> dict:
    """Read .env file and return as dict. Also sets os.environ."""
    env = {}
    if not path.exists():
        return env
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        env[key] = val
        os.environ.setdefault(key, val)
    return env

if __name__ == "__main__":
    e = load_env()
    print(f"loaded {len(e)} env vars from {ENV_PATH}")
    for k, v in e.items():
        masked = v[:4] + "***" + v[-2:] if len(v) > 8 and "KEY" in k else v
        print(f"  {k} = {masked}")
