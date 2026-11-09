from __future__ import annotations
import os
from pathlib import Path
from .exceptions import ConfigError

_loaded = False

def load_dotenv(path: str | None = None):
    global _loaded
    if _loaded:
        return
    try:
        from dotenv import load_dotenv as _ld
    except Exception:
        _loaded = True
        return
    p = Path(path) if path else Path.cwd() / ".env"
    if p.exists():
        _ld(dotenv_path=p, override=False)
    _loaded = True

def get(key: str, default: str | None = None, required: bool = False, dotenv_path: str | None = None) -> str | None:
    load_dotenv(dotenv_path)
    v = os.getenv(key, default)
    if required and (v is None or str(v).strip() == ""):
        raise ConfigError(f"Missing {key}. Set it in .env or environment variables.")
    return v

def require(key: str, dotenv_path: str | None = None) -> str:
    v = get(key, required=True, dotenv_path=dotenv_path)
    return str(v)
