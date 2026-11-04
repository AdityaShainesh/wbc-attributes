"""Config loading, per CLAUDE.md section 5: one YAML file fully specifies a run.

load_config returns the resolved dict, which is embedded verbatim into
results/<run_id>.json so a result is readable without the config file beside it.
"""
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


class ConfigError(Exception):
    """The config is missing something a run needs, or contradicts itself."""


def load_config(path):
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"no such config: {path}")
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    config.setdefault("config_path", str(path.relative_to(ROOT) if path.is_absolute() else path))
    return config


def require(config, *keys):
    """Raise unless every key is present, naming all the missing ones at once."""
    missing = [key for key in keys if key not in config]
    if missing:
        raise ConfigError(
            f"{config.get('config_path', 'config')} is missing: {', '.join(missing)}"
        )
    return config


def git_state():
    """Commit hash and dirty flag, recorded with every result."""
    def run(*args):
        try:
            return subprocess.run(args, cwd=ROOT, capture_output=True, text=True,
                                  check=True).stdout.strip()
        except Exception:
            return None
    return {"commit": run("git", "rev-parse", "HEAD"),
            "dirty": bool(run("git", "status", "--porcelain"))}
