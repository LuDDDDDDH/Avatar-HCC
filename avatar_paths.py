from pathlib import Path
import json
import os

_CONFIG_CACHE = {}
_MISSING = object()


def _config():
    location = os.environ.get("AVATAR_INPUT_CONFIG")
    if not location:
        return {}
    path = Path(location).expanduser().resolve()
    stat = path.stat()
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    if key in _CONFIG_CACHE:
        return _CONFIG_CACHE[key]
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError("Invalid input configuration")
    _CONFIG_CACHE.clear()
    _CONFIG_CACHE[key] = value
    return value


def _get(section, key, default=_MISSING):
    values = _config().get(section, {})
    if not isinstance(values, dict):
        raise ValueError("Invalid input configuration")
    if key in values:
        return values[key]
    if default is _MISSING:
        raise KeyError("Missing configured input: " + key)
    return default


def get_value(key, default=_MISSING):
    return _get("values", key, default)


def get_column(key, default=_MISSING):
    value = _get("columns", key, default)
    if not isinstance(value, str) or not value:
        raise ValueError("Invalid configured column: " + key)
    return value


def get_label(key, default=_MISSING):
    return _get("labels", key, default)


def _root():
    value = os.environ.get("AVATAR_PROJECT_DIR") or os.environ.get("AVATAR_INPUT_ROOT")
    if not value:
        raise RuntimeError("Set AVATAR_PROJECT_DIR or AVATAR_INPUT_ROOT")
    return Path(value).expanduser().resolve()


def get_path(key, default=_MISSING, root=None):
    value = _get("paths", key, default)
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    base = _root() if root is None else Path(root).expanduser().resolve()
    return (base / path).resolve()


def project_path(relative=""):
    root = _root()
    source = Path(__file__).resolve().parent
    if root == source or source.is_relative_to(root) or root.is_relative_to(source):
        raise ValueError("Input root must be separate from the code directory")
    result = get_path(relative, relative, root)
    if not result.is_relative_to(root):
        raise ValueError("Input path escapes the configured root")
    return result
