from __future__ import annotations
from os import PathLike
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
PACKAGE_ROOT = Path(__file__).resolve().parent
ABSOLUTE_PATH = re.compile('(?<![:/\\w])(?:[A-Za-z]:[\\\\/]|\\\\\\\\|/)[^\\r\\n\'\\"<>|,;{}\\[\\]]+')

def _path_label(value: str) -> str:
    if re.match('^[A-Za-z]:[\\\\/]', value) or value.startswith('\\\\'):
        path = PureWindowsPath(value)
        root = PureWindowsPath(str(PACKAGE_ROOT))
    else:
        path = PurePosixPath(value)
        root = PurePosixPath(PACKAGE_ROOT.as_posix())
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name or 'external-directory'

def public_path_display(value):
    if isinstance(value, PathLike):
        return public_path_display(str(value))
    if isinstance(value, dict):
        return {public_path_display(key): public_path_display(item) for key, item in value.items()}
    if isinstance(value, list):
        return [public_path_display(item) for item in value]
    if isinstance(value, tuple):
        return tuple((public_path_display(item) for item in value))
    if isinstance(value, str):
        if re.match('^(?:[A-Za-z]:[\\\\/]|\\\\\\\\|/)', value):
            return _path_label(value)
        return ABSOLUTE_PATH.sub(lambda match: _path_label(match.group(0).rstrip()), value)
    return value
