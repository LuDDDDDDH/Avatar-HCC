from pathlib import Path
import os

def resolve(relative):
    configured = os.environ.get('AVATAR_FROZEN_INPUT_ROOT')
    if not configured:
        raise RuntimeError('Run through tools/run_final_figure.py with isolated frozen inputs.')
    root = Path(configured).expanduser().resolve()
    result = (root / relative).resolve()
    if not result.is_relative_to(root):
        raise ValueError('Input must stay within the isolated tree')
    return str(result)
