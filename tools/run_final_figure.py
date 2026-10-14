from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys

CODE = Path(__file__).resolve().parents[1]
TREE = CODE / 'final_visuals' / 'tree'
ENTRIES = json.loads((CODE / 'final_visuals' / 'entries.json').read_text(encoding='utf-8'))

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load_config(path):
    path = Path(path).expanduser().resolve()
    if path.is_relative_to(CODE):
        raise ValueError('Input configuration must be outside the code directory')
    data = json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(data, dict) or not isinstance(data.get('values'), dict):
        raise ValueError('Configuration requires a values object')
    return path, data

def required(figure, config, root=None):
    key = ENTRIES[figure]['input_key']
    if key not in config['values'] or not isinstance(config['values'][key], list):
        raise ValueError('Configure the figure input key with a file list')
    rows = config['values'][key]
    result = []
    for row in rows:
        if not isinstance(row, dict) or not all(key in row for key in ('target', 'sha256')):
            raise ValueError('Input rows require target and sha256')
        target = Path(row['target'])
        if target.is_absolute() or '..' in target.parts:
            raise ValueError('Input targets must be relative paths without parent traversal')
        if root is not None:
            source = (root / row['source_relative']).resolve()
            if not source.is_relative_to(root):
                raise ValueError('Input path must stay within the input root')
        else:
            source = Path(row['source']).expanduser().resolve()
        result.append({'source': source, 'target': target, 'sha256': row['sha256']})
    if len({str(row['target']) for row in result}) != len(result):
        raise ValueError('Input targets must be unique')
    return result

def preflight(rows):
    return [{'target': str(row['target']), 'exists': row['source'].is_file(),
             'sha256_match': row['source'].is_file() and digest(row['source']) == row['sha256']}
            for row in rows]

def adapt_json(value, stage, config):
    if isinstance(value, str):
        normalized = value.replace('\\', '/')
        for marker, prefix in config['values'].get('figures.absolute_markers', []):
            if marker in normalized:
                normalized = prefix + normalized.split(marker, 1)[1]
                for old, new in config['values'].get('figures.path_aliases', []):
                    normalized = normalized.replace(old, new)
                return str(stage / normalized)
        for old, new in config['values'].get('figures.path_aliases', []):
            normalized = normalized.replace(old, new)
        return normalized if normalized != value.replace('\\', '/') else value
    if isinstance(value, list):
        return [adapt_json(item, stage, config) for item in value]
    if isinstance(value, dict):
        return {adapt_json(key, stage, config): adapt_json(item, stage, config) for key, item in value.items()}
    return value

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--figure', choices=ENTRIES)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--input-root', type=Path)
    parser.add_argument('--workdir', type=Path)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if args.list:
        print(json.dumps(ENTRIES, indent=2))
        return
    if not args.figure:
        parser.error('--figure is required')
    entry = ENTRIES[args.figure]
    if entry.get('mode') == 'arguments':
        parser.error('Use the registered command directly with its asset arguments')
    configured = args.config or os.environ.get('AVATAR_INPUT_CONFIG')
    if not configured:
        parser.error('--config or AVATAR_INPUT_CONFIG is required')
    if not (args.check or args.run):
        parser.error('Choose --check or --run')
    config_path, config = load_config(configured)
    input_root = args.input_root.expanduser().resolve() if args.input_root else None
    rows = required(args.figure, config, input_root)
    report = preflight(rows)
    all_match = all(row['sha256_match'] for row in report)
    print(json.dumps({'figure': args.figure, 'inputs_checked': len(report), 'all_inputs_match': all_match,
                      'missing_or_changed': [row for row in report if not row['sha256_match']]}, indent=2))
    if not args.run:
        return
    if not all_match:
        raise SystemExit('Run stopped: missing or changed inputs')
    if not args.workdir:
        parser.error('--workdir is required for --run')
    stage = args.workdir.expanduser().resolve()
    if stage.exists():
        raise SystemExit('Use a new, non-existing work directory')
    if stage.is_relative_to(CODE) or CODE.is_relative_to(stage):
        raise SystemExit('Work directory must be separate from the code directory')
    for row in rows:
        if row['source'].is_relative_to(stage) or stage.is_relative_to(row['source'].parent):
            raise SystemExit('Work directory must be separate from the input files')
    shutil.copytree(TREE, stage)
    original_hashes = {str(row['source']): digest(row['source']) for row in rows}
    for row in rows:
        target = (stage / row['target']).resolve()
        if not target.is_relative_to(stage):
            raise ValueError('Input target escaped the work directory')
        if target.exists():
            raise ValueError('Input target conflicts with a source file')
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(row['source'], target)
        if target.suffix == '.json':
            payload = json.loads(target.read_text(encoding='utf-8-sig'))
            target.write_text(json.dumps(adapt_json(payload, stage, config), ensure_ascii=False, indent=2), encoding='utf-8')
    entry_relative = Path(entry['entrypoint']).relative_to('final_visuals/tree')
    environment = dict(os.environ)
    environment.update(AVATAR_INPUT_CONFIG=str(config_path), AVATAR_FROZEN_INPUT_ROOT=str(stage),
                       AVATAR_PROJECT_DIR=str(stage), AVATAR_INPUT_ROOT=str(stage), MPLBACKEND='Agg', PYTHONDONTWRITEBYTECODE='1')
    environment['PYTHONPATH'] = os.pathsep.join([str(stage), str(stage / 'figure_library'), str(CODE)])
    result = subprocess.run([sys.executable, str(stage / entry_relative)], cwd=stage, env=environment)
    unchanged = all(digest(Path(path)) == expected for path, expected in original_hashes.items())
    (stage / 'run_status.json').write_text(json.dumps({'figure': args.figure, 'returncode': result.returncode,
        'original_inputs_unchanged': unchanged, 'inputs': report}, indent=2), encoding='utf-8')
    if not unchanged:
        raise SystemExit('Original input integrity check failed')
    raise SystemExit(result.returncode)

if __name__ == '__main__':
    main()
