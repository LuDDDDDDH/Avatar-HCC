from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

CODE = Path(__file__).resolve().parents[1]
CONTRACTS = CODE / 'configuration'
KINDS = {'input': 'input_config', 'tables': 'tables_config', 'registry': 'source_registry'}


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def external(path):
    path = Path(path).expanduser().resolve()
    if path.is_relative_to(CODE) or CODE.is_relative_to(path):
        raise ValueError('Working paths must be separate from the source directory')
    return path


def reference(spec, root):
    pointer = spec.get('$ref')
    if pointer is None:
        return spec
    if not pointer.startswith('#/'):
        raise ValueError('Only local schema references are supported')
    value = root
    for part in pointer[2:].split('/'):
        value = value[part.replace('~1', '/').replace('~0', '~')]
    return {**value, **{key: item for key, item in spec.items() if key != '$ref'}}


def type_matches(value, kind):
    if kind == 'null':
        return value is None
    if kind == 'object':
        return isinstance(value, dict)
    if kind == 'array':
        return isinstance(value, list)
    if kind == 'string':
        return isinstance(value, str)
    if kind == 'boolean':
        return isinstance(value, bool)
    if kind == 'integer':
        return isinstance(value, int) and not isinstance(value, bool)
    if kind == 'number':
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return False


def structural_errors(value, spec, root=None, location='$'):
    root = spec if root is None else root
    spec = reference(spec, root)
    errors = []
    for operator in ('anyOf', 'oneOf', 'allOf'):
        if operator in spec:
            matches = sum(not structural_errors(value, branch, root, location) for branch in spec[operator])
            valid = matches > 0 if operator == 'anyOf' else matches == 1 if operator == 'oneOf' else matches == len(spec[operator])
            if not valid:
                return [location + ': schema alternatives do not match']
    expected = spec.get('type')
    if expected is not None:
        types = expected if isinstance(expected, list) else [expected]
        if not any(type_matches(value, kind) for kind in types):
            return [location + ': invalid value type']
    if 'enum' in spec and value not in spec['enum']:
        errors.append(location + ': invalid choice')
    if 'const' in spec and value != spec['const']:
        errors.append(location + ': invalid constant')
    if isinstance(value, dict):
        properties = spec.get('properties', {})
        for key in spec.get('required', []):
            if key not in value:
                errors.append(location + '.' + key + ': missing field')
        for key, item in value.items():
            definition = properties.get(key)
            if definition is None:
                extra = spec.get('additionalProperties', True)
                if extra is False:
                    errors.append(location + '.' + key + ': unknown field')
                elif isinstance(extra, dict):
                    errors.extend(structural_errors(item, extra, root, location + '.' + key))
            else:
                errors.extend(structural_errors(item, definition, root, location + '.' + key))
    elif isinstance(value, list):
        if len(value) < spec.get('minItems', 0):
            errors.append(location + ': too few items')
        if 'maxItems' in spec and len(value) > spec['maxItems']:
            errors.append(location + ': too many items')
        if spec.get('uniqueItems') and len({json.dumps(item, sort_keys=True) for item in value}) != len(value):
            errors.append(location + ': duplicate items')
        if 'items' in spec:
            for index, item in enumerate(value):
                errors.extend(structural_errors(item, spec['items'], root, f'{location}[{index}]'))
    elif isinstance(value, str):
        if len(value) < spec.get('minLength', 0):
            errors.append(location + ': empty value')
        if 'maxLength' in spec and len(value) > spec['maxLength']:
            errors.append(location + ': value is too long')
        if 'pattern' in spec and re.search(spec['pattern'], value) is None:
            errors.append(location + ': invalid format')
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        if 'minimum' in spec and value < spec['minimum']:
            errors.append(location + ': below minimum')
        if 'maximum' in spec and value > spec['maximum']:
            errors.append(location + ': above maximum')
    return errors


def applies(spec, scope):
    scopes = spec.get('x-scope', [])
    if isinstance(scopes, str):
        scopes = [scopes]
    if scope == 'all' or scope in scopes:
        return True
    return False


def configuration_errors(value, spec, scope, root=None, location='$'):
    root = spec if root is None else root
    spec = reference(spec, root)
    errors = []
    if spec.get('x-required') and applies(spec, scope):
        minimum = spec.get('x-min-items', 1)
        if value is None or isinstance(value, str) and not value.strip() or isinstance(value, list) and len(value) < minimum:
            errors.append(location + ': not configured')
    if 'properties' in spec:
        obj = value if isinstance(value, dict) else {}
        for key, definition in spec['properties'].items():
            errors.extend(configuration_errors(obj.get(key), definition, scope, root, location + '.' + key))
    return errors


def clean(value):
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [clean(item) for item in value]
    return value


def checked(kind, filename, scope='all', structure_only=False):
    config = external(filename)
    data = load(config)
    spec = load(CONTRACTS / (KINDS[kind] + '.schema.json'))
    errors = structural_errors(data, spec)
    if not errors and not structure_only:
        errors = configuration_errors(data, spec, scope)
    return config, data, errors


def file_binding_errors(kind, config, data, scope, input_root=None):
    base = external(input_root or config.parent)
    spec = load(CONTRACTS / (KINDS[kind] + '.schema.json'))
    errors = []
    section = 'paths' if kind == 'input' else 'files' if kind == 'tables' else 'sources'
    properties = spec['properties'][section]['properties']
    for key, value in data.get(section, {}).items():
        definition = properties.get(key, {})
        if value is None or not applies(definition, scope):
            continue
        if kind == 'tables' and definition.get('x-direction') != 'input':
            continue
        if not (base / value).exists():
            errors.append('$.' + section + '.' + key + ': file or directory is unavailable')
    if kind == 'input':
        for key, rows in data.get('values', {}).items():
            if not key.startswith('figures.files.') or not applies(spec['properties']['values']['properties'][key], scope):
                continue
            for index, row in enumerate(rows or []):
                path = base / (row.get('source_relative') or row['source'])
                location = '$.values.' + key + f'[{index}]'
                if not path.is_file():
                    errors.append(location + ': input file is unavailable')
                elif row.get('sha256'):
                    digest = hashlib.sha256(path.read_bytes()).hexdigest()
                    if digest != row['sha256']:
                        errors.append(location + ': input SHA-256 does not match')
    return errors


def init(args):
    target = external(args.directory)
    destinations = [target / (stem + '.json') for stem in KINDS.values()]
    if any(path.exists() for path in destinations):
        raise FileExistsError('Configuration already exists')
    target.mkdir(parents=True, exist_ok=True)
    for stem, destination in zip(KINDS.values(), destinations):
        shutil.copyfile(CONTRACTS / (stem + '.template.json'), destination)
    print(json.dumps({'created': [path.name for path in destinations]}))
    return 0


def check(args):
    config, data, errors = checked(args.kind, args.config, args.scope, args.structure)
    if not errors and args.files:
        errors = file_binding_errors(args.kind, config, data, args.scope, args.input_root)
    print(json.dumps({'valid': not errors, 'issue_count': len(errors), 'issues': errors[:20]}))
    return 2 if errors else 0


def run(args):
    if not args.input_config and not args.table_config:
        raise ValueError('Provide --input-config or --table-config')
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        raise ValueError('Provide a command after --')
    configurations = []
    for kind, filename, variable in (('input', args.input_config, 'AVATAR_INPUT_CONFIG'), ('tables', args.table_config, 'AVATAR_TABLE_CONFIG')):
        if filename:
            config, data, errors = checked(kind, filename, args.scope)
            if not errors and args.files:
                errors = file_binding_errors(kind, config, data, args.scope, args.input_root)
            if errors:
                print(json.dumps({'valid': False, 'issue_count': len(errors), 'issues': errors[:20]}))
                return 2
            configurations.append((kind, config, data, variable))
    base = configurations[0][1].parent
    input_root = external(args.input_root or base)
    if not input_root.is_dir():
        raise FileNotFoundError('Input root is unavailable')
    workspace = external(args.workspace or base / 'work')
    workspace.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(AVATAR_PROJECT_DIR=str(input_root), AVATAR_INPUT_ROOT=str(input_root), PYTHONDONTWRITEBYTECODE='1')
    env['PYTHONPATH'] = str(CODE) + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    with tempfile.TemporaryDirectory(prefix='avatar_config_', dir=workspace) as runtime:
        for kind, config, data, variable in configurations:
            if kind == 'tables':
                env[variable] = str(config)
            else:
                path = Path(runtime) / 'input_config.json'
                path.write_text(json.dumps(clean(data), ensure_ascii=False, indent=2), encoding='utf-8')
                env[variable] = str(path)
        return subprocess.run(command, cwd=CODE, env=env).returncode


def hash_file(args):
    digest = hashlib.sha256()
    with Path(args.file).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    print(digest.hexdigest())
    return 0


def scope_choices():
    scopes = {'all'}
    def visit(value):
        if isinstance(value, dict):
            labels = value.get('x-scope', [])
            scopes.update([labels] if isinstance(labels, str) else labels)
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)
    for stem in KINDS.values():
        visit(load(CONTRACTS / (stem + '.schema.json')))
    return sorted(scopes)


def main():
    parser = argparse.ArgumentParser(description='Initialize, check and load Avatar-HCC input configurations.')
    actions = parser.add_subparsers(dest='action', required=True)
    create = actions.add_parser('init', help='Copy data-free templates to an external working directory.')
    create.add_argument('--directory', required=True, type=Path)
    create.set_defaults(function=init)
    inspect = actions.add_parser('check', help='Check configuration structure and required fields.')
    inspect.add_argument('--kind', choices=KINDS, default='input')
    inspect.add_argument('--config', required=True, type=Path)
    inspect.add_argument('--scope', choices=scope_choices(), default='all')
    inspect.add_argument('--structure', action='store_true', help='Validate the template structure without requiring real input bindings.')
    inspect.add_argument('--files', action='store_true', help='Check input availability and configured figure-file hashes.')
    inspect.add_argument('--input-root', type=Path)
    inspect.set_defaults(function=check)
    execute = actions.add_parser('run', help='Run a command with the configured input environment.')
    execute.add_argument('--input-config', type=Path)
    execute.add_argument('--table-config', type=Path)
    execute.add_argument('--input-root', type=Path)
    execute.add_argument('--workspace', type=Path)
    execute.add_argument('--scope', choices=scope_choices(), default='all')
    execute.add_argument('--files', action='store_true', help='Check input availability before running the command.')
    execute.add_argument('command', nargs=argparse.REMAINDER)
    execute.set_defaults(function=run)
    hashing = actions.add_parser('hash-file', help='Calculate the SHA-256 of an existing input file.')
    hashing.add_argument('--file', required=True, type=Path)
    hashing.set_defaults(function=hash_file)
    args = parser.parse_args()
    try:
        return args.function(args)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.exit(2, str(exc) + '\n')


if __name__ == '__main__':
    sys.exit(main())
