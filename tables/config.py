import argparse
import json
import os
from pathlib import Path


def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=os.environ.get('AVATAR_TABLE_CONFIG'))
    args = parser.parse_args()
    if not args.config:
        parser.error('--config is required')
    path = Path(args.config).resolve()
    config = json.loads(path.read_text(encoding='utf-8'))
    root = path.parent
    workspace = (root / config['workspace']).resolve()

    def file_path(key, output=False):
        result = (root / config['files'][key]).resolve()
        if output:
            if result.exists():
                raise ValueError('Output already exists')
            if not result.is_relative_to(workspace):
                raise ValueError('Output must be within workspace')
            code_root = Path(__file__).resolve().parents[1]
            if result.is_relative_to(code_root):
                raise ValueError('Output cannot overwrite code')
            inputs = [value for name, value in config['files'].items()
                      if name not in {'sourceData', 'authoredChanges', 'assemblyPlan',
                                      'nmOutput', 'inspectionErrors'}]
            if result in [(root / value).resolve() for value in inputs]:
                raise ValueError('Output cannot overwrite an input')
            result.parent.mkdir(parents=True, exist_ok=True)
        return result

    return config, file_path
