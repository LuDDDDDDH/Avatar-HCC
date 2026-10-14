from hashlib import sha256
from pathlib import Path

def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()

def prepare_outputs(outdir, names, inputs):
    source_paths = [Path(p).resolve() for p in inputs]
    for source in source_paths:
        if not source.is_file():
            raise FileNotFoundError(f'Input is not an existing file: {source.name}')
    output_directory = Path(outdir).resolve()
    targets = [output_directory / name for name in names]
    for target in targets:
        if target.resolve() in source_paths:
            raise ValueError('An output would overwrite an input')
        if target.exists():
            raise FileExistsError(f'Refusing to overwrite existing output: {target.name}')
    hashes = {str(p): digest(p) for p in source_paths}
    output_directory.mkdir(parents=True, exist_ok=True)
    return (targets, hashes)

def check_inputs_unchanged(hashes):
    for path, before in hashes.items():
        if digest(path) != before:
            raise RuntimeError(f'Input changed while processing: {Path(path).name}')
