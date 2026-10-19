"""Dependency-free publication checks: syntax, local links, JSON, and asset boundaries.

No model imports, network, GPU calls, or dataset reads. Errors report paths, never
suspected credential values. Run from any directory with Python >=3.10 and bash.
"""
import ast
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BLOCKED_DIRS = {'.git', '__pycache__', '.pytest_cache', '.venv', 'runs', 'logs', 'data', 'Data',
                'datasets', 'external_data', 'models', 'checkpoints', 'cache', 'release', 'wandb'}
BLOCKED_SUFFIXES = {'.pt', '.pth', '.ckpt', '.safetensors', '.bin', '.npz', '.npy', '.mp4', '.mov',
                    '.avi', '.webm', '.tar', '.gz', '.zip', '.parquet', '.jsonl', '.pem', '.key'}


def main():
    # Inspect the Git publication file list, not ignored runtime/config directories.
    q = subprocess.run(['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'],
                       cwd=ROOT, check=True, capture_output=True)
    paths = sorted(set(x.decode() for x in q.stdout.split(b'\0') if x))
    errors = []; counts = {'python': 0, 'shell': 0, 'json': 0, 'markdown': 0, 'files': len(paths)}
    for rel in paths:
        p = ROOT / rel
        if p.is_symlink():
            errors.append(f'{rel}: symlinks not allowed in public snapshot'); continue
        if not p.is_file():
            errors.append(f'{rel}: missing file'); continue
        forbidden = (set(p.relative_to(ROOT).parts) & BLOCKED_DIRS or p.suffix in BLOCKED_SUFFIXES
                     or p.name.endswith('.env') or p.name == '.env' or p.name.startswith('.env.')
                     or p.name in {'cases.json', 'data_manifest.json', 'ego_manifest.json'}
                     or 'credentials' in p.name.lower())
        if forbidden:
            errors.append(f'{rel}: local asset/config must not be published'); continue
        if p.stat().st_size > 2 * 1024 * 1024:
            errors.append(f'{rel}: exceeds 2 MiB code/report limit'); continue
        if p.suffix not in {'.py', '.sh', '.md', '.json', '.txt', '.yml', '.yaml', '.example'}:
            continue
        text = p.read_text()
        # Literal secret formats, not environment-variable names or placeholder config.
        patterns = [r'gh[pousr]_[A-Za-z0-9]{30,}', r'github_pat_[A-Za-z0-9_]{40,}',
                    r'sk-(?:proj-)?[A-Za-z0-9_-]{32,}', r'AKIA[0-9A-Z]{16}',
                    r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
                    r'https?://[^\s/@:]+:[^\s/@]+@']
        if any(re.search(pat, text) for pat in patterns):
            errors.append(f'{rel}: possible embedded credential; content suppressed')
        try:
            if p.suffix == '.py':
                ast.parse(text, filename=rel); counts['python'] += 1
            elif p.suffix == '.sh':
                r = subprocess.run(['bash', '-n', str(p)], capture_output=True)
                if r.returncode:
                    errors.append(f'{rel}: bash syntax failed')
                counts['shell'] += 1
            elif p.suffix == '.json':
                json.loads(text); counts['json'] += 1
            elif p.suffix == '.md':
                counts['markdown'] += 1
                # Skip fenced code; check file targets, not generated heading anchors.
                body = re.sub(r'```.*?```', '', text, flags=re.S)
                for label, target in re.findall(r'\[([^\]\n]+)\]\(([^)\n]+)\)', body):
                    if target.startswith(('http://', 'https://', 'mailto:', '#')):
                        continue
                    dest = (p.parent / target.split('#')[0]).resolve()
                    if not dest.is_relative_to(ROOT) or not dest.exists():
                        errors.append(f'{rel}: broken/local-only link: {label}')
        except (SyntaxError, ValueError) as e:
            errors.append(f'{rel}: {type(e).__name__}')
    if errors:
        print('\n'.join(errors)); return 1
    print(json.dumps({'status': 'passed', **counts}, ensure_ascii=False)); return 0


if __name__ == '__main__':
    sys.exit(main())
