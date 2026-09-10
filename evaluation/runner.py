"""Offline comparison runner with fresh mode workers and shared input manifest."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import platform
from importlib.metadata import version, PackageNotFoundError
from tempfile import TemporaryDirectory
from time import perf_counter
from uuid import uuid4

from .claims import compare
from .harness import Mode
from .media import prepare_media, file_hash


def run(path, *, config, kind='image', start=0, stop=None, stride=1, modes=None, timeout=3600):
    modes = [Mode(mode) for mode in (modes if modes is not None else list(Mode))]
    if not modes or len(set(modes)) != len(modes):
        raise ValueError('provide distinct nonempty modes')
    root = Path(__file__).resolve().parent.parent
    # Hash existing implementation files for reproducibility and scope auditing.
    sources = [root / 'live_app.py', *sorted((root / 'fifth_layer').rglob('*.py'))]
    hashes = {str(p.relative_to(root)): file_hash(p) for p in sources}
    run_id = uuid4().hex
    before = perf_counter()
    with prepare_media(path, kind=kind, start=start, stop=stop, stride=stride) as frames:
        prep_seconds = perf_counter() - before
        results, workers = [], []
        with TemporaryDirectory(prefix='aisthesis-eval-control-') as temporary:
            for mode in modes:
                request_path = Path(temporary) / 'request.json'
                result_path = Path(temporary) / f'{mode.value}.json'
                request_path.write_text(json.dumps(dict(frames=frames, mode=mode.value,
                    config=config, run_id=run_id)), encoding='utf-8')
                environment = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                                   HF_HUB_DISABLE_TELEMETRY='1')
                before_worker = perf_counter()
                try:
                    process = subprocess.run([sys.executable, '-m', 'evaluation.worker',
                        str(request_path), str(result_path)], cwd=root, env=environment,
                        capture_output=True, text=True, timeout=timeout)
                except subprocess.TimeoutExpired:
                    workers.append(dict(mode=mode.value, wall_seconds=perf_counter() - before_worker,
                                        returncode=None, stdout='', stderr='worker timeout'))
                    continue
                workers.append(dict(mode=mode.value, wall_seconds=perf_counter() - before_worker,
                                    returncode=process.returncode, stdout=process.stdout, stderr=process.stderr))
                if process.returncode == 0:
                    results.extend(json.loads(result_path.read_text(encoding='utf-8')))
        comparisons = [compare([r for r in results if r['input']['sequence'] == frame['sequence']])
                       for frame in frames if any(r['input']['sequence'] == frame['sequence'] for r in results)]
    after = {str(p.relative_to(root)): file_hash(p) for p in sources}
    libraries = {}
    for package in ('torch', 'transformers', 'ultralytics', 'opencv-python', 'Pillow'):
        try:
            libraries[package] = version(package)
        except PackageNotFoundError:
            libraries[package] = None
    return dict(schema_version='evaluation-0.1', run_id=run_id, config=config,
        environment=dict(python=sys.version, platform=platform.platform(), libraries=libraries),
        preparation_seconds=prep_seconds, implementation_sha256=hashes,
        production_files_unchanged=hashes == after, workers=workers, results=results,
        comparisons=comparisons, status='ok' if all(w['returncode'] == 0 for w in workers)
        and all(r['status'] == 'ok' for r in results) and hashes == after else 'incomplete_or_rejected',
        interpretation='Infrastructure run only; no ground-truth quality or scientific benefit measured.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input')
    parser.add_argument('--kind', choices=['image', 'video'], default='image')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--stop', type=int)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--mode', choices=[m.value for m in Mode], action='append')
    parser.add_argument('--detector-model', default='yolo11n.pt')
    parser.add_argument('--vlm-model', help='Existing local SmolVLM model snapshot directory')
    parser.add_argument('--detector-device', default='cpu')
    parser.add_argument('--vlm-device', default='cpu')
    parser.add_argument('--detector-confidence', type=float, default=.4)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if Path(args.output).exists():
        parser.error('output already exists; choose a new result path')
    if not 0 <= args.detector_confidence <= 1:
        parser.error('detector-confidence must be between 0 and 1')
    modes = args.mode or [m.value for m in Mode]
    if any(m != Mode.FIFTH_LAYER_ONLY.value for m in modes) and not args.vlm_model:
        parser.error('VLM modes require --vlm-model pointing to an existing local snapshot')
    config = dict(detector_model=str(Path(args.detector_model).resolve()),
        vlm_model=str(Path(args.vlm_model).resolve()) if args.vlm_model else None,
        detector_device=args.detector_device, vlm_device=args.vlm_device,
        detector_confidence=args.detector_confidence)
    result = run(args.input, config=config, kind=args.kind, start=args.start, stop=args.stop,
                 stride=args.stride, modes=modes)
    with open(args.output, 'x', encoding='utf-8') as output:
        json.dump(result, output, indent=2, allow_nan=False)
    print(f'{result["status"]}: {args.output}')
    return 0 if result['status'] == 'ok' else 1


if __name__ == '__main__':
    raise SystemExit(main())
