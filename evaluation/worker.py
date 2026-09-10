"""Internal process entry point; only one mode per interpreter."""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('request')
    parser.add_argument('result')
    args = parser.parse_args()
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1')
    from .adapters import RepositoryAdapters
    from .harness import execute
    request = json.loads(Path(args.request).read_text(encoding='utf-8'))
    result = execute(request['frames'], request['mode'], RepositoryAdapters(request['config']),
                     run_id=request['run_id'], isolation='fresh_process')
    Path(args.result).write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
