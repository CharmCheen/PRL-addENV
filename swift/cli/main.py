# Copyright (c) Alibaba, Inc. and its affiliates.
import importlib.util
import os
import socket
import subprocess
import sys
from typing import Dict, List, Optional

from swift.utils import get_logger

logger = get_logger()

ROUTE_MAPPING: Dict[str, str] = {
    'pt': 'swift.cli.pt',
    'sft': 'swift.cli.sft',
    'infer': 'swift.cli.infer',
    'merge-lora': 'swift.cli.merge_lora',
    'web-ui': 'swift.cli.web_ui',
    'deploy': 'swift.cli.deploy',
    'rlhf': 'swift.cli.rlhf',
    'sample': 'swift.cli.sample',
    'export': 'swift.cli.export',
    'eval': 'swift.cli.eval',
    'app': 'swift.cli.app',
}

ROUTE_MAPPING.update({k.replace('-', '_'): v for k, v in ROUTE_MAPPING.items()})


def use_torchrun() -> bool:
    nproc_per_node = os.getenv('NPROC_PER_NODE')
    nnodes = os.getenv('NNODES')
    if nproc_per_node is None and nnodes is None:
        return False
    return True


def _count_visible_gpus() -> int:
    cvis = os.getenv('CUDA_VISIBLE_DEVICES')
    if cvis is not None:
        cleaned = cvis.replace(' ', '')
        if cleaned:
            parts = [p for p in cleaned.split(',') if p != '']
            if parts:
                return len(parts)
    try:
        out = subprocess.check_output(['nvidia-smi', '-L'], text=True, stderr=subprocess.DEVNULL)
        lines = [line for line in out.splitlines() if line.strip()]
        if lines:
            return len(lines)
    except Exception:
        pass
    return 1


def _pick_free_port() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(('', 0))
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(sock.getsockname()[1])


def _normalize_ddp_env() -> None:
    visible_gpus = max(_count_visible_gpus(), 1)
    env_nproc = os.getenv('NPROC_PER_NODE')
    if env_nproc is None or env_nproc.strip() == '':
        os.environ['NPROC_PER_NODE'] = str(visible_gpus)
    else:
        try:
            nproc = int(env_nproc)
        except ValueError:
            logger.warning(f'Invalid NPROC_PER_NODE={env_nproc!r}; fallback to {visible_gpus}')
            nproc = visible_gpus
        if nproc < 1:
            logger.warning(f'NPROC_PER_NODE must be >=1; fallback to {visible_gpus}')
            nproc = visible_gpus
        if nproc > visible_gpus:
            logger.warning(
                f'NPROC_PER_NODE={nproc} exceeds visible GPUs ({visible_gpus}); '
                f'auto-downgrade to {visible_gpus}'
            )
            nproc = visible_gpus
        os.environ['NPROC_PER_NODE'] = str(nproc)

    if os.getenv('MASTER_PORT') is None:
        os.environ['MASTER_PORT'] = _pick_free_port()


def get_torchrun_args() -> Optional[List[str]]:
    if not use_torchrun():
        return
    _normalize_ddp_env()
    torchrun_args = []
    for env_key in ['NPROC_PER_NODE', 'MASTER_PORT', 'NNODES', 'NODE_RANK', 'MASTER_ADDR']:
        env_val = os.getenv(env_key)
        if env_val is None:
            continue
        torchrun_args += [f'--{env_key.lower()}', env_val]
    return torchrun_args


def _compat_web_ui(argv):
    # [compat]
    method_name = argv[0]
    if method_name in {'web-ui', 'web_ui'} and ('--model' in argv or '--adapters' in argv or '--ckpt_dir' in argv):
        argv[0] = 'app'
        logger.warning('Please use `swift app`.')


def cli_main() -> None:
    argv = sys.argv[1:]
    _compat_web_ui(argv)
    method_name = argv[0]
    argv = argv[1:]
    file_path = importlib.util.find_spec(ROUTE_MAPPING[method_name]).origin
    torchrun_args = get_torchrun_args()
    python_cmd = sys.executable
    if torchrun_args is None or method_name not in {'pt', 'sft', 'rlhf', 'infer'}:
        args = [python_cmd, file_path, *argv]
    else:
        args = [python_cmd, '-m', 'torch.distributed.run', *torchrun_args, file_path, *argv]
    print(f"run sh: `{' '.join(args)}`", flush=True)
    result = subprocess.run(args)
    if result.returncode != 0:
        sys.exit(result.returncode)


if __name__ == '__main__':
    cli_main()
