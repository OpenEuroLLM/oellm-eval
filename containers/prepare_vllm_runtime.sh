#!/usr/bin/env bash
# Run inside an explicitly selected, verified vLLM+Ray image. No GPU required.
# The scratch venv is isolated; engine packages and the base image stay intact.
set -euo pipefail
if [[ ${1:-} != --execute || $# != 5 ]]; then
  echo 'Usage: prepare_vllm_runtime.sh --execute WORK_DIR HARNESS_SOURCE EVALCHEMY_SOURCE HUMANEVAL_SOURCE'
  exit 2
fi
work_dir=$(realpath "$2")
harness_source=$(realpath "$3")
evalchemy_source=$(realpath "$4")
humaneval_source=$(realpath "$5")
source /opt/ray-venv/bin/activate
[[ $(command -v python) == /opt/ray-venv/bin/python ]]
export PYTHONNOUSERSITE=1
# The base image prepends its Ray venv through PYTHONPATH. Clear that override
# so the isolated venv owns pip; base-ray.pth adds Ray after local packages.
unset PYTHONPATH
export PIP_CACHE_DIR="$work_dir/pip-cache"
export TMPDIR="$work_dir/tmp"
mkdir -p "$TMPDIR" "$work_dir/provenance"
if [[ -e "$work_dir/venv" ]]; then
  echo 'Refusing to overwrite an existing runtime; use a fresh work directory.' >&2
  exit 2
fi
python -m venv --system-site-packages "$work_dir/venv"
source "$work_dir/venv/bin/activate"
[[ $(command -v python) == "$work_dir/venv/bin/python" ]]
python - "$work_dir" "$humaneval_source" <<'PY'
import importlib.metadata as m
import json
from pathlib import Path
import sys
import sysconfig
root = Path(sys.argv[1])
site = Path(sysconfig.get_path('purelib'))
(site / 'base-ray.pth').write_text('/opt/ray-venv/lib/python3.12/site-packages\n')
(site / 'human-eval-source.pth').write_text(sys.argv[2]+'\n')
assert Path(m.distribution('pip').locate_file('')).is_relative_to(sys.prefix)
PY
python - "$work_dir" <<'PY'
import importlib.metadata as m
import json
from pathlib import Path
import sys
root = Path(sys.argv[1])
versions = {d.metadata['Name']: d.version for d in m.distributions()}
(root / 'provenance/base-packages.json').write_text(json.dumps(versions, indent=2)+'\n')
protected = {'vllm', 'torch', 'torchvision', 'torchaudio', 'transformers', 'numpy', 'ray', 'triton'}
constraints = [f'{n}=={v}' for n,v in sorted(versions.items())
               if n.lower() in protected or n.lower().startswith(('nvidia-', 'flashinfer', 'flash-attn'))]
(root / 'engine-constraints.txt').write_text('\n'.join(constraints)+'\n')
PY
python -m pip --version
# Legacy pure-Python build scripts may import pkg_resources. Propagate a build
# constraint into pip's temporary build environments. HumanEval is imported
# directly from its pinned source (its published CLI entry point is malformed).
printf '%s\n' 'setuptools==80.9.0' > "$work_dir/build-constraints.txt"
export PIP_CONSTRAINT="$work_dir/build-constraints.txt"
# Source builds are allowed only for these small pure-Python packages. Native
# dependencies must have wheels: never compile CUDA or other native code here.
python -m pip install --only-binary=:all: \
  --no-binary=rouge-score,sqlitedict,word2number,langdetect,human-eval \
  --constraint "$work_dir/engine-constraints.txt" \
  --report "$work_dir/provenance/install.json" \
  "$harness_source[api,math,ifeval]" 'datasets>=4,<5' \
  'fire==0.7.1' 'SQLAlchemy>=2,<3' scipy jsonargparse || {
    python -m pip check > "$work_dir/provenance/pip-check.txt" 2>&1 || true
    exit 1
  }
python -m pip check > "$work_dir/provenance/pip-check.txt" 2>&1 || true
python - "$work_dir" <<'PY'
import importlib.metadata as m
import json
from pathlib import Path
import sys
root=Path(sys.argv[1])
lines=(root/'provenance/pip-check.txt').read_text().splitlines()
known='torch 2.13.0+cu130 has requirement nvidia-nccl-cu13==2.29.7; platform_system == "Linux", but you have nvidia-nccl-cu13 2.30.7.'
unexpected=[line for line in lines if line not in {known, 'No broken requirements found.'}]
if unexpected:
    raise SystemExit('Unexpected dependency conflicts: '+repr(unexpected))
versions={d.metadata['Name']:d.version for d in m.distributions()}
(root/'provenance/packages.json').write_text(json.dumps(versions, indent=2)+'\n')
PY
python -m pip freeze --all > "$work_dir/provenance/pip-freeze.txt"
export PYTHONPATH="$evalchemy_source"
export HF_HOME="$work_dir/hf-cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export NLTK_DATA="$work_dir/nltk_data"
python -m nltk.downloader -d "$NLTK_DATA" punkt punkt_tab
python -c 'import vllm, ray; from lm_eval.models.vllm_causallms import VLLM; from eval import eval; print("Engine, adapter and Evalchemy CLI imports passed")'
python "$evalchemy_source/tests/test_vllm_dp_context.py"
python "$evalchemy_source/tests/test_math_answer_parser.py"
python -m eval.eval --help > "$work_dir/provenance/evalchemy-help.txt"
python -m lm_eval --help > "$work_dir/provenance/harness-help.txt"
touch "$work_dir/provenance/prepare-complete"
