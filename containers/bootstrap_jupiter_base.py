"""Build the JUPITER vLLM+Ray base from public, immutable inputs.

No existing SIF or model-deployment checkout is required. Dry-run by default.
The emitted profile records this build's SIF hash: OCI-to-SIF conversion is
not byte reproducible because Apptainer embeds build metadata.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import urllib.request

from build_vllm_image import sha

OCI = "docker.io/vllm/vllm-openai@sha256:2a7cde230b59f3ce6cab33dd245ba6bee41aa87b38c9fe84f966ff24016813ce"
VLLM_COMMIT = "2cf0a6915ce544dc493a0990f2ea38d81601128a"
WHEEL = "ray-2.48.0-cp312-cp312-manylinux2014_aarch64.whl"
WHEEL_URL = "https://files.pythonhosted.org/packages/b4/a6/e7c969bd371c65b7c233d86f23610489e15164ee7eadb3eb78f9d55eda4d/" + WHEEL
WHEEL_SHA = "f1cf33d260316f92f77558185f1c36fc35506d76ee7fdfed9f5b70f9c4bdba7f"
NCCL_EXCEPTION = 'torch 2.13.0+cu130 has requirement nvidia-nccl-cu13==2.29.7; platform_system == "Linux", but you have nvidia-nccl-cu13 2.30.7.'


def definition(wheel):
    if any(c.isspace() for c in str(wheel)):
        raise ValueError("Build paths must not contain whitespace")
    return f'''Bootstrap: docker
From: {OCI}

%files
    {wheel} /opt/wheels/{WHEEL}

%post
    set -eu
    python3 -m venv --system-site-packages /opt/ray-venv
    . /opt/ray-venv/bin/activate
    test "$(command -v python)" = /opt/ray-venv/bin/python
    python -m pip install --no-index --no-deps /opt/wheels/{WHEEL}
    python -m pip check > /opt/ray-pip-check.txt 2>&1 || true
    python - <<'PY'
from pathlib import Path
lines=Path('/opt/ray-pip-check.txt').read_text().splitlines()
assert lines == [{NCCL_EXCEPTION!r}], lines
PY
    rm /opt/wheels/{WHEEL}

%environment
    export PATH=/opt/ray-venv/bin:$PATH
    export PYTHONPATH=/opt/ray-venv/lib/python3.12/site-packages
    export PYTHONNOUSERSITE=1
    export RAY_USAGE_STATS_ENABLED=0

%labels
    oellm.base.oci {OCI}
    oellm.base.vllm_commit {VLLM_COMMIT}
    oellm.ray.wheel_sha256 {WHEEL_SHA}

%test
    /opt/ray-venv/bin/python - <<'PY'
import os, shutil, subprocess, tempfile
import vllm, ray, transformers
assert vllm.__version__ == '0.28.0'
assert transformers.__version__ == '5.15.1'
assert ray.__version__ == '2.48.0'
help_text=subprocess.check_output(['/opt/ray-venv/bin/ray','start','--help'],text=True)
assert '--include-dashboard' in help_text and '--disable-usage-stats' in help_text
tmp=tempfile.mkdtemp(prefix='ray-',dir='/tmp')
try:
    ray.init(num_cpus=1,include_dashboard=False,_temp_dir=tmp)
    @ray.remote
    def ping(): return 'ok'
    assert ray.get(ping.remote()) == 'ok'
finally:
    ray.shutdown()
    shutil.rmtree(tmp)
PY
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--tmp-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    work, tmp = args.work.resolve(), args.tmp_dir.resolve()
    if not tmp.is_relative_to('/tmp') or tmp == Path('/tmp'):
        parser.error("Use a dedicated local /tmp directory for --tmp-dir")
    image = work / 'vllm-v0.28.0-ray2.48.0-arm64.sif'
    recipe = definition(work / WHEEL)
    command = ['apptainer', 'build', '--mksquashfs-args', '-processors 2', str(image), str(work / 'base.def')]
    plan = {'oci': OCI, 'ray_url': WHEEL_URL, 'ray_sha256': WHEEL_SHA,
            'command': command, 'definition': recipe, 'work': str(work)}
    print(json.dumps(plan, indent=2), flush=True)
    if not args.execute:
        return
    if platform.machine() != 'aarch64':
        parser.error('This recipe requires an aarch64 host')
    # Failed builds remain available for diagnosis; never overwrite them.
    work.mkdir(parents=True, exist_ok=False)
    tmp.mkdir(parents=True, mode=0o700, exist_ok=True)
    wheel = work / WHEEL
    urllib.request.urlretrieve(WHEEL_URL, wheel)
    if wheel.stat().st_size != 69151702 or sha(wheel) != WHEEL_SHA:
        parser.error('Ray wheel size/hash mismatch')
    (work / 'base.def').write_text(recipe)
    env = dict(os.environ, APPTAINER_TMPDIR=str(tmp), TMPDIR=str(tmp),
               APPTAINER_CACHEDIR=str(work / 'apptainer-cache'))
    subprocess.run(command, env=env, check=True)
    labels = json.loads(subprocess.check_output(['apptainer','inspect','--json',str(image)],text=True))['data']['attributes']['labels']
    for key, value in {'org.opencontainers.image.revision': VLLM_COMMIT,
                       'org.label-schema.build-arch': 'arm64', 'oellm.base.oci': OCI,
                       'oellm.ray.wheel_sha256': WHEEL_SHA}.items():
        if labels.get(key) != value:
            parser.error(f'Unexpected image label: {key}')
    profile = {'architecture':'aarch64', 'base_filename':image.name,
               'base_bytes':image.stat().st_size, 'base_sha256':sha(image),
               'vllm':'0.28.0', 'ray':'2.48.0', 'source':OCI}
    (work / 'profiles.json').write_text(json.dumps({'jupiter':profile},indent=2)+'\n')
    plan.update(profile=profile, labels=labels, definition_sha256=sha(work/'base.def'),
                builder_sha256=sha(Path(__file__)))
    (work / 'provenance.json').write_text(json.dumps(plan,indent=2)+'\n')


if __name__ == '__main__':
    main()
