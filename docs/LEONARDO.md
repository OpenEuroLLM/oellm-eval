# Leonardo HPC — Environment Setup Guide

## 1. Setting Up an Account

Create your account at the CINECA UserDB portal:
[https://userdb.hpc.cineca.it/](https://userdb.hpc.cineca.it/)

Once you have set up your UserDB profile and obtained your CINECA account name, follow the steps below.

---

## 2. SSH Access (macOS)

### Install `step` CLI
```zsh
brew install step
```

> If Homebrew is not yet installed, run the following first:
> ```zsh
> /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
> ```

### Bootstrap the CA
```zsh
step ca bootstrap \
  --ca-url=https://sshproxy.hpc.cineca.it \
  --fingerprint 2ae1543202304d3f434bdc1a2c92eff2cd2b02110206ef06317e70c1c1735ecd
```

### Generate an SSH Certificate
Replace `your.name@email.com` with the email you used when registering on UserDB.

> ⚠️ Generate the key **into `~/.ssh/`**, never into a project directory — a
> private key inside a git repository is one `git add -A` away from being
> committed and leaked.

```zsh
step ssh certificate your.name@email.com --provisioner cineca-hpc ~/.ssh/cineca_hpc
```

This writes the private key `~/.ssh/cineca_hpc` plus `cineca_hpc.pub` and
`cineca_hpc-cert.pub`. The certificate is short-lived (~12 h) — rerun the
command when it expires.

### Configure SSH
Add the following to `~/.ssh/config` (create it if it doesn't exist via `nano ~/.ssh/config`):
```
Host leonardo
    HostName login07-ext.leonardo.cineca.it
    User your_cineca_username
    IdentityFile ~/.ssh/cineca_hpc
```

Set correct permissions:
```zsh
chmod 600 ~/.ssh/config
```

You can now connect simply with:
```zsh
ssh leonardo
```

---

## 3. Setting Up the Python Environment on the Cluster

### Load Python module
```zsh
module purge
module load python/3.11.7
```

### Install `uv` via a bootstrap environment
Since the cluster does not have `uv` available by default and PyPI access is restricted outside a venv, first create a temporary environment to install `uv`:
```zsh
python -m venv elliot-env
source elliot-env/bin/activate
pip install uv
```

### Make `uv` permanently available
Copy the `uv` binary to `~/.local/bin` so it persists across all environments:
```zsh
mkdir -p ~/.local/bin
cp $HOME/elliot-env/bin/uv ~/.local/bin/uv
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

Verify:
```zsh
uv --version
```

### Install Python 3.12 via `uv`
The project requires Python 3.12, which is not available as a cluster module. `uv` can fetch it directly:
```zsh
uv python install 3.12
```

### Create the project environment
```zsh
uv venv -p 3.12 elliot-venv
source elliot-venv/bin/activate
```

---

## 4. Install the Project

### Clone the repository
```zsh
git clone https://github.com/elliot-project/elliot-cli.git
```

### Install in editable mode
```zsh
uv pip install -e ./elliot-cli
```

---

## 5. Set HuggingFace Cache Directory

Compute nodes have no internet access, so all models and datasets must be pre-downloaded. Set `HF_HOME` to point to a storage area large enough to hold your models and datasets.

> **Warning:** `$HOME` on Leonardo has a quota of only **50 GB** — far too small for most LLMs and multimodal datasets. Do **not** use `$HOME` as your HF cache directory.

Leonardo provides several larger storage areas ([full details](https://docs.hpc.cineca.it/hpc/hpc_data_storage.html)):

| Area | Quota | Purge policy | Notes |
|------|-------|--------------|-------|
| `$WORK` | 1 TB | 6 months post-project | Persistent, parallel I/O, **recommended** |
| `$FAST` | 1 TB | 6 months post-project | Faster I/O than `$WORK`, no extension option |
| `$SCRATCH` | 20 TB | Files purged after **40 days** of inactivity | Temporary only |

**Recommended:** use `$WORK` for persistent caches (models you reuse across runs):

```zsh
mkdir -p $WORK/hf_cache
export HF_HOME="$WORK/hf_cache"
echo 'export HF_HOME="$WORK/hf_cache"' >> ~/.bashrc
source ~/.bashrc
```

If you need more space for a single campaign and don't need the cache long-term, `$SCRATCH` (up to 20 TB) is an option — but files are **automatically deleted after 40 days of inactivity** and must not be kept alive artificially with `touch`:

```zsh
mkdir -p $SCRATCH/hf_cache
export HF_HOME="$SCRATCH/hf_cache"
```

---

## 6. Verify the Setup

Before scheduling anything, run the built-in diagnostic — it checks cluster
detection, required environment variables, your HF cache, SLURM binaries, and
(with `--venv-path`) which eval engines your venv actually provides:

```zsh
oellm-eval doctor
# with a custom venv and the groups you plan to run:
oellm-eval doctor --venv-path /path/to/.venv --task-groups "open-sci-0.01"
```

---

## 7. Running Evaluations

```zsh
# Run evaluations using a task group (recommended)
oellm-eval schedule \
    --models "microsoft/DialoGPT-medium,EleutherAI/pythia-160m" \
    --task-groups "open-sci-0.01"

# Or specify individual tasks
oellm-eval schedule \
    --models "EleutherAI/pythia-160m" \
    --tasks "hellaswag,mmlu" \
    --n-shot 5
```

---

## Additional Resources

Full instructions are also available here:
[https://iffmd.fz-juelich.de/e-hu5RBHRXG6DTgD9NVjig#Leonardo-Access-and-Usage-LAION-Open-Psi-open-sci-Ontocord-AI-openEuroLLM](https://iffmd.fz-juelich.de/e-hu5RBHRXG6DTgD9NVjig#Leonardo-Access-and-Usage-LAION-Open-Psi-open-sci-Ontocord-AI-openEuroLLM)
