# UrbanCars (OpenCLIP)

Two-attribute spurious robustness, replicating Sarridis et al., *Scaling
Vision-Language Models Fails to Mitigate Bias* (ACM MM 2026).
Reference implementation: <https://github.com/gsarridis/vlm-spurious-robustness> (MIT).

Shares its scoring code, prompts and metric with
[`spurious_robustness`](../spurious_robustness/README.md), which covers ImageNet
and CelebA. Read that README for the prompting policy, the interpretation
baselines, and how worst-group accuracy is computed.

**Pinned metric:** `worst_group_accuracy` (minimum over 8 groups)

```bash
oellm-eval schedule \
  --models laion/CLIP-ViT-B-32-laion2B-s34B-b79K \
  --task-groups spurious-urbancars \
  --venv-path ~/spurious-venv
```

Both suites together:

```bash
oellm-eval schedule \
  --models laion/CLIP-ViT-B-32-laion2B-s34B-b79K \
  --task-groups spurious-robustness,spurious-urbancars \
  --venv-path ~/spurious-venv
```

## Why this is a separate suite

`CLUSTER_ENV_VARS` is validated for *every* task in a suite. UrbanCars is the
only benchmark of the three needing a cluster-local directory, so declaring
`URBANCARS_DATA_DIR` alongside ImageNet and CelebA would have failed those two
on any cluster without an UrbanCars tree. As its own suite, the variable is
checked by the login-node pre-flight before submission — the operator learns at
schedule time rather than after a job has been queued and started.

## Group definitions

Label is car type; background and co-occurring object are both spurious. The 8
groups are the 2x2x2 product, read from the directory names:

```
obj={urban|country}, bg={urban|country}, co={urban|country}
```

The minimum is set by the groups where both shortcuts contradict the label —
`obj=urban, bg=country, co=country` and its mirror.

A group contributing zero samples is absent from the minimum rather than scored
0.0. Runs covering fewer than 8 groups log a warning and record `n_groups`,
because a missing group can only make the minimum look better.

Baselines: random 50%, majority class 50%, and a degenerate always-one-class
predictor scores **0% worst-group** — that is the signal the benchmark exists to
produce.

## Data — this dataset must be built, not downloaded

There is no published UrbanCars dataset. It does not exist on the Hugging Face
Hub, and the reference implementation only *consumes* a prebuilt tree. It is
composited by [Whac-A-Mole](https://github.com/facebookresearch/Whac-A-Mole) via
`scripts/prepare_dataset_models/create_urbancars.sh`, which requires:

- **Stanford Cars** — `cars_train.tgz`, `cars_test.tgz`, `car_devkit.tgz`
  (the original Stanford URLs have been offline since 2024; a mirror is needed)
- **COCO 2017** train + val images
- **LVIS** v1 train/val annotation JSON
- **Places365-standard**, 256x256
- **MaskFormer** panoptic checkpoint plus a clone of its repo (needs detectron2)

The script then runs car-mask segmentation on GPU and composites the result.
That is tens of gigabytes of input and a dependency set that cannot share this
suite's venv, which is why it is out of scope for automatic staging here.

Once built, point the cluster at the test split:

```yaml
URBANCARS_DATA_DIR: "/path/to/urbancars/bg-0.5_co_occur_obj-0.5/test"
```

Expected layout — eight directories of `*.jpg`:

```
obj-urban_bg-urban_co_occur_obj-urban/
obj-urban_bg-urban_co_occur_obj-country/
obj-urban_bg-country_co_occur_obj-urban/
obj-urban_bg-country_co_occur_obj-country/
obj-country_bg-urban_co_occur_obj-urban/
obj-country_bg-urban_co_occur_obj-country/
obj-country_bg-country_co_occur_obj-urban/
obj-country_bg-country_co_occur_obj-country/
```

The label comes from the directory name alone, so the layout *is* the ground
truth: a renamed directory silently relabels its images.

If the built tree is uploaded to an internal Hub dataset repo, it can be staged
automatically like ImageNet and CelebA — that only needs a `dataset_specs` entry
on the task, plus a decision about hosting and redistribution, since the result
is derived from four separately-licensed sources.

## Parity

Verified against the reference implementation by running its
`UrbanCarsBenchmark` on an identical eight-directory tree in the same process:
all 15 shared metrics match exactly, including every per-group accuracy.
