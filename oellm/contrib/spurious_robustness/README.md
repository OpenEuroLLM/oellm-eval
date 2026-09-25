# Spurious robustness (OpenCLIP)

Zero-shot robustness to spurious correlations, replicating Sarridis et al.,
*Scaling Vision-Language Models Fails to Mitigate Bias* (ACM MM 2026).
Reference implementation: <https://github.com/gsarridis/vlm-spurious-robustness> (MIT).

| Task | Spurious attributes | Pinned metric | Groups |
|---|---|---|---|
| `spurious_imagenet` | none | `top1_accuracy` | 1 |
| `spurious_celeba` | one (gender) | `worst_group_accuracy` | 4 |
| `spurious_urbancars` | two (background, co-occurring object) | `worst_group_accuracy` | 8 |

## What this suite evaluates

CLIP-family **dual encoders**, not generative VLMs. An image is assigned to the
class whose text embedding is nearest in cosine similarity. That is the paper's
protocol, and it is why this is a contrib suite: neither lm-eval nor lmms-eval
can score by embedding similarity, and lm-eval's multimodal backends raise on
loglikelihood requests carrying an image, so likelihood-ranked multiple choice
is not an alternative either.

Scores from this suite are comparable with the paper's. They are **not**
comparable with numbers obtained by prompting a generative VLM to name a class
on the same images.

## Group definitions

The worst-group number is the **minimum accuracy over the groups below**, and
over nothing else. Average accuracy is over all images, not the mean of the
per-group accuracies — the groups are heavily imbalanced, so the two differ.

**CelebA — 4 groups.** Label is hair colour (`Blond_Hair`), spurious attribute
is gender (`Male`). Evaluated on the official CelebA **test** split (19,962
images).

| group | images | share |
|---|---:|---:|
| `non-blonde_female` | 9,767 | 48.93% |
| `non-blonde_male` | 7,535 | 37.75% |
| `blonde_female` | 2,480 | 12.42% |
| `blonde_male` | **180** | **0.90%** |

The minimum is almost always set by `blonde_male`, the rarest combination — 180
images, under 1% of the split. A worst-group number therefore rests on a small
sample, and `--limit` runs frequently drop the group entirely.

**UrbanCars — 8 groups.** Label is car type; background and co-occurring object
are both spurious. Groups are the 2x2x2 product, read from the directory names:

```
obj={urban|country}, bg={urban|country}, co={urban|country}
```

The minimum is set by the groups where both shortcuts contradict the label
(`obj=urban, bg=country, co=country` and its mirror).

**ImageNet — 1 group.** No spurious attribute, so worst-group accuracy equals
top-1 by construction. It is reported for uniformity; `top1_accuracy` is the
pinned metric.

A group that contributes **zero samples** is absent from the minimum rather than
scored 0.0 — an unobserved group has no accuracy to be worst, and scoring it as
a total failure would make every subsampled run report 0.0. Runs that cover
fewer groups than the benchmark defines log a warning and record `n_groups`,
because a missing group can only make the minimum look better.

## Prompting and scoring policy

Fixed for every model; changing any of it makes scores incomparable.

- **CelebA / UrbanCars** score a class as the **maximum** cosine similarity over
  that class's prompts. Non-blonde is enumerated as concrete hair colours rather
  than a negation, which CLIP text encoders handle poorly. The UrbanCars prompts
  name car subtypes so that neither spurious attribute ever appears in the
  prompt — otherwise the prompt would leak the shortcut being measured.
- **ImageNet** uses OpenCLIP's `IMAGENET_CLASSNAMES` under all 80
  `OPENAI_IMAGENET_TEMPLATES`, **averaged** per class (`build_zero_shot_classifier`).

The max/average split between benchmarks is deliberate and must not be unified.
Exact prompt strings are in `prompts.py` and are frozen.

## Interpreting a score

| | random | majority class | always one class (worst-group) |
|---|---|---|---|
| ImageNet (1000-way) | 0.10% | 0.10% | — |
| CelebA (2-way) | ~50% | **86.67%** (always non-blonde) | **0%** |
| UrbanCars (2-way) | 50% | 50% | **0%** |

The last column is the point of the benchmark: a model that always answers
"non-blonde" scores 86.67% average accuracy on CelebA and **0% worst-group**,
because both blonde groups score zero. A large gap between average and
worst-group means the model is riding the spurious attribute. Read the two
numbers together — an average accuracy near 87% on CelebA is exactly what a
model that has learned nothing but the class prior would produce.

On ImageNet both baselines are 0.10% because the validation split is exactly
balanced at 50 images per class, so a score in the low single digits is
near-chance rather than a weak result.

For reference, `laion/CLIP-ViT-B-32-laion2B-s34B-b79K` on the full CelebA test
split scores 89.58% average and 80.00% worst-group (`blonde_male`) — above the
majority-class baseline on both counts.

## Parity with the reference implementation

This suite is checked against the reference by running *its* benchmark classes
on identical inputs in the same process and comparing every metric.

**CelebA — exact, on the full 19,962-image test split.** All eleven shared
metrics match bit-for-bit with `laion/CLIP-ViT-B-32-laion2B-s34B-b79K`
(worst-group 0.7944444444444444 on `blonde_male`, average 0.8964532611962729).
Label and group assignment were separately verified element-wise across all
19,962 images.

**UrbanCars — exact, on a real eight-directory tree.** All eight per-group
accuracies, the average, the worst-group value and the worst group itself match
the reference. Only `.jpg` files are samples: the generator writes a
`_mask.png` and a `_co_occur_obj_mask.png` beside every composited image, and
those masks are segmentation output, not photographs. Fixtures include them so
file selection is exercised.

**ImageNet — exact, on a synthetic 1000-synset tree.** Top-1, top-5 and image
count all match. The mapping is the part that matters here: the reference maps
synset to class index through its own `data/imagenet_synsets.txt`, while this
suite relies on ascending synset directory order. That file is sorted, so the
two agree — checked rather than assumed, since a mismatch would silently
scramble all 1000 labels. Images were spread across the whole index range so a
mis-ordered mapping could not pass unnoticed.

One finding worth keeping in mind when changing this code: encoding must run
under `autocast` because the reference does. Without it, embeddings shift by
~1e-2 and the argmax flips on borderline samples. That moved CelebA worst-group
by 0.56 points — a single image out of 180. Because the rarest group holds only
180 images, worst-group accuracy has ~0.6-point granularity and is far more
sensitive to numerical drift than average accuracy is. Expect small movement
between CPU and CUDA runs for the same reason.

## Data

| Dataset | Source | Status |
|---|---|---|
| ImageNet | `ILSVRC/imagenet-1k`, validation | **Gated** — terms accepted per researcher under their own HF account, not by the institution. Or set `IMAGENET_VAL_DIR` to a local ImageFolder tree. |
| CelebA | `tpremoli/CelebA-attrs` | Ungated parquet mirror. CelebA itself is **non-commercial research only**, accepted per researcher. |
| UrbanCars | `URBANCARS_DATA_DIR` | **Not published.** Composited from Stanford Cars + Places + LVIS via [Whac-A-Mole](https://github.com/facebookresearch/Whac-A-Mole) at `bg-0.5_co_occur_obj-0.5`. |

Two sourcing traps, both load-bearing:

- The CelebA mirror stores attributes in the original **−1/+1** encoding, not the
  0/1 that `torchvision.datasets.CelebA` produces. Testing `== 0` for a negative
  class matches nothing and yields a silently empty group.
- The mirror's split **names are swapped** relative to official CelebA. The
  official test split (19,962 images) is published here as `validation`; the
  split named `test` is the official validation partition (19,867 images).

Compute nodes run with `HF_HUB_OFFLINE=1`. ImageNet and CelebA are declared as
dataset specs and pre-staged on the login node. UrbanCars is a local tree and is
never fetched, so it must already exist on the cluster filesystem.

## Cluster setup

Add to `clusters.yaml` for any cluster that runs UrbanCars:

```yaml
URBANCARS_DATA_DIR: "/path/to/urbancars/bg-0.5_co_occur_obj-0.5/test"
```

Neither variable is listed in `CLUSTER_ENV_VARS`: the dispatcher treats that
list as required for *every* task in the suite, so listing them would break
CelebA and ImageNet runs on clusters without an UrbanCars tree. `run()` raises
with the same guidance when the variable is actually needed.

## Install

```bash
uv pip install '.[spurious-robustness]'
```
