"""Dataset loading and group assignment for the three benchmarks.

Every loader is a generator of ``(images, labels, groups)`` batches so that a
50k-image split never has to be held in memory at once. ``groups[i]`` is the
subgroup string for sample ``i``; this is the only place where the spurious
attributes are still available, so the group must be attached here.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from itertools import product
from pathlib import Path

Batch = tuple[list, list[int], list[str]]

# CelebA: hair colour is the label, gender is the spurious attribute.
CELEBA_CLASS_NAMES = ("blonde", "non-blonde")  # index 0 / 1, matching CELEBA_PROMPTS
_CELEBA_GENDER = {0: "female", 1: "male"}

# UrbanCars: car type is the label; background and co-occurring object are both
# spurious. The eight subgroup directory names are the 2x2x2 product.
URBANCARS_CLASS_NAMES = ("urban", "country")  # index 0 / 1, matching URBANCARS_PROMPTS
_URBANCARS_ATTRIBUTES = ("urban", "country")

# The official CelebA test split (19,962 images) is the one the paper evaluates.
# In this mirror it is published under the name "validation" — the repo's
# "test" split is the official *validation* partition (19,867 images). Using the
# split named "test" would silently evaluate the wrong 19,867 images.
CELEBA_REPO = "tpremoli/CelebA-attrs"
CELEBA_SPLIT = "validation"

IMAGENET_REPO = "ILSVRC/imagenet-1k"
IMAGENET_SPLIT = "validation"

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".JPEG")

URBANCARS_EXTENSIONS = (".jpg",)


def _batched(iterable, size: int):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def celeba_label_and_group(blond_attr: int, male_attr: int) -> tuple[int, str]:
    """Map raw CelebA attribute values to (class index, group name).

    The mirror stores attributes in CelebA's original -1/+1 encoding, not the
    0/1 that ``torchvision.datasets.CelebA`` converts to. Testing ``== 0`` for a
    negative class therefore matches nothing and silently produces an empty
    group, so both attributes are read as ``== 1`` and negated explicitly.
    """
    is_blonde = int(blond_attr == 1)
    is_male = int(male_attr == 1)
    # Class index 0 is "blonde"; the attribute is 1 when the hair *is* blonde.
    return (
        1 - is_blonde,
        f"{CELEBA_CLASS_NAMES[1 - is_blonde]}_{_CELEBA_GENDER[is_male]}",
    )


def load_celeba(limit: int | None = None, batch_size: int = 32) -> Iterator[Batch]:
    """CelebA official test split, grouped by hair colour x gender."""
    from datasets import load_dataset

    ds = load_dataset(CELEBA_REPO, split=CELEBA_SPLIT)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    for rows in _batched(ds, batch_size):
        images, labels, groups = [], [], []
        for row in rows:
            label, group = celeba_label_and_group(row["Blond_Hair"], row["Male"])
            labels.append(label)
            groups.append(group)
            images.append(row["image"])
        yield images, labels, groups


def load_imagenet(
    data_dir: str | None = None, limit: int | None = None, batch_size: int = 32
) -> Iterator[Batch]:
    """ILSVRC-2012 validation images.

    ImageNet has no spurious attribute, so every sample lands in a single group
    named ``all`` and worst-group accuracy degenerates to top-1 by construction.

    Reads a local ImageFolder tree when *data_dir* is given, otherwise the HF
    parquet copy (which is gated — the operator must have accepted its terms).
    """
    if data_dir:
        yield from _load_imagenet_folder(data_dir, limit, batch_size)
        return

    from datasets import load_dataset

    ds = load_dataset(IMAGENET_REPO, split=IMAGENET_SPLIT)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    for rows in _batched(ds, batch_size):
        images = [r["image"] for r in rows]
        labels = [int(r["label"]) for r in rows]
        yield images, labels, ["all"] * len(rows)


def _load_imagenet_folder(
    data_dir: str, limit: int | None, batch_size: int
) -> Iterator[Batch]:
    """``<data_dir>/<synset>/*.JPEG``.

    Standard ImageNet class indices are assigned in ascending synset order
    (``n01440764`` -> 0), which is also the order of OpenCLIP's
    ``IMAGENET_CLASSNAMES``. Sorting the synset directories therefore reproduces
    the canonical index without needing a separate mapping file.
    """
    root = Path(data_dir)
    synsets = sorted(p.name for p in root.iterdir() if p.is_dir())
    if len(synsets) != 1000:
        raise ValueError(
            f"expected 1000 synset directories under {data_dir!r}, found {len(synsets)}"
        )

    def samples():
        for idx, synset in enumerate(synsets):
            for fname in sorted(os.listdir(root / synset)):
                if fname.endswith(IMAGE_EXTENSIONS):
                    yield str(root / synset / fname), idx

    stream = samples()
    seen = 0
    for rows in _batched(stream, batch_size):
        if limit is not None and seen >= limit:
            return
        if limit is not None:
            rows = rows[: limit - seen]
        seen += len(rows)
        yield [p for p, _ in rows], [i for _, i in rows], ["all"] * len(rows)


def urbancars_subgroup_dirs(data_root: str) -> dict[str, str]:
    """Map subgroup directory name -> path, for the subgroups that exist."""
    found = {}
    for obj, bg, co in product(_URBANCARS_ATTRIBUTES, repeat=3):
        name = f"obj-{obj}_bg-{bg}_co_occur_obj-{co}"
        path = os.path.join(data_root, name)
        if os.path.isdir(path):
            found[name] = path
    return found


def urbancars_group_label(dirname: str) -> tuple[str, int]:
    """``obj-urban_bg-country_co_occur_obj-country`` -> (readable group, class index)."""
    parts = dirname.split("_")
    obj = parts[0].removeprefix("obj-")
    bg = parts[1].removeprefix("bg-")
    co = parts[-1].removeprefix("obj-")
    if obj not in URBANCARS_CLASS_NAMES:
        raise ValueError(f"unrecognised subgroup directory: {dirname}")
    return f"obj={obj}, bg={bg}, co={co}", URBANCARS_CLASS_NAMES.index(obj)


def load_urbancars(
    data_root: str, limit: int | None = None, batch_size: int = 32
) -> Iterator[Batch]:
    """UrbanCars test split laid out as eight subgroup directories.

    The label comes from the directory name alone, so the layout *is* the
    ground truth: a renamed directory silently relabels its images. The caller
    is responsible for checking that all eight subgroups were found — see
    ``suite.run``.
    """
    subgroups = urbancars_subgroup_dirs(data_root)
    if not subgroups:
        raise FileNotFoundError(
            f"no UrbanCars subgroup directories under {data_root!r}; expected "
            "obj-<urban|country>_bg-<urban|country>_co_occur_obj-<urban|country>"
        )

    def samples():
        for dirname in sorted(subgroups):
            group, label = urbancars_group_label(dirname)
            for fname in sorted(os.listdir(subgroups[dirname])):
                if fname.endswith(URBANCARS_EXTENSIONS):
                    yield os.path.join(subgroups[dirname], fname), label, group

    stream = samples()
    seen = 0
    for rows in _batched(stream, batch_size):
        if limit is not None and seen >= limit:
            return
        if limit is not None:
            rows = rows[: limit - seen]
        seen += len(rows)
        yield (
            [p for p, _, _ in rows],
            [lbl for _, lbl, _ in rows],
            [g for _, _, g in rows],
        )
