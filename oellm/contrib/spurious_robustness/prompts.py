"""Class prompts for the two spurious-correlation benchmarks.

The wording is reproduced verbatim from the reference implementation
(https://github.com/gsarridis/vlm-spurious-robustness, MIT licensed) because it
is the wording that produced the published numbers. Changing any string here
silently makes our scores incomparable with the paper's, so treat these as
frozen: a new phrasing belongs in a new task, not in an edit to these lists.

ImageNet deliberately has no entry. It is classified with OpenCLIP's built-in
``IMAGENET_CLASSNAMES`` under all 80 ``OPENAI_IMAGENET_TEMPLATES``, which is the
standard Radford et al. (2021) protocol and is imported directly rather than
copied.
"""

from __future__ import annotations

# UrbanCars: urban vs. country car type.
#
# The car subtypes stand in for the class names so that neither spurious
# attribute (background, co-occurring object) ever appears in the prompt —
# otherwise the prompt would leak the very shortcut the benchmark measures.
URBANCARS_PROMPTS: dict[str, list[str]] = {
    "urban": ["a photograph of a compact, sports, sedan car"],
    "country": ["a photograph of a truck, jeep, pickup car"],
}

# CelebA: blonde vs. non-blonde hair.
#
# Non-blonde is enumerated as concrete hair colours instead of a negation,
# which CLIP-style text encoders handle poorly ("not blonde" embeds close to
# "blonde"). Scoring takes the maximum over a class's prompts, not the mean —
# see ``zeroshot.class_scores``.
CELEBA_PROMPTS: dict[str, list[str]] = {
    "blonde": [
        "a photo of a person with blonde hair",
        "a photo of a person with light blonde hair",
        "a photo of a person with golden hair",
        "a photo of a person with platinum blonde hair",
    ],
    "non-blonde": [
        "a photo of a person with dark hair",
        "a photo of a person with black hair",
        "a photo of a person with brown hair",
        "a photo of a brunette person",
        "a photo of a person with red hair",
        "a photo of a person with grey hair",
        "a photo of a bald person",
        "a photo of a person with auburn hair",
    ],
}
