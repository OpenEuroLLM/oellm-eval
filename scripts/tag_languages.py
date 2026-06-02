"""Add `languages:` tags to multilingual tasks in task-groups.yaml.

Each task in a known multilingual task group encodes its language in its
`subset` (or task name), but across benchmarks the same language is spelled
several ways (e.g. German is ``deu_Latn``, ``de``, ``German`` and ``deu_latn``).
This script normalises all of those to a single canonical ``lang_Scri`` code
(flores style) and inserts an explicit ``languages:`` list onto every task, so
per-language task groups can be derived from a single source of truth.

It uses only PyYAML (already a project dependency) to *read* the language of
each task, then edits the file line-by-line so the diff is limited to the new
``languages:`` lines and existing comments/formatting are untouched. No extra
dependencies, runs against the prebuilt ``.venv``:

    .venv/bin/python scripts/tag_languages.py            # write tags
    .venv/bin/python scripts/tag_languages.py --check    # dry-run
"""

import argparse
import re
import sys
from pathlib import Path

import yaml

# Task groups whose tasks are language-specific and should be tagged.
MULTILINGUAL_GROUPS = {
    "sib200-eu",
    "belebele-eu-5-shot",
    "belebele-eu-cf",
    "flores-200-eu-to-eng",
    "flores-200-eng-to-eu",
    "global-mmlu-eu",
    "mgsm-eu",
    "arc-challenge-mt-eu",
    "include",
    "global-piqa-eu-completions",
    "global-piqa-eu-prompted",
}

# Map the various spellings used across benchmarks to the canonical lang_Scri code.
ALIAS = {
    # ISO 639-1 two-letter codes (global-mmlu, mgsm, arc-mt)
    "de": "deu_Latn",
    "fr": "fra_Latn",
    "it": "ita_Latn",
    "es": "spa_Latn",
    "pt": "por_Latn",
    "cs": "ces_Latn",
    "el": "ell_Grek",
    "lt": "lit_Latn",
    "nl": "nld_Latn",
    "pl": "pol_Latn",
    "ro": "ron_Latn",
    "ru": "rus_Cyrl",
    "sr": "srp_Cyrl",
    "sv": "swe_Latn",
    "tr": "tur_Latn",
    "uk": "ukr_Cyrl",
    "he": "heb_Hebr",
    "en": "eng_Latn",
    "bg": "bul_Cyrl",
    "da": "dan_Latn",
    "et": "est_Latn",
    "fi": "fin_Latn",
    "hu": "hun_Latn",
    "lv": "lvs_Latn",
    "sk": "slk_Latn",
    "sl": "slv_Latn",
    "is": "isl_Latn",
    "nb": "nob_Latn",
    # full English names (include)
    "albanian": "als_Latn",
    "armenian": "hye_Armn",
    "azerbaijani": "aze_Latn",
    "basque": "eus_Latn",
    "belarusian": "bel_Cyrl",
    "bulgarian": "bul_Cyrl",
    "croatian": "hrv_Latn",
    "dutch": "nld_Latn",
    "estonian": "est_Latn",
    "finnish": "fin_Latn",
    "french": "fra_Latn",
    "georgian": "kat_Geor",
    "german": "deu_Latn",
    "greek": "ell_Grek",
    "hungarian": "hun_Latn",
    "italian": "ita_Latn",
    "lithuanian": "lit_Latn",
    "north macedonian": "mkd_Cyrl",
    "polish": "pol_Latn",
    "portuguese": "por_Latn",
    "russian": "rus_Cyrl",
    "serbian": "srp_Cyrl",
    "spanish": "spa_Latn",
    "turkish": "tur_Latn",
    "ukrainian": "ukr_Cyrl",
}

# Distinct individual-language codes that should fold into a macrolanguage code.
SPECIAL = {"ekk_Latn": "est_Latn"}  # global-piqa uses ekk (Standard Estonian)

DEFAULT_PATH = (
    Path(__file__).resolve().parent.parent / "oellm" / "resources" / "task-groups.yaml"
)


def canon(code: str) -> str | None:
    """Normalise any spelling to a canonical ``lang_Scri`` code, or None."""
    code = str(code).strip()
    low = code.lower()
    if low in ALIAS:
        return ALIAS[low]
    # lowercase ``lang_scri`` or ``lang_scri_region`` (global-piqa)
    parts = low.split("_")
    if len(parts) >= 2 and len(parts[0]) == 3 and len(parts[1]) == 4:
        base = f"{parts[0]}_{parts[1].capitalize()}"
        return SPECIAL.get(base, base)
    # already canonical ``lang_Scri``
    if re.match(r"^[a-z]{3}_[A-Z][a-z]{3}$", code):
        return code
    return None


def flores_langs(task_name: str) -> list[str]:
    """Non-English side(s) of a ``flores200:src-tgt`` pair."""
    pair = task_name.split(":", 1)[1]
    return [lang for lang in pair.split("-") if lang != "eng_Latn"]


def resolve_languages(task: dict) -> list[str]:
    """Determine the canonical language code(s) for a single task entry."""
    name = task["task"]
    if name.startswith("flores200:"):
        return flores_langs(name)
    src = task.get("subset")
    if src is None:
        # fall back to a trailing two-letter code in the task name (e.g. ``_is``)
        m = re.search(r"_([a-z]{2})$", name)
        src = m.group(1) if m else None
    code = canon(src) if src is not None else None
    return [code] if code else []


def build_task_language_map(
    path: Path,
) -> tuple[dict[str, list[str]], list[tuple[str, str]]]:
    """Map ``task name -> languages`` for all multilingual tasks, plus unresolved."""
    data = yaml.safe_load(path.read_text()) or {}
    mapping: dict[str, list[str]] = {}
    unresolved: list[tuple[str, str]] = []
    for gname in MULTILINGUAL_GROUPS:
        for task in data["task_groups"][gname]["tasks"]:
            langs = resolve_languages(task)
            if langs:
                mapping[task["task"]] = langs
            else:
                unresolved.append((gname, task["task"]))
    return mapping, unresolved


# Matches a task list item, capturing indentation and the task name. Names may
# contain colons (``flores200:deu_Latn-eng_Latn``) or spaces
# (``include_base_44_north macedonian``), so capture up to the last non-space.
TASK_LINE = re.compile(r"^(?P<indent>\s*)- task:\s*(?P<name>.+?)\s*$")


def insert_tags(path: Path, mapping: dict[str, list[str]]) -> tuple[int, str]:
    """Insert a ``languages:`` line after each matching task line. Idempotent."""
    lines = path.read_text().splitlines(keepends=True)
    out: list[str] = []
    inserted = 0
    i = 0
    while i < len(lines):
        line = lines[i]
        out.append(line)
        m = TASK_LINE.match(line.rstrip("\n"))
        if m and m.group("name") in mapping:
            key_indent = m.group("indent") + "  "
            # Look ahead within this task block for an existing languages: key.
            already = False
            for nxt in lines[i + 1 :]:
                stripped = nxt.strip()
                if not stripped:
                    continue
                # left the task block (dedent to dash or shallower)
                cur_indent = len(nxt) - len(nxt.lstrip())
                if cur_indent < len(key_indent):
                    break
                if stripped.startswith("- task:"):
                    break
                if stripped.startswith("languages:"):
                    already = True
                    break
            if not already:
                codes = ", ".join(mapping[m.group("name")])
                out.append(f"{key_indent}languages: [{codes}]\n")
                inserted += 1
        i += 1
    return inserted, "".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path", type=Path, default=DEFAULT_PATH, help="Path to task-groups.yaml"
    )
    parser.add_argument(
        "--check", action="store_true", help="Dry run: report counts without writing."
    )
    args = parser.parse_args()

    mapping, unresolved = build_task_language_map(args.path)
    inserted, new_text = insert_tags(args.path, mapping)

    if not args.check:
        args.path.write_text(new_text)

    verb = "Would insert" if args.check else "Inserted"
    print(
        f"Resolved {len(mapping)} multilingual tasks; {verb} {inserted} languages: tags"
    )
    for group, task_name in unresolved:
        print(f"UNRESOLVED: {group} / {task_name}")

    return 1 if unresolved else 0


if __name__ == "__main__":
    sys.exit(main())
