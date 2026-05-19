# Flores200 Low BLEU Score Investigation

## Summary

The flores200 benchmark was producing anomalously low BLEU scores for several
European language pairs. Investigation revealed a **prompt format bug**: the
translation prompt used ambiguous 2-letter ISO language codes that the model
could not reliably associate with the correct target language.

## Observed Scores (Before Fix)

`openeurollm/datamix-9b-80-20` on `eng_Latn` → various targets:

| Target Language | BLEU   | chrF++ | Notes                     |
|----------------|--------|--------|---------------------------|
| French (FR)    | 34.47  | 72.55  | ✅ Works (code unambiguous) |
| Portuguese (PT)| 28.85  | 57.55  | ✅ Works                    |
| Dutch (NL)     | 15.60  | 61.32  | ⚠️  Degraded                |
| Croatian (HR)  | 10.81  | 62.60  | ❌ Wrong language output    |
| Czech (CS)     | 10.62  | 80.38  | ❌ Slovak/Croatian output   |
| Danish (DA)    |  5.05  | 17.43  | ❌ Portuguese output!       |
| Slovenian (SL) |  4.55  | 16.96  | ❌ Wrong language output    |
| Irish (GA)     |  1.84  | 18.12  | ❌ Effectively zero         |
| Estonian (ET)  |  1.53  | 18.60  | ❌ English/Spanish/Turkish  |

**Czech anomaly explained**: Czech and Slovak share ~80% character n-grams, so
when the model generates Slovak instead of Czech, chrF++ remains high but BLEU
is significantly lower.

## Root Cause

The lighteval translation prompt template used 2-letter ISO 639-1 codes:

```
EN: [source text] DA:
```

Most LLMs trained on general text do not reliably map these short codes to
specific languages. The ambiguity is severe for:
- `DA` — looks like a Portuguese/Spanish word meaning "gives/of"
- `ET` — French/Latin "and"; often ignored by the model
- `CS` — confusable with Czech/Slovak/Czechoslovak
- `GA` — extremely rare as a language code in LLM training data

The codes that **do** work (`FR`, `DE`, `PT`) are unambiguous because they
appear very frequently as country/language abbreviations in training corpora.

### Failure Modes Observed (Czech, 200 samples)

1. **Wrong Slavic language** (~significant %): Model generates Slovak or
   Croatian when prompted with `"CS:"`. Example:
   - Prompt: `EN: He is speculated to make a run for president in 2016. CS:`
   - Output: `On spekulira se da će se kandidirati za predsjednika 2016.` (Croatian!)

2. **Multi-language continuation** (~14%): Model generates a correct translation
   then continues with other language pairs on the same line:
   - Output: `Termín brouk se používá... SK: Termín "hmyz" sa používa...`
   - The `stop_sequence=["\n"]` doesn't stop this since it happens inline.

3. **English source echo** (~5-10%): Model repeats the English source verbatim:
   - Prompt: `EN: One was sent to George Washington... ET:`
   - Output: `One was sent to George Washington...`

### Danish (DA) — Most Severe Failure
The model almost entirely responds in Portuguese. The 2-letter code `DA` appears
to be interpreted as the Portuguese word "da" (preposition meaning "of/from"),
causing the model to generate Portuguese content.

## Fix Applied

Changed `lighteval/tasks/templates/translation.py` to use full English language
names via `langcodes.get(lang).display_name()`:

**Before:** `EN: [source text] DA:`
**After:**  `English: [source text] Danish:`

This makes the target language unambiguous for all flores200 language pairs.

### Files Changed

- `patches/lighteval-translation-language-names.patch` — the patch to apply to
  the installed lighteval package
- `scripts/apply_lighteval_patches.sh` — script to apply after venv setup

### How to Apply After Reinstalling the Venv

```bash
source /path/to/venv/bin/activate
bash scripts/apply_lighteval_patches.sh
```

## Enabling Sample-Level Debugging

The `--save-details` flag was added to `oellm/resources/template.sbatch` to
save per-sample model outputs as parquet files. Use `scripts/inspect_details.py`
to examine them:

```bash
python3 scripts/inspect_details.py /path/to/eval_results/TIMESTAMP/ --task ces --n 20
```

Parquet files are located at:
```
<results_dir>/<hash>/details/<model_name>/<datetime>/details_<task>|<shots>_<datetime>.parquet
```

Each row contains:
- `doc.query` — full prompt sent to the model
- `model_response.text` — raw model output (list)
- `model_response.text_post_processed` — post-processed output
- `doc.choices` — reference translations
- `metric` — per-sample bleu/chrf++ scores
