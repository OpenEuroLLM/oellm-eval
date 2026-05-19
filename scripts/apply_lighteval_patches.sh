#!/usr/bin/env bash
# Apply patches to the installed lighteval package in the active venv.
# Run this after setting up the venv with requirements-venv.txt.
#
# Usage:
#   source /path/to/venv/bin/activate
#   bash scripts/apply_lighteval_patches.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCHES_DIR="$SCRIPT_DIR/../patches"

# Find the installed lighteval templates directory
LIGHTEVAL_TEMPLATES=$(python3 -c "
import lighteval.tasks.templates.translation as t
import os; print(os.path.dirname(t.__file__))
" 2>/dev/null)

if [ -z "$LIGHTEVAL_TEMPLATES" ]; then
    echo "[error] Could not find lighteval installation. Is the venv active?"
    exit 1
fi

LIGHTEVAL_ROOT=$(dirname "$(dirname "$LIGHTEVAL_TEMPLATES")")
echo "[info] Found lighteval at: $LIGHTEVAL_ROOT"

# Apply the translation language-names patch
PATCH_FILE="$PATCHES_DIR/lighteval-translation-language-names.patch"
TARGET="$LIGHTEVAL_TEMPLATES/translation.py"

if python3 -c "
import langcodes
from lighteval.tasks.templates.translation import get_translation_prompt_function
from lighteval.tasks.templates.utils.formulation import CFFormulation
from lighteval.utils.language import Language
fn = get_translation_prompt_function(Language('eng'), Language('dan'), lambda l: {'source_text': l.get('s',''), 'target_text': l.get('t','')}, CFFormulation())
res = fn({'s': 'test', 't': 'test'}, 'test')
import json; q = json.loads(res)['query']
assert 'Danish' in q, f'Expected Danish in prompt, got: {q}'
print('[ok] Patch already applied (prompt uses full language names)')
" 2>/dev/null; then
    echo "[info] Patch already applied, skipping."
else
    echo "[apply] Patching $TARGET ..."
    patch -p1 -d "$LIGHTEVAL_ROOT" < "$PATCH_FILE"
    echo "[done] Patch applied successfully."
fi
