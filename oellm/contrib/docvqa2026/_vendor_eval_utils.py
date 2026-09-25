"""Official DocVQA 2026 scorer and prompt, copied verbatim below this docstring
from https://github.com/VLR-CVC/DocVQA2026 (eval_utils.py). Do not edit or
reformat: any change alters competition scores. Excluded from ruff in
pyproject.toml.
"""

import Levenshtein
import re
import string
import ast
from dateutil import parser
from datetime import date

# --- 1. ANLS HELPER FUNCTIONS ---
def get_anls(s1, s2):
    s1 = s1.lower().strip()
    s2 = s2.lower().strip()
    if not s1 and not s2: return 1.0
    if not s1 or not s2: return 0.0
    dist = Levenshtein.distance(s1, s2)
    max_len = max(len(s1), len(s2))
    return 1.0 - (dist / max_len)

def is_string_correct(prediction, ground_truths, threshold=0.80):
    max_score = 0.0
    for gt in ground_truths:
        score = get_anls(prediction, gt)
        if score > max_score:
            max_score = score
    return max_score >= threshold

# --- 2. UNIT PARSING LOGIC ---
def parse_magnitude_unit(text):
    """
    Splits a string into a numeric float and a unit string.
    """
    text = text.lower().strip()
    
    match = re.match(r'^(-?\d+(?:\.\d+)?)\s*(.*)$', text) # Check reaction to sections (S.O.)
    
    if not match:
        return None, None
    
    number_str = match.group(1)
    unit_str = match.group(2).strip()
    
    try:
        val = float(number_str)
        return val, unit_str
    except ValueError:
        return None, None

# --- 3. MAIN EVALUATION ---
def evaluate_docvqa_prediction(raw_prediction, ground_truth):

    if not isinstance(raw_prediction, str):
        raw_prediction = str(raw_prediction)

    marker = "FINAL ANSWER:"
    if marker not in raw_prediction:
        return False, raw_prediction

    extracted_answer = raw_prediction.split(marker)[-1].strip()
    
    gt_candidates = []
    try:
        parsed_gt = ast.literal_eval(str(ground_truth))
        if isinstance(parsed_gt, list):
            gt_candidates = [str(x) for x in parsed_gt]
        else:
            gt_candidates = [str(ground_truth)]
    except (ValueError, SyntaxError):
        gt_candidates = [str(ground_truth)]

    # --- A. STRICT MATCHING LOGIC ---
    def check_strict_match(pred_text, gt_text):
        """
        Returns True ONLY if:
        1. Both are valid numbers.
        2. The numeric values are equal.
        3. The units are identical.
        """
        pred_val, pred_unit = parse_magnitude_unit(pred_text)
        gt_val, gt_unit = parse_magnitude_unit(gt_text)
        
        # Check Number + Unit Match
        if pred_val is not None and gt_val is not None:
            if pred_val == gt_val:
                if pred_unit == gt_unit:
                    return True
                return False
        
        # Check Date Match
        try:
            p_clean = pred_text.strip()
            g_clean = gt_text.strip()
            version_regex = r'^\d+\.\d+\.\d+$'
            if re.match(version_regex, p_clean) or re.match(version_regex, g_clean):
                return p_clean == g_clean
            
            if len(p_clean) >= 6 and len(g_clean) >= 6: # Is >= 6 necessary? If yes is not >=6 12.0.0 has 6 characters (S.O.)
                pred_date = parser.parse(p_clean, fuzzy=False).date()
                gt_date = parser.parse(g_clean, fuzzy=False).date()
                return pred_date == gt_date
            
        except (ValueError, TypeError, OverflowError):
            pass
            
        return False

    # --- B. EXECUTION ---
    # 1. Try Strict Match & Detect Numeric GT

    # ----- This section is unnecesary (S.O.) You can just check the first element
    gt_is_numeric = False
    
    for gt in gt_candidates:
        if check_strict_match(extracted_answer, gt):
            return True, extracted_answer
        
        gt_val, gt_unit = parse_magnitude_unit(gt)
        if gt_val is not None:
            gt_is_numeric = True

    if gt_is_numeric:
        return False, extracted_answer
    #-----

    # 3. RELAXED TEXT MATCH (ANLS) 
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    
    def clean_text(text):
        t = text.lower().translate(translator)
        t = re.sub(r'\b(a|an|the)\b', ' ', t)
        return " ".join(t.split())

    clean_pred = clean_text(extracted_answer)
    clean_gt_candidates = [clean_text(gt) for gt in gt_candidates]

    is_correct = is_string_correct(clean_pred, clean_gt_candidates, threshold=0.9)
    
    return is_correct, extracted_answer

def get_evaluation_prompt() -> str:
    MASTER_PROMPT = (
        "ACT AS an expert Document Visual Question Answering (DocVQA) system. "
        "ANALYZE the provided images to extract precise information.\n\n"
        "### MANDATORY RESPONSE RULES:\n"
        "1. SOURCE ADHERENCE: If the question is unanswerable from the document, respond ONLY with \"Unknown\".\n"
        "2. LIST FORMATTING: List multiple answers in order of appearance, separated by a comma and a single space (e.g., \"Answer A, Answer B\"). Do NOT use \"and\".\n"
        "3. NUMBERS & UNITS:\n"
        "   - Convert units to their standardized abbreviation (e.g., use \"kg\" not \"kilograms\", \"m\" not \"meters\").\n"
        "   - Place a single space between the number and the unit (e.g., \"50 kg\", \"10 USD\").\n"
        "4. PERCENTAGES: For percentages, attach the '%' symbol directly to the number with NO space (e.g., \"50%\", not \"50 %\").\n"
        "5. DATE FORMATTING: Convert all dates to YYYY-MM-DD format (e.g., convert \"Jan 1st 24\" to \"2024-01-01\").\n"
        "6. DECIMAL FORMATTING: Decimals should be separated by a single period (e.g., \"3.14\", not \"3,14\").\n"
        "7. THOUSANDS SEPARATOR: Do NOT use commas as thousands separators (e.g., \"1000\", not \"1,000\").\n"
        "8. NO FILLER: Output ONLY the result. Do not frame with sentences like \"The answer is...\"."

        "\n\n### REASONING PROTOCOL:\n"
        "1. Perform exhaustive step-by-step reasoning to locate and verify the data.\n"
        "2. Verify if the data contains a date, number, or unit.\n"
        "3. Step-by-step, transform the data to match the MANDATORY RESPONSE RULES (e.g., converting date format).\n"

        "\n\n### OUTPUT FORMAT:\n"
        "After your analysis, you MUST provide the final result in the following format:\n"
        "FINAL ANSWER: [Your exact formatted answer]\n"
        "Ensure the content inside [FINAL ANSWER] strictly follows the MANDATORY RESPONSE RULES."
    )
    
    return MASTER_PROMPT
