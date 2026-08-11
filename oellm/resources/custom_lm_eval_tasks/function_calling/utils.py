import json
import re


def _extract_user_content(question):
    if isinstance(question, str):
        return question
    if isinstance(question, list):
        for msg in reversed(question):
            if isinstance(msg, dict) and msg.get("role") == "user":
                return msg.get("content", "")
        if question and isinstance(question[-1], dict):
            return question[-1].get("content", "")
    return str(question)


def _format_functions(functions):
    if isinstance(functions, str):
        try:
            functions = json.loads(functions)
        except (json.JSONDecodeError, ValueError):
            return functions
    if not isinstance(functions, list):
        functions = [functions]

    lines = []
    for fn in functions:
        if not isinstance(fn, dict):
            continue
        name = fn.get("name", "unknown")
        desc = fn.get("description", "")
        params = fn.get("parameters", {})
        properties = params.get("properties", {}) if isinstance(params, dict) else {}
        required = params.get("required", []) if isinstance(params, dict) else []

        param_parts = []
        for param_name, param_info in properties.items():
            param_type = param_info.get("type", "any") if isinstance(param_info, dict) else "any"
            suffix = "" if param_name in required else "?"
            param_parts.append(f"{param_name}: {param_type}{suffix}")

        params_str = ", ".join(param_parts)
        lines.append(f"- {name}({params_str}): {desc}")

    return "\n".join(lines)


def _extract_answer(answer):
    if isinstance(answer, list) and answer:
        answer = answer[0]
    if isinstance(answer, dict):
        name = answer.get("name", "")
        args = answer.get("arguments", {})
        if isinstance(args, dict):
            arg_str = ", ".join(f"{k}={repr(v)}" for k, v in args.items())
        else:
            arg_str = str(args)
        return f"{name}({arg_str})"
    return str(answer).strip()


def doc_to_text(doc):
    question = doc.get("question", doc.get("query", ""))
    functions = doc.get("function", doc.get("functions", []))

    user_content = _extract_user_content(question)
    funcs_text = _format_functions(functions)

    return (
        f"Available functions:\n{funcs_text}\n\n"
        f"User: {user_content}\n\n"
        f"Function call:"
    )


def doc_to_target(doc):
    answer = doc.get("answer", doc.get("answers", ""))
    return _extract_answer(answer)


def process_results(doc, results):
    pred = results[0].strip() if results else ""
    target = doc_to_target(doc)

    fn_match = re.match(r"(\w+)\s*\(", target)
    expected_fn = fn_match.group(1) if fn_match else target.strip()

    pred_fn_match = re.search(r"(\w+)\s*\(", pred)
    pred_fn = pred_fn_match.group(1) if pred_fn_match else pred.split("(")[0].strip()

    return {
        "function_name_acc": int(pred_fn.lower() == expected_fn.lower()),
        "exact_match": int(pred == target),
    }
