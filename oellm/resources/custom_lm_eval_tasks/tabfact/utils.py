"""TabFact prompt construction.

``wenhu/tab_fact`` stores tables in ``table_text``: rows separated by
newlines, cells by ``#``, first row is the header. Serialized here as a
pipe table capped at ``_MAX_ROWS`` rows so pathological tables cannot blow
the context window (the cap is stated in the prompt when it triggers).
"""

_MAX_ROWS = 30


def _serialize_table(table_text: str) -> str:
    rows = [r for r in str(table_text).split("\n") if r.strip()]
    lines: list[str] = []
    for i, row in enumerate(rows):
        cells = [c.strip() for c in row.split("#")]
        lines.append(" | ".join(cells))
        if i == 0 and len(rows) > 1:
            lines.append("-" * min(80, max(3, len(lines[0]))))
        if i + 1 >= _MAX_ROWS and len(rows) > _MAX_ROWS:
            lines.append(f"... ({len(rows) - _MAX_ROWS} more rows)")
            break
    return "\n".join(lines)


def doc_to_text(doc) -> str:
    caption = str(doc.get("table_caption", "") or "").strip()
    table = _serialize_table(doc.get("table_text", ""))
    head = f"Table caption: {caption}\n" if caption else ""
    return (
        f"{head}Table:\n{table}\n"
        f"Statement: {doc['statement']}\n"
        "Question: Is the statement entailed or refuted by the table?\n"
        "Answer:"
    )
