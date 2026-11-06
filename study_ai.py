"""LLM-assisted chapter titles, summaries, keywords, and review questions."""

import json
import re


def _extract_object(text: str) -> dict | None:
    """Accept plain JSON or a fenced/model-prefixed response."""
    match = re.search(r"\{.*\}", text or "", re.S)
    if not match:
        return None
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def enrich_chapters(chapters: list[dict], segs: list[tuple], pairs: list[tuple], chat) -> list[dict]:
    """Ask an injected chat function to improve an offline chapter outline.

    The input is deliberately capped so multi-hour videos stay within ordinary
    context windows. Invalid or partial model output leaves offline data intact.
    """
    inputs = []
    for chapter in chapters[:30]:
        snippets = []
        for (start, _end, _), (original, zh) in zip(segs, pairs):
            if chapter["start"] <= start <= chapter["end"]:
                snippets.append((zh or original).strip())
        transcript = " ".join(x for x in snippets if x)
        inputs.append({
            "number": chapter["number"],
            "start": chapter["start"],
            "transcript": transcript[:1800],
        })
    prompt = json.dumps(inputs, ensure_ascii=False)
    response = chat([
        {
            "role": "system",
            "content": (
                "你是课程编辑。根据按章提供的字幕，为每章生成准确、具体、适合学习的中文信息。"
                "只返回JSON对象，格式为 {\"chapters\":[{\"number\":1,\"title\":\"不超过18字\","
                "\"summary\":\"2至3句核心内容\",\"keywords\":[\"3至5个关键词\"],"
                "\"questions\":[\"1至2个可由本章回答的复习题\"]}]}。"
                "不得添加字幕中没有的信息，不要输出Markdown。"
            ),
        },
        {"role": "user", "content": prompt},
    ])
    parsed = _extract_object(response)
    if not parsed or not isinstance(parsed.get("chapters"), list):
        return chapters
    by_number = {x.get("number"): x for x in parsed["chapters"] if isinstance(x, dict)}
    result = []
    for chapter in chapters:
        item = by_number.get(chapter["number"], {})
        updated = dict(chapter)
        for field in ("title", "summary"):
            if isinstance(item.get(field), str) and item[field].strip():
                updated[field] = item[field].strip()[:500]
        for field in ("keywords", "questions"):
            values = item.get(field)
            if isinstance(values, list):
                updated[field] = [str(x).strip()[:100] for x in values if str(x).strip()][:5]
        result.append(updated)
    return result
