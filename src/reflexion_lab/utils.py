from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Iterable
from .schemas import ContextChunk, QAExample, RunRecord

def normalize_answer(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text

def _looks_like_hotpot_qa_item(item: object) -> bool:
    if not isinstance(item, dict):
        return False
    ctx = item.get("context")
    return "id" in item and "answer" in item and isinstance(ctx, dict) and "sentences" in ctx

def _hotpot_item_to_qa(item: dict) -> QAExample:
    """Map HotpotQA JSON (id, answer, level, context.title + context.sentences) to QAExample."""
    ctx = item["context"]
    titles = ctx["title"]
    sentences = ctx["sentences"]
    if len(titles) != len(sentences):
        msg = f"context title/sentences length mismatch: {len(titles)} vs {len(sentences)}"
        raise ValueError(msg)
    chunks: list[ContextChunk] = []
    for title, sents in zip(titles, sentences):
        if isinstance(sents, str):
            text = sents
        else:
            text = "".join(sents)
        chunks.append(ContextChunk(title=str(title), text=text))
    return QAExample(
        qid=str(item["id"]),
        difficulty=item["level"],
        question=item["question"],
        gold_answer=item["answer"],
        context=chunks,
    )

def _parse_qa_item(item: object) -> QAExample:
    if _looks_like_hotpot_qa_item(item):
        return _hotpot_item_to_qa(item)  # type: ignore[arg-type]
    if isinstance(item, dict):
        return QAExample.model_validate(item)
    raise TypeError(f"Expected dict for dataset row, got {type(item).__name__}")

def load_dataset(path: str | Path) -> list[QAExample]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise TypeError("Dataset root must be a JSON array")
    return [_parse_qa_item(item) for item in raw]

def save_jsonl(path: str | Path, records: Iterable[RunRecord]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(record.model_dump_json() + "\n")
