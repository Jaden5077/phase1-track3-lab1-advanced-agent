from __future__ import annotations

import json
import os
import time

from dotenv import load_dotenv
from openai import OpenAI

from .prompts import ACTOR_SYSTEM, EVALUATOR_SYSTEM, REFLECTOR_SYSTEM
from .schemas import JudgeResult, QAExample, ReflectionEntry
from .utils import normalize_answer

load_dotenv()

# Real runtime does not hardcode per-qid failure modes.
FAILURE_MODE_BY_QID: dict[str, str] = {}


def _client() -> OpenAI:
    return OpenAI(
        base_url=os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:11434/v1"),
        api_key=os.getenv("OPENAI_API_KEY", "ollama"),
    )


def _model() -> str:
    return os.getenv("REFLEXION_MODEL", "qwen2.5-coder")


def _extract_json(text: str) -> dict:
    body = text.strip()
    if body.startswith("```"):
        lines = body.splitlines()
        if lines and lines[-1].strip() == "```":
            body = "\n".join(lines[1:-1])
        else:
            body = "\n".join(lines[1:])
    start = body.find("{")
    end = body.rfind("}") + 1
    if start == -1 or end <= start:
        raise ValueError(f"No JSON object found in response: {text[:200]}")
    return json.loads(body[start:end])


def actor_answer(
    example: QAExample,
    attempt_id: int,
    agent_type: str,
    reflection_memory: list[str],
) -> tuple[str, int, int]:
    context_text = "\n\n".join(f"[{c.title}]\n{c.text}" for c in example.context)
    reflection_hint = (
        "\n".join(f"- {item}" for item in reflection_memory)
        if reflection_memory
        else "(none)"
    )
    user_msg = (
        f"Attempt: {attempt_id}\n"
        f"Agent type: {agent_type}\n"
        f"Reflection memory:\n{reflection_hint}\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {example.question}\n\n"
        "Return only the final short answer."
    )

    t0 = time.perf_counter()
    response = _client().chat.completions.create(
        model=_model(),
        messages=[
            {"role": "system", "content": ACTOR_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.1,
        max_tokens=80,
    )
    latency_ms = int((time.perf_counter() - t0) * 1000)
    answer = (response.choices[0].message.content or "").strip()
    for prefix in ("Answer:", "Final answer:", "The answer is"):
        if answer.lower().startswith(prefix.lower()):
            answer = answer[len(prefix) :].strip()
    total_tokens = response.usage.total_tokens if response.usage else 0
    return answer, total_tokens, latency_ms


def evaluator(example: QAExample, answer: str) -> JudgeResult:
    if normalize_answer(example.gold_answer) == normalize_answer(answer):
        return JudgeResult(
            score=1,
            reason="Exact match after normalization.",
            missing_evidence=[],
            spurious_claims=[],
        )

    user_msg = (
        f"Question: {example.question}\n"
        f"Gold answer: {example.gold_answer}\n"
        f"Predicted answer: {answer}\n\n"
        "Return strict JSON with keys: score, reason, missing_evidence, spurious_claims."
    )
    try:
        response = _client().chat.completions.create(
            model=_model(),
            messages=[
                {"role": "system", "content": EVALUATOR_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            temperature=0,
            max_tokens=250,
        )
        data = _extract_json((response.choices[0].message.content or "").strip())
        score = 1 if int(data.get("score", 0)) == 1 else 0
        return JudgeResult(
            score=score,
            reason=str(data.get("reason", "No reason provided.")),
            missing_evidence=list(data.get("missing_evidence", [])),
            spurious_claims=list(data.get("spurious_claims", [])),
        )
    except Exception as exc:  # noqa: BLE001
        fallback_score = 1 if normalize_answer(example.gold_answer) == normalize_answer(answer) else 0
        return JudgeResult(
            score=fallback_score,
            reason=f"Fallback exact-match due to evaluator parsing/runtime error: {exc}",
            missing_evidence=[],
            spurious_claims=[],
        )


def reflector(example: QAExample, attempt_id: int, judge: JudgeResult) -> ReflectionEntry:
    user_msg = (
        f"Question: {example.question}\n"
        f"Gold answer: {example.gold_answer}\n"
        f"Attempt id: {attempt_id}\n"
        f"Judge reason: {judge.reason}\n"
        f"Missing evidence: {judge.missing_evidence}\n"
        f"Spurious claims: {judge.spurious_claims}\n\n"
        "Return strict JSON with keys: failure_reason, lesson, next_strategy."
    )
    try:
        response = _client().chat.completions.create(
            model=_model(),
            messages=[
                {"role": "system", "content": REFLECTOR_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.3,
            max_tokens=250,
        )
        data = _extract_json((response.choices[0].message.content or "").strip())
        return ReflectionEntry(
            attempt_id=attempt_id,
            failure_reason=str(data.get("failure_reason", judge.reason)),
            lesson=str(data.get("lesson", "Need stronger evidence-grounded reasoning.")),
            next_strategy=str(data.get("next_strategy", "Complete all reasoning hops before answering.")),
        )
    except Exception:
        return ReflectionEntry(
            attempt_id=attempt_id,
            failure_reason=judge.reason,
            lesson="The previous answer was not fully grounded in context.",
            next_strategy="Re-read context and explicitly complete all hops before the final answer.",
        )
