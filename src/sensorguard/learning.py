"""Interactive learning check for the ML-readiness gate."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


QUESTIONS = (
    "Why are TWF, HDF, PWF, OSF, and RNF excluded from the model features?",
    "What is the difference between precision and recall?",
    "Why is the probability threshold selected on validation data instead of test data?",
    "Why does the random forest outperforming the PyTorch MLP matter?",
    "What happens during a PyTorch forward pass, backward pass, and optimizer step?",
    "Name one limitation that prevents the test result from proving real-factory performance.",
)


def save_answers(answers: list[str], output_path: str | Path) -> dict[str, object]:
    if len(answers) != len(QUESTIONS):
        raise ValueError(f"expected {len(QUESTIONS)} answers")
    cleaned = [answer.strip() for answer in answers]
    if any(not answer for answer in cleaned):
        raise ValueError("every learning-check answer must be non-empty")
    payload: dict[str, object] = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "awaiting_review",
        "responses": [
            {"question": question, "answer": answer}
            for question, answer in zip(QUESTIONS, cleaned, strict=True)
        ],
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def run_interactive_check(output_path: str | Path) -> Path:
    print("SensorGuard ML-readiness check")
    print("Answer in your own words. Short, specific answers are better than memorized paragraphs.\n")
    answers: list[str] = []
    for index, question in enumerate(QUESTIONS, start=1):
        print(f"{index}. {question}")
        answer = input("> ").strip()
        while not answer:
            print("Please enter an answer before continuing.")
            answer = input("> ").strip()
        answers.append(answer)
        print()
    save_answers(answers, output_path)
    destination = Path(output_path)
    print(f"Saved your answers to {destination}")
    print("Ask Codex to review that file before starting the medical ML project.")
    return destination

