"""Write small dependency-free SVG summaries."""

from __future__ import annotations

import html
from pathlib import Path


def save_model_comparison(
    scores: dict[str, float],
    output_path: str | Path,
    *,
    metric_name: str,
) -> None:
    if not scores:
        raise ValueError("at least one score is required")
    width = 760
    row_height = 62
    padding = 170
    height = 70 + row_height * len(scores)
    rows: list[str] = []
    for index, (name, score) in enumerate(scores.items()):
        if not 0.0 <= score <= 1.0:
            raise ValueError("scores must be between zero and one")
        y = 55 + index * row_height
        bar_width = score * (width - padding - 45)
        rows.append(
            f'<text x="12" y="{y+21}" font-family="sans-serif" font-size="15">{html.escape(name)}</text>'
            f'<rect x="{padding}" y="{y}" width="{bar_width:.2f}" height="28" fill="#2563eb"/>'
            f'<text x="{padding+bar_width+8:.2f}" y="{y+20}" font-family="sans-serif" font-size="14">{score:.3f}</text>'
        )
    markup = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{width/2}" y="28" text-anchor="middle" font-family="sans-serif" font-size="20">Validation {html.escape(metric_name)}</text>
{''.join(rows)}
</svg>
"""
    Path(output_path).write_text(markup, encoding="utf-8")


def save_confusion_matrix(
    matrix: list[list[int]],
    output_path: str | Path,
    *,
    title: str,
) -> None:
    if len(matrix) != 2 or any(len(row) != 2 for row in matrix):
        raise ValueError("binary confusion matrix must be 2x2")
    labels = (("TN", matrix[0][0]), ("FP", matrix[0][1]), ("FN", matrix[1][0]), ("TP", matrix[1][1]))
    cells: list[str] = []
    for index, (label, value) in enumerate(labels):
        column = index % 2
        row = index // 2
        x = 180 + column * 180
        y = 90 + row * 120
        fill = "#dbeafe" if label in {"TN", "TP"} else "#fee2e2"
        cells.append(
            f'<rect x="{x}" y="{y}" width="160" height="100" fill="{fill}" stroke="#334155"/>'
            f'<text x="{x+80}" y="{y+38}" text-anchor="middle" font-family="sans-serif" font-size="16">{label}</text>'
            f'<text x="{x+80}" y="{y+72}" text-anchor="middle" font-family="sans-serif" font-size="24">{value}</text>'
        )
    markup = f"""<svg xmlns="http://www.w3.org/2000/svg" width="700" height="380" viewBox="0 0 700 380">
<rect width="100%" height="100%" fill="white"/>
<text x="350" y="32" text-anchor="middle" font-family="sans-serif" font-size="20">{html.escape(title)}</text>
<text x="350" y="360" text-anchor="middle" font-family="sans-serif" font-size="14">Predicted class</text>
<text x="35" y="210" text-anchor="middle" transform="rotate(-90 35 210)" font-family="sans-serif" font-size="14">Actual class</text>
{''.join(cells)}
</svg>
"""
    Path(output_path).write_text(markup, encoding="utf-8")

