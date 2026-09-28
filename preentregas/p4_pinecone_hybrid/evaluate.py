"""Precision@5 y Recall@5 sobre un golden set explícito."""

import json
from pathlib import Path
from typing import Any

from .rag import RAGSystem


def precision_at_k(retrieved: list[str], relevant: set[str], k: int = 5) -> float:
    return len(set(retrieved[:k]) & relevant) / k


def recall_at_k(retrieved: list[str], relevant: set[str], k: int = 5) -> float:
    if not relevant:
        raise ValueError("Cada caso debe tener documentos relevantes etiquetados")
    return len(set(retrieved[:k]) & relevant) / len(relevant)


async def evaluate(system: RAGSystem, golden_set: list[dict[str, Any]]) -> dict:
    if not golden_set:
        raise ValueError("El golden set no puede estar vacío")
    rows = []
    for case in golden_set:
        found = await system.retrieve(case["question"], top_k=5)
        sources = [chunk.source for chunk in found]
        relevant = set(case["relevant_sources"])
        rows.append({
            "question": case["question"],
            "retrieved_sources": sources,
            "relevant_sources": sorted(relevant),
            "precision_at_5": precision_at_k(sources, relevant),
            "recall_at_5": recall_at_k(sources, relevant),
        })
    return {
        "cases": rows,
        "mean_precision_at_5": sum(row["precision_at_5"] for row in rows) / len(rows),
        "mean_recall_at_5": sum(row["recall_at_5"] for row in rows) / len(rows),
    }


def load_golden_set(path: Path) -> list[dict[str, Any]]:
    cases = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or len(cases) < 5:
        raise ValueError("El golden set debe contener al menos cinco preguntas")
    for case in cases:
        if not case.get("question") or not case.get("relevant_sources"):
            raise ValueError("Cada caso necesita pregunta y fuentes relevantes")
    return cases
