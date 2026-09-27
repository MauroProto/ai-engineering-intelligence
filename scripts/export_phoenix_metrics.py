"""Exporta solo metricas del proyecto de carga desde Phoenix local, en lectura."""
import json
import sqlite3
from datetime import datetime
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROJECT = "coderhouse-deepseek-verificado-2026-09-26"


def main():
    connection = sqlite3.connect(f"file:{ROOT}/.data/phoenix/phoenix.db?mode=ro", uri=True)
    query = """SELECT t.trace_id,s.name,s.span_kind,s.start_time,s.end_time,
                      s.llm_token_count_prompt,s.llm_token_count_completion,sc.total_cost
               FROM spans s JOIN traces t ON t.id=s.trace_rowid
               JOIN projects p ON p.id=t.project_rowid
               LEFT JOIN span_costs sc ON sc.span_rowid=s.id
               WHERE p.name=? ORDER BY s.start_time"""
    spans = [{"trace_id": t, "name": n, "kind": k,
              "seconds": (datetime.fromisoformat(e)-datetime.fromisoformat(s)).total_seconds(),
              "input_tokens": i or 0, "output_tokens": o or 0, "cost_usd": c}
             for t,n,k,s,e,i,o,c in connection.execute(query, (PROJECT,))]
    connection.close()
    roots = [s for s in spans if s["name"] == "task.execute"]
    assert len(roots) == 5, "Usar el proyecto aislado de cinco consultas"
    report = {"project": PROJECT, "source": "Phoenix SQLite en modo lectura; no se modifican trazas",
              "root_traces": len(roots), "span_count": len(spans),
              "llm_calls": sum(s["kind"] == "LLM" for s in spans),
              "p95_seconds": float(np.percentile([s["seconds"] for s in roots],95)),
              "cost_usd": sum(s["cost_usd"] or 0 for s in spans), "spans": spans}
    benchmark = json.loads((ROOT/"evidence/benchmark.json").read_text())
    assert abs(report["cost_usd"]-benchmark["cost_usd"]) < 1e-10
    assert {s["trace_id"] for s in roots} == {j["trace_id"] for j in benchmark["jobs"]}
    (ROOT/"evidence/phoenix_metrics.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print({k:v for k,v in report.items() if k != "spans"})


if __name__ == "__main__":
    main()
