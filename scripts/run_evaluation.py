#!/usr/bin/env python3
"""Run RAG evaluation over test queries and print metrics summary.

Outputs:
  - Console summary table
  - JSON report (outputs/evaluations/metrics_report.json)
  - Markdown summary (outputs/evaluations/metrics_report.md)
"""
import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

# Ensure project root is on sys.path when running as a script
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import create_pipeline_from_saved
from src.queries.test_queries import get_all_queries
from src.evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    mean_reciprocal_rank,
    contains_expected_phrases,
    simple_hallucination_flag,
    extract_sources_from_chunks,
    citation_coverage,
)

load_dotenv()

REPORT_DIR = Path("outputs/evaluations")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

print("\n================ EVALUATION: RAG SYSTEM ================")
print("Loading pipeline (expects pre-built index)...")
pipeline = create_pipeline_from_saved()

queries = get_all_queries()
print(f"Total queries: {len(queries)}")

results_summary = []

for q in queries:
    qid = q['id']
    query_text = q['query']
    expected_sources = q.get('expected_sources', [])
    expected_phrases = q.get('expected_answer_contains', [])

    # Run pipeline query
    result = pipeline.query(query_text, top_k=5, verbose=False)

    retrieved_sources = extract_sources_from_chunks(result['retrieved_chunks'])
    context_concat = "\n".join(c['text'] for c in result['retrieved_chunks'])
    answer = result['answer']

    p5 = precision_at_k(retrieved_sources, expected_sources, 5)
    r5 = recall_at_k(retrieved_sources, expected_sources, 5)
    mrr = mean_reciprocal_rank(retrieved_sources, expected_sources)
    phrase_cov = contains_expected_phrases(answer, expected_phrases)
    hallucinated = simple_hallucination_flag(answer, context_concat) if not q.get('hallucination_test') else False
    cite_cov = citation_coverage(answer)

    results_summary.append({
        'id': qid,
        'query': query_text,
        'category': q.get('category'),
        'difficulty': q.get('difficulty'),
        'precision_at_5': p5,
        'recall_at_5': r5,
        'mrr': mrr,
        'expected_sources': expected_sources,
        'retrieved_sources': retrieved_sources,
        'phrase_coverage': phrase_cov,
        'citation_coverage': cite_cov,
        'hallucination_flag': hallucinated,
        'provider': result.get('provider'),
        'model': result.get('model'),
        'answer': (answer[:500] + '...') if len(answer) > 500 else answer,
        'error': result.get('error')
    })

# Aggregate metrics
avg_precision = sum(r['precision_at_5'] for r in results_summary) / len(results_summary)
avg_recall = sum(r['recall_at_5'] for r in results_summary) / len(results_summary)
avg_mrr = sum(r['mrr'] for r in results_summary) / len(results_summary)
avg_phrase_cov = sum(r['phrase_coverage'] for r in results_summary) / len(results_summary)
avg_cite_cov = sum(r['citation_coverage'] for r in results_summary) / len(results_summary)
hallucination_rate = sum(1 for r in results_summary if r['hallucination_flag']) / len(results_summary)

report = {
    'aggregate': {
        'avg_precision_at_5': avg_precision,
        'avg_recall_at_5': avg_recall,
        'avg_mrr': avg_mrr,
        'avg_phrase_coverage': avg_phrase_cov,
        'avg_citation_coverage': avg_cite_cov,
        'hallucination_rate': hallucination_rate,
        'num_queries': len(results_summary)
    },
    'per_query': results_summary
}

# Write JSON
json_path = REPORT_DIR / 'metrics_report.json'
with open(json_path, 'w') as f:
    json.dump(report, f, indent=2)
print(f"Saved JSON report: {json_path}")

# Write Markdown summary
md_path = REPORT_DIR / 'metrics_report.md'
with open(md_path, 'w') as f:
    f.write("# RAG Evaluation Report\n\n")
    f.write("## Aggregate Metrics\n")
    for k, v in report['aggregate'].items():
        f.write(f"- **{k}**: {v:.4f}\n" if isinstance(v, float) else f"- **{k}**: {v}\n")
    f.write("\n## Per-Query Results\n")
    for r in results_summary:
        f.write(f"\n### Query {r['id']}: {r['query']}\n")
        f.write(f"- Category: {r['category']} | Difficulty: {r['difficulty']}\n")
        f.write(f"- Provider: {r['provider']} | Model: {r['model']}\n")
        f.write(f"- Precision@5: {r['precision_at_5']:.3f} | Recall@5: {r['recall_at_5']:.3f} | MRR: {r['mrr']:.3f}\n")
        f.write(f"- Phrase Coverage: {r['phrase_coverage']:.3f} | Citation Coverage: {r['citation_coverage']:.3f}\n")
        f.write(f"- Hallucination Flag: {r['hallucination_flag']}\n")
        f.write(f"- Expected Sources: {', '.join(r['expected_sources'])}\n")
        f.write(f"- Retrieved Sources: {', '.join(r['retrieved_sources'])}\n")
        if r.get('error'):
            f.write(f"- Error: {r['error']}\n")
        f.write("- Answer (first 350 chars):\n\n")
        snippet = r['answer'][:350] if r.get('answer') else ''
        f.write(f"> {snippet}\n")
print(f"Saved Markdown summary: {md_path}")

print("\nEvaluation complete.")
