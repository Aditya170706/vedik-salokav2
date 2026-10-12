"""Research benchmark suite for the Vedic Shloka Intelligence system.

Sub-packages:
  benchmarks.retrieval — retrieval quality benchmark (Recall@k / Precision@k / MRR)
                         and knowledge-graph contribution ablations.
  benchmarks.formula   — formula / mathematical-correctness benchmark
                         (5 decoupled evaluation stages).

All benchmarks are deterministic, write machine-readable JSON + Markdown
tables into ``benchmarks/results/`` and never mutate the shipped database
artifacts under ``database/``.
"""
