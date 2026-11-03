# 06: Ragas Evaluation Suite

**What to build:** An automated quantitative evaluation harness (`evals/run_evals.py`) using Ragas to measure retrieval and generation quality (Context Precision, Context Recall, Faithfulness, Answer Relevance) against a benchmark dataset.

**Blocked by:** 04: Agentic Retrieval, Confidence Fallback & Query Rewrite (Component 3)

**Status:** ready-for-agent

- [ ] Synthetic golden dataset created under `evals/data/golden_dataset.json` covering multi-document policy queries and edge cases.
- [ ] Evaluation runner script executes queries through the retrieval pipeline and computes Ragas metrics: `context_precision`, `context_recall`, `faithfulness`, and `answer_relevancy`.
- [ ] Evaluation report is generated and output as structured JSON summary and markdown table showing per-query and aggregate scores.
- [ ] Configurable threshold assertions flag quality regressions if aggregate scores fall below defined targets.
- [ ] Runner can be executed directly as a CLI command or integrated into automated test runs.
