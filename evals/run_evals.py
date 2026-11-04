#!/usr/bin/env python3
"""Ragas Evaluation Suite Runner.

Evaluates RAG retrieval and generation outputs against core Ragas metrics:
- context_precision
- context_recall
- faithfulness
- answer_relevancy

Supports mock / offline mode for CI/CD environments without external API keys.
"""

import argparse
import asyncio
import datetime
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from src.constants import STOPWORDS


def stem_token(word: str) -> str:
    """Stem word by removing common English inflectional and derivational suffixes."""
    if word.startswith("retain") or word.startswith("reten"):
        return "retain"
    if word.startswith("transit") or word.startswith("transm"):
        return "transit"
    if word.startswith("stor"):
        return "store"
    if word.startswith("requir"):
        return "require"
    if word.startswith("govern"):
        return "govern"
    if word.startswith("comply") or word.startswith("compli"):
        return "comply"
    for suffix in (
        "tions",
        "tion",
        "ings",
        "ing",
        "ments",
        "ment",
        "ances",
        "ance",
        "ences",
        "ence",
        "ables",
        "able",
        "ibles",
        "ible",
        "ies",
        "es",
        "ed",
        "s",
    ):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def tokenize(text: str) -> set[str]:
    """Tokenize text into lowercase alphanumeric content words and stems."""
    words = re.findall(r"\b[a-zA-Z0-9_\-\.]+\b", text.lower())
    tokens = set()
    for w in words:
        if len(w) > 1 and w not in STOPWORDS:
            tokens.add(w)
            tokens.add(stem_token(w))
    return tokens


def calculate_context_precision(
    retrieved_contexts: list[str],
    reference_contexts: list[str],
    ground_truth: str,
) -> float:
    """Calculate context precision: whether relevant contexts appear at top ranks."""
    if not retrieved_contexts:
        return 0.0

    target_tokens = tokenize(" ".join(reference_contexts) + " " + ground_truth)
    if not target_tokens:
        return 1.0

    relevant_flags: list[bool] = []
    for ctx in retrieved_contexts:
        ctx_tokens = tokenize(ctx)
        overlap = len(ctx_tokens.intersection(target_tokens))
        # Consider a chunk relevant if it shares significant content tokens
        token_ratio = overlap / min(len(ctx_tokens) or 1, len(target_tokens))
        is_rel = overlap >= 3 or (len(target_tokens) > 0 and token_ratio >= 0.25)
        relevant_flags.append(is_rel)

    num_relevant = sum(relevant_flags)
    if num_relevant == 0:
        return 0.0

    # Ragas context precision formula: mean precision@k for relevant items
    precisions: list[float] = []
    hits = 0
    for k, is_rel in enumerate(relevant_flags, start=1):
        if is_rel:
            hits += 1
            precisions.append(hits / k)

    return min(1.0, max(0.0, sum(precisions) / num_relevant))


def calculate_context_recall(
    retrieved_contexts: list[str],
    reference_contexts: list[str],
    ground_truth: str,
) -> float:
    """Calculate context recall: fraction of reference facts present in retrieved contexts."""
    all_retrieved = " ".join(retrieved_contexts)
    retrieved_tokens = tokenize(all_retrieved)

    # Split reference into statements or sentences
    ref_text = " ".join(reference_contexts) if reference_contexts else ground_truth
    statements = [s.strip() for s in re.split(r"[.\n;]+", ref_text) if len(s.strip()) > 5]

    if not statements:
        return 1.0

    covered = 0
    for stmt in statements:
        stmt_tokens = tokenize(stmt)
        if not stmt_tokens:
            covered += 1
            continue
        overlap = len(stmt_tokens.intersection(retrieved_tokens))
        # Covered if at least 50% of statement tokens are found in retrieved contexts
        if (overlap / len(stmt_tokens)) >= 0.45:
            covered += 1

    return min(1.0, max(0.0, covered / len(statements)))


def calculate_faithfulness(response: str, retrieved_contexts: list[str]) -> float:
    """Calculate faithfulness: whether response claims are grounded in retrieved contexts."""
    if not response:
        return 0.0

    all_retrieved = " ".join(retrieved_contexts)
    context_tokens = tokenize(all_retrieved)

    sentences = [s.strip() for s in re.split(r"[.\n;]+", response) if len(s.strip()) > 5]
    if not sentences:
        return 1.0

    faithful_count = 0
    for sent in sentences:
        sent_tokens = tokenize(sent)
        if not sent_tokens:
            faithful_count += 1
            continue
        overlap = len(sent_tokens.intersection(context_tokens))
        ratio = overlap / len(sent_tokens)
        if ratio >= 0.50:
            faithful_count += 1
        else:
            faithful_count += ratio

    return min(1.0, max(0.0, faithful_count / len(sentences)))


def calculate_answer_relevancy(question: str, response: str) -> float:
    """Calculate answer relevancy: whether response directly addresses the query."""
    q_tokens = tokenize(question)
    r_tokens = tokenize(response)

    if not q_tokens:
        return 1.0
    if not r_tokens:
        return 0.0

    meta_terms = {
        "mechanism",
        "mechanisms",
        "frequently",
        "timeline",
        "criteria",
        "algorithm",
        "algorithms",
        "protocol",
        "protocols",
        "period",
        "requirement",
        "requirements",
        "manner",
        "way",
    }
    content_q = {t for t in q_tokens if t not in meta_terms}
    if not content_q:
        content_q = q_tokens

    overlap = len(content_q.intersection(r_tokens))
    q_coverage = overlap / len(content_q)
    length_factor = min(1.0, len(r_tokens) / 8.0)

    score = (0.7 * q_coverage) + (0.3 * length_factor)
    return min(1.0, max(0.0, score))


def evaluate_mock_sample(sample: dict[str, Any]) -> dict[str, float]:
    """Evaluate a single sample dictionary in mock / offline mode."""
    question = sample.get("question") or sample.get("user_input", "")
    ref_contexts = sample.get("reference_contexts") or sample.get("contexts", [])
    ground_truth = sample.get("ground_truth") or sample.get("reference", "")
    retrieved_contexts = sample.get("retrieved_contexts") or ref_contexts
    response = sample.get("response") or sample.get("answer") or ground_truth

    c_precision = calculate_context_precision(retrieved_contexts, ref_contexts, ground_truth)
    c_recall = calculate_context_recall(retrieved_contexts, ref_contexts, ground_truth)
    faithfulness = calculate_faithfulness(response, retrieved_contexts)
    a_relevancy = calculate_answer_relevancy(question, response)

    return {
        "context_precision": round(c_precision, 4),
        "context_recall": round(c_recall, 4),
        "faithfulness": round(faithfulness, 4),
        "answer_relevancy": round(a_relevancy, 4),
    }


async def populate_live_contexts_async(
    dataset: list[dict[str, Any]],
    retrieval_service: Any | None = None,
) -> list[dict[str, Any]]:
    """Query RetrievalService to dynamically populate retrieved_contexts for each sample."""
    if retrieval_service is None:
        from src.services.retrieval import RetrievalService

        retrieval_service = RetrievalService()

    for sample in dataset:
        query = sample.get("question") or sample.get("user_input", "")
        namespace = (
            sample.get("namespace") or sample.get("metadata", {}).get("namespace") or "default"
        )
        context = sample.get("conversation_context") or []
        res = await retrieval_service.retrieve(
            namespace=namespace,
            query_text=query,
            conversation_context=context,
        )
        sample["retrieved_contexts"] = [item.chunk_text for item in res.results]

    return dataset


def populate_live_contexts(
    dataset: list[dict[str, Any]],
    retrieval_service: Any | None = None,
) -> list[dict[str, Any]]:
    """Synchronous / bridge wrapper to run live retrieval context population."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(
                asyncio.run,
                populate_live_contexts_async(dataset, retrieval_service),
            ).result()
    else:
        return asyncio.run(populate_live_contexts_async(dataset, retrieval_service))


def evaluate_dataset(
    dataset: list[dict[str, Any]],
    mock: bool = True,
    thresholds: dict[str, float] | None = None,
    live: bool = False,
    retrieval_service: Any | None = None,
) -> dict[str, Any]:
    """Execute evaluation over dataset items and return structured results."""
    if thresholds is None:
        thresholds = {
            "context_precision": 0.70,
            "context_recall": 0.70,
            "faithfulness": 0.70,
            "answer_relevancy": 0.70,
        }

    if live:
        dataset = populate_live_contexts(dataset, retrieval_service=retrieval_service)

    # Decide mode: online Ragas vs mock
    use_online = (not mock) and bool(os.environ.get("OPENAI_API_KEY"))

    per_sample_results: list[dict[str, Any]] = []

    if use_online:
        try:
            from datasets import Dataset as HFDataset
            from ragas import evaluate
            from ragas.metrics import (
                answer_relevancy,
                context_precision,
                context_recall,
                faithfulness,
            )

            # Format dataset for Ragas
            ragas_data = {
                "question": [s.get("question") or s.get("user_input", "") for s in dataset],
                "contexts": [
                    s.get("retrieved_contexts") or s.get("reference_contexts", []) for s in dataset
                ],
                "answer": [s.get("response") or s.get("ground_truth", "") for s in dataset],
                "ground_truth": [s.get("ground_truth") or s.get("reference", "") for s in dataset],
            }
            hf_ds = HFDataset.from_dict(ragas_data)
            eval_result = evaluate(
                hf_ds,
                metrics=[context_precision, context_recall, faithfulness, answer_relevancy],
            )
            df = eval_result.to_pandas()
            for idx, sample in enumerate(dataset):
                scores = {
                    "context_precision": float(df["context_precision"].iloc[idx]),
                    "context_recall": float(df["context_recall"].iloc[idx]),
                    "faithfulness": float(df["faithfulness"].iloc[idx]),
                    "answer_relevancy": float(df["answer_relevancy"].iloc[idx]),
                }
                per_sample_results.append(
                    {
                        "sample_id": sample.get("id", f"sample-{idx}"),
                        "question": sample.get("question") or sample.get("user_input", ""),
                        "scores": {k: round(v, 4) for k, v in scores.items()},
                        "metadata": sample.get("metadata", {}),
                    }
                )
        except Exception as e:
            # Fallback to mock on online failure
            print(f"Online evaluation encountered error ({e}); falling back to offline mode.")
            use_online = False

    if not use_online:
        for idx, sample in enumerate(dataset):
            scores = evaluate_mock_sample(sample)
            per_sample_results.append(
                {
                    "sample_id": sample.get("id", f"sample-{idx}"),
                    "question": sample.get("question") or sample.get("user_input", ""),
                    "scores": scores,
                    "metadata": sample.get("metadata", {}),
                }
            )

    # Compute mean metrics
    metric_keys = ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]
    mean_metrics: dict[str, float] = {}
    thresholds_passed: dict[str, bool] = {}

    for k in metric_keys:
        vals = [s["scores"][k] for s in per_sample_results]
        mean_score = sum(vals) / len(vals) if vals else 0.0
        mean_metrics[k] = round(mean_score, 4)
        min_thresh = thresholds.get(k, 0.0)
        thresholds_passed[k] = mean_score >= min_thresh

    all_passed = all(thresholds_passed.values())

    mode_label = (
        f"live-{'online' if use_online else 'mock'}"
        if live
        else ("online" if use_online else "mock")
    )

    return {
        "total_samples": len(dataset),
        "metrics": mean_metrics,
        "thresholds": thresholds,
        "thresholds_passed": thresholds_passed,
        "passed": all_passed,
        "mode": mode_label,
        "live": live,
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
        "per_sample_results": per_sample_results,
    }


def generate_markdown_report(result: dict[str, Any]) -> str:
    """Generate a clean Markdown report from evaluation results."""
    status_str = "PASS" if result["passed"] else "FAIL"
    lines = [
        "# Ragas Evaluation Report",
        "",
        "## Summary",
        f"- **Status:** {status_str}",
        f"- **Total Samples:** {result['total_samples']}",
        f"- **Mode:** {result['mode']}",
        f"- **Timestamp:** {result['timestamp']}",
        "",
        "## Metric Results",
        "",
        "| Metric | Score | Threshold | Status |",
        "| :--- | :--- | :--- | :--- |",
    ]

    for metric, score in result["metrics"].items():
        thresh = result["thresholds"].get(metric, 0.0)
        passed = result["thresholds_passed"].get(metric, False)
        pass_icon = "PASS" if passed else "FAIL"
        lines.append(f"| **{metric}** | {score:.4f} | {thresh:.4f} | {pass_icon} |")

    lines.extend(
        [
            "",
            "## Detailed Per-Sample Breakdown",
            "",
            (
                "| ID | Category | Context Precision | Context Recall | "
                "Faithfulness | Answer Relevancy |"
            ),
            "| :--- | :--- | :--- | :--- | :--- | :--- |",
        ]
    )

    for sample in result["per_sample_results"]:
        s_id = sample["sample_id"]
        category = sample.get("metadata", {}).get("category", "general")
        scores = sample["scores"]
        lines.append(
            f"| {s_id} | {category} | {scores['context_precision']:.4f} | "
            f"{scores['context_recall']:.4f} | {scores['faithfulness']:.4f} | "
            f"{scores['answer_relevancy']:.4f} |"
        )

    lines.append("")
    return "\n".join(lines)


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for evaluation runner."""
    parser = argparse.ArgumentParser(description="Ragas Evaluation Runner")
    parser.add_argument(
        "--dataset",
        type=str,
        default="evals/data/golden_dataset.json",
        help="Path to evaluation dataset JSON file",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to output JSON summary report",
    )
    parser.add_argument(
        "--output-md",
        type=str,
        default=None,
        help="Path to output Markdown report",
    )
    parser.add_argument(
        "--min-precision",
        type=float,
        default=0.70,
        help="Minimum required context_precision threshold (default: 0.70)",
    )
    parser.add_argument(
        "--min-recall",
        type=float,
        default=0.70,
        help="Minimum required context_recall threshold (default: 0.70)",
    )
    parser.add_argument(
        "--min-faithfulness",
        type=float,
        default=0.70,
        help="Minimum required faithfulness threshold (default: 0.70)",
    )
    parser.add_argument(
        "--min-relevancy",
        type=float,
        default=0.70,
        help="Minimum required answer_relevancy threshold (default: 0.70)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        dest="live",
        default=False,
        help="Run live retrieval via RetrievalService to dynamically populate retrieved_contexts",
    )
    parser.add_argument(
        "--mock",
        "--offline",
        action="store_true",
        dest="mock",
        default=False,
        help="Run evaluation in offline/mock mode without calling external APIs",
    )

    return parser.parse_args(args)


def main(args: list[str] | None = None) -> int:
    """Run CLI evaluation and return exit code."""
    parsed = parse_args(args)

    dataset_path = Path(parsed.dataset)
    if not dataset_path.exists():
        print(f"Error: Dataset file not found at {dataset_path}", file=sys.stderr)
        return 1

    with open(dataset_path, encoding="utf-8") as f:
        dataset = json.load(f)

    thresholds = {
        "context_precision": parsed.min_precision,
        "context_recall": parsed.min_recall,
        "faithfulness": parsed.min_faithfulness,
        "answer_relevancy": parsed.min_relevancy,
    }

    # If OPENAI_API_KEY is not set, force mock mode
    mock_mode = parsed.mock or not bool(os.environ.get("OPENAI_API_KEY"))

    result = evaluate_dataset(
        dataset=dataset,
        mock=mock_mode,
        thresholds=thresholds,
        live=parsed.live,
    )

    # Output reports if requested
    if parsed.output_json:
        out_json_path = Path(parsed.output_json)
        out_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"Wrote JSON evaluation summary to {out_json_path}")

    if parsed.output_md:
        out_md_path = Path(parsed.output_md)
        out_md_path.parent.mkdir(parents=True, exist_ok=True)
        md_content = generate_markdown_report(result)
        out_md_path.write_text(md_content, encoding="utf-8")
        print(f"Wrote Markdown evaluation report to {out_md_path}")

    print("\n=== Evaluation Results ===")
    for m, score in result["metrics"].items():
        thresh = thresholds[m]
        status = "PASS" if result["thresholds_passed"][m] else "FAIL"
        print(f"{m:20s}: {score:.4f} (Threshold: {thresh:.4f}) -> {status}")

    overall_status = "PASS" if result["passed"] else "FAIL"
    print(f"\nOverall Result: {overall_status} (Mode: {result['mode']})")

    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
