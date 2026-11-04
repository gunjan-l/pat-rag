import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from evals.run_evals import evaluate_dataset, main, parse_args


def test_golden_dataset_structure():
    dataset_path = Path("evals/data/golden_dataset.json")
    assert dataset_path.exists(), f"Dataset file does not exist at {dataset_path}"

    with open(dataset_path, encoding="utf-8") as f:
        data = json.load(f)

    assert isinstance(data, list), "Dataset root must be a list of evaluation samples"
    assert len(data) >= 5, "Dataset should have at least 5 benchmark samples"

    multi_doc_count = 0
    for sample in data:
        assert "id" in sample, "Sample must have an 'id'"
        has_q = "question" in sample or "user_input" in sample
        assert has_q, "Sample must have 'question' or 'user_input'"
        has_ctx = "reference_contexts" in sample or "contexts" in sample
        assert has_ctx, "Sample must have 'reference_contexts'"
        has_gt = "ground_truth" in sample or "reference" in sample
        assert has_gt, "Sample must have 'ground_truth'"

        contexts = sample.get("reference_contexts") or sample.get("contexts")
        assert isinstance(contexts, list), "Contexts must be a list of strings"
        assert len(contexts) > 0, "Contexts must not be empty"

        # Check for multi-document policy queries
        metadata = sample.get("metadata", {})
        if (
            metadata.get("is_multi_document")
            or len(metadata.get("source_documents", [])) > 1
            or sample.get("category") == "multi-document"
        ):
            multi_doc_count += 1

    assert multi_doc_count >= 1, "Dataset must include at least one multi-document policy query"


def test_mock_eval_metric_calculation(tmp_path):
    dataset_path = Path("evals/data/golden_dataset.json")
    with open(dataset_path, encoding="utf-8") as f:
        data = json.load(f)

    # Run evaluation in mock mode
    result = evaluate_dataset(
        dataset=data,
        mock=True,
    )

    assert "metrics" in result
    metrics = result["metrics"]

    # Must contain the 4 core Ragas metrics
    required_metrics = [
        "context_precision",
        "context_recall",
        "faithfulness",
        "answer_relevancy",
    ]
    for metric_name in required_metrics:
        assert metric_name in metrics, f"Metric {metric_name} missing from results"
        score = metrics[metric_name]
        assert isinstance(score, float), f"Metric {metric_name} should be a float"
        assert 0.0 <= score <= 1.0, f"Metric {metric_name} score {score} out of bounds [0, 1]"

    assert "per_sample_results" in result
    assert len(result["per_sample_results"]) == len(data)
    for sample_res in result["per_sample_results"]:
        assert "sample_id" in sample_res
        assert "scores" in sample_res
        for metric_name in required_metrics:
            assert metric_name in sample_res["scores"]
            s = sample_res["scores"][metric_name]
            assert 0.0 <= s <= 1.0


def test_runner_threshold_enforcement_pass(tmp_path):
    output_json = tmp_path / "summary.json"
    output_md = tmp_path / "report.md"

    # Set low thresholds that should easily pass
    exit_code = main(
        [
            "--dataset",
            "evals/data/golden_dataset.json",
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
            "--min-precision",
            "0.5",
            "--min-recall",
            "0.5",
            "--min-faithfulness",
            "0.5",
            "--min-relevancy",
            "0.5",
            "--mock",
        ]
    )

    assert exit_code == 0
    assert output_json.exists()
    assert output_md.exists()

    with open(output_json, encoding="utf-8") as f:
        summary = json.load(f)

    assert summary["passed"] is True
    assert summary["thresholds_passed"]["context_precision"] is True
    assert summary["thresholds_passed"]["context_recall"] is True
    assert summary["thresholds_passed"]["faithfulness"] is True
    assert summary["thresholds_passed"]["answer_relevancy"] is True


def test_runner_threshold_enforcement_fail(tmp_path):
    output_json = tmp_path / "summary_fail.json"
    output_md = tmp_path / "report_fail.md"

    # Set impossibly high threshold (1.01) to force failure
    exit_code = main(
        [
            "--dataset",
            "evals/data/golden_dataset.json",
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
            "--min-precision",
            "1.01",
            "--mock",
        ]
    )

    assert exit_code == 1
    assert output_json.exists()

    with open(output_json, encoding="utf-8") as f:
        summary = json.load(f)

    assert summary["passed"] is False
    assert summary["thresholds_passed"]["context_precision"] is False


def test_runner_report_generation(tmp_path):
    output_json = tmp_path / "eval_out.json"
    output_md = tmp_path / "eval_out.md"

    exit_code = main(
        [
            "--dataset",
            "evals/data/golden_dataset.json",
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
            "--mock",
        ]
    )
    assert exit_code == 0

    # Check JSON output structure
    assert output_json.exists()
    with open(output_json, encoding="utf-8") as f:
        data = json.load(f)

    assert "total_samples" in data
    assert data["total_samples"] >= 5
    assert "metrics" in data
    assert "thresholds" in data
    assert "passed" in data
    assert "per_sample_results" in data

    # Check Markdown report
    assert output_md.exists()
    md_content = output_md.read_text(encoding="utf-8")
    assert "# Ragas Evaluation Report" in md_content
    assert "context_precision" in md_content
    assert "context_recall" in md_content
    assert "faithfulness" in md_content
    assert "answer_relevancy" in md_content
    assert "Pass/Fail Status" in md_content or "Status:" in md_content


def test_parse_args_defaults():
    args = parse_args([])
    assert args.dataset == "evals/data/golden_dataset.json"
    assert args.output_json is None
    assert args.output_md is None
    assert args.min_precision == 0.70
    assert args.min_recall == 0.70
    assert args.min_faithfulness == 0.70
    assert args.min_relevancy == 0.70
    assert args.mock is False
    assert args.live is False


def test_parse_args_custom():
    args = parse_args(
        [
            "--dataset",
            "custom.json",
            "--output-json",
            "out.json",
            "--output-md",
            "out.md",
            "--min-precision",
            "0.85",
            "--min-recall",
            "0.80",
            "--min-faithfulness",
            "0.90",
            "--min-relevancy",
            "0.75",
            "--mock",
            "--live",
        ]
    )
    assert args.dataset == "custom.json"
    assert args.output_json == "out.json"
    assert args.output_md == "out.md"
    assert args.min_precision == 0.85
    assert args.min_recall == 0.80
    assert args.min_faithfulness == 0.90
    assert args.min_relevancy == 0.75
    assert args.mock is True
    assert args.live is True


def test_missing_dataset_file(tmp_path):
    exit_code = main(["--dataset", str(tmp_path / "non_existent.json")])
    assert exit_code == 1


@pytest.mark.parametrize(
    "failing_flag",
    [
        "--min-recall",
        "--min-faithfulness",
        "--min-relevancy",
    ],
)
def test_individual_threshold_failures(tmp_path, failing_flag):
    output_json = tmp_path / f"fail_{failing_flag.strip('-')}.json"
    exit_code = main(
        [
            "--dataset",
            "evals/data/golden_dataset.json",
            "--output-json",
            str(output_json),
            failing_flag,
            "1.05",
            "--mock",
        ]
    )
    assert exit_code == 1
    assert output_json.exists()
    with open(output_json, encoding="utf-8") as f:
        summary = json.load(f)
    assert summary["passed"] is False


def test_cli_subprocess_execution(tmp_path):
    out_json = tmp_path / "subp_out.json"
    cmd = [
        sys.executable,
        "evals/run_evals.py",
        "--dataset",
        "evals/data/golden_dataset.json",
        "--output-json",
        str(out_json),
        "--min-precision",
        "0.5",
        "--mock",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0
    assert out_json.exists()


def test_live_eval_mode_populates_retrieved_contexts():
    mock_retrieval_service = AsyncMock()
    mock_item = MagicMock()
    mock_item.chunk_text = (
        "Accreditation Policy 2026, Section 4.2 (Data Retention): All clinical audit records "
        "and related patient review documentation must be retained for a mandatory minimum "
        "period of 7 years following final disposition."
    )
    mock_response = MagicMock()
    mock_response.results = [mock_item]
    mock_retrieval_service.retrieve.return_value = mock_response

    sample = {
        "id": "live-sample-1",
        "question": "What is the mandatory data retention period for clinical audit records?",
        "reference_contexts": [
            "Accreditation Policy 2026, Section 4.2 (Data Retention): All clinical audit records "
            "must be retained for 7 years."
        ],
        "ground_truth": "Retained for 7 years.",
        "response": "Clinical audit records must be retained for 7 years.",
        "retrieved_contexts": [],
    }

    result = evaluate_dataset(
        dataset=[sample],
        mock=True,
        live=True,
        retrieval_service=mock_retrieval_service,
    )

    assert result["live"] is True
    assert "live" in result["mode"]
    assert sample["retrieved_contexts"] == [mock_item.chunk_text]
    mock_retrieval_service.retrieve.assert_awaited_once()
