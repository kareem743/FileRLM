import json

from evals.run_rlm_eval import BenchmarkCategory, default_benchmark_plan, main


def test_default_benchmark_plan_covers_four_paper_categories() -> None:
    scenarios = default_benchmark_plan()
    categories = {scenario.category for scenario in scenarios}

    assert len(scenarios) == 4
    assert categories == {
        BenchmarkCategory.SEARCH,
        BenchmarkCategory.LINEAR,
        BenchmarkCategory.PAIRWISE,
        BenchmarkCategory.DEEP_REASONING,
    }


def test_all_scenarios_share_the_expected_scale_targets() -> None:
    scenarios = default_benchmark_plan()

    for scenario in scenarios:
        assert scenario.context_sizes == (100_000, 500_000, 1_000_000, 10_000_000)
        assert "recursion_depth" in scenario.required_metrics
        assert "runtime_seconds" in scenario.required_metrics


def test_eval_cli_can_emit_the_plan_as_json(capsys) -> None:
    exit_code = main(["--plan-json"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert len(payload) == 4
    assert payload[0]["name"] == "s_niah_smoke"
