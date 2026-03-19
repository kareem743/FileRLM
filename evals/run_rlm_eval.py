from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Iterable


class BenchmarkCategory(str, Enum):
    SEARCH = "search"
    LINEAR = "linear"
    PAIRWISE = "pairwise"
    DEEP_REASONING = "deep_reasoning"


@dataclass(slots=True)
class EvaluationScenario:
    name: str
    category: BenchmarkCategory
    description: str
    context_sizes: tuple[int, ...]
    required_metrics: tuple[str, ...]


def default_benchmark_plan() -> list[EvaluationScenario]:
    """Return the first evaluation plan aligned with the paper summary."""

    shared_sizes = (100_000, 500_000, 1_000_000, 10_000_000)
    shared_metrics = ("accuracy", "subcall_cost", "recursion_depth", "runtime_seconds")

    return [
        EvaluationScenario(
            name="s_niah_smoke",
            category=BenchmarkCategory.SEARCH,
            description="Locate a hidden fact in a long synthetic context.",
            context_sizes=shared_sizes,
            required_metrics=shared_metrics,
        ),
        EvaluationScenario(
            name="oolong_linear_smoke",
            category=BenchmarkCategory.LINEAR,
            description="Aggregate evidence spread across the entire context.",
            context_sizes=shared_sizes,
            required_metrics=shared_metrics,
        ),
        EvaluationScenario(
            name="oolong_pairs_smoke",
            category=BenchmarkCategory.PAIRWISE,
            description="Recover entity pairs that require pairwise reasoning.",
            context_sizes=shared_sizes,
            required_metrics=("f1", "subcall_cost", "recursion_depth", "runtime_seconds"),
        ),
        EvaluationScenario(
            name="codeqa_smoke",
            category=BenchmarkCategory.DEEP_REASONING,
            description="Answer a multi-step question over a local code or docs fixture.",
            context_sizes=shared_sizes,
            required_metrics=shared_metrics,
        ),
    ]


def plan_as_dict(scenarios: Iterable[EvaluationScenario]) -> list[dict[str, object]]:
    return [asdict(scenario) for scenario in scenarios]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Planning-stage RLM evaluation harness.")
    parser.add_argument(
        "--plan-json",
        action="store_true",
        help="Print the benchmark plan as JSON instead of running evaluations.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.plan_json:
        print(json.dumps(plan_as_dict(default_benchmark_plan()), indent=2))
        return 0

    raise SystemExit(
        "Evaluation execution is not wired yet. Implement the RLM engine and baseline adapters next."
    )


if __name__ == "__main__":
    raise SystemExit(main())
