from pathlib import Path

from file_rlm.contracts import AnswerResult, QuestionRequest, REPLExecutionResult


def test_question_request_defaults_match_the_planned_root_loop() -> None:
    request = QuestionRequest(file_path=Path("sample.txt"), question="What is the answer?")

    assert request.file_path == Path("sample.txt")
    assert request.question == "What is the answer?"
    assert request.max_root_iterations == 26
    assert request.max_recursion_depth == 1


def test_answer_result_tracks_observability_fields() -> None:
    result = AnswerResult(
        answer="needle found",
        iterations=3,
        subcall_count=2,
        recursion_depth=1,
        prompt_chars=120_000,
        trace=["inspect prefix", "query chunk", "finalize"],
    )

    assert result.answer == "needle found"
    assert result.iterations == 3
    assert result.subcall_count == 2
    assert result.recursion_depth == 1
    assert result.prompt_chars == 120_000
    assert len(result.trace) == 3


def test_repl_execution_result_tracks_stdout_and_state_keys() -> None:
    result = REPLExecutionResult(stdout="found it", state_keys=("buffer", "context"))

    assert result.stdout == "found it"
    assert result.state_keys == ("buffer", "context")
