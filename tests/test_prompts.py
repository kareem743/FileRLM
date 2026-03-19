from pathlib import Path

from file_rlm.loaders import LoadedDocument
from file_rlm.prompts import build_follow_up_prompt, build_initial_user_prompt, build_root_system_prompt
from file_rlm.repl_runtime import REPLExecutionResult


def test_root_prompt_mentions_context_llm_query_and_qwen_limits() -> None:
    prompt = build_root_system_prompt(max_chars_per_subquery=24_000)

    assert "context" in prompt
    assert "llm_query" in prompt
    assert "truncated outputs" in prompt
    assert "24k characters" in prompt
    assert "FINAL(" in prompt
    assert "FINAL_VAR(" in prompt
    assert "Do not use import" in prompt
    assert "Do not use open()" in prompt
    assert "filesystem access" in prompt
    assert "`re`" in prompt


def test_initial_user_prompt_contains_question_and_document_metadata() -> None:
    document = LoadedDocument(
        path=Path("paper.pdf"),
        text="summary",
        file_type="pdf",
        char_count=7,
        line_count=1,
    )

    prompt = build_initial_user_prompt(document=document, question="What is the summary?")

    assert "paper.pdf" in prompt
    assert "pdf" in prompt
    assert "What is the summary?" in prompt
    assert "Do not ask for the full context inline" in prompt
    assert "do not open the file path" in prompt.lower()
    assert "use the already-loaded `context` variable" in prompt


def test_follow_up_prompt_surfaces_stdout_and_state_keys() -> None:
    result = REPLExecutionResult(stdout="Found a likely answer", state_keys=("buffer", "context"))

    prompt = build_follow_up_prompt(question="Where is the answer?", execution=result)

    assert "Found a likely answer" in prompt
    assert "buffer, context" in prompt
    assert "Where is the answer?" in prompt


def test_follow_up_prompt_surfaces_repl_errors() -> None:
    result = REPLExecutionResult(
        stdout="",
        state_keys=("context",),
        error="NameError: missing_value",
    )

    prompt = build_follow_up_prompt(question="Why did this fail?", execution=result)

    assert "NameError: missing_value" in prompt
    assert "Why did this fail?" in prompt
