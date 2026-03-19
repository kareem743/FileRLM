from __future__ import annotations

from file_rlm.contracts import REPLExecutionResult
from file_rlm.loaders import LoadedDocument


def build_root_system_prompt(*, max_chars_per_subquery: int) -> str:
    return (
        "You are a Recursive Language Model operating over a persistent Python REPL. "
        "The full user document is stored in a variable named `context`. "
        "The question is stored in `query`. "
        "You can inspect, transform, and analyze the document by writing Python in ```repl``` blocks. "
        "You also have access to `llm_query(text)` for semantic analysis of chunks inside the isolated REPL. "
        "Use the preloaded variables and helpers already available in the REPL: `context`, `query`, `metadata`, `llm_query`, and `re`. "
        "Do not use import statements. Do not use open(). Do not use filesystem access. The document is already loaded in `context`. "
        "You will only see truncated outputs from the REPL, so keep long intermediate buffers in variables. "
        "Because the model is qwen3:8b, keep each `llm_query` payload within about "
        f"24k characters (roughly {max_chars_per_subquery:,} characters) and batch related evidence instead of making many tiny calls. "
        "Look through the context methodically before answering. "
        "When you are ready, return either FINAL(your answer) or FINAL_VAR(variable_name)."
    )


def build_initial_user_prompt(*, document: LoadedDocument, question: str) -> str:
    return (
        f"Question: {question}\n"
        f"Document path: {document.path}\n"
        f"Document type: {document.file_type}\n"
        f"Document characters: {document.char_count}\n"
        f"Document lines: {document.line_count}\n\n"
        "Start by inspecting the document structure in REPL. "
        "Do not ask for the full context inline. "
        "The document path is metadata only: do not open the file path, and use the already-loaded `context` variable instead. "
        "Use REPL code or FINAL(...)."
    )


def build_follow_up_prompt(*, question: str, execution: REPLExecutionResult) -> str:
    state_keys = ", ".join(execution.state_keys)
    error_block = ""
    if execution.error:
        error_block = f"\nLast REPL error:\n{execution.error}\n"
    return (
        f"Question: {question}\n"
        f"Last REPL stdout:\n{execution.stdout or '[no stdout]'}\n"
        f"{error_block}"
        f"Available state keys: {state_keys}\n"
        "Continue with another ```repl``` block or finish with FINAL(...) / FINAL_VAR(...)."
    )
