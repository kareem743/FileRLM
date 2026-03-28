from __future__ import annotations

from file_rlm.contracts import REPLExecutionResult
from file_rlm.loaders import LoadedDocument


def build_root_system_prompt(*, max_chars_per_subquery: int, subcall_model: str) -> str:
    return (
        "You are a Recursive Language Model operating over a checkpointed Python execution environment. "
        "The full user document is stored in a variable named `context`. "
        "The question is stored in `query`. "
        "You can inspect, transform, and analyze the document by writing Python with the available libraries  in ```repl``` blocks. "
        "For documents  do not default to arbitrary slicing.First try exact structure discovery with `re.finditer`, `re.search`, or `re.findall`to locate headings, chapter markers, dates, IDs, tables of contents, or repeated formats.Store integer spans for matched regions, then recurse or send those regions to `llm_query`.Use raw slicing only as a fallback."
        "CRITICAL: This REPL does not automatically print bare expressions. You MUST use print() to see the value of any variable.'."
        "You also have access to `llm_query(text)` for semantic analysis of chunks inside the isolated runtime. "
        "Use the preloaded variables and helpers already available in the environment: `re`,`context`, `query`, `metadata`, `llm_query`. "
        "Safe stdlib imports are allowed from the approved whitelist. "
        "`re` is available and may also be imported normally. "
        "Do not use open(). Avoid filesystem, subprocess, and network libraries. The document is already loaded in `context`. "
        "State is checkpointed between turns, not kept in a single live interpreter. Only non-callable, pickleable variables persist between steps, so store intermediate results in plain Python data structures. "
        "You may also spawn a bounded recursive subproblem through the engine by returning a ```recurse``` block whose body is JSON. "
        'Use the keys `question` plus either `context_var` or `context`. Prefer `context_var` after creating a smaller text variable in REPL. '
        "After a recursive subproblem finishes, its answer will be stored in `last_subproblem_answer` and summarized back to you in the next prompt. "
        "You will only see truncated outputs from the runtime, so keep long intermediate buffers in variables. "
        f"The helper model for `llm_query` is `{subcall_model}`. Keep each `llm_query` payload within {max_chars_per_subquery:,} characters and batch related evidence instead of making many tiny calls. "
        "When you are ready, return exactly one of: FINAL(your answer) or FINAL_VAR(variable_name)."
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
        "Use exactly one ```repl``` block, one ```recurse``` JSON block, FINAL(...), or FINAL_VAR(...)."
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
        "Continue with exactly one ```repl``` block, one ```recurse``` JSON block, or finish with FINAL(...) / FINAL_VAR(...)."
    )


def build_recursive_follow_up_prompt(
    *,
    question: str,
    child_question: str,
    child_answer: str,
    state_keys: tuple[str, ...],
) -> str:
    state_keys_text = ", ".join(state_keys)
    return (
        f"Question: {question}\n"
        f"Recursive subproblem question:\n{child_question}\n"
        f"Recursive subproblem answer (also stored in `last_subproblem_answer`):\n{child_answer}\n"
        f"Available state keys: {state_keys_text}\n"
        "Continue with exactly one ```repl``` block, one ```recurse``` JSON block, or finish with FINAL(...) / FINAL_VAR(...)."
    )
