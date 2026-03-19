from __future__ import annotations

import json
import re
from typing import Callable

from file_rlm.config import ModelSettings, RuntimeLimits
from file_rlm.contracts import AnswerResult, QuestionRequest, REPLRuntime
from file_rlm.loaders import DocumentLoader, LoadedDocument
from file_rlm.prompts import (
    build_follow_up_prompt,
    build_initial_user_prompt,
    build_recursive_follow_up_prompt,
    build_root_system_prompt,
)

FINAL_VAR_RE = re.compile(r"FINAL_VAR\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)")
FINAL_RE = re.compile(r"FINAL\((.*?)\)", re.DOTALL)
REPL_BLOCK_RE = re.compile(r"```repl\s*(.*?)```", re.DOTALL | re.IGNORECASE)
RECURSE_BLOCK_RE = re.compile(r"```recurse\s*(.*?)```", re.DOTALL | re.IGNORECASE)


class RLMEngine:
    """Root RLM loop that coordinates the loader, model, and isolated runtime."""

    def __init__(
        self,
        *,
        document_loader: DocumentLoader,
        ollama_client,
        runtime_factory: Callable[[], REPLRuntime],
        model_settings: ModelSettings,
        limits: RuntimeLimits,
    ) -> None:
        self.document_loader = document_loader
        self.ollama_client = ollama_client
        self.runtime_factory = runtime_factory
        self.model_settings = model_settings
        self.limits = limits

    def _extract_action(self, response: str) -> tuple[str, str]:
        final_var_matches = list(FINAL_VAR_RE.finditer(response))
        if final_var_matches:
            return ("final_var", final_var_matches[-1].group(1))

        final_matches = list(FINAL_RE.finditer(response))
        if final_matches:
            return ("final", final_matches[-1].group(1).strip())

        recurse_blocks = RECURSE_BLOCK_RE.findall(response)
        if recurse_blocks:
            joined = "\n\n".join(block.strip() for block in recurse_blocks if block.strip())
            return ("recurse", joined)

        repl_blocks = REPL_BLOCK_RE.findall(response)
        if repl_blocks:
            joined = "\n\n".join(block.strip() for block in repl_blocks if block.strip())
            return ("repl", joined)

        return ("invalid", response.strip())

    def _document_metadata(self, document: LoadedDocument) -> dict[str, object]:
        return {
            "path": str(document.path),
            "file_type": document.file_type,
            "char_count": document.char_count,
            "line_count": document.line_count,
        }

    def _clamp_iterations(self, requested_iterations: int) -> int:
        return max(1, min(requested_iterations, self.limits.max_root_iterations))

    def _clamp_recursion_depth(self, requested_depth: int) -> int:
        return max(0, min(requested_depth, self.limits.max_recursion_depth))

    def _parse_recurse_payload(self, payload: str) -> dict[str, str]:
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid recurse JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError("Recurse payload must be a JSON object.")

        question = data.get("question")
        context_var = data.get("context_var")
        context_text = data.get("context")

        if not isinstance(question, str) or not question.strip():
            raise ValueError("Recurse payload must include a non-empty string `question`.")
        if context_var is None and context_text is None:
            raise ValueError("Recurse payload must include either `context_var` or `context`.")
        if context_var is not None and not isinstance(context_var, str):
            raise ValueError("`context_var` must be a string when provided.")
        if context_text is not None and not isinstance(context_text, str):
            raise ValueError("`context` must be a string when provided.")

        parsed: dict[str, str] = {"question": question.strip()}
        if context_var is not None:
            parsed["context_var"] = context_var.strip()
        if context_text is not None:
            parsed["context"] = context_text
        return parsed

    def _resolve_child_context(self, runtime: REPLRuntime, recurse_spec: dict[str, str]) -> tuple[str, dict[str, object]]:
        if "context_var" in recurse_spec:
            var_name = recurse_spec["context_var"]
            try:
                value = runtime.get_variable(var_name)
            except KeyError as exc:
                raise ValueError(f"Recurse context variable `{var_name}` does not exist.") from exc
            if not isinstance(value, str):
                raise ValueError(f"Recurse context variable `{var_name}` must contain a string.")
            return value, {"source": "runtime_variable", "context_var": var_name, "char_count": len(value)}

        context_text = recurse_spec["context"]
        return context_text, {"source": "inline_recurse_context", "char_count": len(context_text)}

    def _answer_from_context(
        self,
        *,
        context_text: str,
        question: str,
        metadata: dict[str, object],
        effective_iterations: int,
        remaining_depth: int,
        remaining_recursive_calls: int,
        current_depth: int,
        progress_callback: Callable[[str], None] | None = None,
        initial_user_prompt: str | None = None,
    ) -> AnswerResult:
        runtime = self.runtime_factory()
        runtime.initialize(context=context_text, question=question, metadata=metadata)
        system_prompt = build_root_system_prompt(
            max_chars_per_subquery=self.limits.max_chars_per_subquery,
            subcall_model=self.model_settings.subcall_model,
        )
        user_prompt = initial_user_prompt or (
            f"Question: {question}\n"
            f"Context characters: {len(context_text)}\n"
            f"Metadata: {metadata}\n\n"
            "Use exactly one ```repl``` block, one ```recurse``` JSON block, FINAL(...), or FINAL_VAR(...)."
        )
        trace: list[str] = []
        emit = progress_callback or (lambda _message: None)

        try:
            for iteration in range(1, effective_iterations + 1):
                emit(
                    f"Depth {current_depth} step {iteration}/{effective_iterations}: querying root model"
                )
                response = self.ollama_client.generate(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    model=self.model_settings.root_model,
                    temperature=self.model_settings.temperature,
                    timeout_seconds=self.model_settings.request_timeout_seconds,
                )
                trace.append(f"depth={current_depth}:assistant:{response[:200]}")
                action, payload = self._extract_action(response)

                if action == "final":
                    emit(f"Depth {current_depth} step {iteration}/{effective_iterations}: Final answer")
                    return AnswerResult(
                        answer=payload,
                        iterations=iteration,
                        subcall_count=int(runtime.get_variable("__subcall_count")),
                        recursion_depth=current_depth,
                        prompt_chars=len(context_text),
                        trace=trace,
                    )

                if action == "final_var":
                    try:
                        answer_value = runtime.get_variable(payload)
                    except KeyError:
                        emit(
                            f"Depth {current_depth} step {iteration}/{effective_iterations}: final variable missing `{payload}`"
                        )
                        user_prompt = (
                            f"The previous response referred to FINAL_VAR({payload}), but that variable does not exist in the current state.\n"
                            "Reply with one ```repl``` block, one ```recurse``` JSON block, FINAL(...), or FINAL_VAR(...) using an existing variable."
                        )
                        continue

                    emit(
                        f"Depth {current_depth} step {iteration}/{effective_iterations}: Final answer from variable `{payload}`"
                    )
                    return AnswerResult(
                        answer=str(answer_value),
                        iterations=iteration,
                        subcall_count=int(runtime.get_variable("__subcall_count")),
                        recursion_depth=current_depth,
                        prompt_chars=len(context_text),
                        trace=trace,
                    )

                if action == "recurse":
                    try:
                        recurse_spec = self._parse_recurse_payload(payload)
                        child_context, child_metadata = self._resolve_child_context(runtime, recurse_spec)
                    except ValueError as exc:
                        emit(
                            f"Depth {current_depth} step {iteration}/{effective_iterations}: recurse payload error\n{exc}"
                        )
                        user_prompt = (
                            f"The previous recurse block was invalid: {exc}\n"
                            "Reply with exactly one valid ```recurse``` JSON block, one ```repl``` block, FINAL(...), or FINAL_VAR(...)."
                        )
                        continue

                    if remaining_depth <= 0:
                        emit(
                            f"Depth {current_depth} step {iteration}/{effective_iterations}: recursion depth limit reached"
                        )
                        user_prompt = (
                            "You attempted to recurse, but the remaining recursion depth is 0. "
                            "Continue in the current scope with REPL or finish with FINAL(...)/FINAL_VAR(...)."
                        )
                        continue

                    if remaining_recursive_calls <= 0:
                        emit(
                            f"Depth {current_depth} step {iteration}/{effective_iterations}: recursive call budget exhausted"
                        )
                        user_prompt = (
                            "You attempted to recurse, but the recursive call budget is exhausted. "
                            "Continue in the current scope with REPL or finish with FINAL(...)/FINAL_VAR(...)."
                        )
                        continue

                    child_question = recurse_spec["question"]
                    child_iterations = min(effective_iterations, self.limits.max_child_iterations)
                    emit(
                        f"Depth {current_depth} step {iteration}/{effective_iterations}: spawning recursive subproblem\n{child_question}"
                    )
                    child_result = self._answer_from_context(
                        context_text=child_context,
                        question=child_question,
                        metadata=child_metadata,
                        effective_iterations=child_iterations,
                        remaining_depth=remaining_depth - 1,
                        remaining_recursive_calls=remaining_recursive_calls - 1,
                        current_depth=current_depth + 1,
                        progress_callback=emit,
                    )
                    runtime.set_variable("last_subproblem_answer", child_result.answer)
                    runtime.set_variable("last_subproblem_question", child_question)
                    trace.extend(child_result.trace)
                    trace.append(
                        f"depth={current_depth}:child_answer:{child_result.answer[:200]}"
                    )
                    user_prompt = build_recursive_follow_up_prompt(
                        question=question,
                        child_question=child_question,
                        child_answer=child_result.answer[: self.limits.max_observation_chars],
                        state_keys=runtime.list_variables(),
                    )
                    continue

                if action == "repl":
                    if len(payload) > self.limits.max_repl_code_chars:
                        emit(
                            f"Depth {current_depth} step {iteration}/{effective_iterations}: REPL code too large ({len(payload)} chars)"
                        )
                        user_prompt = (
                            f"The previous ```repl``` block was too large at {len(payload)} characters. "
                            f"Stay within {self.limits.max_repl_code_chars} characters and continue with one smaller ```repl``` block, one ```recurse``` JSON block, or FINAL(...)."
                        )
                        continue

                    emit(
                        f"Depth {current_depth} step {iteration}/{effective_iterations}: Generated REPL code\n{payload}"
                    )
                    execution = runtime.execute(payload)
                    trace.append(f"depth={current_depth}:repl:{execution.stdout[:200]}")
                    if execution.error:
                        emit(
                            f"Depth {current_depth} step {iteration}/{effective_iterations}: REPL error\n{execution.error}"
                        )
                    if execution.stdout:
                        emit(
                            f"Depth {current_depth} step {iteration}/{effective_iterations}: REPL stdout\n{execution.stdout}"
                        )
                    user_prompt = build_follow_up_prompt(question=question, execution=execution)
                    continue

                invalid_payload = payload[: self.limits.max_observation_chars]
                emit(
                    f"Depth {current_depth} step {iteration}/{effective_iterations}: Invalid model output\n{invalid_payload}"
                )
                user_prompt = (
                    f"The previous response was invalid:\n{invalid_payload}\n\n"
                    "Reply with exactly one ```repl``` block, one ```recurse``` JSON block, FINAL(...), or FINAL_VAR(...)."
                )

            raise RuntimeError(
                "The RLM engine reached its iteration limit without a final answer. "
                f"effective_iterations={effective_iterations}, current_depth={current_depth}, "
                f"remaining_depth={remaining_depth}, remaining_recursive_calls={remaining_recursive_calls}."
            )
        finally:
            runtime.close()

    def answer(
        self,
        request: QuestionRequest,
        progress_callback: Callable[[str], None] | None = None,
    ) -> AnswerResult:
        document = self.document_loader.load(request.file_path)
        effective_iterations = self._clamp_iterations(request.max_root_iterations)
        effective_recursion_depth = self._clamp_recursion_depth(request.max_recursion_depth)
        initial_prompt = build_initial_user_prompt(document=document, question=request.question)
        return self._answer_from_context(
            context_text=document.text,
            question=request.question,
            metadata=self._document_metadata(document),
            effective_iterations=effective_iterations,
            remaining_depth=effective_recursion_depth,
            remaining_recursive_calls=self.limits.max_recursive_calls,
            current_depth=0,
            progress_callback=progress_callback,
            initial_user_prompt=initial_prompt,
        )
