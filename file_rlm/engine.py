from __future__ import annotations

import re
from typing import Callable

from file_rlm.config import ModelSettings, RuntimeLimits
from file_rlm.contracts import AnswerResult, QuestionRequest, REPLRuntime
from file_rlm.loaders import DocumentLoader, LoadedDocument
from file_rlm.prompts import build_follow_up_prompt, build_initial_user_prompt, build_root_system_prompt

FINAL_VAR_RE = re.compile(r"FINAL_VAR\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)")
FINAL_RE = re.compile(r"FINAL\((.*)\)", re.DOTALL)
REPL_BLOCK_RE = re.compile(r"```repl\s*(.*?)```", re.DOTALL | re.IGNORECASE)


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
        final_var = FINAL_VAR_RE.search(response)
        if final_var:
            return ("final_var", final_var.group(1))

        final = FINAL_RE.search(response)
        if final:
            return ("final", final.group(1).strip())

        repl_blocks = REPL_BLOCK_RE.findall(response)
        if repl_blocks:
            return ("repl", "\n\n".join(block.strip() for block in repl_blocks if block.strip()))

        return ("invalid", response.strip())

    def _document_metadata(self, document: LoadedDocument) -> dict[str, object]:
        return {
            "path": str(document.path),
            "file_type": document.file_type,
            "char_count": document.char_count,
            "line_count": document.line_count,
        }

    def answer(
        self,
        request: QuestionRequest,
        progress_callback: Callable[[str], None] | None = None,
    ) -> AnswerResult:
        document = self.document_loader.load(request.file_path)
        runtime = self.runtime_factory()
        runtime.initialize(
            context=document.text,
            question=request.question,
            metadata=self._document_metadata(document),
        )

        system_prompt = build_root_system_prompt(
            max_chars_per_subquery=self.limits.max_chars_per_subquery
        )
        user_prompt = build_initial_user_prompt(document=document, question=request.question)
        trace: list[str] = []
        emit = progress_callback or (lambda _message: None)

        try:
            for iteration in range(1, request.max_root_iterations + 1):
                emit(f"Step {iteration}/{request.max_root_iterations}: querying root model")
                response = self.ollama_client.generate(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    model=self.model_settings.root_model,
                )
                trace.append(f"assistant:{response[:200]}")
                action, payload = self._extract_action(response)

                if action == "final":
                    emit(f"Step {iteration}/{request.max_root_iterations}: Final answer\n{payload}")
                    return AnswerResult(
                        answer=payload,
                        iterations=iteration,
                        subcall_count=int(runtime.get_variable("__subcall_count")),
                        recursion_depth=request.max_recursion_depth,
                        prompt_chars=document.char_count,
                        trace=trace,
                    )

                if action == "final_var":
                    emit(
                        f"Step {iteration}/{request.max_root_iterations}: Final answer from variable `{payload}`"
                    )
                    return AnswerResult(
                        answer=str(runtime.get_variable(payload)),
                        iterations=iteration,
                        subcall_count=int(runtime.get_variable("__subcall_count")),
                        recursion_depth=request.max_recursion_depth,
                        prompt_chars=document.char_count,
                        trace=trace,
                    )

                if action == "repl":
                    emit(
                        f"Step {iteration}/{request.max_root_iterations}: Generated REPL code\n{payload}"
                    )
                    execution = runtime.execute(payload)
                    trace.append(f"repl:{execution.stdout[:200]}")
                    if execution.error:
                        emit(
                            f"Step {iteration}/{request.max_root_iterations}: REPL error\n{execution.error}"
                        )
                    if execution.stdout:
                        emit(
                            f"Step {iteration}/{request.max_root_iterations}: REPL stdout\n{execution.stdout}"
                        )
                    user_prompt = build_follow_up_prompt(
                        question=request.question,
                        execution=execution,
                    )
                    continue

                emit(
                    f"Step {iteration}/{request.max_root_iterations}: Invalid model output\n{payload}"
                )
                user_prompt = (
                    f"The previous response was invalid:\n{payload}\n\n"
                    "Reply with a ```repl``` block, FINAL(...), or FINAL_VAR(...)."
                )

            raise RuntimeError("The RLM engine reached its iteration limit without a final answer.")
        finally:
            runtime.close()
