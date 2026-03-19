from __future__ import annotations

import json
import pickle
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent

from file_rlm.config import DockerSettings, ModelSettings, RuntimeLimits
from file_rlm.contracts import REPLExecutionResult


@dataclass(slots=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


def _default_runner(command: list[str], workdir: Path, timeout_seconds: int) -> CommandResult:
    try:
        completed = subprocess.run(
            command,
            cwd=workdir,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        return CommandResult(
            returncode=124,
            stdout=exc.stdout or "",
            stderr=f"Timed out after {timeout_seconds} seconds.",
        )

    return CommandResult(
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


class DockerREPLRuntime:
    """Executes model-generated Python inside an ephemeral Docker container with checkpointed state."""

    def __init__(
        self,
        *,
        docker: DockerSettings,
        model_settings: ModelSettings,
        limits: RuntimeLimits,
        runner=_default_runner,
        workspace_root: Path | None = None,
    ) -> None:
        self.docker = docker
        self.model_settings = model_settings
        self.limits = limits
        self.runner = runner
        self.workspace_root = workspace_root
        self._state: dict[str, object] = {}

    def initialize(self, *, context: str, question: str, metadata: dict[str, object]) -> None:
        self._state = {
            "context": context,
            "query": question,
            "metadata": metadata,
            "__subcall_count": 0,
        }

    def _build_docker_command(self, workdir: Path) -> list[str]:
        mount = f"{workdir.resolve().as_posix()}:{self.docker.workdir}"
        return [
            "docker",
            "run",
            "--rm",
            "--read-only",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=64m",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            "--pids-limit",
            str(self.docker.pids_limit),
            "--memory",
            self.docker.memory_limit,
            "--cpus",
            self.docker.cpu_limit,
            "--user",
            self.docker.user,
            "-v",
            mount,
            "-w",
            self.docker.workdir,
            self.docker.image,
            "python",
            "runner.py",
        ]

    def _write_runner_script(self, workdir: Path) -> None:
        script = dedent(
            f"""
            import io
            import json
            import pickle
            import re
            import urllib.request
            from contextlib import redirect_stdout

            MODEL = {self.model_settings.subcall_model!r}
            OLLAMA_URL = {self.docker.ollama_url!r}
            MAX_STDOUT = {self.limits.max_stdout_chars}
            MAX_SUBQUERY_CHARS = {self.limits.max_chars_per_subquery}
            MAX_SUBCALLS = {self.limits.max_subcalls}
            TEMPERATURE = {self.model_settings.temperature}
            REQUEST_TIMEOUT = {self.model_settings.request_timeout_seconds}

            with open("state_in.pkl", "rb") as fh:
                state = pickle.load(fh)

            with open("user_code.py", "r", encoding="utf-8") as fh:
                user_code = fh.read()

            safe_builtins = {{
                "len": len,
                "range": range,
                "min": min,
                "max": max,
                "sum": sum,
                "sorted": sorted,
                "enumerate": enumerate,
                "print": print,
                "str": str,
                "int": int,
                "float": float,
                "bool": bool,
                "list": list,
                "dict": dict,
                "set": set,
                "tuple": tuple,
                "zip": zip,
                "any": any,
                "all": all,
                "abs": abs,
                "round": round,
            }}

            def blocked_import(*args, **kwargs):
                raise ImportError(
                    "Imports are disabled in this REPL. Use the preloaded variables "
                    "`context`, `query`, `metadata`, `llm_query`, and `re`."
                )

            safe_builtins["__import__"] = blocked_import

            def llm_query(text: str) -> str:
                if not isinstance(text, str):
                    raise TypeError("llm_query(text) expects a string argument.")
                if len(text) > MAX_SUBQUERY_CHARS:
                    raise ValueError(
                        f"llm_query payload exceeds the {{MAX_SUBQUERY_CHARS}}-character limit. "
                        f"Received {{len(text)}} characters."
                    )

                current_count = int(state.get("__subcall_count", 0))
                if current_count >= MAX_SUBCALLS:
                    raise RuntimeError(
                        f"llm_query subcall limit reached ({{MAX_SUBCALLS}}). Consolidate evidence and stop making more subcalls."
                    )
                state["__subcall_count"] = current_count + 1

                payload = json.dumps({{
                    "model": MODEL,
                    "prompt": text,
                    "stream": False,
                    "options": {{"temperature": TEMPERATURE}},
                }}).encode("utf-8")
                req = urllib.request.Request(
                    url=f"{{OLLAMA_URL}}/api/generate",
                    data=payload,
                    headers={{"Content-Type": "application/json"}},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as response:
                    body = json.loads(response.read().decode("utf-8"))
                return body["response"]

            env = {{
                "__builtins__": safe_builtins,
                "context": state["context"],
                "query": state["query"],
                "metadata": state["metadata"],
                "llm_query": llm_query,
                "re": re,
            }}

            for key, value in state.items():
                if not key.startswith("__"):
                    env[key] = value

            stdout_buffer = io.StringIO()
            error = None

            try:
                with redirect_stdout(stdout_buffer):
                    exec(user_code, env, env)
            except Exception as exc:
                error = f"{{type(exc).__name__}}: {{exc}}"

            new_state = {{"__subcall_count": state.get("__subcall_count", 0)}}
            for key, value in env.items():
                if key in {{"__builtins__", "llm_query", "re"}}:
                    continue
                if callable(value):
                    continue
                try:
                    pickle.dumps(value)
                except Exception:
                    continue
                new_state[key] = value

            stdout = stdout_buffer.getvalue()[:MAX_STDOUT]
            with open("state_out.pkl", "wb") as fh:
                pickle.dump(new_state, fh)
            with open("result.json", "w", encoding="utf-8") as fh:
                json.dump({{
                    "stdout": stdout,
                    "state_keys": sorted(new_state.keys()),
                    "error": error,
                }}, fh)
            """
        ).strip()
        (workdir / "runner.py").write_text(script, encoding="utf-8")

    def execute(self, code: str) -> REPLExecutionResult:
        if not self._state:
            raise RuntimeError("Runtime must be initialized before execution.")
        if len(code) > self.limits.max_repl_code_chars:
            raise RuntimeError(
                f"REPL code exceeds the {self.limits.max_repl_code_chars}-character limit."
            )

        base_dir = self.workspace_root
        if base_dir is None:
            temp_dir = tempfile.TemporaryDirectory()
            workdir = Path(temp_dir.name)
            cleanup = temp_dir
        else:
            workdir = self.workspace_root / "docker_runtime"
            workdir.mkdir(parents=True, exist_ok=True)
            cleanup = None

        workdir.chmod(0o777)

        try:
            (workdir / "state_in.pkl").write_bytes(pickle.dumps(self._state))
            (workdir / "user_code.py").write_text(code, encoding="utf-8")
            self._write_runner_script(workdir)

            command = self._build_docker_command(workdir)
            result = self.runner(command, workdir, self.limits.repl_timeout_seconds)
            if result.returncode != 0:
                raise RuntimeError(result.stderr or "Docker REPL execution failed.")

            payload = json.loads((workdir / "result.json").read_text(encoding="utf-8"))
            self._state = pickle.loads((workdir / "state_out.pkl").read_bytes())
            return REPLExecutionResult(
                stdout=payload["stdout"],
                state_keys=tuple(payload["state_keys"]),
                error=payload["error"],
            )
        finally:
            if cleanup is not None:
                cleanup.cleanup()

    def get_variable(self, name: str) -> object:
        return self._state[name]

    def set_variable(self, name: str, value: object) -> None:
        try:
            pickle.dumps(value)
        except Exception as exc:
            raise TypeError(f"Value for `{name}` is not pickleable and cannot persist across steps.") from exc
        self._state[name] = value

    def list_variables(self) -> tuple[str, ...]:
        return tuple(sorted(self._state.keys()))

    def close(self) -> None:
        return None
