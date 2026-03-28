import threading
import time

from file_rlm.config import ModelSettings
from file_rlm.llama_cpp_client import LlamaCppClient


class FakeLlama:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.active_calls = 0
        self.max_active_calls = 0
        self.call_thread_ids: list[int] = []

    def create_chat_completion(self, *, messages, temperature):
        del messages, temperature
        with self._lock:
            self.active_calls += 1
            self.max_active_calls = max(self.max_active_calls, self.active_calls)
            self.call_thread_ids.append(threading.get_ident())
        try:
            time.sleep(0.05)
            return {"choices": [{"message": {"content": "OK"}}]}
        finally:
            with self._lock:
                self.active_calls -= 1


class FakeSubcallLlama:
    def create_completion(self, *, prompt, temperature, max_tokens):
        del temperature, max_tokens
        return {"choices": [{"text": f"SUB:{prompt}"}]}


def test_generate_serializes_native_llama_calls() -> None:
    client = LlamaCppClient(settings=ModelSettings())
    fake_llm = FakeLlama()
    client._root_llm = fake_llm

    barrier = threading.Barrier(2)
    results: list[str] = []

    def run_call() -> None:
        barrier.wait(timeout=1)
        results.append(
            client.generate(
                system_prompt="Reply with OK",
                user_prompt="Say OK",
                model="ignored",
            )
        )

    threads = [threading.Thread(target=run_call) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert results == ["OK", "OK"]
    assert fake_llm.max_active_calls == 1
    assert len(set(fake_llm.call_thread_ids)) == 2


def test_generate_subcall_uses_loaded_subcall_model() -> None:
    client = LlamaCppClient(settings=ModelSettings())
    client._subcall_llm = FakeSubcallLlama()

    result = client.generate_subcall(prompt="needle")

    assert result == "SUB:needle"
