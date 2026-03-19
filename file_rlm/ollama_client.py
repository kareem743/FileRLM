from __future__ import annotations

import json
from urllib import request


class OllamaHTTPClient:
    """Small HTTP adapter for Ollama's chat endpoint."""

    def __init__(self, host: str) -> None:
        self.host = host.rstrip("/")

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: str,
        temperature: float = 0.0,
        timeout_seconds: int = 120,
    ) -> str:
        payload = {
            "model": model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "options": {"temperature": temperature},
        }
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(
            url=f"{self.host}/api/chat",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=timeout_seconds) as response:
            body = json.loads(response.read().decode("utf-8"))

        return body["message"]["content"]
