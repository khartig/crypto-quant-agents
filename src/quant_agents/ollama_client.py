from __future__ import annotations

import json
from typing import Any
from urllib import error, request


class OllamaClient:
    def __init__(self, base_url: str, timeout_seconds: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def _request_json(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        req = request.Request(
            url=f"{self.base_url}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method=method,
        )
        effective_timeout = self.timeout_seconds if timeout_seconds is None else max(1.0, float(timeout_seconds))
        try:
            with request.urlopen(req, timeout=effective_timeout) as response:
                body = response.read().decode("utf-8")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama HTTP error {exc.code}: {detail}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"Ollama connection error: {exc.reason}") from exc

        try:
            decoded = json.loads(body)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Ollama returned non-JSON response.") from exc
        if not isinstance(decoded, dict):
            raise RuntimeError("Ollama returned unexpected response format.")
        return decoded

    def list_models(self) -> list[str]:
        decoded = self._request_json("GET", "/api/tags")
        models = decoded.get("models", [])
        if not isinstance(models, list):
            return []
        names: list[str] = []
        for model in models:
            if not isinstance(model, dict):
                continue
            name = model.get("name")
            if isinstance(name, str) and name.strip():
                names.append(name.strip())
        return names

    def generate(
        self,
        *,
        model: str,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.1,
        format_json: bool = False,
        num_predict: int | None = None,
        timeout_seconds: float | None = None,
    ) -> str:
        options: dict[str, Any] = {"temperature": temperature}
        if num_predict is not None:
            options["num_predict"] = max(1, int(num_predict))
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": options,
        }
        if system:
            payload["system"] = system
        if format_json:
            payload["format"] = "json"

        decoded = self._request_json("POST", "/api/generate", payload, timeout_seconds=timeout_seconds)
        response = decoded.get("response")
        if not isinstance(response, str):
            raise RuntimeError("Ollama response did not include text output.")
        return response.strip()
