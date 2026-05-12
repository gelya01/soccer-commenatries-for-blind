from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DEFAULT_OPENROUTER_BASE = "https://openrouter.ai/api/v1"
DEFAULT_CHAIN_SYSTEM_PROMPT = (
    "Ты спортивный тифлокомментатор. На входе цепочка событий StatsBomb. "
    "Выбери только ключевые события и кратко прокомментируй их без домыслов. "
    'Верни строго JSON: {"summary":"...","items":[{"event_id":"...","timestamp":"...","action":"...","commentary":"..."}]}.'
)


@dataclass
class OpenRouterCommentaryResult:
    parsed: dict[str, Any]
    raw_content: str
    raw_response: dict[str, Any]


class OpenRouterAPIError(RuntimeError):
    pass


class OpenRouterClient:
    def __init__(
        self,
        api_key: str | None = None,
        *,
        api_base: str = DEFAULT_OPENROUTER_BASE,
        app_title: str = "statsbomb_toolkit",
        referer: str = "http://localhost",
        timeout: int = 60,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("Set api_key or env OPENROUTER_API_KEY.")

        self.api_base = api_base.rstrip("/")
        self.app_title = app_title
        self.referer = referer
        self.timeout = timeout

        self._session = requests.Session()
        retry = Retry(
            total=3,
            connect=3,
            read=3,
            status=3,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(["GET", "POST"]),
            backoff_factor=1.0,
        )
        self._session.mount("https://", HTTPAdapter(max_retries=retry))

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.referer,
            "X-OpenRouter-Title": self.app_title,
        }

    @staticmethod
    def _raise_api_error(resp: requests.Response, context: str) -> None:
        try:
            payload = resp.json()
        except Exception:
            payload = resp.text
        raise OpenRouterAPIError(
            f"{context} failed: HTTP {resp.status_code}: {payload}"
        )

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any]:
        # Try direct parse first.
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

        # Common case: fenced json block.
        fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
        if fenced:
            return json.loads(fenced.group(1))

        # Fallback: first json object in text.
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            raise json.JSONDecodeError("No JSON object found", text, 0)
        return json.loads(m.group(0))

    def list_models(self, *, user_only: bool = True) -> list[dict[str, Any]]:
        endpoint = "/models/user" if user_only else "/models"
        resp = self._session.get(
            f"{self.api_base}{endpoint}",
            headers=self._headers(),
            timeout=self.timeout,
        )
        if resp.status_code >= 400:
            self._raise_api_error(resp, "List models")
        data = resp.json()
        return data.get("data", [])

    def chat_completion(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 500,
        response_format: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format is not None:
            payload["response_format"] = response_format

        resp = self._session.post(
            f"{self.api_base}/chat/completions",
            headers=self._headers(),
            json=payload,
            timeout=self.timeout,
        )
        if resp.status_code >= 400:
            self._raise_api_error(resp, "Chat completion")
        return resp.json()

    def generate_chain_commentary(
        self,
        chain_payload: Any,
        *,
        model: str,
        system_prompt: str = DEFAULT_CHAIN_SYSTEM_PROMPT,
        temperature: float = 0.2,
        max_tokens: int = 600,
        force_json: bool = True,
    ) -> OpenRouterCommentaryResult:
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": json.dumps(chain_payload, ensure_ascii=False),
            },
        ]
        response_format = {"type": "json_object"} if force_json else None
        raw = self.chat_completion(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
        )
        text = raw["choices"][0]["message"]["content"]
        parsed = self._extract_json(text)
        return OpenRouterCommentaryResult(
            parsed=parsed, raw_content=text, raw_response=raw
        )
