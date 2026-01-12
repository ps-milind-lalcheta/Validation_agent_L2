from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Optional, Type, TypeVar

import httpx
from pydantic import BaseModel, ValidationError


T = TypeVar("T", bound=BaseModel)


@dataclass
class LLMConfig:
    base_url: str
    api_key: str
    model: str
    timeout_s: float = 60.0
    temperature: float = 0.2
    max_tokens: int = 8000


class LLMClient:
    """OpenAI-compatible Chat Completions client."""

    def __init__(self, config: Optional[LLMConfig] = None):
        if config is None:
            config = LLMConfig(
                base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
                api_key=os.getenv("LLM_API_KEY", ""),
                model=os.getenv("LLM_MODEL", "gpt-4.1-mini"),
            )
        if not config.api_key:
            raise ValueError("LLM_API_KEY is required.")
        self.config = config

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.config.api_key}", "Content-Type": "application/json"}

    def chat(self, system: str, user: str) -> str:
        url = f"{self.config.base_url}/chat/completions"
        payload = {
            "model": self.config.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        
        import time
        max_retries = 5
        base_delay = 2.0
        
        for attempt in range(max_retries + 1):
            with httpx.Client(timeout=self.config.timeout_s) as client:
                r = client.post(url, headers=self._headers(), json=payload)
                if r.status_code == 429:
                    if attempt < max_retries:
                        time.sleep(base_delay * (2 ** attempt))
                        continue
                r.raise_for_status()
                data = r.json()
                # Enforce rate limit (max 15 RPM -> 1 req / 4 sec)
                time.sleep(4.0)
                # print(f"DEBUG: Response data: {json.dumps(data, indent=2)}")
                if "content" not in data["choices"][0]["message"]:
                     raise RuntimeError(f"Missing 'content' in response: {json.dumps(data)}")
                return data["choices"][0]["message"]["content"]
        raise RuntimeError(f"Failed after {max_retries} retries due to rate limiting.")

    def chat_json(self, system: str, user: str, schema: Type[T], retries: int = 3) -> T:
        last_text: Optional[str] = None
        for attempt in range(retries + 1):
            text = self.chat(system=system, user=user) if attempt == 0 else self._repair_json(system, last_text, schema)
            last_text = text
            try:
                obj = self._parse_json(text)
                return schema.model_validate(obj)
            except (ValidationError, json.JSONDecodeError):
                if attempt >= retries:
                    raise
        raise RuntimeError("Unexpected JSON validation failure.")

    def _parse_json(self, text: str) -> Any:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            cleaned = cleaned.replace("json\n", "", 1).strip()
        return json.loads(cleaned)

    def _repair_json(self, system: str, bad_text: str | None, schema: Type[T]) -> str:
        schema_json = schema.model_json_schema()
        repair_system = system + "\nYou are also a JSON repair assistant. Output ONLY corrected JSON."
        repair_user = f"""The previous output was invalid JSON or didn't match schema.

INVALID_OUTPUT:
{bad_text}

TARGET_SCHEMA_JSON:
{json.dumps(schema_json, ensure_ascii=False)}

Return ONLY valid JSON matching the schema. No markdown. No commentary.
"""
        return self.chat(system=repair_system, user=repair_user)
