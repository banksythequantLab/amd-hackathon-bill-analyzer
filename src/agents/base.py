"""
Agent base class for Bill Analyzer.

Pattern:
  - Every agent has a name, a target chunk, and either the spine or reasoner endpoint
  - Agents call an LLM with a system prompt + user prompt
  - Agents may call tools (e.g. fetch_usc) which are passed in by the orchestrator
  - Agents return a typed Pydantic model (or dict if model is None)
  - Failures are retried up to N times with exponential backoff
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import httpx
from pydantic import BaseModel


# Default endpoints — overridden in production via the orchestrator
SPINE_ENDPOINT    = "http://165.245.134.1:8001/v1"
REASONER_ENDPOINT = "http://165.245.134.1:8003/v1"
VISION_ENDPOINT   = "http://165.245.134.1:8002/v1"


@dataclass
class AgentResult:
    """Container for what an agent produces."""
    agent_name: str
    chunk_id: str
    output: dict | str   # parsed JSON dict, or raw string if no schema
    raw_response: str    # full model response text
    elapsed_ms: float
    prompt_tokens: int
    completion_tokens: int
    tool_calls: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class AgentBase:
    """Subclass and override `system_prompt`, `user_prompt`, `output_schema`."""

    name: str = "base"
    target_endpoint: str = SPINE_ENDPOINT  # override per-agent
    target_model: str = "spine"
    temperature: float = 0.0
    max_tokens: int = 2000

    # Pydantic schema for structured output. None = unstructured string.
    output_schema: Optional[type[BaseModel]] = None

    def system_prompt(self) -> str:
        raise NotImplementedError

    def user_prompt(self, chunk_text: str, chunk_id: str, **context) -> str:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Core LLM call (synchronous, OpenAI-compatible)
    # ------------------------------------------------------------------
    def call_llm(self, messages: list[dict], stream: bool = False) -> dict:
        payload = {
            "model": self.target_model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        with httpx.Client(timeout=600.0) as client:
            r = client.post(f"{self.target_endpoint}/chat/completions", json=payload)
            r.raise_for_status()
            return r.json()

    # ------------------------------------------------------------------
    # Tool dispatch — subclasses set self.tools = {"name": callable}
    # ------------------------------------------------------------------
    tools: dict[str, Callable] = {}

    def call_tool(self, tool_name: str, **kwargs) -> Any:
        if tool_name not in self.tools:
            raise ValueError(f"Unknown tool: {tool_name}")
        return self.tools[tool_name](**kwargs)

    # ------------------------------------------------------------------
    # Output parsing — handles Thinking-mode reasoning chains by taking
    # the LAST balanced JSON object in the response
    # ------------------------------------------------------------------
    @staticmethod
    def extract_json(content: str) -> Optional[dict]:
        # Strip <think>...</think> blocks (Qwen3 Thinking variants) — both closed
        # tags and the leading-`<think>` / trailing-`</think>` stragglers.
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
        # Some responses emit just the OPEN tag mid-stream and keep thinking until
        # max_tokens is hit. If we still see an unmatched <think>, drop everything
        # before it (the closing tag is missing because the model was truncated).
        if "<think>" in content and "</think>" not in content:
            # No closing tag — entire response was thinking. Nothing to extract.
            return None
        # Also strip a stray closing tag with no opener.
        content = content.replace("</think>", "")

        # Try fenced ```json``` block first (last one wins)
        fences = list(re.finditer(r"```(?:json)?\s*(\{.+?\})\s*```", content, re.DOTALL))
        if fences:
            try:
                return json.loads(fences[-1].group(1))
            except json.JSONDecodeError:
                pass

        # Try LAST balanced top-level JSON
        last_close = content.rfind("}")
        while last_close > 0:
            depth = 0
            for i in range(last_close, -1, -1):
                ch = content[i]
                if ch == "}":
                    depth += 1
                elif ch == "{":
                    depth -= 1
                    if depth == 0:
                        candidate = content[i:last_close + 1]
                        try:
                            return json.loads(candidate)
                        except json.JSONDecodeError:
                            break
            last_close = content.rfind("}", 0, last_close)
        return None

    # ------------------------------------------------------------------
    # Run the agent against a chunk
    # ------------------------------------------------------------------
    def run(self, chunk_text: str, chunk_id: str, max_retries: int = 2,
            **context) -> AgentResult:
        sys_prompt = self.system_prompt()
        usr_prompt = self.user_prompt(chunk_text, chunk_id, **context)

        last_err = None
        last_raw = ""
        for attempt in range(max_retries + 1):
            t_start = time.perf_counter()
            try:
                response = self.call_llm([
                    {"role": "system", "content": sys_prompt},
                    {"role": "user",   "content": usr_prompt},
                ])
                elapsed_ms = (time.perf_counter() - t_start) * 1000

                content = response["choices"][0]["message"]["content"]
                last_raw = content
                usage = response.get("usage", {})

                # Structured output
                if self.output_schema is not None:
                    parsed = self.extract_json(content)
                    if parsed is None:
                        last_err = f"attempt {attempt}: response was not parseable JSON"
                        continue
                    try:
                        validated = self.output_schema(**parsed)
                        output = validated.model_dump()
                    except Exception as e:
                        last_err = f"attempt {attempt}: schema validation failed: {e}"
                        continue
                else:
                    output = content

                return AgentResult(
                    agent_name=self.name,
                    chunk_id=chunk_id,
                    output=output,
                    raw_response=content,
                    elapsed_ms=elapsed_ms,
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                )
            except (httpx.HTTPError, httpx.TimeoutException) as e:
                last_err = f"attempt {attempt}: HTTP error: {e}"
                if attempt < max_retries:
                    time.sleep(2 ** attempt)
                continue

        # All retries exhausted
        return AgentResult(
            agent_name=self.name,
            chunk_id=chunk_id,
            output={},
            raw_response=last_raw,
            elapsed_ms=0.0,
            prompt_tokens=0,
            completion_tokens=0,
            errors=[last_err or "unknown"],
        )
