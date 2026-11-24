from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=4000)
    system_prompt: Optional[str] = Field(default=None, max_length=8000)

    @field_validator("message", mode="before")
    @classmethod
    def _strip_message(cls, v: Any) -> Any:
        # length rules apply after trimming whitespace
        return v.strip() if isinstance(v, str) else v

    @field_validator("system_prompt", mode="after")
    @classmethod
    def _blank_prompt_is_none(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v = v.strip()
        return v or None


class ChatMetadata(BaseModel):
    resolution_source: str
    model: str
    latency_ms: int
    input_tokens: int
    output_tokens: int
    key_id: Optional[str] = None
    fallback_attempts: int = 0


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    metadata: ChatMetadata
