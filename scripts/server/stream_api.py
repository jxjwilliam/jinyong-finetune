from __future__ import annotations

import json
import os
from typing import Any, AsyncGenerator

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    max_tokens: int = 256
    temperature: float = 0.7


def ollama_base_url() -> str:
    return os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")


def ollama_model() -> str:
    return os.getenv("OLLAMA_MODEL", "jinyong")


app = FastAPI(title="JinYong Streaming API", version="0.1.0")


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/generate")
async def generate(req: GenerateRequest) -> dict[str, Any]:
    payload = {
        "model": ollama_model(),
        "prompt": req.prompt,
        "stream": False,
        "options": {
            "num_predict": req.max_tokens,
            "temperature": req.temperature,
        },
    }
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(f"{ollama_base_url()}/api/generate", json=payload)
        if resp.status_code >= 400:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        body = resp.json()
        return {"text": body.get("response", ""), "raw": body}


async def ollama_stream(req: GenerateRequest) -> AsyncGenerator[dict[str, str], None]:
    payload = {
        "model": ollama_model(),
        "prompt": req.prompt,
        "stream": True,
        "options": {
            "num_predict": req.max_tokens,
            "temperature": req.temperature,
        },
    }
    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream("POST", f"{ollama_base_url()}/api/generate", json=payload) as resp:
            if resp.status_code >= 400:
                detail = await resp.aread()
                yield {"event": "error", "data": detail.decode("utf-8", errors="ignore")}
                return
            async for line in resp.aiter_lines():
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                chunk = str(obj.get("response", ""))
                if chunk:
                    yield {"event": "token", "data": chunk}
                if bool(obj.get("done", False)):
                    yield {"event": "done", "data": "true"}
                    return


@app.post("/v1/generate/stream")
async def generate_stream(req: GenerateRequest) -> EventSourceResponse:
    return EventSourceResponse(ollama_stream(req))


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("SERVER_HOST", "127.0.0.1")
    port = int(os.getenv("SERVER_PORT", "8000"))
    uvicorn.run("scripts.server.stream_api:app", host=host, port=port, reload=False)

