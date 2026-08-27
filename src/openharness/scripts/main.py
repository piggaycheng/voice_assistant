import asyncio
import os

import uvicorn
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="OpenHarness HTTP Bridge")

BRIDGE_KEY = os.getenv("OPENHARNESS_BRIDGE_KEY", "")
OPENHARNESS_COMMAND = os.getenv("OPENHARNESS_COMMAND", "oh")
PERMISSION_MODE = os.getenv("OPENHARNESS_PERMISSION_MODE", "plan")
MAX_TURNS = int(os.getenv("OPENHARNESS_MAX_TURNS", "10"))
PROCESS_TIMEOUT = float(os.getenv("OPENHARNESS_TIMEOUT", "120"))


class QueryRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=20000)


class QueryResponse(BaseModel):
    answer: str


def verify_api_key(authorization: str | None) -> None:
    if BRIDGE_KEY and authorization != f"Bearer {BRIDGE_KEY}":
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.get("/")
async def health_check():
    return {
        "status": "running",
        "service": "openharness-http-bridge",
        "authentication_required": bool(BRIDGE_KEY),
    }


@app.post("/query", response_model=QueryResponse)
async def query_openharness(
    request: QueryRequest,
    authorization: str | None = Header(default=None),
):
    verify_api_key(authorization)

    command = [
        OPENHARNESS_COMMAND,
        "-p",
        request.prompt,
        "--output-format",
        "text",
        "--permission-mode",
        PERMISSION_MODE,
        "--max-turns",
        str(MAX_TURNS),
    ]

    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            process.communicate(),
            timeout=PROCESS_TIMEOUT,
        )
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail="OpenHarness CLI is not installed") from error
    except TimeoutError as error:
        process.kill()
        await process.communicate()
        raise HTTPException(status_code=504, detail="OpenHarness request timed out") from error

    if process.returncode != 0:
        error_message = stderr.decode(errors="replace").strip()
        raise HTTPException(
            status_code=502,
            detail=error_message or f"OpenHarness exited with code {process.returncode}",
        )

    answer = stdout.decode(errors="replace").strip()
    if not answer:
        raise HTTPException(status_code=502, detail="OpenHarness returned an empty response")

    return QueryResponse(answer=answer)


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=os.getenv("OPENHARNESS_BRIDGE_HOST", "0.0.0.0"),
        port=int(os.getenv("OPENHARNESS_BRIDGE_PORT", "8010")),
    )