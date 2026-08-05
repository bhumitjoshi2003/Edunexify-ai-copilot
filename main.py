"""
main.py — FastAPI application entry point.

This service is internal — it should never be exposed directly to the internet.
Spring Boot is the only caller. All business auth lives in Spring Boot.
"""
from fastapi import FastAPI

import memory
from routers import chat

app = FastAPI(
    title="Edunexify AI Service",
    version="0.1.0",
    # Disable the public /docs and /redoc pages in prod to avoid exposing internals.
    # Set docs_url="/docs" locally if you want the Swagger UI during development.
    docs_url="/docs",
    redoc_url=None,
)

app.include_router(chat.router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.on_event("shutdown")
async def shutdown() -> None:
    await memory.close()
