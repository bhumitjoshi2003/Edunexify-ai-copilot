"""
main.py — FastAPI application entry point.

This service is internal — it should never be exposed directly to the internet.
Spring Boot is the only caller. All business auth lives in Spring Boot.
"""
from fastapi import FastAPI

import checkpointer
import memory
from routers import (
    chat,
    knowledge_base,
    workflows_attendance_reminders,
    workflows_fee_reminders,
    workflows_teacher_attendance_reminders,
    workflows_leave_decisions,
)

app = FastAPI(
    title="Edunexify AI Service",
    version="0.1.1",
    # Disable the public /docs and /redoc pages in prod to avoid exposing internals.
    # Set docs_url="/docs" locally if you want the Swagger UI during development.
    docs_url="/docs",
    redoc_url=None,
)

app.include_router(chat.router)
app.include_router(knowledge_base.router)
app.include_router(workflows_fee_reminders.router)
app.include_router(workflows_attendance_reminders.router)
app.include_router(workflows_teacher_attendance_reminders.router)
app.include_router(workflows_leave_decisions.router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.on_event("shutdown")
async def shutdown() -> None:
    await memory.close()
    await checkpointer.close()
