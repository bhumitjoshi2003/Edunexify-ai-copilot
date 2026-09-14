"""
main.py — FastAPI application entry point.

This service is internal — it should never be exposed directly to the internet.
Spring Boot is the only caller. All business auth lives in Spring Boot.
"""
from fastapi import FastAPI, Request

import checkpointer
import memory
from observability import (
    REQUEST_ID_HEADER, initialize_sentry, normalize_request_id, reset_request_id, set_request_id,
)
from routers import (
    chat,
    knowledge_base,
    workflows_attendance_reminders,
    workflows_fee_reminders,
    workflows_teacher_attendance_reminders,
    workflows_leave_decisions,
)

initialize_sentry()

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


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = normalize_request_id(request.headers.get(REQUEST_ID_HEADER))
    token = set_request_id(request_id)
    try:
        response = await call_next(request)
        response.headers[REQUEST_ID_HEADER] = request_id
        return response
    finally:
        reset_request_id(token)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.on_event("shutdown")
async def shutdown() -> None:
    await memory.close()
    await checkpointer.close()
