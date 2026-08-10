"""
schemas/workflows.py — Pydantic models for the /workflows/fee-reminders endpoints.

Mirrors schemas/chat.py's shape (UserContext reused as-is) so Spring Boot's proxy code stays
consistent between the chat and workflow call sites.
"""
from typing import Literal

from pydantic import BaseModel

from schemas.chat import UserContext


class WorkflowStartRequest(BaseModel):
    session: str
    className: str | None = None
    user: UserContext
    accessToken: str


class WorkflowResumeRequest(BaseModel):
    decision: Literal["approved", "rejected"]
    schoolId: int
    adminUserId: str
    accessToken: str


class DefaulterOut(BaseModel):
    studentId: str
    studentName: str
    className: str
    totalDue: float | None = None  # None = amount unknown, not ₹0 — see graphs/state.py


class SendOutcomeOut(BaseModel):
    studentId: str
    status: str


class SendResultOut(BaseModel):
    sentCount: int
    failedCount: int
    outcomes: list[SendOutcomeOut] = []
    error: str | None = None


class WorkflowStateResponse(BaseModel):
    """Returned by start, resume, and the status GET — same shape from all three so
    Angular has one type to render regardless of which call produced it."""
    workflowId: str
    status: str
    draftSummary: str
    defaulterCount: int
    totalAmountDue: float | None = None  # None = amount unknown, not ₹0
    defaulters: list[DefaulterOut] = []
    sendResult: SendResultOut | None = None
    error: str | None = None
