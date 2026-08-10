"""
graphs/state.py — state schema for the fee-reminder-reminder HITL workflow.

Pydantic BaseModel, not TypedDict: this state is checkpointed to Redis and reloaded in a
*separate* HTTP request (possibly hours later, possibly after a deploy) — Pydantic validates
on load, so a corrupted/stale blob fails loudly instead of producing a deep KeyError inside a
node. Confirmed via a smoke test that LangGraph's StateGraph accepts a Pydantic model directly.

Deliberately excluded from this state: accessToken, X-Internal-Secret, or anything else
secret. The access token is threaded through per-invocation via `config["configurable"]`
instead (see graphs/fee_reminder_workflow.py) precisely so it's never checkpointed and a
resume always uses a *fresh* token from the request that triggered it, not a stale one from
whenever the workflow started.
"""
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class WorkflowStatus(str, Enum):
    PENDING_APPROVAL = "pending_approval"
    REJECTED = "rejected"
    SENT = "sent"
    PARTIALLY_SENT = "partially_sent"
    FAILED = "failed"
    NO_DEFAULTERS = "no_defaulters"


class DefaulterLite(BaseModel):
    """Minimal fields only — no parent contact info. Spring already has that; the graph
    never needs it, so it never sits in a Redis checkpoint."""
    studentId: str
    studentName: str
    className: str
    # None means "amount unknown" (no FeeStructure configured for this class/session in
    # Spring) — never coerce this to 0.0, that would misreport a real ₹0 due.
    totalDue: float | None = None


class SendOutcome(BaseModel):
    studentId: str
    status: str  # "sent" | "failed"


class SendResult(BaseModel):
    sentCount: int = 0
    failedCount: int = 0
    outcomes: list[SendOutcome] = Field(default_factory=list)  # capped, see fee_reminder_workflow.py
    error: str | None = None


class FeeReminderWorkflowState(BaseModel):
    workflow_id: str
    school_id: int
    admin_user_id: str
    session: str
    class_name: str | None = None

    status: WorkflowStatus = WorkflowStatus.PENDING_APPROVAL
    defaulter_count: int = 0
    # None means no defaulter in this batch has a known amount — see DefaulterLite.totalDue.
    total_amount_due: float | None = None
    defaulters: list[DefaulterLite] = Field(default_factory=list)

    draft_summary: str = ""

    approval_decision: str | None = None  # "approved" | "rejected"
    approved_by: str | None = None
    approved_at: str | None = None

    send_result: SendResult | None = None

    error: str | None = None
    retry_count: int = 0

    created_at: str = Field(default_factory=_now_iso)
    updated_at: str = Field(default_factory=_now_iso)
