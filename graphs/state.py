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


# ─── Attendance Reminder workflow ───────────────────────────────────────────────
# SendOutcome/SendResult above are already domain-neutral (studentId/status,
# sentCount/failedCount/outcomes/error) — reused as-is, not redefined here.

class AttendanceWorkflowStatus(str, Enum):
    PENDING_APPROVAL = "pending_approval"
    REJECTED = "rejected"
    SENT = "sent"
    PARTIALLY_SENT = "partially_sent"
    FAILED = "failed"
    NO_LOW_ATTENDANCE_STUDENTS = "no_low_attendance_students"


class LowAttendanceStudentLite(BaseModel):
    """Minimal fields only — no parent contact info, same reasoning as DefaulterLite."""
    studentId: str
    studentName: str
    className: str
    attendancePercentage: float
    daysPresent: int
    daysAbsent: int
    totalWorkingDays: int

    # Populated only for a CONSECUTIVE_ABSENCE batch; None means "this student wasn't selected
    # by a streak", never "a streak of zero". Kept optional rather than defaulted to 0 so the
    # approval card can tell those two apart — see AttendanceCriterion below.
    consecutiveAbsentDays: int | None = None
    absentDates: list[str] | None = None


class AttendanceReminderWorkflowState(BaseModel):
    workflow_id: str
    school_id: int
    admin_user_id: str
    session: str
    class_name: str | None = None
    threshold: float = 75.0

    status: AttendanceWorkflowStatus = AttendanceWorkflowStatus.PENDING_APPROVAL
    student_count: int = 0
    students: list[LowAttendanceStudentLite] = Field(default_factory=list)

    draft_summary: str = ""

    approval_decision: str | None = None  # "approved" | "rejected"
    approved_by: str | None = None
    approved_at: str | None = None

    send_result: SendResult | None = None

    error: str | None = None
    retry_count: int = 0

    created_at: str = Field(default_factory=_now_iso)
    updated_at: str = Field(default_factory=_now_iso)


# ─── Teacher-scoped Attendance Reminder workflow ────────────────────────────────
# Reuses AttendanceWorkflowStatus/LowAttendanceStudentLite/SendResult as-is — same
# domain-neutral shapes as the admin workflow. Kept as its own state class (rather than
# reusing AttendanceReminderWorkflowState) because class_name here is REQUIRED, never
# optional: a teacher-initiated batch always belongs to exactly one class, resolved
# server-side by Spring from the teacher's own classTeacher field before this state is
# ever constructed — see AiTeacherAttendanceWorkflowController.start(). Naming the field
# teacher_user_id (not admin_user_id) keeps that guarantee visible in the state itself.

class AttendanceCriterion(str, Enum):
    """Which attendance pattern selects a teacher's reminder batch.

    These are genuinely different questions, not two settings of one knob: a student absent the
    last three days running may sit comfortably above the percentage threshold, and a student at
    40% for the year may have attended every day this week. Keeping them as an explicit enum
    (rather than, say, inferring from which parameter was supplied) means the criterion survives
    into the checkpoint, the Spring audit row, and the approval card the teacher actually reads.
    """
    BELOW_THRESHOLD = "BELOW_THRESHOLD"
    CONSECUTIVE_ABSENCE = "CONSECUTIVE_ABSENCE"


class TeacherAttendanceReminderWorkflowState(BaseModel):
    workflow_id: str
    school_id: int
    teacher_user_id: str
    class_name: str
    session: str

    criterion: AttendanceCriterion = AttendanceCriterion.BELOW_THRESHOLD
    threshold: float = 75.0              # used when criterion is BELOW_THRESHOLD
    min_consecutive_days: int | None = None  # used when criterion is CONSECUTIVE_ABSENCE

    status: AttendanceWorkflowStatus = AttendanceWorkflowStatus.PENDING_APPROVAL
    student_count: int = 0
    students: list[LowAttendanceStudentLite] = Field(default_factory=list)

    draft_summary: str = ""

    approval_decision: str | None = None  # "approved" | "rejected"
    approved_by: str | None = None
    approved_at: str | None = None

    send_result: SendResult | None = None

    error: str | None = None
    retry_count: int = 0

    created_at: str = Field(default_factory=_now_iso)
    updated_at: str = Field(default_factory=_now_iso)


# ─── Leave Decision workflow ────────────────────────────────────────────────────
# Structurally a sibling of the reminder workflows, with one categorical difference worth
# stating plainly: a reminder batch has no recipient parameter at all, so the model cannot
# choose who is contacted. A leave decision necessarily names specific requests, so the
# model's ids are a PROPOSAL that Spring re-resolves and re-validates — never an instruction.
# See AiLeaveWorkflowController's Javadoc for the full defence-in-depth chain.

class LeaveWorkflowStatus(str, Enum):
    PENDING_APPROVAL = "pending_approval"
    REJECTED = "rejected"
    APPLIED = "applied"
    PARTIALLY_APPLIED = "partially_applied"
    FAILED = "failed"
    NO_ACTIONABLE_REQUESTS = "no_actionable_requests"


class LeaveRequestLite(BaseModel):
    """One leave request as shown on the approval card. No contact details — the card is for
    identifying WHICH request, and Spring already holds everything needed to act on it."""
    leaveId: int
    studentId: str
    studentName: str
    className: str
    leaveDate: str
    reason: str
    # Current status at draft time. Anything other than PENDING is shown to the reviewer but
    # excluded from the actionable set — it is someone else's decision, already made.
    status: str


class LeaveOutcome(BaseModel):
    leaveId: int
    outcome: str  # applied | skipped_not_pending | skipped_not_found | skipped_wrong_class | failed


class LeaveApplyResult(BaseModel):
    appliedCount: int = 0
    skippedCount: int = 0
    failedCount: int = 0
    outcomes: list[LeaveOutcome] = Field(default_factory=list)
    error: str | None = None


class LeaveDecisionWorkflowState(BaseModel):
    workflow_id: str
    school_id: int
    requester_user_id: str
    requester_role: str
    # Set for a TEACHER batch, None for a school-wide admin batch. Resolved by Spring from the
    # teacher's own classTeacher field, never client- or LLM-suppliable.
    class_name: str | None = None

    decision: str = "APPROVED"  # APPROVED | REJECTED — what to do WITH the leaves
    requested_leave_ids: list[int] = Field(default_factory=list)

    status: LeaveWorkflowStatus = LeaveWorkflowStatus.PENDING_APPROVAL
    # Everything resolved, actionable or not — the reviewer sees the full picture.
    leave_requests: list[LeaveRequestLite] = Field(default_factory=list)
    # Subset that is still PENDING and in scope; only these can actually change.
    actionable_leave_ids: list[int] = Field(default_factory=list)

    draft_summary: str = ""

    approval_decision: str | None = None  # "approved" | "rejected" — the BATCH decision
    approved_by: str | None = None
    approved_at: str | None = None

    apply_result: LeaveApplyResult | None = None

    error: str | None = None
    retry_count: int = 0

    created_at: str = Field(default_factory=_now_iso)
    updated_at: str = Field(default_factory=_now_iso)
