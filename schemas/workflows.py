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


# ─── Attendance Reminder workflow ───────────────────────────────────────────────

class AttendanceWorkflowStartRequest(BaseModel):
    session: str
    className: str | None = None
    threshold: float = 75.0
    user: UserContext
    accessToken: str


class AttendanceWorkflowResumeRequest(BaseModel):
    decision: Literal["approved", "rejected"]
    schoolId: int
    adminUserId: str
    accessToken: str


class LowAttendanceStudentOut(BaseModel):
    studentId: str
    studentName: str
    className: str
    attendancePercentage: float
    daysPresent: int
    daysAbsent: int
    totalWorkingDays: int
    # Present only on a CONSECUTIVE_ABSENCE batch — see LowAttendanceStudentLite in graphs/state.py.
    consecutiveAbsentDays: int | None = None
    absentDates: list[str] | None = None


class AttendanceWorkflowStateResponse(BaseModel):
    """Returned by start, resume, and the status GET — same shape from all three, mirroring
    WorkflowStateResponse's rationale above."""
    workflowId: str
    status: str
    draftSummary: str
    studentCount: int
    threshold: float
    students: list[LowAttendanceStudentOut] = []
    sendResult: SendResultOut | None = None
    error: str | None = None


# ─── Teacher-scoped Attendance Reminder workflow ────────────────────────────────

class TeacherAttendanceWorkflowStartRequest(BaseModel):
    session: str
    # Always resolved server-side by Spring from the teacher's own classTeacher field
    # (see AiTeacherAttendanceWorkflowController.start) — never client- or LLM-suppliable,
    # same guarantee tools/teacher_attendance.py relies on for user.className.
    className: str
    # Which attendance pattern selects the batch. Spring validates this against a closed set
    # before forwarding, so an unexpected value never reaches the graph.
    criterion: Literal["BELOW_THRESHOLD", "CONSECUTIVE_ABSENCE"] = "BELOW_THRESHOLD"
    threshold: float = 75.0
    minConsecutiveDays: int | None = None
    user: UserContext
    accessToken: str


class TeacherAttendanceWorkflowResumeRequest(BaseModel):
    decision: Literal["approved", "rejected"]
    schoolId: int
    teacherUserId: str
    accessToken: str


class TeacherAttendanceWorkflowStateResponse(BaseModel):
    """Returned by start, resume, and the status GET — same shape from all three, mirroring
    WorkflowStateResponse's rationale above. The top-level className field (absent from
    AttendanceWorkflowStateResponse, which only carries className per-student) doubles as
    routers/chat.py's _workflow_kind discriminator between the admin and teacher variants."""
    workflowId: str
    status: str
    draftSummary: str
    studentCount: int
    threshold: float
    className: str
    criterion: str = "BELOW_THRESHOLD"
    minConsecutiveDays: int | None = None
    students: list[LowAttendanceStudentOut] = []
    sendResult: SendResultOut | None = None
    error: str | None = None


# ─── Leave Decision workflow ────────────────────────────────────────────────────

class LeaveDecisionStartRequest(BaseModel):
    decision: Literal["APPROVED", "REJECTED"]
    leaveIds: list[int]
    # Resolved server-side by Spring from the teacher's own classTeacher field; None for a
    # school-wide admin batch. Never client- or LLM-suppliable.
    className: str | None = None
    user: UserContext
    accessToken: str


class LeaveDecisionResumeRequest(BaseModel):
    decision: Literal["approved", "rejected"]
    schoolId: int
    requesterUserId: str
    accessToken: str


class LeaveRequestOut(BaseModel):
    leaveId: int
    studentId: str
    studentName: str
    className: str
    leaveDate: str
    reason: str
    status: str


class LeaveOutcomeOut(BaseModel):
    leaveId: int
    outcome: str


class LeaveApplyResultOut(BaseModel):
    appliedCount: int
    skippedCount: int
    failedCount: int
    outcomes: list[LeaveOutcomeOut] = []
    error: str | None = None


class LeaveDecisionStateResponse(BaseModel):
    """Returned by start, resume, and the status GET — one shape from all three, matching how
    the reminder workflows already behave."""
    workflowId: str
    status: str
    draftSummary: str
    decision: str
    className: str | None = None
    # Distinct counts: leaveCount is everything shown on the card, actionableCount is the subset
    # that can actually change. They differ whenever a request was already decided by someone else.
    leaveCount: int
    actionableCount: int
    leaveRequests: list[LeaveRequestOut] = []
    applyResult: LeaveApplyResultOut | None = None
    error: str | None = None
