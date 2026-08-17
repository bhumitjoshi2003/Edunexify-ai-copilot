"""
routers/workflows_attendance_reminders.py — start/resume/status endpoints for the Low-Attendance
Warning LangGraph workflow.

Direct structural mirror of routers/workflows_fee_reminders.py — same trust model (gated by
X-Internal-Secret, userId/role/schoolId come from the request body exactly as Spring put them
there), same idempotent-resume/resume-lock pattern. See that file's module docstring for the
full rationale, not repeated here.
"""
import logging
import uuid

from fastapi import APIRouter, Header, HTTPException
from langgraph.types import Command

import checkpointer
from config import settings
from graphs.attendance_reminder_workflow import graph
from graphs.state import AttendanceReminderWorkflowState
from schemas.workflows import (
    AttendanceWorkflowResumeRequest,
    AttendanceWorkflowStartRequest,
    AttendanceWorkflowStateResponse,
    LowAttendanceStudentOut,
    SendOutcomeOut,
    SendResultOut,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/workflows/attendance-reminders")

_ADMIN_ROLES = {"ADMIN", "SUPER_ADMIN"}


def _check_secret(x_internal_secret: str) -> None:
    if x_internal_secret != settings.internal_secret:
        raise HTTPException(status_code=401, detail="Unauthorized")


def _state_to_response(workflow_id: str, values: dict) -> AttendanceWorkflowStateResponse:
    send_result = values.get("send_result")
    return AttendanceWorkflowStateResponse(
        workflowId=workflow_id,
        status=_status_str(values.get("status")),
        draftSummary=values.get("draft_summary") or "",
        studentCount=values.get("student_count") or 0,
        threshold=values.get("threshold") or 75.0,
        students=[
            LowAttendanceStudentOut(
                studentId=s.studentId, studentName=s.studentName, className=s.className,
                attendancePercentage=s.attendancePercentage, daysPresent=s.daysPresent,
                daysAbsent=s.daysAbsent, totalWorkingDays=s.totalWorkingDays,
            )
            for s in (values.get("students") or [])
        ],
        sendResult=(
            SendResultOut(
                sentCount=send_result.sentCount,
                failedCount=send_result.failedCount,
                outcomes=[SendOutcomeOut(studentId=o.studentId, status=o.status) for o in send_result.outcomes],
                error=send_result.error,
            )
            if send_result else None
        ),
        error=values.get("error"),
    )


def _status_str(status) -> str:
    # status comes back as the AttendanceWorkflowStatus enum member from a live graph
    # result, but as a plain string once round-tripped through checkpoint (de)serialization.
    return status.value if hasattr(status, "value") else str(status)


_TERMINAL_STATUSES = {"rejected", "sent", "partially_sent", "failed", "no_low_attendance_students"}


@router.post("/start", response_model=AttendanceWorkflowStateResponse)
async def start_workflow(
    request: AttendanceWorkflowStartRequest,
    x_internal_secret: str = Header(alias="X-Internal-Secret"),
) -> AttendanceWorkflowStateResponse:
    _check_secret(x_internal_secret)
    if request.user.role not in _ADMIN_ROLES:
        raise HTTPException(status_code=403, detail="Only admins can start this workflow.")

    workflow_id = uuid.uuid4().hex
    initial_state = AttendanceReminderWorkflowState(
        workflow_id=workflow_id,
        school_id=request.user.schoolId,
        admin_user_id=request.user.userId,
        session=request.session,
        class_name=request.className,
        threshold=request.threshold,
    )
    config = {"configurable": {"thread_id": workflow_id, "access_token": request.accessToken}}

    result = await graph.ainvoke(initial_state, config=config)
    return _state_to_response(workflow_id, result)


@router.post("/{workflow_id}/resume", response_model=AttendanceWorkflowStateResponse)
async def resume_workflow(
    workflow_id: str,
    request: AttendanceWorkflowResumeRequest,
    x_internal_secret: str = Header(alias="X-Internal-Secret"),
) -> AttendanceWorkflowStateResponse:
    _check_secret(x_internal_secret)
    if request.schoolId is None:
        raise HTTPException(status_code=400, detail="schoolId is required")

    config = {"configurable": {"thread_id": workflow_id, "access_token": request.accessToken}}

    snapshot = await graph.aget_state(config)
    if not snapshot or not snapshot.values:
        raise HTTPException(status_code=404, detail="Unknown workflow.")

    values = snapshot.values
    # Defense-in-depth tenant check — Spring already verified this against its own
    # ai_attendance_reminder_batch row before ever calling us, but never trust a single layer.
    if values.get("school_id") != request.schoolId:
        raise HTTPException(status_code=403, detail="Workflow does not belong to this school.")

    current_status = _status_str(values.get("status"))
    if current_status in _TERMINAL_STATUSES:
        # Idempotent short-circuit: a duplicate/retried approve or reject must never
        # re-invoke the graph (which would re-send emails) — just replay the stored result.
        return _state_to_response(workflow_id, values)

    if not await checkpointer.acquire_resume_lock(workflow_id, settings.workflow_resume_lock_seconds):
        raise HTTPException(status_code=409, detail="This workflow is already being processed. Please wait a moment.")

    try:
        # Re-check status after acquiring the lock — the concurrent request that lost the
        # lock race may have been the one that just finished by the time we get here.
        snapshot = await graph.aget_state(config)
        current_status = _status_str(snapshot.values.get("status"))
        if current_status in _TERMINAL_STATUSES:
            return _state_to_response(workflow_id, snapshot.values)

        result = await graph.ainvoke(Command(resume=request.decision), config=config)
    finally:
        await checkpointer.release_resume_lock(workflow_id)

    return _state_to_response(workflow_id, result)


@router.get("/{workflow_id}", response_model=AttendanceWorkflowStateResponse)
async def get_workflow_status(
    workflow_id: str,
    x_internal_secret: str = Header(alias="X-Internal-Secret"),
) -> AttendanceWorkflowStateResponse:
    _check_secret(x_internal_secret)
    config = {"configurable": {"thread_id": workflow_id, "access_token": ""}}
    snapshot = await graph.aget_state(config)
    if not snapshot or not snapshot.values:
        raise HTTPException(status_code=404, detail="Unknown workflow.")
    return _state_to_response(workflow_id, snapshot.values)
