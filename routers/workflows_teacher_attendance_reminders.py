"""
routers/workflows_teacher_attendance_reminders.py — start/resume/status endpoints for the
teacher-scoped Low-Attendance Warning LangGraph workflow.

Direct structural mirror of routers/workflows_attendance_reminders.py (the ADMIN, school-wide
variant) — same trust model (gated by X-Internal-Secret, userId/role/schoolId come from the
request body exactly as Spring put them there), same idempotent-resume/resume-lock pattern. See
that file's module docstring for the full rationale, not repeated here.

The one load-bearing difference: role is restricted to TEACHER (not ADMIN/SUPER_ADMIN), and the
resume tenant check additionally requires the SAME teacher who started the workflow — schoolId
alone is not enough here, since (unlike an admin batch, which any admin in the school may
approve) a teacher-scoped batch belongs to exactly one teacher's one class.
"""
import logging
import uuid

from fastapi import APIRouter, Header, HTTPException
from langgraph.types import Command

import checkpointer
from config import settings
from graphs.state import AttendanceCriterion, TeacherAttendanceReminderWorkflowState
from graphs.teacher_attendance_reminder_workflow import graph
from schemas.workflows import (
    LowAttendanceStudentOut,
    SendOutcomeOut,
    SendResultOut,
    TeacherAttendanceWorkflowResumeRequest,
    TeacherAttendanceWorkflowStartRequest,
    TeacherAttendanceWorkflowStateResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/workflows/teacher-attendance-reminders")


def _check_secret(x_internal_secret: str) -> None:
    if x_internal_secret != settings.internal_secret:
        raise HTTPException(status_code=401, detail="Unauthorized")


def _state_to_response(workflow_id: str, values: dict) -> TeacherAttendanceWorkflowStateResponse:
    send_result = values.get("send_result")
    return TeacherAttendanceWorkflowStateResponse(
        workflowId=workflow_id,
        status=_status_str(values.get("status")),
        draftSummary=values.get("draft_summary") or "",
        studentCount=values.get("student_count") or 0,
        threshold=values.get("threshold") or 75.0,
        className=values.get("class_name") or "",
        criterion=_status_str(values.get("criterion")) or "BELOW_THRESHOLD",
        minConsecutiveDays=values.get("min_consecutive_days"),
        students=[
            LowAttendanceStudentOut(
                studentId=s.studentId, studentName=s.studentName, className=s.className,
                attendancePercentage=s.attendancePercentage, daysPresent=s.daysPresent,
                daysAbsent=s.daysAbsent, totalWorkingDays=s.totalWorkingDays,
                consecutiveAbsentDays=s.consecutiveAbsentDays, absentDates=s.absentDates,
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


@router.post("/start", response_model=TeacherAttendanceWorkflowStateResponse)
async def start_workflow(
    request: TeacherAttendanceWorkflowStartRequest,
    x_internal_secret: str = Header(alias="X-Internal-Secret"),
) -> TeacherAttendanceWorkflowStateResponse:
    _check_secret(x_internal_secret)
    if request.user.role != "TEACHER":
        raise HTTPException(status_code=403, detail="Only teachers can start this workflow.")
    if not request.className:
        # Spring already refuses to call us without a resolved className (see
        # AiTeacherAttendanceWorkflowController.start) — this is defense-in-depth only.
        raise HTTPException(status_code=403, detail="You are not assigned as a class teacher.")

    workflow_id = uuid.uuid4().hex
    initial_state = TeacherAttendanceReminderWorkflowState(
        workflow_id=workflow_id,
        school_id=request.user.schoolId,
        teacher_user_id=request.user.userId,
        class_name=request.className,
        session=request.session,
        criterion=AttendanceCriterion(request.criterion),
        threshold=request.threshold,
        min_consecutive_days=request.minConsecutiveDays,
    )
    config = {"configurable": {"thread_id": workflow_id, "access_token": request.accessToken}}

    result = await graph.ainvoke(initial_state, config=config)
    return _state_to_response(workflow_id, result)


@router.post("/{workflow_id}/resume", response_model=TeacherAttendanceWorkflowStateResponse)
async def resume_workflow(
    workflow_id: str,
    request: TeacherAttendanceWorkflowResumeRequest,
    x_internal_secret: str = Header(alias="X-Internal-Secret"),
) -> TeacherAttendanceWorkflowStateResponse:
    _check_secret(x_internal_secret)
    if request.schoolId is None:
        raise HTTPException(status_code=400, detail="schoolId is required")

    config = {"configurable": {"thread_id": workflow_id, "access_token": request.accessToken}}

    snapshot = await graph.aget_state(config)
    if not snapshot or not snapshot.values:
        raise HTTPException(status_code=404, detail="Unknown workflow.")

    values = snapshot.values
    # Defense-in-depth tenant + ownership check — Spring already verified both against its
    # own ai_teacher_attendance_reminder_batch row before ever calling us, but never trust a
    # single layer. Unlike the admin workflow, schoolId alone isn't enough: a teacher-scoped
    # batch belongs to exactly one teacher, not "any admin in the school".
    if values.get("school_id") != request.schoolId:
        raise HTTPException(status_code=403, detail="Workflow does not belong to this school.")
    if values.get("teacher_user_id") != request.teacherUserId:
        raise HTTPException(status_code=403, detail="This workflow does not belong to you.")

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


@router.get("/{workflow_id}", response_model=TeacherAttendanceWorkflowStateResponse)
async def get_workflow_status(
    workflow_id: str,
    x_internal_secret: str = Header(alias="X-Internal-Secret"),
) -> TeacherAttendanceWorkflowStateResponse:
    _check_secret(x_internal_secret)
    config = {"configurable": {"thread_id": workflow_id, "access_token": ""}}
    snapshot = await graph.aget_state(config)
    if not snapshot or not snapshot.values:
        raise HTTPException(status_code=404, detail="Unknown workflow.")
    return _state_to_response(workflow_id, snapshot.values)
