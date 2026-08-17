"""
routers/workflows_leave_decisions.py — start/resume/status endpoints for the Leave Decision
LangGraph workflow.

Structural mirror of routers/workflows_teacher_attendance_reminders.py — same trust model (gated
by X-Internal-Secret, identity comes from the request body exactly as Spring put it there), same
idempotent-resume/resume-lock pattern. See that file's module docstring for the full rationale.

The resume tenant check requires the same requester who started the batch, matching the teacher
attendance workflow rather than the admin ones: a leave batch names specific students' requests,
so "any admin in the school may finish it" is a wider door than this surface needs. Spring
independently enforces the same rule (and, for a teacher, current class ownership) before ever
calling here.
"""
import logging
import uuid

from fastapi import APIRouter, Header, HTTPException
from langgraph.types import Command

import checkpointer
from config import settings
from graphs.leave_decision_workflow import graph
from graphs.state import LeaveDecisionWorkflowState
from schemas.workflows import (
    LeaveApplyResultOut,
    LeaveDecisionResumeRequest,
    LeaveDecisionStartRequest,
    LeaveDecisionStateResponse,
    LeaveOutcomeOut,
    LeaveRequestOut,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/workflows/leave-decisions")

_ALLOWED_ROLES = {"ADMIN", "TEACHER"}


def _check_secret(x_internal_secret: str) -> None:
    if x_internal_secret != settings.internal_secret:
        raise HTTPException(status_code=401, detail="Unauthorized")


def _status_str(status) -> str:
    return status.value if hasattr(status, "value") else str(status)


def _state_to_response(workflow_id: str, values: dict) -> LeaveDecisionStateResponse:
    apply_result = values.get("apply_result")
    requests = values.get("leave_requests") or []
    return LeaveDecisionStateResponse(
        workflowId=workflow_id,
        status=_status_str(values.get("status")),
        draftSummary=values.get("draft_summary") or "",
        decision=values.get("decision") or "APPROVED",
        className=values.get("class_name"),
        leaveCount=len(requests),
        actionableCount=len(values.get("actionable_leave_ids") or []),
        leaveRequests=[
            LeaveRequestOut(
                leaveId=r.leaveId, studentId=r.studentId, studentName=r.studentName,
                className=r.className, leaveDate=r.leaveDate, reason=r.reason, status=r.status,
            )
            for r in requests
        ],
        applyResult=(
            LeaveApplyResultOut(
                appliedCount=apply_result.appliedCount,
                skippedCount=apply_result.skippedCount,
                failedCount=apply_result.failedCount,
                outcomes=[LeaveOutcomeOut(leaveId=o.leaveId, outcome=o.outcome) for o in apply_result.outcomes],
                error=apply_result.error,
            )
            if apply_result else None
        ),
        error=values.get("error"),
    )


_TERMINAL_STATUSES = {"rejected", "applied", "partially_applied", "failed", "no_actionable_requests"}


@router.post("/start", response_model=LeaveDecisionStateResponse)
async def start_workflow(
    request: LeaveDecisionStartRequest,
    x_internal_secret: str = Header(alias="X-Internal-Secret"),
) -> LeaveDecisionStateResponse:
    _check_secret(x_internal_secret)
    if request.user.role not in _ALLOWED_ROLES:
        raise HTTPException(status_code=403, detail="Only admins and teachers can decide leave requests.")
    if not request.leaveIds:
        raise HTTPException(status_code=400, detail="At least one leave request id is required.")

    workflow_id = uuid.uuid4().hex
    initial_state = LeaveDecisionWorkflowState(
        workflow_id=workflow_id,
        school_id=request.user.schoolId,
        requester_user_id=request.user.userId,
        requester_role=request.user.role,
        class_name=request.className,
        decision=request.decision,
        requested_leave_ids=request.leaveIds,
    )
    config = {"configurable": {"thread_id": workflow_id, "access_token": request.accessToken}}

    result = await graph.ainvoke(initial_state, config=config)
    return _state_to_response(workflow_id, result)


@router.post("/{workflow_id}/resume", response_model=LeaveDecisionStateResponse)
async def resume_workflow(
    workflow_id: str,
    request: LeaveDecisionResumeRequest,
    x_internal_secret: str = Header(alias="X-Internal-Secret"),
) -> LeaveDecisionStateResponse:
    _check_secret(x_internal_secret)

    config = {"configurable": {"thread_id": workflow_id, "access_token": request.accessToken}}

    snapshot = await graph.aget_state(config)
    if not snapshot or not snapshot.values:
        raise HTTPException(status_code=404, detail="Unknown workflow.")

    values = snapshot.values
    # Defense in depth — Spring already checked both against its own batch row.
    if values.get("school_id") != request.schoolId:
        raise HTTPException(status_code=403, detail="Workflow does not belong to this school.")
    if values.get("requester_user_id") != request.requesterUserId:
        raise HTTPException(status_code=403, detail="This workflow does not belong to you.")

    current_status = _status_str(values.get("status"))
    if current_status in _TERMINAL_STATUSES:
        # Idempotent short-circuit: a duplicate approve/reject must never re-invoke the graph
        # (which would re-apply decisions) — just replay the stored result.
        return _state_to_response(workflow_id, values)

    if not await checkpointer.acquire_resume_lock(workflow_id, settings.workflow_resume_lock_seconds):
        raise HTTPException(status_code=409, detail="This workflow is already being processed. Please wait a moment.")

    try:
        snapshot = await graph.aget_state(config)
        current_status = _status_str(snapshot.values.get("status"))
        if current_status in _TERMINAL_STATUSES:
            return _state_to_response(workflow_id, snapshot.values)

        result = await graph.ainvoke(Command(resume=request.decision), config=config)
    finally:
        await checkpointer.release_resume_lock(workflow_id)

    return _state_to_response(workflow_id, result)


@router.get("/{workflow_id}", response_model=LeaveDecisionStateResponse)
async def get_workflow_status(
    workflow_id: str,
    x_internal_secret: str = Header(alias="X-Internal-Secret"),
) -> LeaveDecisionStateResponse:
    _check_secret(x_internal_secret)
    config = {"configurable": {"thread_id": workflow_id, "access_token": ""}}
    snapshot = await graph.aget_state(config)
    if not snapshot or not snapshot.values:
        raise HTTPException(status_code=404, detail="Unknown workflow.")
    return _state_to_response(workflow_id, snapshot.values)
