"""
graphs/leave_decision_workflow.py — the Leave Approve/Reject human-in-the-loop workflow.

    resolve_leave_requests -> prepare_decision_draft -> request_approval (interrupt)
        -[resume: approved]-> apply_decisions -> finish
        -[resume: rejected]-> finish
    (resolve error, or nothing actionable) -> finish / handle_failure
    (apply hard error, not a partial failure) -> handle_failure -> finish

Structural mirror of graphs/teacher_attendance_reminder_workflow.py — same interrupt()/
Command(resume=...) mechanism, same three-way terminal-status logic, same
hallucination-resistant draft prompt. See fee_reminder_workflow.py's module docstring for the
rationale behind the shared pieces, not repeated here.

WHAT IS DIFFERENT, AND WHY IT MATTERS
Every reminder workflow rests on a property that cannot hold here: they take no recipient
parameter at all, so the model is structurally incapable of choosing who gets contacted. A leave
decision is inherently about *specific* requests, so the model must name ids. That safety
property is therefore replaced, not preserved:

  * resolve_leave_requests never trusts the proposed ids. It asks Spring to resolve them
    (GET /api/leaves/for-review?ids=...), which returns only rows inside the caller's school and,
    for a teacher, inside their own class. Anything the model invented, guessed, or remembered
    from another school simply does not come back.
  * Of what does come back, only rows still PENDING become actionable. A request someone else
    already decided is shown to the reviewer but excluded from the set that can change.
  * Spring re-runs both checks again at apply time (see LeaveDecisionService) — this node's
    filtering is for an honest approval card, not for safety on its own.
  * Nothing is applied until a human approves the card.

So the model's ids are a proposal. The human reading the card, and Spring's revalidation, are
what actually authorise a change.
"""
import logging
from datetime import datetime, timezone

import httpx
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from openai import AsyncOpenAI

from checkpointer import RedisCheckpointSaver
from config import settings
from graphs.state import (
    LeaveApplyResult,
    LeaveDecisionWorkflowState,
    LeaveOutcome,
    LeaveRequestLite,
    LeaveWorkflowStatus,
)

logger = logging.getLogger(__name__)

_async_client = AsyncOpenAI(api_key=settings.deepseek_api_key, base_url="https://api.deepseek.com")

MAX_OUTCOMES_STORED = 50


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _touch(state: LeaveDecisionWorkflowState) -> dict:
    return {"updated_at": _now_iso()}


# ─── Nodes ─────────────────────────────────────────────────────────────────────

async def resolve_leave_requests(state: LeaveDecisionWorkflowState, config: RunnableConfig) -> dict:
    access_token = config["configurable"]["access_token"]

    if not state.requested_leave_ids:
        return {"leave_requests": [], "actionable_leave_ids": [], **_touch(state)}

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{settings.spring_boot_url}/api/leaves/for-review",
                params={"ids": [str(i) for i in state.requested_leave_ids]},
                cookies={"accessToken": access_token},
                timeout=15.0,
            )
    except httpx.HTTPError as e:
        logger.warning("resolve_leave_requests: Spring call failed: %s", e)
        return {"error": f"Could not reach leave data: {e}", **_touch(state)}

    if response.status_code == 403:
        return {"error": "Access denied. You can only act on leave requests for your own class.", **_touch(state)}
    if response.status_code != 200:
        return {"error": f"Spring Boot returned {response.status_code} resolving leave requests.", **_touch(state)}

    raw = response.json()
    requests = [
        LeaveRequestLite(
            leaveId=r["id"],
            studentId=r.get("studentId") or "",
            studentName=r.get("studentName") or r.get("studentId") or "",
            className=r.get("className") or "",
            leaveDate=r.get("leaveDate") or "",
            reason=r.get("reason") or "",
            status=r.get("status") or "PENDING",
        )
        for r in raw
    ]
    # Oldest first, matching the review queue the teacher/admin sees elsewhere.
    requests.sort(key=lambda r: (r.leaveDate, r.leaveId))
    actionable = [r.leaveId for r in requests if r.status == "PENDING"]

    return {
        "leave_requests": requests,
        "actionable_leave_ids": actionable,
        **_touch(state),
    }


def route_after_resolve(state: LeaveDecisionWorkflowState) -> str:
    if state.error:
        return "handle_failure"
    if not state.actionable_leave_ids:
        return "no_actionable_requests"
    return "prepare_decision_draft"


async def prepare_decision_draft(state: LeaveDecisionWorkflowState, config: RunnableConfig) -> dict:
    verb = "approve" if state.decision == "APPROVED" else "reject"
    actionable = [r for r in state.leave_requests if r.status == "PENDING"]
    already_decided = [r for r in state.leave_requests if r.status != "PENDING"]

    fallback = (
        f"{len(actionable)} leave request(s) will be {state.decision.lower()}"
        + (f" for class {state.class_name}" if state.class_name else "")
        + (f". {len(already_decided)} request(s) were already decided and will be left unchanged"
           if already_decided else "")
        + ". Review and approve to apply."
    )

    # Real per-request details, so the model never has to invent a name or date to fill a gap —
    # the same hallucination fix the reminder workflows needed (see their prompts).
    examples = "; ".join(f"{r.studentName} on {r.leaveDate}" for r in actionable[:5])
    skipped_note = (
        f"\nalreadyDecidedCount: {len(already_decided)} (these will NOT be changed)"
        if already_decided else ""
    )
    prompt = (
        f"Write a short, professional 2-3 sentence summary for a school {state.requester_role.lower()} "
        f"reviewing a batch of leave requests they are about to {verb}. This text is shown only to "
        "the reviewer. State how many requests will be affected and what will happen to them. You "
        "may mention a couple of students by name — if you do, use exactly the names and dates "
        "given in examples below; never invent, alter, or round a date. If any requests were "
        "already decided, say plainly that they will be left unchanged. End by inviting the "
        "reviewer to approve or reject applying this.\n\n"
        f"decision: {state.decision}\n"
        f"actionableCount: {len(actionable)}\n"
        f"scope: {('class ' + state.class_name) if state.class_name else 'school-wide'}"
        f"{skipped_note}\n"
        f"examples (real student names and leave dates): {examples}"
    )

    try:
        response = await _async_client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=200,
            timeout=15.0,
        )
        draft = (response.choices[0].message.content or "").strip() or fallback
    except Exception as e:  # cosmetic text only — never block the workflow on a DeepSeek hiccup
        logger.warning("prepare_decision_draft: DeepSeek call failed, using fallback: %s", e)
        draft = fallback

    return {"draft_summary": draft, **_touch(state)}


async def request_approval(state: LeaveDecisionWorkflowState) -> dict:
    # Nothing but the interrupt call belongs here — a resumed node re-executes its whole body.
    decision = interrupt({
        "workflowId": state.workflow_id,
        "decision": state.decision,
        "className": state.class_name,
        "actionableCount": len(state.actionable_leave_ids),
        "draftSummary": state.draft_summary,
    })
    return {"approval_decision": decision}


def route_after_approval(state: LeaveDecisionWorkflowState) -> str:
    if state.approval_decision == "approved":
        return "apply_decisions"
    if state.approval_decision == "rejected":
        return "rejected"
    # Defensive only — the router validates the decision before ever invoking resume.
    return "handle_failure"


async def apply_decisions(state: LeaveDecisionWorkflowState, config: RunnableConfig) -> dict:
    access_token = config["configurable"]["access_token"]

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{settings.spring_boot_url}/api/ai/workflows/leave-decisions/{state.workflow_id}/dispatch",
                # Informational only: Spring applies the ids from its own batch row, never these,
                # so a tampered body cannot widen what the human approved.
                json={"leaveIds": state.actionable_leave_ids},
                cookies={"accessToken": access_token},
                timeout=60.0,
            )
    except httpx.HTTPError as e:
        logger.warning("apply_decisions: Spring dispatch call failed: %s", e)
        return {"apply_result": LeaveApplyResult(error=f"Could not reach Spring to apply decisions: {e}"), **_touch(state)}

    if response.status_code != 200:
        return {"apply_result": LeaveApplyResult(error=f"Spring Boot returned {response.status_code} applying decisions."), **_touch(state)}

    body = response.json()
    outcomes = [
        LeaveOutcome(leaveId=o["leaveId"], outcome=o["outcome"])
        for o in (body.get("outcomes") or [])[:MAX_OUTCOMES_STORED]
    ]
    return {
        "apply_result": LeaveApplyResult(
            appliedCount=body.get("appliedCount", 0),
            skippedCount=body.get("skippedCount", 0),
            failedCount=body.get("failedCount", 0),
            outcomes=outcomes,
        ),
        **_touch(state),
    }


def route_after_apply(state: LeaveDecisionWorkflowState) -> str:
    if state.apply_result and state.apply_result.error:
        return "handle_failure"
    return "finish"


async def handle_failure(state: LeaveDecisionWorkflowState) -> dict:
    error = state.error or (state.apply_result.error if state.apply_result else None) or "Unknown error"
    return {"status": LeaveWorkflowStatus.FAILED, "error": error, **_touch(state)}


async def finish(state: LeaveDecisionWorkflowState) -> dict:
    if state.status == LeaveWorkflowStatus.FAILED:
        return {}  # handle_failure already set the terminal status
    if state.approval_decision == "rejected":
        return {"status": LeaveWorkflowStatus.REJECTED, **_touch(state)}
    if not state.actionable_leave_ids:
        return {"status": LeaveWorkflowStatus.NO_ACTIONABLE_REQUESTS, **_touch(state)}
    if state.apply_result:
        # A batch where everything was SKIPPED is still a success: declining to overwrite
        # decisions other people already made is the workflow behaving correctly. Only a genuine
        # failure (an exception applying a still-PENDING request) degrades the status.
        if state.apply_result.failedCount == 0:
            status = LeaveWorkflowStatus.APPLIED
        elif state.apply_result.appliedCount == 0:
            status = LeaveWorkflowStatus.FAILED
        else:
            status = LeaveWorkflowStatus.PARTIALLY_APPLIED
        return {"status": status, **_touch(state)}
    return {}


# ─── Graph assembly ────────────────────────────────────────────────────────────

def _build_graph():
    builder = StateGraph(LeaveDecisionWorkflowState)

    builder.add_node("resolve_leave_requests", resolve_leave_requests)
    builder.add_node("prepare_decision_draft", prepare_decision_draft)
    builder.add_node("request_approval", request_approval)
    builder.add_node("apply_decisions", apply_decisions)
    builder.add_node("handle_failure", handle_failure)
    builder.add_node("finish", finish)

    builder.add_edge(START, "resolve_leave_requests")
    builder.add_conditional_edges("resolve_leave_requests", route_after_resolve, {
        "handle_failure": "handle_failure",
        "no_actionable_requests": "finish",
        "prepare_decision_draft": "prepare_decision_draft",
    })
    builder.add_edge("prepare_decision_draft", "request_approval")
    builder.add_conditional_edges("request_approval", route_after_approval, {
        "apply_decisions": "apply_decisions",
        "rejected": "finish",
        "handle_failure": "handle_failure",
    })
    builder.add_conditional_edges("apply_decisions", route_after_apply, {
        "handle_failure": "handle_failure",
        "finish": "finish",
    })
    builder.add_edge("handle_failure", "finish")
    builder.add_edge("finish", END)

    return builder


_serde = JsonPlusSerializer(
    allowed_msgpack_modules=[
        ("graphs.state", "LeaveWorkflowStatus"),
        ("graphs.state", "LeaveRequestLite"),
        ("graphs.state", "LeaveOutcome"),
        ("graphs.state", "LeaveApplyResult"),
        ("graphs.state", "LeaveDecisionWorkflowState"),
    ],
)
_checkpointer = RedisCheckpointSaver(ttl_seconds=settings.workflow_checkpoint_ttl_seconds, serde=_serde)
graph = _build_graph().compile(checkpointer=_checkpointer)
