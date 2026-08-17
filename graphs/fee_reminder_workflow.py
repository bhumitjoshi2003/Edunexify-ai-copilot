"""
graphs/fee_reminder_workflow.py — the Fee Defaulter Reminder human-in-the-loop workflow.

    load_defaulters -> prepare_reminder_draft -> request_admin_approval (interrupt)
        -[resume: approved]-> send_reminders -> finish
        -[resume: rejected]-> finish
    (load_defaulters error, or defaulter_count == 0) -> finish / handle_failure
    (send_reminders hard error, not a partial failure) -> handle_failure -> finish

This is the ONLY graph in this codebase — the existing chat/tool-calling loop in
routers/chat.py is untouched. LangGraph earns its keep here specifically for the
interrupt()/Command(resume=...) mechanism: it persists state at the pause point and lets a
*separate* HTTP request resume exactly where it left off, which is what makes "approve this
from a different request, possibly hours later" tractable without hand-rolling checkpoint
serialization ourselves (see checkpointer.py for how that persistence layer works).

Critical rule, confirmed against LangGraph's actual resume semantics: a node containing
interrupt() re-executes its *entire body* from the top when the graph is resumed, not just
"returns from where interrupt() was called". request_admin_approval below does nothing but
call interrupt() and return its result — no side effects, no idempotency logic — precisely
because of this. Idempotency (the resume-lock, the terminal-status short-circuit) belongs in
routers/workflows_fee_reminders.py, wrapping the ainvoke() call, not inside the graph.

The access token is threaded through via `config["configurable"]["access_token"]` on every
invocation (start AND resume), never through state — configurable values are per-invocation
and are never checkpointed, so even a re-executed node only ever sees the token from the
*current* call. This is what makes resume work after the original JWT has long expired.
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
from graphs.state import DefaulterLite, FeeReminderWorkflowState, SendOutcome, SendResult, WorkflowStatus

logger = logging.getLogger(__name__)

_async_client = AsyncOpenAI(api_key=settings.deepseek_api_key, base_url="https://api.deepseek.com")

# Outcomes are capped for the same reason tools/admin_fees.py caps "mostOverdue" at 10 — a
# batch of hundreds shouldn't produce a multi-KB per-student list just for a summary card.
# sentCount/failedCount (not this list) are the authoritative totals.
MAX_OUTCOMES_STORED = 50


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _touch(state: FeeReminderWorkflowState) -> dict:
    return {"updated_at": _now_iso()}


# ─── Nodes ─────────────────────────────────────────────────────────────────────

async def load_defaulters(state: FeeReminderWorkflowState, config: RunnableConfig) -> dict:
    access_token = config["configurable"]["access_token"]

    params: dict = {"session": state.session}
    if state.class_name:
        params["className"] = state.class_name

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{settings.spring_boot_url}/api/student-fees/overdue",
                params=params,
                cookies={"accessToken": access_token},
                timeout=15.0,
            )
    except httpx.HTTPError as e:
        logger.warning("load_defaulters: Spring call failed: %s", e)
        return {"error": f"Could not reach fee data: {e}", **_touch(state)}

    if response.status_code == 403:
        return {"error": "Access denied fetching fee defaulters.", **_touch(state)}
    if response.status_code != 200:
        return {"error": f"Spring Boot returned {response.status_code} loading defaulters.", **_touch(state)}

    raw = response.json()
    defaulters = [
        DefaulterLite(
            studentId=d["studentId"],
            studentName=d.get("studentName") or d["studentId"],
            className=d.get("className") or "",
            # None (not 0.0) when Spring has no FeeStructure configured for this student's
            # class/session — see OverdueStudentDto.totalDue. Never coerce to zero here.
            totalDue=d.get("totalDue"),
        )
        for d in raw
    ]
    known_dues = [d.totalDue for d in defaulters if d.totalDue is not None]
    total_due = round(sum(known_dues), 2) if known_dues else None

    return {
        "defaulters": defaulters,
        "defaulter_count": len(defaulters),
        "total_amount_due": total_due,
        **_touch(state),
    }


def route_after_load(state: FeeReminderWorkflowState) -> str:
    if state.error:
        return "handle_failure"
    if state.defaulter_count == 0:
        return "no_defaulters"
    return "prepare_reminder_draft"


async def prepare_reminder_draft(state: FeeReminderWorkflowState, config: RunnableConfig) -> dict:
    # total_amount_due is None when no defaulter in this batch has a FeeStructure configured
    # for the session — that's "unknown", not "₹0 due". Never fabricate a figure here.
    amount_clause = f" totaling ₹{state.total_amount_due:,.2f}" if state.total_amount_due is not None else ""
    fallback = (
        f"{state.defaulter_count} parent(s) owe fees{amount_clause}"
        f" for session {state.session}"
        + (f" (class {state.class_name})" if state.class_name else "")
        + ". Review and approve to send reminder emails using the standard reminder template."
    )

    # Real per-student amounts (or "amount unknown"), not just names — mirrors the fix
    # applied to attendance_reminder_workflow.py's identical prompt shape after E2E testing
    # showed a bare-names example list gives the model room to invent a plausible-looking
    # but wrong figure for a named student. None means no FeeStructure configured, not ₹0.
    sample_examples = "; ".join(
        f"{d.studentName} (₹{d.totalDue:,.2f})" if d.totalDue is not None else f"{d.studentName} (amount unknown)"
        for d in state.defaulters[:5]
    )
    amount_line = (
        f"totalAmountDue: {state.total_amount_due}"
        if state.total_amount_due is not None
        else "totalAmountDue: unknown — do NOT state or estimate a rupee amount, just omit it"
    )
    prompt = (
        "Write a short, professional 2-3 sentence summary for a SCHOOL ADMIN reviewing a batch "
        "of fee reminder emails before sending. This text is shown only to the admin, never to "
        "parents. State how many students/parents are affected and the total amount due (if "
        "known — see below). You may mention a couple of example students by name — if you "
        "state a specific rupee amount for a named student, it MUST be exactly the figure given "
        "for them in exampleStudents below (or omit the amount entirely if theirs is unknown); "
        "never estimate, round to a different value, or invent one. End by inviting them to "
        "approve or reject sending.\n\n"
        f"defaulterCount: {state.defaulter_count}\n"
        f"{amount_line}\n"
        f"session: {state.session}\n"
        f"className: {state.class_name or 'all classes'}\n"
        f"exampleStudents (name and their real amount due): {sample_examples}"
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
        logger.warning("prepare_reminder_draft: DeepSeek call failed, using fallback: %s", e)
        draft = fallback

    return {"draft_summary": draft, **_touch(state)}


async def request_admin_approval(state: FeeReminderWorkflowState) -> dict:
    # Nothing but the interrupt call belongs here — see module docstring for why.
    decision = interrupt({
        "workflowId": state.workflow_id,
        "defaulterCount": state.defaulter_count,
        "totalAmountDue": state.total_amount_due,
        "draftSummary": state.draft_summary,
        "session": state.session,
        "className": state.class_name,
    })
    return {"approval_decision": decision}


def route_after_approval(state: FeeReminderWorkflowState) -> str:
    if state.approval_decision == "approved":
        return "send_reminders"
    if state.approval_decision == "rejected":
        return "rejected"
    # Defensive only — routers/workflows_fee_reminders.py validates the decision
    # value before ever invoking resume, so this should be unreachable.
    return "handle_failure"


async def send_reminders(state: FeeReminderWorkflowState, config: RunnableConfig) -> dict:
    access_token = config["configurable"]["access_token"]
    student_ids = [d.studentId for d in state.defaulters]

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{settings.spring_boot_url}/api/ai/workflows/fee-reminders/{state.workflow_id}/dispatch",
                json={"studentIds": student_ids, "session": state.session},
                cookies={"accessToken": access_token},
                timeout=60.0,  # a real bulk email loop — give it real headroom
            )
    except httpx.HTTPError as e:
        logger.warning("send_reminders: Spring dispatch call failed: %s", e)
        return {
            "send_result": SendResult(error=f"Could not reach Spring to send reminders: {e}"),
            **_touch(state),
        }

    if response.status_code != 200:
        return {
            "send_result": SendResult(error=f"Spring Boot returned {response.status_code} dispatching reminders."),
            **_touch(state),
        }

    body = response.json()
    outcomes = [
        SendOutcome(studentId=o["studentId"], status=o["status"])
        for o in (body.get("outcomes") or [])[:MAX_OUTCOMES_STORED]
    ]
    return {
        "send_result": SendResult(
            sentCount=body.get("sentCount", 0),
            failedCount=body.get("failedCount", 0),
            outcomes=outcomes,
        ),
        **_touch(state),
    }


def route_after_send(state: FeeReminderWorkflowState) -> str:
    if state.send_result and state.send_result.error:
        return "handle_failure"
    return "finish"


async def handle_failure(state: FeeReminderWorkflowState) -> dict:
    error = state.error or (state.send_result.error if state.send_result else None) or "Unknown error"
    return {"status": WorkflowStatus.FAILED, "error": error, **_touch(state)}


async def finish(state: FeeReminderWorkflowState) -> dict:
    if state.status == WorkflowStatus.FAILED:
        return {}  # handle_failure already set the terminal status
    if state.approval_decision == "rejected":
        return {"status": WorkflowStatus.REJECTED, **_touch(state)}
    if state.defaulter_count == 0:
        return {"status": WorkflowStatus.NO_DEFAULTERS, **_touch(state)}
    if state.send_result:
        # Three-way, not two-way: failedCount > 0 alone conflates "some sent, some failed"
        # with "every single send failed" — confirmed live during E2E testing (SMTP down ->
        # sentCount=0/failedCount=N reported as "partially_sent", which reads as if some
        # parents WERE reached). Mirrors the sent/failed logic AiWorkflowController.dispatch()
        # already uses on the Spring side for the exact same distinction.
        if state.send_result.failedCount == 0:
            status = WorkflowStatus.SENT
        elif state.send_result.sentCount == 0:
            status = WorkflowStatus.FAILED
        else:
            status = WorkflowStatus.PARTIALLY_SENT
        return {"status": status, **_touch(state)}
    return {}


# ─── Graph assembly ────────────────────────────────────────────────────────────

def _build_graph():
    builder = StateGraph(FeeReminderWorkflowState)

    builder.add_node("load_defaulters", load_defaulters)
    builder.add_node("prepare_reminder_draft", prepare_reminder_draft)
    builder.add_node("request_admin_approval", request_admin_approval)
    builder.add_node("send_reminders", send_reminders)
    builder.add_node("handle_failure", handle_failure)
    builder.add_node("finish", finish)

    builder.add_edge(START, "load_defaulters")
    builder.add_conditional_edges("load_defaulters", route_after_load, {
        "handle_failure": "handle_failure",
        "no_defaulters": "finish",
        "prepare_reminder_draft": "prepare_reminder_draft",
    })
    builder.add_edge("prepare_reminder_draft", "request_admin_approval")
    builder.add_conditional_edges("request_admin_approval", route_after_approval, {
        "send_reminders": "send_reminders",
        "rejected": "finish",
        "handle_failure": "handle_failure",
    })
    builder.add_conditional_edges("send_reminders", route_after_send, {
        "handle_failure": "handle_failure",
        "finish": "finish",
    })
    builder.add_edge("handle_failure", "finish")
    builder.add_edge("finish", END)

    return builder


# Explicitly allowlist this module's custom Enum/Pydantic types for checkpoint
# (de)serialization — without this, LangGraph's strict msgpack serde warns now and
# will refuse to deserialize them at all in a future version.
_serde = JsonPlusSerializer(
    allowed_msgpack_modules=[
        ("graphs.state", "WorkflowStatus"),
        ("graphs.state", "DefaulterLite"),
        ("graphs.state", "SendOutcome"),
        ("graphs.state", "SendResult"),
        ("graphs.state", "FeeReminderWorkflowState"),
    ],
)
_checkpointer = RedisCheckpointSaver(ttl_seconds=settings.workflow_checkpoint_ttl_seconds, serde=_serde)
graph = _build_graph().compile(checkpointer=_checkpointer)
