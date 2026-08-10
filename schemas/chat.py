"""
schemas/chat.py — Pydantic models for the /chat endpoint.

Pydantic validates incoming JSON automatically. If Spring Boot sends a malformed
payload, FastAPI returns a 422 with a clear error — no manual validation needed.
"""
from pydantic import BaseModel


class UserContext(BaseModel):
    """
    The authenticated user's identity, extracted by Spring Boot from the JWT
    and forwarded here so Python never needs to validate tokens itself.
    """
    userId: str
    role: str        # STUDENT | TEACHER | ADMIN | SUB_ADMIN | SUPER_ADMIN
    schoolId: int
    name: str | None = None
    className: str | None = None  # class name for STUDENTs / assigned class for TEACHERs


class ChatRequest(BaseModel):
    """
    Payload sent by Spring Boot to this service.
    The accessToken is the raw JWT cookie value — Python forwards it when
    calling Spring Boot APIs so those endpoints apply their own auth checks.

    conversationId scopes short-term memory (see memory.py) together with
    user.schoolId and user.userId. It's generated and held by Angular for the
    lifetime of one chat panel session — a fresh chat ("New chat") means a
    fresh conversationId, which means fresh (empty) memory.
    """
    message: str
    conversationId: str
    user: UserContext
    accessToken: str


class ChatResponse(BaseModel):
    reply: str
    # Populated only when this turn started a workflow (e.g. the fee-reminder LangGraph
    # workflow, via tools/admin_fee_workflows.py) — Spring's own response shape, passed
    # through as-is. When set, `reply` is a short synthetic note for conversation memory,
    # not meant to be shown as prose; Angular should render `workflow` as the approval
    # card instead of `reply` as text.
    workflow: dict | None = None
