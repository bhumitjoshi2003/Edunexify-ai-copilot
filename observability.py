import re
import uuid
from contextvars import ContextVar

import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.starlette import StarletteIntegration

from config import settings

REQUEST_ID_HEADER = "X-Request-ID"
_VALID_REQUEST_ID = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)
_ALLOWED_TAGS = {"service", "operation", "exception.class", "request_id", "environment", "release"}


def initialize_sentry() -> None:
    if not settings.sentry_dsn:
        return
    sentry_sdk.init(
        dsn=settings.sentry_dsn,
        environment=settings.sentry_environment,
        release=settings.release_sha or None,
        send_default_pii=False,
        traces_sample_rate=0.0,
        profiles_sample_rate=0.0,
        before_send=scrub_event,
        # Disabled: the SDK's default (500-599) auto-reports any HTTPException by
        # status code, duplicating our manual capture_unexpected() calls and — worse —
        # reporting expected provider errors that route handlers deliberately convert
        # to a 502/503 without treating them as unexpected. Real uncaught exceptions
        # still reach Sentry via the SDK's ASGI-level capture, which isn't gated by this.
        integrations=[
            StarletteIntegration(failed_request_status_codes=set()),
            FastApiIntegration(failed_request_status_codes=set()),
        ],
    )


def scrub_event(event: dict, hint: dict) -> dict:
    for key in ("request", "user", "breadcrumbs", "extra", "contexts"):
        event.pop(key, None)
    event["message"] = "Unexpected application error"
    for value in event.get("exception", {}).get("values", []):
        value["value"] = "Unexpected application error"
    event["tags"] = {k: v for k, v in event.get("tags", {}).items() if k in _ALLOWED_TAGS}
    return event


def normalize_request_id(value: str | None) -> str:
    return value if value and _VALID_REQUEST_ID.fullmatch(value) else str(uuid.uuid4())


def set_request_id(value: str) -> object:
    return _request_id.set(value)


def reset_request_id(token: object) -> None:
    _request_id.reset(token)


def capture_unexpected(operation: str, failure: Exception) -> None:
    if not settings.sentry_dsn:
        return
    with sentry_sdk.push_scope() as scope:
        scope.set_tag("service", "ai")
        scope.set_tag("operation", operation if re.fullmatch(r"[A-Za-z0-9._-]{1,80}", operation) else "unknown")
        scope.set_tag("exception.class", type(failure).__name__)
        request_id = _request_id.get()
        if request_id:
            scope.set_tag("request_id", request_id)
        sentry_sdk.capture_exception(failure)
