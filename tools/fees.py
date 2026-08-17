"""
tools/fees.py — Tool that fetches a student's fee summary from Spring Boot.

Endpoint used: GET /api/student-fees/{studentId}/{year}
  - Returns a list of StudentFees records — one per academic month (up to 12).
  - Each record has: month (1–12), paid (bool), amountPaid, takesBus, etc.
  - Spring Boot's StudentFeesController ignores the studentId path param for
    STUDENT role and substitutes authService.getUserId() — so a student can
    never see another student's fees even if they guess a different studentId.

IMPORTANT: `month` in StudentFees is the ACADEMIC month number (1 = first month
of the school's academic year, not January). A school starting in April has
month=1 → April, month=2 → May, ..., month=9 → December, month=10 → January.
We fetch the school's academicYearStartMonth from /api/school/settings and use
it to convert academic month numbers to correct calendar month names.

We post-process the raw list into a compact summary dict so the LLM receives
clean, pre-aggregated data rather than a verbose 12-item array. This reduces
token usage and produces more accurate, consistent answers.
"""
import logging

import httpx

from academic_calendar import (
    academic_month_to_calendar_name,
    fetch_academic_year_start_month,
    session_label_for_start_month,
)
from config import settings
from schemas.chat import UserContext

logger = logging.getLogger(__name__)


async def get_fee_summary(
    user: UserContext,
    access_token: str,
    session: str | None = None,
) -> dict:
    """
    Fetches month-by-month fee records for the student and returns a summary.

    Spring Boot enforces: STUDENT role always receives their own data.
    The studentId in the URL is overridden server-side for STUDENT role,
    so there is no risk of cross-student data access.
    """
    # Fetched once and reused below for both session resolution (if the caller
    # didn't name one) and converting academic month numbers to calendar names.
    start_month = await fetch_academic_year_start_month(access_token)
    resolved_session = session or session_label_for_start_month(start_month)

    url = f"{settings.spring_boot_url}/api/student-fees/{user.userId}/{resolved_session}"

    async with httpx.AsyncClient() as client:
        response = await client.get(
            url,
            cookies={"accessToken": access_token},
            timeout=10.0,
        )

    if response.status_code == 404:
        return {
            "session": resolved_session,
            "message": f"No fee records found for session {resolved_session}. Fees may not have been generated yet.",
        }
    if response.status_code == 403:
        return {"error": "Access denied. You can only view your own fees."}
    if response.status_code != 200:
        return {"error": f"Could not fetch fees (status {response.status_code})."}

    records: list[dict] = response.json()
    if not records:
        return {
            "session": resolved_session,
            "message": f"No fee records found for session {resolved_session}.",
        }

    # ── Aggregate into a summary the LLM can reason about easily ──────────────
    paid_months: list[str] = []
    unpaid_months: list[str] = []
    total_paid: float = 0.0

    for r in records:
        academic_month: int = r.get("month", 0)
        month_name = academic_month_to_calendar_name(academic_month, start_month)
        is_paid: bool = r.get("paid") or r.get("manuallyPaid") or False
        amount_paid = r.get("amountPaid") or r.get("manualPaymentReceived") or 0

        if is_paid:
            paid_months.append(month_name)
            try:
                total_paid += float(amount_paid)
            except (TypeError, ValueError):
                pass
        else:
            unpaid_months.append(month_name)

    return {
        "session": resolved_session,
        "academicYearStartMonth": start_month,
        "totalMonths": len(records),
        "paidMonths": paid_months,
        "unpaidMonths": unpaid_months,
        "paidCount": len(paid_months),
        "unpaidCount": len(unpaid_months),
        "totalAmountPaid": round(total_paid, 2),
        "hasPendingFees": len(unpaid_months) > 0,
        "takesBus": records[0].get("takesBus", False) if records else False,
    }
