"""
tools/admin_fees.py — School-wide fee defaulter insights for the ADMIN role.

Endpoint used: GET /api/student-fees/overdue?session=&className=
  - Returns List<OverdueStudentDto>: every active student with unpaid,
    past-due fee months for the session — already computed by
    FeeReminderService (the same data fee-reminders.component.ts uses to
    send reminder notifications), so no new backend logic needed.
  - @PreAuthorize'd to ADMIN/SUPER_ADMIN; schoolId comes from the JWT.

We turn the raw per-student list into class-level totals plus a capped
"most overdue" list, since a school-wide defaulter list can otherwise run
to hundreds of rows.
"""
from collections import defaultdict

import httpx

from config import settings
from schemas.chat import UserContext
from tools.attendance import current_academic_session


async def get_fee_defaulters(
    user: UserContext,
    access_token: str,
    session: str | None = None,
    className: str | None = None,
) -> dict:
    resolved_session = session or await current_academic_session(access_token)

    params: dict = {"session": resolved_session}
    if className:
        params["className"] = className

    url = f"{settings.spring_boot_url}/api/student-fees/overdue"

    async with httpx.AsyncClient() as client:
        response = await client.get(
            url,
            params=params,
            cookies={"accessToken": access_token},
            timeout=10.0,
        )

    if response.status_code == 403:
        return {"error": "Access denied. Fee defaulter data is only available to admins."}
    if response.status_code != 200:
        return {"error": f"Spring Boot returned {response.status_code}: {response.text}"}

    defaulters: list[dict] = response.json()
    if not defaulters:
        return {
            "session": resolved_session,
            "className": className,
            "defaulterCount": 0,
            "message": "No students have pending fees for this session.",
        }

    # totalDue is null for a student whose class has no FeeStructure configured for this
    # session — that means "amount unknown", not "₹0 due". Never coerce it to 0 in a sum:
    # that would silently understate the total and imply those students owe nothing.
    known_dues = [d["totalDue"] for d in defaulters if d.get("totalDue") is not None]
    unknown_count = len(defaulters) - len(known_dues)
    total_due = round(sum(known_dues), 2) if known_dues else None

    by_class: dict[str, dict] = defaultdict(lambda: {"defaulterCount": 0, "totalDue": 0.0, "unknownCount": 0})
    for d in defaulters:
        entry = by_class[d.get("className", "Unknown")]
        entry["defaulterCount"] += 1
        if d.get("totalDue") is not None:
            entry["totalDue"] += d["totalDue"]
        else:
            entry["unknownCount"] += 1

    class_breakdown = sorted(
        (
            {
                "className": cls,
                "defaulterCount": v["defaulterCount"],
                "totalDue": round(v["totalDue"], 2) if v["unknownCount"] < v["defaulterCount"] else None,
                "amountUnknownCount": v["unknownCount"],
            }
            for cls, v in by_class.items()
        ),
        key=lambda x: -x["defaulterCount"],
    )

    most_overdue = sorted(defaulters, key=lambda d: -(d.get("daysOverdue") or 0))[:10]

    return {
        "session": resolved_session,
        "className": className,
        "defaulterCount": len(defaulters),
        # None means no student in this result has a configured fee structure for the
        # session — the model must say the amount is unknown, never state ₹0.
        "totalAmountDue": total_due,
        "amountUnknownCount": unknown_count,
        "byClass": class_breakdown,
        "mostOverdue": [
            {
                "studentName": d["studentName"],
                "className": d["className"],
                "unpaidMonths": d.get("unpaidMonths") or [],
                "totalDue": d.get("totalDue"),
                "daysOverdue": d.get("daysOverdue"),
            }
            for d in most_overdue
        ],
        "truncated": len(defaulters) > 10,
    }
