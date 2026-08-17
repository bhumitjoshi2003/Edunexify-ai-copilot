"""
tools/admin_staff_attendance.py — Read-only staff/teacher attendance insights
for the ADMIN role.

Endpoints used (all under TeacherAttendanceController in the Java backend —
the same calendar/holiday/TeacherLeave-aware TeacherAttendanceService the
Staff Attendance page and admin dashboard widget already use as their
authoritative source; nothing here re-derives a status or a working day):
  - GET /api/teacher-checkin/today-summary  (ADMIN-only) — used ONLY to
    resolve "today" (and "this month") in the school's OWN configured
    timezone, never via Python's local date.today(). A single-day staff
    attendance query is exactly the kind of fact that timezone mismatch
    could get wrong — this mirrors the reasoning academic_calendar.py
    documents for session labels, just applied more strictly since here the
    wrong day is a wrong, nameable claim about real people ("Alex was
    absent") rather than a coarse date-range label.
  - GET /api/teacher-checkin/date/{date}  (ADMIN+TEACHER) — every teacher's
    row for one date, real rows or the service's own ABSENT/ON_LEAVE
    synthesis for settled working days. We only bucket the already-resolved
    `status` field by name here.
  - GET /api/teacher-checkin/summary?month&year  (ADMIN-only, whole-school)
    — school-wide totals plus every teacher's per-day rows for the month.
    "Which teachers have the most absences" is answered by GROUPING those
    already-computed statuses by teacherId (the same plain aggregation
    get_fee_defaulters already does by class) — never by recomputing
    ON_TIME/LATE/ABSENT/HALF_DAY/ON_LEAVE ourselves.
"""
import httpx

from config import settings
from schemas.chat import UserContext

_STATUS_BUCKETS = {
    "ON_TIME": "presentTeachers",
    "LATE": "lateTeachers",
    "ABSENT": "absentTeachers",
    "HALF_DAY": "halfDayTeachers",
    "ON_LEAVE": "onLeaveTeachers",
}


async def _resolve_school_today(access_token: str) -> dict | None:
    """
    Resolves today's date/month/year in the school's own configured timezone
    via /today-summary — the same call the admin dashboard widget uses.
    Returns None on any failure (403/network) so callers can report it
    the same way they report any other backend error, rather than silently
    falling back to a possibly-wrong local date.
    """
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{settings.spring_boot_url}/api/teacher-checkin/today-summary",
            cookies={"accessToken": access_token},
            timeout=10.0,
        )
    if response.status_code != 200:
        return None
    today_str = response.json()["date"]  # "YYYY-MM-DD"
    year, month, _ = today_str.split("-")
    return {"date": today_str, "month": int(month), "year": int(year)}


async def get_staff_attendance_by_date(
    user: UserContext,
    access_token: str,
    date: str | None = None,
) -> dict:
    resolved_date = date
    if not resolved_date:
        today = await _resolve_school_today(access_token)
        if today is None:
            return {"error": "Access denied, or could not resolve today's date. Staff attendance data is only available to admins."}
        resolved_date = today["date"]

    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{settings.spring_boot_url}/api/teacher-checkin/date/{resolved_date}",
            cookies={"accessToken": access_token},
            timeout=10.0,
        )

    if response.status_code == 403:
        return {"error": "Access denied. Staff attendance data is only available to admins."}
    if response.status_code != 200:
        return {"error": f"Spring Boot returned {response.status_code}: {response.text}"}

    records: list[dict] = response.json()

    buckets: dict[str, list[dict]] = {v: [] for v in _STATUS_BUCKETS.values()}
    for r in records:
        key = _STATUS_BUCKETS.get(r.get("status"))
        if key is None:
            continue
        buckets[key].append({
            "teacherId": r["teacherId"],
            "teacherName": r["teacherName"],
            "checkInTime": r.get("checkInTime"),
            "checkOutTime": r.get("checkOutTime"),
        })
    for lst in buckets.values():
        lst.sort(key=lambda t: t["teacherName"])

    if not records:
        return {
            "date": resolved_date,
            # An empty list here means "not a settled working day" (weekend/holiday) OR
            # "today, before it's over" — either way it is NOT "everyone was present".
            # The model must not report 0 absences/lates as a positive result in this case.
            "note": (
                "No staff attendance records exist for this date. This is expected on a "
                "non-working day (weekend or holiday), or for today before the day is over — "
                "it does not mean every teacher was present."
            ),
            **{k: [] for k in buckets},
            "presentCount": 0, "lateCount": 0, "absentCount": 0,
            "halfDayCount": 0, "onLeaveCount": 0,
        }

    return {
        "date": resolved_date,
        "presentCount": len(buckets["presentTeachers"]),
        "lateCount": len(buckets["lateTeachers"]),
        "absentCount": len(buckets["absentTeachers"]),
        "halfDayCount": len(buckets["halfDayTeachers"]),
        "onLeaveCount": len(buckets["onLeaveTeachers"]),
        **buckets,
    }


async def get_staff_attendance_month_summary(
    user: UserContext,
    access_token: str,
    month: int | None = None,
    year: int | None = None,
) -> dict:
    if month is None or year is None:
        today = await _resolve_school_today(access_token)
        if today is None:
            return {"error": "Access denied, or could not resolve the current month. Staff attendance data is only available to admins."}
        month = month or today["month"]
        year = year or today["year"]

    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{settings.spring_boot_url}/api/teacher-checkin/summary",
            params={"month": month, "year": year},
            cookies={"accessToken": access_token},
            timeout=10.0,
        )

    if response.status_code == 403:
        return {"error": "Access denied. Staff attendance data is only available to admins."}
    if response.status_code != 200:
        return {"error": f"Spring Boot returned {response.status_code}: {response.text}"}

    data = response.json()
    records: list[dict] = data.get("records") or []

    per_teacher: dict[str, dict] = {}
    for r in records:
        tid = r["teacherId"]
        entry = per_teacher.setdefault(tid, {
            "teacherId": tid,
            "teacherName": r["teacherName"],
            "presentDays": 0,
            "lateDays": 0,
            "absentDays": 0,
            "halfDayDays": 0,
            "onLeaveDays": 0,
        })
        status = r.get("status")
        if status in ("ON_TIME", "LATE"):
            entry["presentDays"] += 1
        if status == "LATE":
            entry["lateDays"] += 1
        elif status == "ABSENT":
            entry["absentDays"] += 1
        elif status == "HALF_DAY":
            entry["halfDayDays"] += 1
        elif status == "ON_LEAVE":
            entry["onLeaveDays"] += 1

    ranked = sorted(per_teacher.values(), key=lambda t: -t["absentDays"])

    if not records:
        return {
            "month": month,
            "year": year,
            "totalWorkingDays": data.get("totalWorkingDays", 0),
            "message": "No staff attendance data found for this month yet.",
        }

    return {
        "month": month,
        "year": year,
        # School-wide totals, straight from the backend — never recomputed here.
        "totalWorkingDays": data.get("totalWorkingDays"),
        "presentDays": data.get("presentDays"),
        "lateDays": data.get("lateDays"),
        "absentDays": data.get("absentDays"),
        "halfDayDays": data.get("halfDayDays"),
        "onLeaveDays": data.get("onLeaveDays"),
        "onTimePercentage": data.get("onTimePercentage"),
        "totalTeachersChecked": len(per_teacher),
        # Capped — a large staff list could otherwise run long.
        "teachersWithMostAbsences": ranked[:10],
        "truncated": len(ranked) > 10,
    }
