"""
tools/admin_dashboard.py — School-wide overview tools for the ADMIN role.

  GET /api/dashboard/stats                              -> DashboardStatsDto   (single flat object, already compact)
  GET /api/dashboard/class-stats                         -> List<ClassStatsDto> (this month only)
  GET /api/attendance/summary/school?type=year&session=  -> List<ClassAttendanceSummaryDTO> (whole session, every class in one call)

All three are @PreAuthorize'd to ADMIN/SUPER_ADMIN and derive schoolId from
the JWT server-side — same guarantee the student/teacher tools rely on, just
at ADMIN's broader scope.

get_class_attendance_comparison defaults to the full academic session rather
than "this calendar month": the month-only class-stats endpoint reports 0%
for every class if attendance simply hasn't been marked yet this month,
indistinguishable from a genuine 0% rate unless you also check workingDays
(0 = no data, >0 = real numbers). A session-wide comparison is what admins
mean by "which classes have the lowest attendance" far more often than a
possibly-empty current month, so that's the default; "month" is still
available as an explicit choice.
"""
import httpx

from config import settings
from schemas.chat import UserContext
from tools.attendance import current_academic_session


async def get_school_overview(user: UserContext, access_token: str) -> dict:
    """Fetches the top-level school snapshot: student/teacher counts, fees collected
    this month, overdue student count, today's attendance rate, pending leave requests."""
    url = f"{settings.spring_boot_url}/api/dashboard/stats"

    async with httpx.AsyncClient() as client:
        response = await client.get(url, cookies={"accessToken": access_token}, timeout=10.0)

    if response.status_code == 403:
        return {"error": "Access denied. School overview is only available to admins."}
    if response.status_code != 200:
        return {"error": f"Spring Boot returned {response.status_code}: {response.text}"}

    return response.json()


async def get_class_attendance_comparison(
    user: UserContext,
    access_token: str,
    type: str = "year",
    session: str | None = None,
) -> dict:
    cookies = {"accessToken": access_token}

    if type == "month":
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{settings.spring_boot_url}/api/dashboard/class-stats",
                cookies=cookies,
                timeout=10.0,
            )
        if response.status_code == 403:
            return {"error": "Access denied. Class attendance comparison is only available to admins."}
        if response.status_code != 200:
            return {"error": f"Spring Boot returned {response.status_code}: {response.text}"}

        classes: list[dict] = response.json()
        if not classes:
            return {"message": "No classes with active students found."}

        no_data_yet = [c["className"] for c in classes if c.get("workingDays", 0) == 0]
        measurable = [c for c in classes if c.get("workingDays", 0) > 0]

        if not measurable:
            return {
                "period": "current calendar month",
                "message": "No attendance has been recorded yet this month for any class.",
                "classesWithNoAttendanceRecordedYet": no_data_yet,
            }

        by_rate_asc = sorted(measurable, key=lambda c: c["attendanceRate"])
        school_avg = round(sum(c["attendanceRate"] for c in measurable) / len(measurable), 1)

        return {
            "period": "current calendar month",
            "classCount": len(measurable),
            "schoolAverageAttendancePercentage": school_avg,
            "lowestAttendanceClasses": by_rate_asc[:3],
            "highestAttendanceClasses": by_rate_asc[-3:][::-1],
            "allClasses": by_rate_asc,
            "classesWithNoAttendanceRecordedYet": no_data_yet,
        }

    if type != "year":
        return {"error": f"Unknown type '{type}'. Use 'year' or 'month'."}

    resolved_session = session or await current_academic_session(access_token)

    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{settings.spring_boot_url}/api/attendance/summary/school",
            params={"type": "year", "session": resolved_session},
            cookies=cookies,
            timeout=10.0,
        )

    if response.status_code == 403:
        return {"error": "Access denied. Class attendance comparison is only available to admins."}
    if response.status_code != 200:
        return {"error": f"Spring Boot returned {response.status_code}: {response.text}"}

    students: list[dict] = response.json()
    if not students:
        return {"session": resolved_session, "message": "No attendance data recorded yet for this session."}

    by_class: dict[str, list[float]] = {}
    counts: dict[str, int] = {}
    for s in students:
        by_class.setdefault(s["className"], []).append(s["attendancePercentage"])
        counts[s["className"]] = counts.get(s["className"], 0) + 1

    per_class = [
        {"className": cls, "studentCount": counts[cls], "attendanceRate": round(sum(pcts) / len(pcts), 1)}
        for cls, pcts in by_class.items()
    ]
    by_rate_asc = sorted(per_class, key=lambda c: c["attendanceRate"])
    school_avg = round(sum(c["attendanceRate"] for c in per_class) / len(per_class), 1)

    return {
        "period": f"session {resolved_session}",
        "classCount": len(per_class),
        "schoolAverageAttendancePercentage": school_avg,
        "lowestAttendanceClasses": by_rate_asc[:3],
        "highestAttendanceClasses": by_rate_asc[-3:][::-1],
        "allClasses": by_rate_asc,
    }
