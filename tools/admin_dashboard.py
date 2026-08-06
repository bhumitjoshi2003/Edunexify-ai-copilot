"""
tools/admin_dashboard.py — School-wide overview tools for the ADMIN role.

Both tools here are thin wrappers around Spring Boot's existing DashboardController
(the same data admin-dashboard.component.ts already renders as charts) — no new
backend logic, just reshaping for the LLM:

  GET /api/dashboard/stats        -> DashboardStatsDto   (single flat object, already compact)
  GET /api/dashboard/class-stats  -> List<ClassStatsDto>  (per-class student count + attendance rate)

Both endpoints are @PreAuthorize'd to ADMIN/SUPER_ADMIN and derive schoolId
from the JWT server-side (via DashboardService's SecurityUtil) — same
guarantee the student/teacher tools rely on, just at ADMIN's broader scope.

Note: class-stats' attendanceRate is always "this calendar month" — Spring
Boot doesn't currently accept a period param for it. We surface that as-is
rather than pretending it's configurable.
"""
import httpx

from config import settings
from schemas.chat import UserContext


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


async def get_class_attendance_comparison(user: UserContext, access_token: str) -> dict:
    """Fetches this month's attendance rate for every class, for cross-class comparison."""
    url = f"{settings.spring_boot_url}/api/dashboard/class-stats"

    async with httpx.AsyncClient() as client:
        response = await client.get(url, cookies={"accessToken": access_token}, timeout=10.0)

    if response.status_code == 403:
        return {"error": "Access denied. Class attendance comparison is only available to admins."}
    if response.status_code != 200:
        return {"error": f"Spring Boot returned {response.status_code}: {response.text}"}

    classes: list[dict] = response.json()
    if not classes:
        return {"message": "No classes with active students found."}

    by_rate_asc = sorted(classes, key=lambda c: c["attendanceRate"])
    school_avg = round(sum(c["attendanceRate"] for c in classes) / len(classes), 1)

    return {
        "period": "current calendar month",
        "classCount": len(classes),
        "schoolAverageAttendancePercentage": school_avg,
        "lowestAttendanceClasses": by_rate_asc[:3],
        "highestAttendanceClasses": by_rate_asc[-3:][::-1],
        "allClasses": by_rate_asc,
    }
