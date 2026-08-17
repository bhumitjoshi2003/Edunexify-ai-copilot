"""
tools/admin_attendance.py — School-wide low-attendance lookup for the ADMIN role.

Endpoint used: GET /api/attendance/summary/school?type=year&session=
  - Returns List<ClassAttendanceSummaryDTO> (now includes className per row) —
    every student in the school in one call, computed server-side by looping
    classes internally (AttendanceService.getSchoolSummary), rather than this
    tool fanning out one HTTP request per class itself.
  - @PreAuthorize'd to ADMIN/SUPER_ADMIN; schoolId comes from the JWT.
"""
import httpx

from config import settings
from schemas.chat import UserContext
from tools.attendance import current_academic_session


async def get_school_low_attendance_students(
    user: UserContext,
    access_token: str,
    threshold: float = 75.0,
    session: str | None = None,
) -> dict:
    resolved_session = session or await current_academic_session(access_token)

    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{settings.spring_boot_url}/api/attendance/summary/school",
            params={"type": "year", "session": resolved_session},
            cookies={"accessToken": access_token},
            timeout=10.0,
        )

    if response.status_code == 403:
        return {"error": "Access denied. School-wide attendance data is only available to admins."}
    if response.status_code != 200:
        return {"error": f"Spring Boot returned {response.status_code}: {response.text}"}

    all_students: list[dict] = response.json()
    if not all_students:
        return {"session": resolved_session, "message": "No attendance data found for this session yet."}

    below_threshold = sorted(
        (s for s in all_students if s["attendancePercentage"] < threshold),
        key=lambda s: s["attendancePercentage"],
    )

    return {
        "session": resolved_session,
        "threshold": threshold,
        "totalStudentsChecked": len(all_students),
        "studentsBelowThreshold": len(below_threshold),
        # Capped — a school-wide list could otherwise run to hundreds of rows.
        "students": [
            {
                "studentId": s["studentId"],
                "studentName": s["studentName"],
                "className": s["className"],
                "attendancePercentage": round(s["attendancePercentage"], 1),
            }
            for s in below_threshold[:20]
        ],
        "truncated": len(below_threshold) > 20,
    }
