"""
tools/admin_attendance.py — School-wide low-attendance lookup for the ADMIN role.

There's no single Spring Boot endpoint for "every student below X% attendance
across the whole school", so this composes two existing endpoints:

  1. GET /api/dashboard/class-stats                 -> the school's class list
  2. GET /api/attendance/summary/class/{className}   -> per-student attendance, per class

Unlike the TEACHER version of this tool (tools/teacher_attendance.py), which is
hard-scoped to the caller's own classTeacher assignment, AttendanceController
lets ADMIN callers request *any* className in their school — the teacher-only
restriction in that endpoint simply doesn't apply to the ADMIN role. So this
tool fans the per-class call out across every class in the school concurrently
and merges the results, still relying entirely on Spring Boot's own schoolId
scoping and role checks for each call.
"""
import asyncio

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
    resolved_session = session or current_academic_session()
    cookies = {"accessToken": access_token}

    async with httpx.AsyncClient() as client:
        class_stats_resp = await client.get(
            f"{settings.spring_boot_url}/api/dashboard/class-stats",
            cookies=cookies,
            timeout=10.0,
        )
        if class_stats_resp.status_code != 200:
            return {"error": f"Spring Boot returned {class_stats_resp.status_code}: {class_stats_resp.text}"}

        class_names = [c["className"] for c in class_stats_resp.json()]
        if not class_names:
            return {"message": "No classes with active students found."}

        async def _fetch_class(class_name: str) -> list[dict]:
            resp = await client.get(
                f"{settings.spring_boot_url}/api/attendance/summary/class/{class_name}",
                params={"type": "year", "session": resolved_session},
                cookies=cookies,
                timeout=10.0,
            )
            if resp.status_code != 200:
                return []
            return [{**student, "className": class_name} for student in resp.json()]

        per_class_results = await asyncio.gather(*[_fetch_class(c) for c in class_names])

    all_students = [student for class_list in per_class_results for student in class_list]
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
