"""
tools/teacher_attendance.py — Class-wide attendance tools for the TEACHER role.

Endpoint used: GET /api/attendance/summary/class/{className}
  - Returns List<ClassAttendanceSummaryDTO>: one entry per student in the class
    (studentId, studentName, totalWorkingDays, daysPresent, daysAbsent, attendancePercentage).
  - Spring Boot's AttendanceController enforces that a TEACHER caller's own
    `classTeacher` field must equal the requested className — a teacher can
    never pull another class's attendance, regardless of what the LLM sends.

We never take className as a tool argument from the LLM. It always comes from
`user.className`, which Spring Boot itself resolved from the teacher's
classTeacher field when building the request (see AiProxyController). This
means there is no argument the model could pass that widens access — the
same guarantee the student tools rely on with user.userId.

Both tools below share one fetch of the raw per-student list and differ only
in how they post-process it, so the LLM gets compact, purpose-built output
instead of the full class roster either way.
"""
import httpx

from config import settings
from schemas.chat import UserContext
from tools.attendance import current_academic_session


async def _fetch_class_attendance(
    user: UserContext,
    access_token: str,
    type: str,
    session: str | None,
    month: int | None,
    year: int | None,
) -> list[dict] | dict:
    """Returns the raw per-student list, or a dict with an 'error' key."""
    class_name = user.className
    if not class_name:
        return {"error": "You are not assigned as a class teacher, so class-wide attendance isn't available to you."}

    params: dict = {"type": type}
    if type == "year":
        params["session"] = session or current_academic_session()
    elif type == "month":
        if month is None or year is None:
            return {"error": "month and year are required when type is 'month'"}
        params["month"] = month
        params["year"] = year
    else:
        return {"error": f"Unknown type '{type}'. Use 'year' or 'month'."}

    url = f"{settings.spring_boot_url}/api/attendance/summary/class/{class_name}"

    async with httpx.AsyncClient() as client:
        response = await client.get(
            url,
            params=params,
            cookies={"accessToken": access_token},
            timeout=10.0,
        )

    if response.status_code == 403:
        return {"error": "Access denied. You can only view attendance for your assigned class."}
    if response.status_code != 200:
        return {"error": f"Spring Boot returned {response.status_code}: {response.text}"}

    return response.json()


async def get_class_attendance_summary(
    user: UserContext,
    access_token: str,
    type: str,
    session: str | None = None,
    month: int | None = None,
    year: int | None = None,
) -> dict:
    """Aggregate attendance overview for the teacher's whole class."""
    raw = await _fetch_class_attendance(user, access_token, type, session, month, year)
    if isinstance(raw, dict):
        return raw  # error dict

    if not raw:
        return {"className": user.className, "message": "No students found in this class."}

    percentages = [s["attendancePercentage"] for s in raw]
    class_avg = round(sum(percentages) / len(percentages), 1)
    by_pct_asc = sorted(raw, key=lambda s: s["attendancePercentage"])

    def _slim(s: dict) -> dict:
        return {"studentName": s["studentName"], "attendancePercentage": round(s["attendancePercentage"], 1)}

    return {
        "className": user.className,
        "period": {"type": type, "session": session, "month": month, "year": year},
        "studentCount": len(raw),
        "classAveragePercentage": class_avg,
        "studentsBelow75Percent": sum(1 for p in percentages if p < 75),
        "lowestAttendance": [_slim(s) for s in by_pct_asc[:3]],
        "highestAttendance": [_slim(s) for s in by_pct_asc[-3:][::-1]],
    }


async def get_low_attendance_students(
    user: UserContext,
    access_token: str,
    threshold: float = 75.0,
    type: str = "year",
    session: str | None = None,
    month: int | None = None,
    year: int | None = None,
) -> dict:
    """Returns only the students below `threshold`% attendance, lowest first."""
    raw = await _fetch_class_attendance(user, access_token, type, session, month, year)
    if isinstance(raw, dict):
        return raw  # error dict

    if not raw:
        return {"className": user.className, "message": "No students found in this class."}

    below = sorted(
        (s for s in raw if s["attendancePercentage"] < threshold),
        key=lambda s: s["attendancePercentage"],
    )

    return {
        "className": user.className,
        "period": {"type": type, "session": session, "month": month, "year": year},
        "threshold": threshold,
        "totalStudents": len(raw),
        "studentsBelowThreshold": len(below),
        "students": [
            {
                "studentId": s["studentId"],
                "studentName": s["studentName"],
                "attendancePercentage": round(s["attendancePercentage"], 1),
                "daysPresent": s["daysPresent"],
                "daysAbsent": s["daysAbsent"],
                "totalWorkingDays": s["totalWorkingDays"],
            }
            for s in below
        ],
    }
