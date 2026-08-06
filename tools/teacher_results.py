"""
tools/teacher_results.py — Class-wide exam performance tool for the TEACHER role.

One Spring Boot call: GET /api/marks/class/{className}/exam-performance?session=&examName=
  -> ClassExamPerformanceDTO: class average, ranked student list, subject
     averages, and studentsWithNoMarksEntered — resolved (latest or named
     exam) and aggregated server-side in one call.
     MarkController.checkTeacherClassAccess enforces the teacher can only
     pull results for their own class — a 403 otherwise. className is always
     user.className (the teacher's own assigned class), never LLM-supplied.

Students with no mark entered are excluded from the ranking/averages
server-side (rank == 0 in the underlying data) rather than counted as having
scored 0% — MarkService.computeClassExamPerformance handles that distinction,
this tool just reshapes the already-correct result for the LLM.
"""
import httpx

from config import settings
from schemas.chat import UserContext
from tools.fees import current_academic_session


async def get_class_performance_summary(
    user: UserContext,
    access_token: str,
    session: str | None = None,
    examName: str | None = None,
) -> dict:
    class_name = user.className
    if not class_name:
        return {"error": "You are not assigned as a class teacher, so class performance data isn't available to you."}

    resolved_session = session or current_academic_session()

    params: dict = {"session": resolved_session}
    if examName:
        params["examName"] = examName

    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{settings.spring_boot_url}/api/marks/class/{class_name}/exam-performance",
            params=params,
            cookies={"accessToken": access_token},
            timeout=10.0,
        )

    if response.status_code == 403:
        return {"error": "Access denied. You can only view performance data for your assigned class."}
    if response.status_code != 200:
        return {"error": f"Spring Boot returned {response.status_code}: {response.text}"}

    data: dict = response.json()
    if data.get("noExamsYet") or "error" in data:
        return data  # already shaped: {noExamsYet, message} or {error, availableExams}

    ranked: list[dict] = data.get("studentsRanked") or []

    def _slim(s: dict) -> dict:
        return {"studentName": s["studentName"], "percentage": round(s["percentage"], 1)}

    subject_performance = sorted(
        (
            {"subject": name, "classAveragePercentage": pct}
            for name, pct in (data.get("subjectAverages") or {}).items()
        ),
        key=lambda x: x["classAveragePercentage"],
    )

    return {
        "className": data.get("className", class_name),
        "session": resolved_session,
        "examName": data.get("examName"),
        "studentsWithMarksEntered": len(ranked),
        "studentsWithNoMarksEntered": data.get("studentsWithNoMarksEntered") or [],
        "classAveragePercentage": data.get("classAveragePercentage"),
        "topPerformers": [_slim(s) for s in ranked[:3]],
        "studentsNeedingAttention": [_slim(s) for s in ranked[-3:][::-1]] if len(ranked) > 3 else [],
        "subjectPerformance": subject_performance,
        "weakestSubject": subject_performance[0] if subject_performance else None,
        "strongestSubject": subject_performance[-1] if subject_performance else None,
    }
