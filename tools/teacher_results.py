"""
tools/teacher_results.py — Class-wide exam performance tool for the TEACHER role.

Two Spring Boot calls are chained here:
  1. GET /api/exams?session=&className=       -> List<ExamConfig>, used only to
     resolve an exam name (or "latest") to an examConfigId. className is always
     user.className (the teacher's own assigned class), never LLM-supplied.
  2. GET /api/marks/class/{className}/exam/{examConfigId}
     -> List<ClassStudentResultDTO>, the actual per-student, per-subject marks.
     Spring Boot's MarkController.checkTeacherClassAccess enforces the teacher
     can only pull results for their own class — a 403 otherwise.

ExamConfig has no explicit exam date, only an auto-increment id, so "latest
exam" is taken as the highest id for the session — the same "last in list"
convention tools/results.py already uses for a student's most recent exam.

We reduce two verbose payloads (exam list + full per-student/per-subject
marks) into class-level and subject-level aggregates so the LLM never has to
average dozens of student records itself.
"""
from collections import defaultdict

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
    cookies = {"accessToken": access_token}

    async with httpx.AsyncClient() as client:
        exams_resp = await client.get(
            f"{settings.spring_boot_url}/api/exams",
            params={"session": resolved_session, "className": class_name},
            cookies=cookies,
            timeout=10.0,
        )

        if exams_resp.status_code != 200:
            return {"error": f"Spring Boot returned {exams_resp.status_code}: {exams_resp.text}"}

        exams: list[dict] = exams_resp.json()
        if not exams:
            return {
                "className": class_name,
                "session": resolved_session,
                "noExamsYet": True,
                "message": f"No exams are configured for {class_name} in session {resolved_session} yet.",
            }

        chosen = None
        if examName:
            chosen = next((e for e in exams if e.get("examName", "").lower() == examName.lower()), None)
            if chosen is None:
                return {
                    "error": f"No exam named '{examName}' found for {class_name} in {resolved_session}.",
                    "availableExams": [e.get("examName") for e in exams],
                }
        else:
            # No explicit date field on ExamConfig — highest id is the most recently created exam.
            chosen = max(exams, key=lambda e: e["id"])

        results_resp = await client.get(
            f"{settings.spring_boot_url}/api/marks/class/{class_name}/exam/{chosen['id']}",
            cookies=cookies,
            timeout=10.0,
        )

    if results_resp.status_code == 403:
        return {"error": "Access denied. You can only view performance data for your assigned class."}
    if results_resp.status_code != 200:
        return {"error": f"Spring Boot returned {results_resp.status_code}: {results_resp.text}"}

    results: list[dict] = results_resp.json()
    if not results:
        return {
            "className": class_name,
            "examName": chosen.get("examName"),
            "message": "No students found for this exam.",
        }

    scored = [r for r in results if r.get("percentage") is not None]
    class_avg = round(sum(r["percentage"] for r in scored) / len(scored), 1) if scored else None

    by_pct_desc = sorted(scored, key=lambda r: r["percentage"], reverse=True)

    def _slim_student(r: dict) -> dict:
        return {"studentName": r["studentName"], "percentage": round(r["percentage"], 1)}

    # Per-subject class average % — averaged only over students with an entered mark.
    subject_scores: dict[str, list[float]] = defaultdict(list)
    for r in results:
        for s in r.get("subjects") or []:
            obtained = s.get("marksObtained")
            max_marks = s.get("maxMarks")
            if obtained is not None and max_marks:
                subject_scores[s["subjectName"]].append(obtained / max_marks * 100)

    subject_performance = sorted(
        (
            {"subject": name, "classAveragePercentage": round(sum(scores) / len(scores), 1)}
            for name, scores in subject_scores.items()
        ),
        key=lambda x: x["classAveragePercentage"],
    )

    return {
        "className": class_name,
        "session": resolved_session,
        "examName": chosen.get("examName"),
        "studentCount": len(results),
        "studentsWithMarksEntered": len(scored),
        "classAveragePercentage": class_avg,
        "topPerformers": [_slim_student(r) for r in by_pct_desc[:3]],
        "studentsNeedingAttention": [_slim_student(r) for r in by_pct_desc[-3:][::-1]] if len(by_pct_desc) > 3 else [],
        "subjectPerformance": subject_performance,
        "weakestSubject": subject_performance[0] if subject_performance else None,
        "strongestSubject": subject_performance[-1] if subject_performance else None,
    }
