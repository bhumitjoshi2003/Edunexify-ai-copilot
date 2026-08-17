"""
tools/results.py — Fetches a student's exam results from Spring Boot.

Endpoint used: GET /api/marks/student/{studentId}/results?session=YYYY-YYYY
  - Returns List<ExamResultDTO> — one entry per exam in the session.
  - Each ExamResultDTO has: examName, subjects (with rank, classAverage,
    marksObtained, maxMarks), totalMarksObtained, totalMaxMarks,
    percentage, overallRank.
  - Spring Boot's MarkController enforces STUDENT role can only see their
    own results regardless of the studentId path param.

We post-process into a compact summary so the LLM gets pre-aggregated,
clean data (latest exam highlight, per-exam breakdown, cross-exam best/worst
subject) rather than the full verbose payload.
"""
import httpx
from collections import defaultdict

from config import settings
from schemas.chat import UserContext
from academic_calendar import current_academic_session


def _pct(obtained: float | None, max_marks: float | None) -> float | None:
    """Return percentage rounded to 1 dp, or None if inputs are missing."""
    if obtained is None or not max_marks:
        return None
    return round(obtained / max_marks * 100, 1)


def _grade(pct: float | None) -> str:
    """Simple letter-grade helper for display."""
    if pct is None:
        return "N/A"
    if pct >= 90:
        return "A+"
    if pct >= 80:
        return "A"
    if pct >= 70:
        return "B"
    if pct >= 60:
        return "C"
    if pct >= 50:
        return "D"
    return "F"


async def get_results_summary(
    user: UserContext,
    access_token: str,
    session: str | None = None,
) -> dict:
    """
    Fetches all exam results for the student in the given session and
    returns a structured, LLM-friendly summary.

    Included in output:
    - Per-exam breakdown with subject-level marks, rank, class average
    - Highlights: highest/lowest subject per exam
    - Cross-exam aggregates: best and weakest subject overall (by avg %)
    - Latest exam pointer (last in the returned list)
    """
    resolved_session = session or await current_academic_session(access_token)
    url = f"{settings.spring_boot_url}/api/marks/student/{user.userId}/results"

    async with httpx.AsyncClient() as client:
        response = await client.get(
            url,
            params={"session": resolved_session},
            cookies={"accessToken": access_token},
            timeout=10.0,
        )

    if response.status_code == 403:
        return {"error": "Access denied. You can only view your own results."}
    if response.status_code != 200:
        # Return the raw Spring Boot response body so the error can be diagnosed
        return {"error": f"Spring Boot returned {response.status_code}: {response.text}"}

    exams: list[dict] = response.json()
    if not exams:
        return {
            "session": resolved_session,
            "noResultsYet": True,
            "message": (
                f"No exam results found for session {resolved_session}. "
                "Exams may not have been configured or marks may not have been entered yet."
            ),
        }

    # ── Per-exam summary ───────────────────────────────────────────────────────
    exam_summaries: list[dict] = []

    # Track subject performance across all exams for cross-exam aggregates
    # subject_name → list of (obtained, max) tuples where obtained is not None
    subject_scores: dict[str, list[tuple[float, float]]] = defaultdict(list)

    for exam in exams:
        exam_name: str = exam.get("examName", "Unknown Exam")
        subjects_raw: list[dict] = exam.get("subjects") or []

        subject_details: list[dict] = []
        scored_subjects: list[dict] = []  # subjects with marks for highlight

        for s in subjects_raw:
            obtained = s.get("marksObtained")
            max_m = s.get("maxMarks")
            avg = s.get("classAverage")
            rank = s.get("rank", 0)
            subj_name: str = s.get("subjectName", "Unknown")

            subj_pct = _pct(obtained, max_m)
            avg_pct = _pct(avg, max_m) if avg is not None else None

            detail = {
                "subject": subj_name,
                "obtained": obtained,
                "maxMarks": max_m,
                "percentage": subj_pct,
                "grade": _grade(subj_pct),
                "rank": rank if rank and rank > 0 else None,
                "classAvgPercentage": avg_pct,
            }
            subject_details.append(detail)

            if obtained is not None and max_m:
                scored_subjects.append(detail)
                subject_scores[subj_name].append((obtained, float(max_m)))

        # Highest and lowest subject for this exam
        highest = max(scored_subjects, key=lambda x: x["percentage"]) if scored_subjects else None
        lowest = min(scored_subjects, key=lambda x: x["percentage"]) if scored_subjects else None

        total_obtained = exam.get("totalMarksObtained")
        total_max = exam.get("totalMaxMarks")
        overall_pct = exam.get("percentage") or _pct(total_obtained, total_max)
        overall_rank = exam.get("overallRank", 0) or None

        exam_summaries.append({
            "examName": exam_name,
            "totalObtained": total_obtained,
            "totalMax": total_max,
            "percentage": round(overall_pct, 1) if overall_pct else None,
            "grade": _grade(overall_pct),
            "overallRank": overall_rank if overall_rank and overall_rank > 0 else None,
            "subjects": subject_details,
            "highestSubject": (
                {"subject": highest["subject"], "percentage": highest["percentage"]} if highest else None
            ),
            "lowestSubject": (
                {"subject": lowest["subject"], "percentage": lowest["percentage"]} if lowest else None
            ),
        })

    # ── Cross-exam aggregates ──────────────────────────────────────────────────
    subject_avg_pct: list[dict] = []
    for subj_name, scores in subject_scores.items():
        avg_pct = round(sum(o / m * 100 for o, m in scores) / len(scores), 1)
        subject_avg_pct.append({"subject": subj_name, "avgPercentage": avg_pct})

    best_subject = max(subject_avg_pct, key=lambda x: x["avgPercentage"]) if subject_avg_pct else None
    weakest_subject = min(subject_avg_pct, key=lambda x: x["avgPercentage"]) if subject_avg_pct else None

    return {
        "session": resolved_session,
        "examCount": len(exam_summaries),
        "latestExam": exam_summaries[-1]["examName"] if exam_summaries else None,
        "exams": exam_summaries,
        "bestSubjectOverall": best_subject,
        "weakestSubjectOverall": weakest_subject,
    }
