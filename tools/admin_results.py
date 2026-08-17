"""
tools/admin_results.py — Exam performance tools for the ADMIN role.

Both tools are thin wrappers around consolidated Spring Boot endpoints
(MarkService.computeClassExamPerformance / getSchoolPerformanceSummary):

  GET /api/marks/class/{className}/exam-performance?session=&examName=
    -> ClassExamPerformanceDTO: class average, ranked student list
       (top/lowest scorer = first/last entry), subject averages, and
       studentsWithNoMarksEntered — resolved and aggregated server-side in
       one call instead of the old client-side "list exams, pick one, fetch
       results, aggregate" round trip.

  GET /api/marks/school/performance-summary?session=
    -> SchoolPerformanceSummaryDTO: every active class's own latest exam,
       already aggregated, plus classesWithNoExamConfigured and
       classesWithExamButNoMarksEntered kept separate (collapsing them would
       hide whether a class was skipped because nothing's set up yet or
       because it's set up but ungraded).

Both endpoints exclude students with no mark entered from rankings/averages
server-side (rank == 0 in the underlying data) rather than counting them as
having scored 0% — that distinction used to get lost when this aggregation
happened in Python from raw per-student rows.

get_school_performance_summary still builds a school-wide top/bottom
performer list here, since that's a cross-class view specific to what the
LLM needs and isn't something a single class-scoped DTO would return.
"""
import httpx

from config import settings
from schemas.chat import UserContext
from tools.attendance import current_academic_session


async def get_class_exam_results(
    user: UserContext,
    access_token: str,
    className: str,
    session: str | None = None,
    examName: str | None = None,
) -> dict:
    """Per-student breakdown for one named class's exam — top/lowest scorer, full ranked list."""
    resolved_session = session or await current_academic_session(access_token)

    params: dict = {"session": resolved_session}
    if examName:
        params["examName"] = examName

    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{settings.spring_boot_url}/api/marks/class/{className}/exam-performance",
            params=params,
            cookies={"accessToken": access_token},
            timeout=10.0,
        )

    if response.status_code == 403:
        return {"error": f"Access denied for class {className}."}
    if response.status_code != 200:
        return {"error": f"Spring Boot returned {response.status_code}: {response.text}"}

    data: dict = response.json()
    if data.get("noExamsYet") or "error" in data:
        return data  # already shaped: {noExamsYet, message} or {error, availableExams}

    ranked: list[dict] = data.get("studentsRanked") or []

    return {
        "className": data.get("className", className),
        "session": resolved_session,
        "examName": data.get("examName"),
        "studentsWithMarksEntered": len(ranked),
        "studentsWithNoMarksEntered": data.get("studentsWithNoMarksEntered") or [],
        "classAveragePercentage": data.get("classAveragePercentage"),
        "topScorer": ranked[0] if ranked else None,
        "lowestScorer": ranked[-1] if ranked else None,
        "allStudentsRanked": ranked,
        "subjectPerformance": sorted(
            (
                {"subject": name, "classAveragePercentage": pct}
                for name, pct in (data.get("subjectAverages") or {}).items()
            ),
            key=lambda x: x["classAveragePercentage"],
        ),
    }


async def get_school_performance_summary(
    user: UserContext,
    access_token: str,
    session: str | None = None,
) -> dict:
    resolved_session = session or await current_academic_session(access_token)

    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{settings.spring_boot_url}/api/marks/school/performance-summary",
            params={"session": resolved_session},
            cookies={"accessToken": access_token},
            timeout=15.0,
        )

    if response.status_code == 403:
        return {"error": "Access denied. School performance data is only available to admins."}
    if response.status_code != 200:
        return {"error": f"Spring Boot returned {response.status_code}: {response.text}"}

    data: dict = response.json()
    class_results: list[dict] = data.get("classResults") or []

    if not class_results:
        return {
            "session": resolved_session,
            "noResultsYet": True,
            "message": f"No exam results are available yet for any class in session {resolved_session}.",
            "classesWithNoExamConfigured": data.get("classesWithNoExamConfigured") or [],
            "classesWithExamButNoMarksEntered": data.get("classesWithExamButNoMarksEntered") or [],
        }

    by_avg_asc = sorted(class_results, key=lambda c: c["classAveragePercentage"])

    # School-wide subject averages: average each class's subject average (not
    # every student), so a large class doesn't drown out a small one's signal.
    subject_scores_all: dict[str, list[float]] = {}
    for c in class_results:
        for subject, avg in (c.get("subjectAverages") or {}).items():
            subject_scores_all.setdefault(subject, []).append(avg)

    subject_school_wide = sorted(
        (
            {"subject": s, "schoolAveragePercentage": round(sum(v) / len(v), 1)}
            for s, v in subject_scores_all.items()
        ),
        key=lambda x: x["schoolAveragePercentage"],
    )

    # School-wide student leaderboard, built from the same per-class ranked
    # lists above. Percentages are compared across different exams per class,
    # so this is an approximation, not a single ranked exam — the "note"
    # field says so explicitly for the LLM to relay.
    all_students = [
        {"studentName": s["studentName"], "percentage": s["percentage"], "className": c["className"]}
        for c in class_results
        for s in (c.get("studentsRanked") or [])
    ]
    by_student_pct_desc = sorted(all_students, key=lambda s: s["percentage"], reverse=True)

    return {
        "session": resolved_session,
        "classesEvaluated": len(class_results),
        "classesWithNoExamConfigured": data.get("classesWithNoExamConfigured") or [],
        "classesWithExamButNoMarksEntered": data.get("classesWithExamButNoMarksEntered") or [],
        "schoolAveragePercentage": round(
            sum(c["classAveragePercentage"] for c in class_results) / len(class_results), 1
        ),
        "bestPerformingClass": {
            "className": by_avg_asc[-1]["className"],
            "examName": by_avg_asc[-1]["examName"],
            "percentage": by_avg_asc[-1]["classAveragePercentage"],
        },
        "worstPerformingClass": {
            "className": by_avg_asc[0]["className"],
            "examName": by_avg_asc[0]["examName"],
            "percentage": by_avg_asc[0]["classAveragePercentage"],
        },
        "classBreakdown": [
            {"className": c["className"], "examName": c["examName"], "percentage": c["classAveragePercentage"]}
            for c in by_avg_asc
        ],
        "weakestSubjectsSchoolWide": subject_school_wide[:5],
        "schoolWideTopPerformers": by_student_pct_desc[:5],
        "schoolWideNeedingAttention": by_student_pct_desc[-5:][::-1],
        "note": "Top/bottom performers compare each class's own latest exam — classes may be on different exams, so this is an approximation, not one single ranked test.",
    }
