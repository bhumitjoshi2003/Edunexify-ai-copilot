"""
tools/admin_results.py — School-wide exam performance tool for the ADMIN role.

Composes three existing endpoints, none of them new:
  1. GET /api/dashboard/class-stats                              -> the school's class list
  2. GET /api/exams?session=&className=                          -> that class's exams
  3. GET /api/marks/class/{className}/exam/{examConfigId}        -> per-student, per-subject marks

MarkController's class-results endpoint only restricts TEACHER callers to
their own classTeacher assignment (checkTeacherClassAccess returns null,
i.e. "allowed", for any non-TEACHER role) — so ADMIN can already fetch any
class's results, same as it can already fetch any class's attendance.

Each class may be on a different "latest exam" (a Class 10 final exam and a
Class 5 unit test aren't the same event) — we resolve "latest" per class
independently, the same highest-id convention tools/teacher_results.py uses,
and surface each class's examName alongside its score so that's visible
rather than silently assumed to be the same exam school-wide.

Fetches for different classes run concurrently (each class's own two calls
are inherently sequential — you need the examConfigId before you can fetch
results) to keep latency reasonable for schools with many classes.
"""
import asyncio
from collections import defaultdict

import httpx

from config import settings
from schemas.chat import UserContext
from tools.attendance import current_academic_session


async def _class_latest_exam_performance(
    client: httpx.AsyncClient, class_name: str, session: str, cookies: dict
) -> dict | None:
    exams_resp = await client.get(
        f"{settings.spring_boot_url}/api/exams",
        params={"session": session, "className": class_name},
        cookies=cookies,
        timeout=10.0,
    )
    if exams_resp.status_code != 200:
        return None
    exams: list[dict] = exams_resp.json()
    if not exams:
        return None

    latest = max(exams, key=lambda e: e["id"])

    results_resp = await client.get(
        f"{settings.spring_boot_url}/api/marks/class/{class_name}/exam/{latest['id']}",
        cookies=cookies,
        timeout=10.0,
    )
    if results_resp.status_code != 200:
        return None
    results: list[dict] = results_resp.json()

    scored = [r for r in results if r.get("percentage") is not None]
    if not scored:
        return None

    subject_scores: dict[str, list[float]] = defaultdict(list)
    for r in results:
        for s in r.get("subjects") or []:
            obtained = s.get("marksObtained")
            max_marks = s.get("maxMarks")
            if obtained is not None and max_marks:
                subject_scores[s["subjectName"]].append(obtained / max_marks * 100)

    return {
        "className": class_name,
        "examName": latest.get("examName"),
        "classAveragePercentage": round(sum(r["percentage"] for r in scored) / len(scored), 1),
        "studentCount": len(scored),
        "subjectAverages": {name: round(sum(v) / len(v), 1) for name, v in subject_scores.items()},
    }


async def get_school_performance_summary(
    user: UserContext,
    access_token: str,
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

        per_class = await asyncio.gather(
            *[_class_latest_exam_performance(client, c, resolved_session, cookies) for c in class_names]
        )

    per_class = [c for c in per_class if c is not None]
    if not per_class:
        return {
            "session": resolved_session,
            "noResultsYet": True,
            "message": f"No exam results are available yet for any class in session {resolved_session}.",
        }

    by_avg_asc = sorted(per_class, key=lambda c: c["classAveragePercentage"])

    # School-wide subject averages: average each class's subject average (not
    # every student), so a large class doesn't drown out a small one's signal.
    subject_scores_all: dict[str, list[float]] = defaultdict(list)
    for c in per_class:
        for subject, avg in c["subjectAverages"].items():
            subject_scores_all[subject].append(avg)

    subject_school_wide = sorted(
        (
            {"subject": s, "schoolAveragePercentage": round(sum(v) / len(v), 1)}
            for s, v in subject_scores_all.items()
        ),
        key=lambda x: x["schoolAveragePercentage"],
    )

    return {
        "session": resolved_session,
        "classesEvaluated": len(per_class),
        "classesWithNoExamYet": len(class_names) - len(per_class),
        "schoolAveragePercentage": round(sum(c["classAveragePercentage"] for c in per_class) / len(per_class), 1),
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
    }
