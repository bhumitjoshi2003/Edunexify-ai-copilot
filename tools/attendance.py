"""
tools/attendance.py — Tool that fetches student attendance from Spring Boot.

Design principle: each tool function is a plain async function that makes an
HTTP call to an existing Spring Boot endpoint. The function:
  - Receives the user context and the forwarded accessToken
  - Builds the correct URL and query params
  - Returns the parsed JSON dict, which is then serialised and sent back to the LLM

Why forward the accessToken cookie instead of a service-level token?
Because Spring Boot's existing auth checks (student can only see own data,
schoolId scoping) are tied to the JWT. Forwarding it means we get all those
guarantees for free without duplicating any logic here.
"""
from datetime import date

import httpx

from config import settings
from schemas.chat import UserContext


def current_academic_session() -> str:
    """
    Derive the current academic session string (e.g. '2026-2027').
    Edunexify academic years start in April (month 4).
    August 2026 → April 2026 is the start → session is '2026-2027'.
    """
    today = date.today()
    if today.month >= 4:
        return f"{today.year}-{today.year + 1}"
    return f"{today.year - 1}-{today.year}"


async def get_attendance_summary(
    user: UserContext,
    access_token: str,
    type: str,
    session: str | None = None,
    month: int | None = None,
    year: int | None = None,
) -> dict:
    """
    Calls: GET /api/attendance/summary/student/{userId}
    with ?type=year&session=YYYY-YYYY  OR  ?type=month&month=M&year=YYYY

    Spring Boot already enforces:
      - Students can only fetch their own attendance (returns 403 otherwise)
      - schoolId is taken from the JWT — no cross-tenant data possible

    We pass the accessToken as a cookie header so the request is treated
    identically to one coming from the Angular frontend.
    """
    url = f"{settings.spring_boot_url}/api/attendance/summary/student/{user.userId}"

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

    async with httpx.AsyncClient() as client:
        response = await client.get(
            url,
            params=params,
            cookies={"accessToken": access_token},
            timeout=10.0,
        )

    if response.status_code == 403:
        return {"error": "Access denied. You can only view your own attendance."}
    if response.status_code != 200:
        return {"error": f"Spring Boot returned {response.status_code}: {response.text}"}

    return response.json()
