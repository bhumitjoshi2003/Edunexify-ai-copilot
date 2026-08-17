"""
academic_calendar.py — single source of truth for converting between calendar
dates/months and a school's academic-year numbering.

A school's academic year can start in any configured month (April, January,
July, December, ...) — see Spring Boot's School.academicYearStartMonth. This
module mirrors the same conventions already used server-side (see
FeeCalculationService.academicMonthStart/parseSession and
StudentFeesService.getAcademicYear in the Java backend):
  - academic month 1 = whatever calendar month the school configured as its
    start month; academic month 12 = the calendar month right before it.
  - a session label like "2025-2026" is just two opaque years — which one is
    "current" depends entirely on the school's start month, never on a fixed
    April/January cutover.

Previously this logic was duplicated (both hardcoding "today.month >= 4") in
tools/fees.py and tools/attendance.py, and re-exported from the latter into
7+ other tool modules — this file replaces both copies with one
implementation that fetches the school's real academicYearStartMonth instead
of assuming April.
"""
import logging
from datetime import date

import httpx

from config import settings

logger = logging.getLogger(__name__)

_MONTH_NAMES = [
    "", "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

_DEFAULT_START_MONTH = 4  # only used if /api/school/settings is unreachable


def academic_month_to_calendar_name(academic_month: int, start_month: int) -> str:
    """
    Convert an academic month number (1-based, relative to the school's year start)
    to a calendar month name.

    Example: start_month=4 (April), academic_month=1 -> April
             start_month=4,          academic_month=10 -> January
             start_month=1 (January), academic_month=1 -> January
    """
    if not (1 <= academic_month <= 12):
        return f"Month {academic_month}"
    calendar_month = (start_month - 1 + academic_month - 1) % 12 + 1
    return _MONTH_NAMES[calendar_month]


async def fetch_academic_year_start_month(access_token: str) -> int:
    """
    Fetches the school's academic year start month from /api/school/settings.
    Returns 4 (April) only as a last-resort fallback if the call fails —
    never as a silent assumption when the real value is available.
    """
    url = f"{settings.spring_boot_url}/api/school/settings"
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                url,
                cookies={"accessToken": access_token},
                timeout=5.0,
            )
        if response.status_code == 200:
            start_month = response.json().get("academicYearStartMonth", _DEFAULT_START_MONTH)
            return int(start_month)
    except Exception as e:
        logger.warning("Could not fetch school settings: %s", e)
    return _DEFAULT_START_MONTH


def session_label_for_start_month(start_month: int, today: date | None = None) -> str:
    """
    Derives the current academic session label (e.g. "2025-2026") for a school
    with the given start month, as of `today` (defaults to date.today()).
    Mirrors StudentFeesService.getAcademicYear / FeeReminderService.getAcademicYear
    in the Java backend exactly: whichever academic year `today` currently falls
    within, labeled as startYear-(startYear+1).
    """
    today = today or date.today()
    if today.month >= start_month:
        return f"{today.year}-{today.year + 1}"
    return f"{today.year - 1}-{today.year}"


async def current_academic_session(access_token: str) -> str:
    """
    The session label ("2025-2026") for whatever academic year is currently
    running, for the calling user's school — fetches the school's real
    academicYearStartMonth first, rather than assuming April.
    """
    start_month = await fetch_academic_year_start_month(access_token)
    return session_label_for_start_month(start_month)
