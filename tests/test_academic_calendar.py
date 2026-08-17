"""
Unit tests for academic_calendar.py — no pytest dependency (none exists in this
service yet), plain stdlib unittest so `python3 -m unittest tests.test_academic_calendar`
runs with no extra install.

Covers January-, April-, July-, and December-start schools, per the fee-module
academic-calendar audit: academic month 1 must always resolve to whatever
calendar month the school actually configured as its start, never a hardcoded
April assumption.
"""
import unittest
from datetime import date

from academic_calendar import (
    academic_month_to_calendar_name,
    session_label_for_start_month,
)


class AcademicMonthToCalendarNameTests(unittest.TestCase):
    def test_april_start_school_month_1_is_april(self):
        self.assertEqual(academic_month_to_calendar_name(1, start_month=4), "April")
        self.assertEqual(academic_month_to_calendar_name(9, start_month=4), "December")
        self.assertEqual(academic_month_to_calendar_name(10, start_month=4), "January")
        self.assertEqual(academic_month_to_calendar_name(12, start_month=4), "March")

    def test_january_start_school_month_1_is_january(self):
        self.assertEqual(academic_month_to_calendar_name(1, start_month=1), "January")
        self.assertEqual(academic_month_to_calendar_name(12, start_month=1), "December")

    def test_july_start_school_month_1_is_july(self):
        self.assertEqual(academic_month_to_calendar_name(1, start_month=7), "July")
        self.assertEqual(academic_month_to_calendar_name(6, start_month=7), "December")
        self.assertEqual(academic_month_to_calendar_name(7, start_month=7), "January")
        self.assertEqual(academic_month_to_calendar_name(12, start_month=7), "June")

    def test_december_start_school_month_1_is_december(self):
        self.assertEqual(academic_month_to_calendar_name(1, start_month=12), "December")
        self.assertEqual(academic_month_to_calendar_name(2, start_month=12), "January")
        self.assertEqual(academic_month_to_calendar_name(12, start_month=12), "November")

    def test_out_of_range_month_does_not_crash(self):
        self.assertEqual(academic_month_to_calendar_name(0, start_month=4), "Month 0")
        self.assertEqual(academic_month_to_calendar_name(13, start_month=4), "Month 13")


class SessionLabelForStartMonthTests(unittest.TestCase):
    def test_april_start_matches_original_hardcoded_behavior(self):
        # This is the exact cutover the old `today.month >= 4` check encoded —
        # asserting it here proves April-start schools see no behavior change.
        self.assertEqual(session_label_for_start_month(4, today=date(2026, 3, 31)), "2025-2026")
        self.assertEqual(session_label_for_start_month(4, today=date(2026, 4, 1)), "2026-2027")
        self.assertEqual(session_label_for_start_month(4, today=date(2026, 8, 15)), "2026-2027")

    def test_january_start_school(self):
        self.assertEqual(session_label_for_start_month(1, today=date(2026, 1, 1)), "2026-2027")
        self.assertEqual(session_label_for_start_month(1, today=date(2026, 12, 31)), "2026-2027")

    def test_july_start_school(self):
        self.assertEqual(session_label_for_start_month(7, today=date(2026, 6, 30)), "2025-2026")
        self.assertEqual(session_label_for_start_month(7, today=date(2026, 7, 1)), "2026-2027")

    def test_december_start_school(self):
        self.assertEqual(session_label_for_start_month(12, today=date(2026, 11, 30)), "2025-2026")
        self.assertEqual(session_label_for_start_month(12, today=date(2026, 12, 1)), "2026-2027")


if __name__ == "__main__":
    unittest.main()
