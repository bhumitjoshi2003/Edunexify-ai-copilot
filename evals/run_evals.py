"""
evals/run_evals.py — golden-dataset eval runner for the AI Copilot.

Calls the *exact same* in-process turn-processing function routers/chat.py
already uses for POST /chat (not a reimplementation), with tracing enabled —
assertions run against the same Trace object built for production
observability, so this needs no separate instrumentation.

Requires live infra (deliberate, not a shortcut): a running Spring Boot +
Postgres with the test users/school seeded, and real DEEPSEEK_API_KEY /
OPENAI_API_KEY in .env — mocking the LLM or the tool HTTP calls would defeat
the point of testing real tool-selection and RAG-grounding behavior. Costs
real API money per run; intended to be run manually before a prompt/model/
tool/RAG change ships, not on every commit.

Usage:
    python evals/run_evals.py [--case ID] [--category CATEGORY]
"""
import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import yaml
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

load_dotenv()  # the app's own .env — DEEPSEEK_API_KEY, OPENAI_API_KEY, INTERNAL_SECRET, SPRING_BOOT_URL
load_dotenv(Path(__file__).resolve().parent.parent / ".env.eval")  # test-user credentials only

import memory  # noqa: E402
import tracing as tracing_module  # noqa: E402
from config import settings  # noqa: E402
from routers.chat import chat  # noqa: E402
from schemas.chat import ChatRequest, UserContext  # noqa: E402

DATASET_PATH = Path(__file__).resolve().parent / "golden_dataset.yaml"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

_login_cache: dict[str, dict] = {}


def _resolve_test_user(key: str) -> dict:
    prefix = f"EVAL_USER_{key.upper()}"
    user_id = os.environ.get(f"{prefix}_ID")
    password = os.environ.get(f"{prefix}_PASSWORD")
    school_id = os.environ.get(f"{prefix}_SCHOOL_ID")
    if not user_id or not password or not school_id:
        raise RuntimeError(
            f"No test user configured for '{key}' — set {prefix}_ID, {prefix}_PASSWORD, "
            f"{prefix}_SCHOOL_ID in .env.eval (see .env.eval.example)."
        )
    return {"userId": user_id, "password": password, "schoolId": int(school_id)}


async def _login(user_id: str, password: str) -> str:
    """Real login against Spring Boot — returns the accessToken cookie value."""
    if user_id in _login_cache:
        return _login_cache[user_id]["accessToken"]

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{settings.spring_boot_url}/api/auth/login",
            json={"userId": user_id, "password": password},
            timeout=10.0,
        )
    if response.status_code != 200:
        raise RuntimeError(f"Login failed for {user_id}: {response.status_code} {response.text[:200]}")

    access_token = response.cookies.get("accessToken")
    if not access_token:
        raise RuntimeError(f"Login for {user_id} returned no accessToken cookie.")

    _login_cache[user_id] = {"accessToken": access_token}
    return access_token


async def _run_case(case: dict) -> dict:
    case_id = case["id"]
    result = {"id": case_id, "category": case.get("category"), "passed": False, "failures": []}

    try:
        test_user = _resolve_test_user(case["test_user"])
    except RuntimeError as e:
        result["failures"].append(str(e))
        result["skipped"] = True
        return result

    try:
        access_token = await _login(test_user["userId"], test_user["password"])
    except RuntimeError as e:
        result["failures"].append(str(e))
        return result

    user = UserContext(
        userId=test_user["userId"],
        role=case["role"],
        schoolId=test_user["schoolId"],
    )
    request = ChatRequest(
        message=case["message"],
        conversationId=f"eval-{case_id}-{int(time.time())}",
        user=user,
        accessToken=access_token,
    )

    # Capture the Trace built inside chat() without changing that function's
    # signature — Trace.finish() still runs normally (so eval runs get logged/
    # shipped like any real turn), we just grab a reference to `self` first.
    captured: dict = {}
    original_finish = tracing_module.Trace.finish

    async def _capturing_finish(self):
        captured["trace"] = self
        await original_finish(self)

    tracing_module.Trace.finish = _capturing_finish
    try:
        response = await chat(request, settings.internal_secret)
    except Exception as e:
        result["failures"].append(f"chat() raised: {e}")
        return result
    finally:
        tracing_module.Trace.finish = original_finish

    trace = captured.get("trace")
    reply = response.reply
    result["reply_preview"] = reply[:300]

    _assert_case(case.get("expect", {}), reply, trace, result["failures"])
    result["passed"] = not result["failures"]
    return result


def _normalize(text: str) -> str:
    """Strips **bold** markers before substring matching. '**bold**' is the only
    markdown emphasis the model ever produces (see chat-markdown.pipe.ts on the
    frontend) — without this, a correct "the school **may** request..." fails a
    literal "may request" check purely because the bold wrapper splits the two
    words, which is a formatting artifact, not a wording difference."""
    return text.replace("**", "")


# A forbidden phrase inside one of these sentences is very likely part of a
# correct denial/hedge ("I can't confirm they're on approved leave", "...tell me
# whether it was approved leave or not") rather than the incorrect unhedged claim
# the check is actually trying to catch ("They are on approved leave."). Simple
# same-sentence marker check — not full negation parsing, but enough to stop
# naive substring matching from failing a reply for correctly declining to answer.
_HEDGE_MARKERS = (
    "n't", "not ", "no ", "cannot", "unable", "whether", "don't have",
    "doesn't have", "no leave", "not available", "unclear", "can't tell",
)


def _forbidden_present(text_lower: str, phrase_lower: str) -> bool:
    start = 0
    while True:
        idx = text_lower.find(phrase_lower, start)
        if idx == -1:
            return False
        sentence_start = max(text_lower.rfind(".", 0, idx), text_lower.rfind("\n", 0, idx)) + 1
        sentence_end = idx + len(phrase_lower)
        sentence = text_lower[sentence_start:sentence_end]
        if not any(marker in sentence for marker in _HEDGE_MARKERS):
            return True  # an unhedged occurrence — genuine violation
        start = idx + 1  # hedged here; keep looking for a later, unhedged occurrence


def _assert_case(expect: dict, reply: str, trace, failures: list[str]) -> None:
    reply_lower = _normalize(reply).lower()

    # Scans what the system actually produced (reply + tool call summaries + RAG
    # evidence) — deliberately NOT trace.user_message. A forbidden-phrase check
    # must never scan the test question itself: a question like "Is S101 on
    # approved leave?" legitimately contains the exact phrase we're checking the
    # *answer* never asserts, and scanning the echoed question back would fail
    # the case regardless of how the model actually responded.
    if trace:
        scan_blob = _normalize(json.dumps({
            "finalReply": trace.final_reply,
            "toolsCalled": trace.tool_calls,
            "ragResults": trace.rag_results,
        }, default=str)).lower()
    else:
        scan_blob = reply_lower

    tools_called = {t["name"] for t in (trace.tool_calls if trace else [])}
    for tool in expect.get("tools_called", []):
        if tool not in tools_called:
            failures.append(f"expected tool '{tool}' to be called; actual: {sorted(tools_called)}")

    for phrase in expect.get("forbidden_substrings", []):
        if _forbidden_present(scan_blob, phrase.lower()):
            failures.append(f"forbidden substring found (unhedged): {phrase!r}")

    required_any = expect.get("required_substrings_any", [])
    if required_any and not any(p.lower() in reply_lower for p in required_any):
        failures.append(f"none of the required substrings found: {required_any}")

    if "expect_citation" in expect:
        has_citation = "📄" in reply
        if expect["expect_citation"] and not has_citation:
            failures.append("expected a citation (📄) but reply has none")
        if not expect["expect_citation"] and has_citation:
            failures.append("expected no citation, but reply has one")

    if expect.get("expect_no_rag_results"):
        rag_results = trace.rag_results if trace else []
        if rag_results:
            failures.append(f"expected no RAG results, got {len(rag_results)}")

    if "min_rag_similarity" in expect:
        sims = [r.get("similarity", 0) for r in (trace.rag_results if trace else [])]
        if not sims or max(sims) < expect["min_rag_similarity"]:
            failures.append(f"expected max RAG similarity >= {expect['min_rag_similarity']}, got {sims}")


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", help="Run only this case id")
    parser.add_argument("--category", help="Run only cases in this category")
    args = parser.parse_args()

    try:
        cases = yaml.safe_load(DATASET_PATH.read_text())
        if args.case:
            cases = [c for c in cases if c["id"] == args.case]
        if args.category:
            cases = [c for c in cases if c.get("category") == args.category]

        results = []
        for case in cases:
            r = await _run_case(case)
            results.append(r)
            if r.get("skipped"):
                status = "SKIP "
            elif r["passed"]:
                status = "PASS "
            else:
                status = "FAIL "
            print(f"{status} {r['id']} ({r.get('category')})")
            for f in r["failures"]:
                print(f"       - {f}")

        passed = sum(1 for r in results if r["passed"])
        skipped = sum(1 for r in results if r.get("skipped"))
        failed = len(results) - passed - skipped
        print(f"\n{passed} passed, {failed} failed, {skipped} skipped, {len(results)} total")

        RESULTS_DIR.mkdir(exist_ok=True)
        out_path = RESULTS_DIR / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
        out_path.write_text(json.dumps(results, indent=2, default=str))
        print(f"Results written to {out_path}")

        return 1 if failed > 0 else 0
    finally:
        # chat() (via memory.get_history/append_turn) lazily opens a module-level
        # redis.asyncio connection that's never torn down on its own. Left alone,
        # it gets garbage-collected *after* asyncio.run() has already closed this
        # loop, and redis-py's __del__ then throws "Event loop is closed" trying to
        # clean itself up. Closing it here, while the loop is still alive, avoids that.
        await memory.close()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
