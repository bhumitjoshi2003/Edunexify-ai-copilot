"""Offline contract tests: no credentials, network, Redis, or model calls."""
import importlib.util
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from fastapi import HTTPException
from pydantic import ValidationError

class WisdomTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.create = AsyncMock()
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=self.create)))
        spec = importlib.util.spec_from_file_location('wisdom_test_router', Path(__file__).parents[1] / 'routers/wisdom.py')
        self.module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'config': SimpleNamespace(settings=SimpleNamespace(internal_secret='test-secret')), 'routers.chat': SimpleNamespace(_async_client=client)}):
            spec.loader.exec_module(self.module)
        self.request = self.module.DraftRequest(theme='Focus', translation='TEST VERIFIED TRANSLATION PLACEHOLDER')
        self.candidates = [
            self.module.VerseCandidate(id=1, chapter=2, verse=47, themes='duty, action'),
            self.module.VerseCandidate(id=2, chapter=16, verse=1, themes='fearlessness, purity'),
        ]
        self.suggest_request = self.module.SuggestRequest(theme='anger and self-control', candidates=self.candidates)

    def response(self, data):
        self.create.return_value = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(data)))])

    async def test_secret_required_before_model_call(self):
        with self.assertRaises(HTTPException) as error:
            await self.module.draft(self.request, 'wrong')
        self.assertEqual(error.exception.status_code, 403)
        self.create.assert_not_awaited()

    async def test_editorial_only_response_no_tools_or_publication(self):
        value = {'simpleMeaning':'Meaning','understanding':'Explanation with example','lesson':'Takeaway'}
        self.response(value)
        result = await self.module.draft(self.request, 'test-secret')
        self.assertEqual(result.model_dump(), value)
        self.assertNotIn('tools', self.create.call_args.kwargs)
        self.assertIn('TEST VERIFIED TRANSLATION PLACEHOLDER', self.create.call_args.kwargs['messages'][1]['content'])

    async def test_hallucinated_scripture_fields_rejected(self):
        self.response({'simpleMeaning':'Meaning','understanding':'Example','lesson':'Lesson','sanskrit':'invented'})
        with self.assertRaises(HTTPException) as error:
            await self.module.draft(self.request, 'test-secret')
        self.assertEqual(error.exception.status_code, 502)

    async def test_publication_fields_rejected(self):
        self.response({'simpleMeaning':'Meaning','understanding':'Example','lesson':'Lesson','publishAt':'now'})
        with self.assertRaises(HTTPException):
            await self.module.draft(self.request, 'test-secret')

    def test_client_cannot_supply_scripture_or_publish_instruction_fields(self):
        for field in ('sanskrit','verse','schoolId','publishAt'):
            with self.assertRaises(ValidationError):
                self.module.DraftRequest(theme='test',translation='test',**{field:'untrusted'})

    async def test_missing_sections_rejected(self):
        self.response({'simpleMeaning':'Meaning'})
        with self.assertRaises(HTTPException):
            await self.module.draft(self.request, 'test-secret')

    async def test_suggest_verses_secret_required_before_model_call(self):
        with self.assertRaises(HTTPException) as error:
            await self.module.suggest_verses(self.suggest_request, 'wrong')
        self.assertEqual(error.exception.status_code, 403)
        self.create.assert_not_awaited()

    async def test_suggest_verses_empty_candidates_short_circuits_without_calling_model(self):
        empty_request = self.module.SuggestRequest(theme='anger', candidates=[])
        result = await self.module.suggest_verses(empty_request, 'test-secret')
        self.assertEqual(result, [])
        self.create.assert_not_awaited()

    async def test_suggest_verses_returns_ranked_candidates(self):
        self.response({'suggestions': [
            {'verseId': 2, 'rationale': 'Speaks directly to fearlessness under pressure.'},
            {'verseId': 1, 'rationale': 'Frames duty without attachment to outcome.'},
        ]})
        result = await self.module.suggest_verses(self.suggest_request, 'test-secret')
        self.assertEqual([s.verseId for s in result], [2, 1])
        self.assertIn('anger and self-control', self.create.call_args.kwargs['messages'][1]['content'])

    async def test_suggest_verses_filters_out_a_hallucinated_id_not_in_candidates(self):
        self.response({'suggestions': [
            {'verseId': 1, 'rationale': 'Valid candidate.'},
            {'verseId': 999, 'rationale': 'An id the model invented, not in our verified table.'},
        ]})
        result = await self.module.suggest_verses(self.suggest_request, 'test-secret')
        self.assertEqual([s.verseId for s in result], [1])

    async def test_suggest_verses_invalid_model_output_rejected(self):
        self.response({'not_suggestions': 'garbage'})
        result = await self.module.suggest_verses(self.suggest_request, 'test-secret')
        self.assertEqual(result, [])

    def test_suggest_request_accepts_up_to_1000_candidates_covering_the_full_gita_corpus(self):
        many = [self.module.VerseCandidate(id=i, chapter=(i % 18) + 1, verse=i, themes='') for i in range(1, 701)]
        request = self.module.SuggestRequest(theme='test', candidates=many)
        self.assertEqual(len(request.candidates), 700)

    def test_suggest_request_rejects_more_than_1000_candidates(self):
        too_many = [self.module.VerseCandidate(id=i, chapter=1, verse=i, themes='') for i in range(1, 1002)]
        with self.assertRaises(ValidationError):
            self.module.SuggestRequest(theme='test', candidates=too_many)

    async def test_suggest_verses_uses_excerpt_for_semantic_matching_not_only_theme_tags(self):
        candidate_with_excerpt = self.module.VerseCandidate(
            id=3, chapter=18, verse=66, themes='',
            excerpt='Abandon all varieties of duty and surrender unto Me alone.')
        request = self.module.SuggestRequest(theme='surrender', candidates=[candidate_with_excerpt])
        self.response({'suggestions': [{'verseId': 3, 'rationale': 'Matches the theme of surrender.'}]})
        result = await self.module.suggest_verses(request, 'test-secret')
        self.assertEqual([s.verseId for s in result], [3])
        sent_content = self.create.call_args.kwargs['messages'][1]['content']
        self.assertIn('surrender unto Me alone', sent_content)

if __name__ == '__main__':
    unittest.main()
