"""Offline tests for the pieces we added on top of the template.

Run:  uv run python -m unittest test_ojbot -v
"""

from __future__ import annotations

import json
import os
import unittest
from datetime import datetime, timedelta, timezone

import market_anchor as ma

NOW = datetime(2026, 10, 31, 12, 0, tzinfo=timezone.utc)


def _market(question: str, yes: float, end: datetime, closed=False, vol=50_000, slug="ev"):
    return {
        "question": question,
        "outcomes": json.dumps(["Yes", "No"]),
        "outcomePrices": json.dumps([str(yes), str(1 - yes)]),
        "endDate": end.isoformat(),
        "closed": closed,
        "active": True,
        "volumeNum": vol,
        "slug": slug,
    }


class SimilarityTests(unittest.TestCase):
    def test_same_question_scores_high(self):
        a = "Will the next Google Gemini Pro model be released by October 31, 2026?"
        b = "Will the next Google Gemini Pro model be released by October 31?"
        self.assertGreater(ma.similarity(a, b), 0.7)

    def test_different_question_scores_low(self):
        a = "Will the US government shut down before November 16, 2026?"
        b = "Will Bitcoin close above $100,000 on November 16, 2026?"
        self.assertLess(ma.similarity(a, b), 0.3)


class PickAnchorTests(unittest.TestCase):
    q = "Will the next Google Gemini Pro model be released by October 31, 2026?"
    close = NOW + timedelta(days=1)

    def test_picks_matching_open_market(self):
        events = [{"title": "Gemini", "slug": "gemini", "markets": [
            _market("Will the next Google Gemini Pro model be released by October 31?", 0.42, NOW + timedelta(days=1)),
        ]}]
        a = ma.pick_anchor(self.q, self.close, events, now=NOW)
        self.assertIsNotNone(a)
        self.assertAlmostEqual(a.price, 0.42)
        self.assertIn("polymarket.com/event/gemini", a.url)

    def test_skips_closed_and_dissimilar_and_far_dates(self):
        events = [{"title": "x", "slug": "x", "markets": [
            _market("Will the next Google Gemini Pro model be released by October 31?", 0.42, NOW + timedelta(days=1), closed=True),
            _market("Will Bitcoin close above $100,000 on October 31?", 0.42, NOW + timedelta(days=1)),
            _market("Will the next Google Gemini Pro model be released by December 31?", 0.42, NOW + timedelta(days=61)),
        ]}]
        self.assertIsNone(ma.pick_anchor(self.q, self.close, events, now=NOW))

    def test_skips_illiquid_and_degenerate_prices(self):
        events = [{"title": "x", "slug": "x", "markets": [
            _market("Will the next Google Gemini Pro model be released by October 31?", 0.42, NOW + timedelta(days=1), vol=50),
            _market("Will the next Google Gemini Pro model be released by October 31?", 1.0, NOW + timedelta(days=1)),
        ]}]
        self.assertIsNone(ma.pick_anchor(self.q, self.close, events, now=NOW))

    def test_prefers_most_similar(self):
        events = [{"title": "x", "slug": "x", "markets": [
            _market("Will the next Gemini Pro model be released by October 31?", 0.30, NOW + timedelta(days=1)),
            _market("Will the next Google Gemini Pro model be released by October 31, 2026?", 0.55, NOW + timedelta(days=1)),
        ]}]
        a = ma.pick_anchor(self.q, self.close, events, now=NOW)
        self.assertAlmostEqual(a.price, 0.55)

    def test_no_close_time_still_matches(self):
        events = [{"title": "x", "slug": "x", "markets": [
            _market("Will the next Google Gemini Pro model be released by October 31?", 0.42, NOW + timedelta(days=1)),
        ]}]
        self.assertIsNotNone(ma.pick_anchor(self.q, None, events, now=NOW))


class NumberGuardTests(unittest.TestCase):
    def test_threshold_mismatch_rejected(self):
        self.assertFalse(ma.numbers_agree("Will Bitcoin reach $150,000 before January 1, 2027?",
                                          "Will Bitcoin reach $130,000 in January?"))

    def test_year_is_optional_but_day_is_not(self):
        self.assertTrue(ma.numbers_agree("Will X happen by October 31, 2026?", "Will X happen by October 31?"))
        self.assertFalse(ma.numbers_agree("Will X happen by October 31, 2026?", "Will X happen by October 30?"))

    def test_pick_anchor_uses_guard(self):
        q = "Will Bitcoin reach $150,000 before January 1, 2027?"
        events = [{"title": "x", "slug": "x", "markets": [
            _market("Will Bitcoin reach $130,000 before January 1?", 0.5, NOW + timedelta(days=1)),
        ]}]
        self.assertIsNone(ma.pick_anchor(q, NOW + timedelta(days=1), events, now=NOW))


class BlendTests(unittest.TestCase):
    def test_blend_sits_between_and_leans_market(self):
        p = ma.blend(0.30, 0.60)
        self.assertGreater(p, 0.45)
        self.assertLess(p, 0.60)

    def test_blend_clips_extremes(self):
        self.assertEqual(ma.blend(0.001, 0.001), 0.03)
        self.assertEqual(ma.blend(0.999, 0.999), 0.97)

    def test_equal_inputs_unchanged(self):
        self.assertAlmostEqual(ma.blend(0.4, 0.4), 0.4, places=6)


class BotWiringTests(unittest.TestCase):
    def setUp(self):
        os.environ.setdefault("METACULUS_TOKEN", "test-token")
        os.environ.setdefault("OPENROUTER_API_KEY", "test-key")

    def test_pool_rotates_across_families(self):
        import ojbot

        bot = ojbot.OjBot(
            research_reports_per_question=1,
            predictions_per_research_report=4,
            publish_reports_to_metaculus=False,
            pool=["openrouter/a/one", "openrouter/b/two", "openrouter/c/three"],
            llms={"default": "openrouter/a/one", "summarizer": ojbot.CHEAP_MODEL,
                  "researcher": "no_research", "parser": ojbot.CHEAP_MODEL},
        )
        seen = [bot.get_llm("default", "llm").model for _ in range(6)]
        self.assertEqual(seen, ["openrouter/a/one", "openrouter/b/two", "openrouter/c/three"] * 2)
        # other purposes are untouched
        self.assertEqual(bot.get_llm("parser", "string_name"), ojbot.CHEAP_MODEL)

    def test_model_pool_env_override(self):
        import ojbot

        os.environ["MODEL_POOL"] = " openrouter/x/y , openrouter/z/w "
        try:
            self.assertEqual(ojbot.model_pool(), ["openrouter/x/y", "openrouter/z/w"])
        finally:
            del os.environ["MODEL_POOL"]
        self.assertEqual(ojbot.model_pool(), ojbot.DEFAULT_POOL)

    def test_researcher_falls_back_to_search_model(self):
        import ojbot

        for k in ("ASKNEWS_CLIENT_ID", "ASKNEWS_SECRET"):
            os.environ.pop(k, None)
        self.assertEqual(ojbot.researcher_name(), ojbot.SEARCH_MODEL)
        os.environ["ASKNEWS_CLIENT_ID"] = "a"
        os.environ["ASKNEWS_SECRET"] = "b"
        try:
            self.assertEqual(ojbot.researcher_name(), "asknews/news-summaries")
        finally:
            del os.environ["ASKNEWS_CLIENT_ID"], os.environ["ASKNEWS_SECRET"]


if __name__ == "__main__":
    unittest.main()
