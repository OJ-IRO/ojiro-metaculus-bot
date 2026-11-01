"""Our FutureEval bot: the Metaculus template plus three changes.

1. Cross-family ensemble: each forecast sample rotates through a pool of
   frontier models from different providers (via OpenRouter); the framework
   takes the median, so model-specific errors partly cancel.
2. Research: AskNews when keys exist (free for tournament bots), otherwise a
   Perplexity search model through OpenRouter.
3. Market anchor: when a Polymarket market asks the same binary question, its
   price is added to the research and blended into the final probability.

Run:  uv run python ojbot.py --mode test_questions | tournament | metaculus_cup
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import logging
import os
from datetime import datetime, timezone
from typing import Literal

from bot_helpers import (
    check_environment,
    print_run_summary_banner,
    print_startup_banner,
    silence_noisy_dependencies,
)

silence_noisy_dependencies()

from forecasting_tools import (  # noqa: E402
    BinaryQuestion,
    GeneralLlm,
    MetaculusClient,
    MetaculusQuestion,
)

import market_anchor  # noqa: E402
from main import FallTemplateBot2026  # noqa: E402

logger = logging.getLogger(__name__)

# Balanced for cost (~$0.10-0.15 per question) and family diversity.
DEFAULT_POOL = [
    "openrouter/openai/gpt-5.6-terra",
    "openrouter/anthropic/claude-sonnet-5.5",
    "openrouter/google/gemini-3.8-flash",
    "openrouter/x-ai/grok-4.7",
]
CHEAP_MODEL = "openrouter/openai/gpt-5.4-nano"
SEARCH_MODEL = "openrouter/perplexity/sonar"
BINARY_FLOOR, BINARY_CEIL = 0.03, 0.97


def model_pool() -> list[str]:
    raw = os.getenv("MODEL_POOL", "").strip()
    return [m.strip() for m in raw.split(",") if m.strip()] or list(DEFAULT_POOL)


def researcher_name() -> str:
    if os.getenv("ASKNEWS_CLIENT_ID") and os.getenv("ASKNEWS_SECRET"):
        return "asknews/news-summaries"
    return SEARCH_MODEL


class OjBot(FallTemplateBot2026):
    _max_concurrent_questions = 2
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    def __init__(self, *args, pool: list[str] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._pool = [
            GeneralLlm(model=m, temperature=0.3, timeout=120, allowed_tries=2)
            for m in (pool or model_pool())
        ]
        self._cycle = itertools.cycle(range(len(self._pool)))
        self._anchors: dict[str, market_anchor.Anchor] = {}

    # ---- ensemble: rotate the "default" model across families -------------
    def next_model(self) -> GeneralLlm:
        return self._pool[next(self._cycle)]

    def get_llm(self, purpose: str = "default", guarantee_type=None):  # type: ignore[override]
        if purpose == "default" and guarantee_type == "llm" and self._pool:
            return self.next_model()
        return super().get_llm(purpose, guarantee_type)  # type: ignore[arg-type]

    # ---- research + market anchor ------------------------------------------
    async def run_research(self, question: MetaculusQuestion) -> str:
        research = await super().run_research(question)
        if isinstance(question, BinaryQuestion):
            close = getattr(question, "scheduled_resolution_time", None) or getattr(
                question, "close_time", None
            )
            if close and close.tzinfo is None:
                close = close.replace(tzinfo=timezone.utc)
            anchor = await asyncio.to_thread(
                market_anchor.find_anchor, question.question_text, close
            )
            if anchor:
                self._anchors[str(question.id_of_post)] = anchor
                research = f"{research}\n\n{market_anchor.anchor_note(anchor)}"
                logger.info(
                    f"Market anchor for {question.page_url}: {anchor.price:.2f} "
                    f"(sim {anchor.similarity:.2f}) {anchor.url}"
                )
        return research

    # ---- aggregation: blend with the market, keep away from the extremes --
    async def _aggregate_predictions(self, predictions, question):  # type: ignore[override]
        aggregate = await super()._aggregate_predictions(predictions, question)
        if isinstance(question, BinaryQuestion) and isinstance(aggregate, float):
            anchor = self._anchors.get(str(question.id_of_post))
            if anchor:
                blended = market_anchor.blend(aggregate, anchor.price)
                logger.info(
                    f"Blend for {question.page_url}: ensemble {aggregate:.3f} + "
                    f"market {anchor.price:.3f} -> {blended:.3f}"
                )
                aggregate = blended
            aggregate = min(max(aggregate, BINARY_FLOOR), BINARY_CEIL)
        return aggregate


def build_bot(publish: bool) -> OjBot:
    return OjBot(
        research_reports_per_question=1,
        predictions_per_research_report=4,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=publish,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=True,
        extra_metadata_in_explanation=True,
        llms={
            "default": GeneralLlm(model=DEFAULT_POOL[0], temperature=0.3, timeout=120),
            "summarizer": CHEAP_MODEL,
            "researcher": researcher_name(),
            "parser": CHEAP_MODEL,
        },
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    parser = argparse.ArgumentParser(description="Run our forecasting bot")
    parser.add_argument(
        "--mode",
        choices=["tournament", "metaculus_cup", "test_questions"],
        default="tournament",
    )
    parser.add_argument("--dry-run", action="store_true", help="do not publish")
    args = parser.parse_args()
    run_mode: Literal["tournament", "metaculus_cup", "test_questions"] = args.mode

    check_environment(strict=True)
    publish = not args.dry_run
    print_startup_banner(run_mode, will_publish=publish)
    logger.info(f"Model pool: {model_pool()} | researcher: {researcher_name()}")

    bot = build_bot(publish)
    client = MetaculusClient()
    if run_mode == "tournament":
        reports = asyncio.run(
            bot.forecast_on_tournament(client.CURRENT_AI_COMPETITION_ID, return_exceptions=True)
        ) + asyncio.run(
            bot.forecast_on_tournament(client.CURRENT_MINIBENCH_ID, return_exceptions=True)
        )
        url = "https://www.metaculus.com/tournament/fall-futureeval-2026/"
    elif run_mode == "metaculus_cup":
        bot.skip_previously_forecasted_questions = False
        reports = asyncio.run(
            bot.forecast_on_tournament(client.CURRENT_METACULUS_CUP_ID, return_exceptions=True)
        )
        url = "https://www.metaculus.com/tournament/metaculus-cup-fall-2026/"
    else:
        bot.skip_previously_forecasted_questions = False
        reports = asyncio.run(
            bot.forecast_on_tournament("bot-testing-area", return_exceptions=True)
        )
        url = "https://www.metaculus.com/tournament/bot-testing-area/"

    bot.log_report_summary(reports)
    print_run_summary_banner(reports, will_publish=publish, tournament_url=url)
