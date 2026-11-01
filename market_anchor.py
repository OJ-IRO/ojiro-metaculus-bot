"""Prediction-market anchor for Metaculus questions.

Finds a Polymarket market that asks the same thing as a Metaculus binary
question and returns its current YES price. Pure functions are separated from
network access so they can be unit-tested with canned API responses.

Matching is deliberately strict: a wrong market is far worse than no market.
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import requests

logger = logging.getLogger(__name__)

GAMMA_SEARCH = "https://gamma-api.polymarket.com/public-search"
MIN_SIMILARITY = 0.6  # Jaccard on content words
MAX_CLOSE_GAP_DAYS = 4.0
MARKET_WEIGHT = 0.6  # weight of the market price in the final logit blend
PRICE_FLOOR, PRICE_CEIL = 0.03, 0.97

_STOP = {
    "will", "the", "a", "an", "of", "in", "on", "by", "to", "be", "is", "at", "for",
    "or", "and", "before", "after", "than", "as", "this", "that", "its", "it", "their",
    "with", "from", "any", "does", "do", "has", "have", "between", "there", "more",
    "least", "end", "during",
}


@dataclass(frozen=True)
class Anchor:
    price: float            # YES probability implied by the market
    market_question: str
    url: str
    similarity: float


def content_words(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9][a-z0-9.%$-]*", text.lower())
    return {w.strip(".-") for w in words if w not in _STOP and len(w) > 1}


def similarity(a: str, b: str) -> float:
    wa, wb = content_words(a), content_words(b)
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def _parse_dt(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def _yes_price(market: dict) -> float | None:
    outcomes = market.get("outcomes")
    prices = market.get("outcomePrices")
    try:
        outcomes = json.loads(outcomes) if isinstance(outcomes, str) else outcomes
        prices = json.loads(prices) if isinstance(prices, str) else prices
        if not outcomes or not prices:
            return None
        for o, p in zip(outcomes, prices):
            if str(o).strip().lower() == "yes":
                return float(p)
        return float(prices[0])
    except (ValueError, TypeError):
        return None


def numbers(text: str) -> set[str]:
    """Numeric tokens that must agree between the two questions (thresholds, days)."""
    out = set()
    for tok in re.findall(r"\d[\d,]*\.?\d*", text):
        tok = tok.replace(",", "").rstrip(".")
        if tok and not (len(tok) == 4 and tok.startswith("20")):  # years are optional
            out.add(tok)
    return out


def numbers_agree(question_text: str, market_question: str) -> bool:
    return numbers(question_text) <= numbers(market_question)


def pick_anchor(
    question_text: str,
    question_close: datetime | None,
    events: list[dict],
    now: datetime | None = None,
) -> Anchor | None:
    """Choose the best matching open market from a gamma public-search payload."""
    now = now or datetime.now(timezone.utc)
    best: Anchor | None = None
    for ev in events or []:
        for m in ev.get("markets") or []:
            if m.get("closed") or not m.get("active", True):
                continue
            mq = m.get("question") or ev.get("title") or ""
            sim = similarity(question_text, mq)
            if sim < MIN_SIMILARITY or not numbers_agree(question_text, mq):
                continue
            end = _parse_dt(m.get("endDate") or ev.get("endDate"))
            if end and end < now:
                continue
            if question_close and end:
                gap = abs((end - question_close).total_seconds()) / 86400
                if gap > MAX_CLOSE_GAP_DAYS:
                    continue
            price = _yes_price(m)
            if price is None or not (0.0 < price < 1.0):
                continue
            # Illiquid or dead markets sit at round numbers with no volume.
            vol = float(m.get("volumeNum") or m.get("volume") or 0)
            if vol < 1000:
                continue
            slug = ev.get("slug") or m.get("slug") or ""
            cand = Anchor(price=price, market_question=mq,
                          url=f"https://polymarket.com/event/{slug}", similarity=sim)
            if best is None or cand.similarity > best.similarity:
                best = cand
    return best


def fetch_events(query: str, timeout: float = 15.0) -> list[dict]:
    r = requests.get(GAMMA_SEARCH, params={"q": query[:200], "limit_per_type": 6},
                     timeout=timeout, headers={"User-Agent": "metaculus-bot/1.0"})
    r.raise_for_status()
    return r.json().get("events") or []


def find_anchor(question_text: str, question_close: datetime | None) -> Anchor | None:
    """Network wrapper; never raises."""
    try:
        anchor = pick_anchor(question_text, question_close, fetch_events(question_text))
        if anchor is None:
            keywords = " ".join(sorted(content_words(question_text), key=len, reverse=True)[:8])
            if keywords:
                anchor = pick_anchor(question_text, question_close, fetch_events(keywords))
        return anchor
    except Exception as exc:  # noqa: BLE001 - anchor is optional
        logger.info(f"No market anchor ({exc!r})")
        return None


def _logit(p: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def blend(p_model: float, p_market: float, w_market: float = MARKET_WEIGHT) -> float:
    """Logit-space blend, clipped to a safe band for log scoring."""
    z = w_market * _logit(p_market) + (1 - w_market) * _logit(p_model)
    p = 1 / (1 + math.exp(-z))
    return min(max(p, PRICE_FLOOR), PRICE_CEIL)


def anchor_note(anchor: Anchor) -> str:
    return (
        f"Prediction market check: Polymarket currently prices YES at {anchor.price:.0%} "
        f"on the closely matching market \"{anchor.market_question}\" ({anchor.url}). "
        "Liquid market prices are usually better calibrated than individual reasoning; "
        "deviate only with a concrete reason."
    )
