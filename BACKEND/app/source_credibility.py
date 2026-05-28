"""
Source Credibility Scoring — Aarushi Sharma
Ranks retrieved sources so the research agent can prioritise high-quality references.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from urllib.parse import urlparse
from typing import Optional
import httpx
from bs4 import BeautifulSoup


# ── Domain reputation tiers ───────────────────────────────────────────────────

TRUSTED_DOMAINS: dict[str, int] = {
    "nature.com": 20, "science.org": 20, "cell.com": 20,
    "pubmed.ncbi.nlm.nih.gov": 20, "arxiv.org": 15, "scholar.google.com": 15,
    "jstor.org": 15, "springer.com": 12, "wiley.com": 12, "elsevier.com": 12,
    "ieee.org": 15, "acm.org": 15,
    "who.int": 18, "cdc.gov": 18, "nih.gov": 18, "nasa.gov": 18,
    "un.org": 15, "europa.eu": 15, "gov.uk": 15, "nist.gov": 15,
    "bbc.com": 12, "reuters.com": 14, "apnews.com": 14, "nytimes.com": 12,
    "theguardian.com": 12, "ft.com": 13, "economist.com": 13,
    "wikipedia.org": 8,
    "stackoverflow.com": 6,
    "docs.python.org": 10, "developer.mozilla.org": 10,
}

UNTRUSTED_PATTERNS: list[str] = [
    r"medium\.com", r"substack\.com", r"wordpress\.com",
    r"blogspot\.com", r"wix\.com", r"weebly\.com",
    r"quora\.com", r"reddit\.com", r"yahoo\.answers",
    r"\bforum\b", r"\bblog\b",
]


@dataclass
class SourceScore:
    url: str
    domain: str
    total_score: float
    domain_score: float
    freshness_score: float
    content_quality_score: float
    https_bonus: float
    tier: str
    notes: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"[{self.tier}] {self.domain} → {self.total_score:.1f}/100  "
            f"({', '.join(self.notes)})"
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lstrip("www.")
    except Exception:
        return url


def _https_bonus(url: str) -> float:
    return 5.0 if url.startswith("https://") else 0.0


def _domain_score(domain: str) -> tuple[float, list[str]]:
    notes: list[str] = []
    for trusted, bonus in TRUSTED_DOMAINS.items():
        if domain == trusted or domain.endswith("." + trusted):
            notes.append(f"trusted domain +{bonus}")
            return float(bonus), notes
    for pattern in UNTRUSTED_PATTERNS:
        if re.search(pattern, domain, re.I):
            notes.append(f"low-quality pattern ({pattern})")
            return -10.0, notes
    if domain.endswith(".edu"):
        notes.append("edu TLD +8")
        return 8.0, notes
    if domain.endswith(".gov"):
        notes.append("gov TLD +10")
        return 10.0, notes
    if domain.endswith(".org"):
        notes.append("org TLD +4")
        return 4.0, notes
    notes.append("unknown domain, neutral")
    return 0.0, notes


def _fetch_metadata(url: str, timeout: int = 8) -> dict:
    try:
        resp = httpx.get(url, timeout=timeout, follow_redirects=True,
                         headers={"User-Agent": "ResearchAgentCredibility/1.0"})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        text = soup.get_text(separator=" ")
        word_count = len(text.split())
        has_author = bool(
            soup.find(attrs={"rel": "author"}) or
            soup.find(class_=re.compile(r"author", re.I)) or
            re.search(r"\bby\s+[A-Z][a-z]+ [A-Z][a-z]+", text)
        )
        year_match = re.search(r"\b(20[12]\d)\b", text)
        year = int(year_match.group(1)) if year_match else None
        has_date = bool(
            soup.find("time") or
            soup.find(class_=re.compile(r"date|publish", re.I)) or
            year_match
        )
        return {"word_count": word_count, "has_author": has_author,
                "has_date": has_date, "year": year}
    except Exception:
        return {"word_count": 0, "has_author": False, "has_date": False, "year": None}


def _freshness_score(year: Optional[int]) -> tuple[float, str]:
    if year is None:
        return 0.0, "no date found"
    import datetime
    current_year = datetime.date.today().year
    age = current_year - year
    if age <= 1:
        return 15.0, f"very fresh ({year})"
    if age <= 3:
        return 10.0, f"recent ({year})"
    if age <= 7:
        return 5.0, f"somewhat dated ({year})"
    return 0.0, f"old ({year})"


def _content_quality_score(meta: dict) -> tuple[float, list[str]]:
    score = 0.0
    notes: list[str] = []
    if meta["has_author"]:
        score += 10
        notes.append("has author")
    if meta["has_date"]:
        score += 5
        notes.append("has date")
    wc = meta["word_count"]
    if wc > 2000:
        score += 15
        notes.append(f"long ({wc} words)")
    elif wc > 800:
        score += 8
        notes.append(f"medium ({wc} words)")
    elif wc > 200:
        score += 3
        notes.append(f"short ({wc} words)")
    else:
        notes.append("very short content")
    return score, notes


def _tier(score: float) -> str:
    if score >= 65:
        return "High"
    if score >= 35:
        return "Medium"
    return "Low"


# ── Public API ────────────────────────────────────────────────────────────────

def score_source(url: str, fetch_content: bool = True) -> SourceScore:
    """Score a single URL for credibility."""
    domain = _get_domain(url)
    d_score, d_notes = _domain_score(domain)
    https_b = _https_bonus(url)

    if fetch_content:
        meta = _fetch_metadata(url)
    else:
        meta = {"word_count": 0, "has_author": False, "has_date": False, "year": None}

    fresh_score, fresh_note = _freshness_score(meta["year"])
    cq_score, cq_notes = _content_quality_score(meta)

    base = 50.0
    total = min(100.0, max(0.0, base + d_score + https_b + fresh_score + cq_score - 50))

    notes = d_notes + [fresh_note] + cq_notes
    return SourceScore(
        url=url, domain=domain,
        total_score=round(total, 2),
        domain_score=d_score,
        freshness_score=fresh_score,
        content_quality_score=cq_score,
        https_bonus=https_b,
        tier=_tier(total),
        notes=notes,
    )


def rank_sources(urls: list[str], fetch_content: bool = True) -> list[SourceScore]:
    """Score and rank a list of URLs, highest first."""
    scores = [score_source(url, fetch_content) for url in urls]
    return sorted(scores, key=lambda s: s.total_score, reverse=True)


def filter_sources(
    urls: list[str],
    min_tier: str = "Medium",
    fetch_content: bool = True,
) -> list[SourceScore]:
    """Return only sources at or above the specified tier."""
    tier_order = {"High": 2, "Medium": 1, "Low": 0}
    threshold = tier_order.get(min_tier, 1)
    ranked = rank_sources(urls, fetch_content)
    return [s for s in ranked if tier_order.get(s.tier, 0) >= threshold]


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    urls = sys.argv[1:] or [
        "https://arxiv.org/abs/2005.14165",
        "https://en.wikipedia.org/wiki/Artificial_intelligence",
        "https://some-random-blog.blogspot.com/post",
    ]
    print("Scoring sources ...\n")
    for s in rank_sources(urls):
        print(s)
