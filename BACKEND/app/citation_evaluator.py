"""
Citation Accuracy Evaluation Module — Aarushi Sharma
Validates that citations in the report actually match the source content.
"""

import re
import httpx
from dataclasses import dataclass, field
from typing import Optional
from bs4 import BeautifulSoup


@dataclass
class CitationCheckResult:
    citation_id: str
    url: str
    claim: str
    is_valid: bool
    confidence: float
    reason: str
    matched_excerpt: Optional[str] = None


@dataclass
class EvaluationReport:
    total_citations: int
    valid_citations: int
    invalid_citations: int
    accuracy_score: float
    results: list[CitationCheckResult] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"=== Citation Accuracy Report ===",
            f"Total citations  : {self.total_citations}",
            f"Valid            : {self.valid_citations}",
            f"Invalid          : {self.invalid_citations}",
            f"Accuracy score   : {self.accuracy_score:.1%}",
            "",
        ]
        for r in self.results:
            status = "✅" if r.is_valid else "❌"
            lines.append(f"{status} [{r.citation_id}] {r.url}")
            lines.append(f"   Claim     : {r.claim[:120]}...")
            lines.append(f"   Reason    : {r.reason}")
            if r.matched_excerpt:
                lines.append(f"   Match     : {r.matched_excerpt[:100]}...")
            lines.append("")
        return "\n".join(lines)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_text_from_url(url: str, timeout: int = 10) -> str:
    """Fetch a URL and return its plain text."""
    try:
        resp = httpx.get(url, timeout=timeout, follow_redirects=True,
                         headers={"User-Agent": "ResearchAgentBot/1.0"})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        return " ".join(soup.get_text(separator=" ").split())
    except Exception as exc:
        return f"FETCH_ERROR: {exc}"


def _keyword_overlap(claim: str, page_text: str) -> tuple[float, Optional[str]]:
    """Simple keyword-overlap heuristic."""
    stopwords = {"the", "a", "an", "is", "in", "on", "at", "of", "and",
                 "to", "for", "was", "were", "be", "been", "with", "that",
                 "this", "it", "by", "as", "from", "are"}
    claim_words = {w.lower() for w in re.findall(r"\b\w+\b", claim)
                   if w.lower() not in stopwords and len(w) > 3}
    if not claim_words:
        return 0.0, None

    page_lower = page_text.lower()
    matched = {w for w in claim_words if w in page_lower}
    confidence = len(matched) / len(claim_words)

    best_sentence = None
    best_hit = 0
    for sentence in re.split(r"[.!?]", page_text):
        hits = sum(1 for w in matched if w in sentence.lower())
        if hits > best_hit:
            best_hit = hits
            best_sentence = sentence.strip()

    return confidence, best_sentence if best_hit >= 2 else None


# ── Public API ────────────────────────────────────────────────────────────────

def parse_citations_from_report(report_markdown: str) -> list[dict]:
    """Extract citations from a Markdown report."""
    citations = []

    ref_pattern = re.compile(
        r"\[(\d+)\]\s+(https?://\S+)(?:\s*[—–-]\s*(.+))?", re.MULTILINE
    )
    for m in ref_pattern.finditer(report_markdown):
        citations.append({
            "id": m.group(1),
            "url": m.group(2).rstrip(").,"),
            "context": m.group(3) or "",
        })

    inline_pattern = re.compile(r"\[([^\]]+)\]\((https?://[^\)]+)\)")
    seen_urls = {c["url"] for c in citations}
    for i, m in enumerate(inline_pattern.finditer(report_markdown), start=len(citations) + 1):
        url = m.group(2)
        if url not in seen_urls:
            citations.append({"id": f"L{i}", "url": url, "context": m.group(1)})
            seen_urls.add(url)

    return citations


def evaluate_citations(
    report_markdown: str,
    claim_map: Optional[dict[str, str]] = None,
    confidence_threshold: float = 0.35,
) -> EvaluationReport:
    """Main entry point — evaluates all citations in a report."""
    citations = parse_citations_from_report(report_markdown)
    results: list[CitationCheckResult] = []

    for cit in citations:
        cid = cit["id"]
        url = cit["url"]
        claim = (claim_map or {}).get(cid, cit["context"]) or "general reference"

        page_text = _extract_text_from_url(url)
        if page_text.startswith("FETCH_ERROR"):
            results.append(CitationCheckResult(
                citation_id=cid, url=url, claim=claim,
                is_valid=False, confidence=0.0,
                reason=f"Could not fetch URL: {page_text}",
            ))
            continue

        confidence, excerpt = _keyword_overlap(claim, page_text)
        is_valid = confidence >= confidence_threshold
        reason = (
            f"Keyword overlap {confidence:.0%} — above threshold"
            if is_valid else
            f"Keyword overlap {confidence:.0%} — below threshold"
        )
        results.append(CitationCheckResult(
            citation_id=cid, url=url, claim=claim,
            is_valid=is_valid, confidence=confidence,
            reason=reason, matched_excerpt=excerpt,
        ))

    valid = sum(1 for r in results if r.is_valid)
    total = len(results)
    return EvaluationReport(
        total_citations=total,
        valid_citations=valid,
        invalid_citations=total - valid,
        accuracy_score=valid / total if total else 0.0,
        results=results,
    )


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python citation_evaluator.py <report.md>")
        sys.exit(1)
    with open(sys.argv[1]) as f:
        md = f.read()
    report = evaluate_citations(md)
    print(report.summary())
