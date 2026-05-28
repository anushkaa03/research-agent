"""
Self-Critique System — Aarushi Sharma
Agent reviews its own generated report, identifies gaps, and triggers follow-up searches.
"""

from __future__ import annotations
import os
import re
from dataclasses import dataclass, field
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

CRITIQUE_SYSTEM_PROMPT = """You are a rigorous academic research editor.
You will be given a research report and the original research topic.
Your job is to identify:
1. Factual gaps — important aspects of the topic that are not covered.
2. Unsupported claims — statements made without citations.
3. Shallow sections — areas covered only superficially that need deeper treatment.
4. Missing perspectives — viewpoints, counterarguments, or alternative interpretations absent from the report.
5. Citation weaknesses — over-reliance on low-quality or very old sources.

Return your critique as JSON with this exact schema:
{
  "overall_quality": "High|Medium|Low",
  "overall_summary": "<2-3 sentence assessment>",
  "gaps": [{"title": "...", "description": "...", "suggested_search_query": "..."}],
  "unsupported_claims": ["<claim text>"],
  "shallow_sections": ["<section heading or description>"],
  "missing_perspectives": ["<perspective>"],
  "recommended_follow_up_queries": ["<search query>"]
}
Return ONLY the JSON, no markdown fences."""


IMPROVEMENT_SYSTEM_PROMPT = """You are an expert research writer.
You will be given a research report, the original topic, and a critique identifying its weaknesses.
Improve the report by:
- Adding or expanding sections to fill identified gaps.
- Adding [Needs Citation] markers next to unsupported claims.
- Deepening shallow sections with more analysis.
- Incorporating the missing perspectives noted in the critique.
Do NOT invent new facts; mark any additions that need further research with [VERIFY].
Return the complete improved Markdown report."""


@dataclass
class CritiqueResult:
    overall_quality: str
    overall_summary: str
    gaps: list[dict] = field(default_factory=list)
    unsupported_claims: list[str] = field(default_factory=list)
    shallow_sections: list[str] = field(default_factory=list)
    missing_perspectives: list[str] = field(default_factory=list)
    recommended_follow_up_queries: list[str] = field(default_factory=list)

    def follow_up_queries(self) -> list[str]:
        queries = list(self.recommended_follow_up_queries)
        for gap in self.gaps:
            q = gap.get("suggested_search_query", "")
            if q and q not in queries:
                queries.append(q)
        return queries

    def __str__(self) -> str:
        lines = [
            f"=== Self-Critique Report ===",
            f"Overall quality : {self.overall_quality}",
            f"Summary         : {self.overall_summary}",
            "",
        ]
        if self.gaps:
            lines.append("📌 Gaps:")
            for g in self.gaps:
                lines.append(f"  • {g['title']}: {g['description']}")
        if self.unsupported_claims:
            lines.append("\n⚠️  Unsupported claims:")
            for c in self.unsupported_claims:
                lines.append(f"  • {c[:120]}")
        if self.shallow_sections:
            lines.append("\n📉 Shallow sections:")
            for s in self.shallow_sections:
                lines.append(f"  • {s}")
        if self.missing_perspectives:
            lines.append("\n🔭 Missing perspectives:")
            for p in self.missing_perspectives:
                lines.append(f"  • {p}")
        if self.recommended_follow_up_queries:
            lines.append("\n🔍 Recommended follow-up searches:")
            for q in self.recommended_follow_up_queries:
                lines.append(f"  • {q}")
        return "\n".join(lines)


# ── Core functions ────────────────────────────────────────────────────────────

def critique_report(report_markdown: str, topic: str) -> CritiqueResult:
    """Ask Groq to critique the given research report."""
    import json

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=1500,
        messages=[
            {"role": "system", "content": CRITIQUE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"TOPIC: {topic}\n\nREPORT:\n{report_markdown}"
            }
        ]
    )
    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("`").strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return CritiqueResult(
            overall_quality="Unknown",
            overall_summary=raw[:300],
        )

    return CritiqueResult(
        overall_quality=data.get("overall_quality", "Unknown"),
        overall_summary=data.get("overall_summary", ""),
        gaps=data.get("gaps", []),
        unsupported_claims=data.get("unsupported_claims", []),
        shallow_sections=data.get("shallow_sections", []),
        missing_perspectives=data.get("missing_perspectives", []),
        recommended_follow_up_queries=data.get("recommended_follow_up_queries", []),
    )


def improve_report(
    report_markdown: str,
    topic: str,
    critique: CritiqueResult,
) -> str:
    """Ask Groq to produce an improved version of the report."""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=4000,
        messages=[
            {"role": "system", "content": IMPROVEMENT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"TOPIC: {topic}\n\n"
                    f"ORIGINAL REPORT:\n{report_markdown}\n\n"
                    f"CRITIQUE:\n{critique}"
                )
            }
        ]
    )
    return response.choices[0].message.content.strip()


def self_critique_loop(
    report_markdown: str,
    topic: str,
    search_fn=None,
    max_rounds: int = 2,
) -> dict:
    """
    Full self-critique loop:
      1. Critique the report.
      2. Optionally run follow-up searches.
      3. Produce an improved report.
      4. Repeat up to max_rounds.
    """
    current_report = report_markdown
    all_critiques: list[CritiqueResult] = []
    follow_up_content: dict[str, str] = {}

    for round_num in range(1, max_rounds + 1):
        print(f"\n[Self-Critique] Round {round_num}/{max_rounds}")

        critique = critique_report(current_report, topic)
        all_critiques.append(critique)
        print(critique)

        if critique.overall_quality == "High" and not critique.gaps:
            print("[Self-Critique] Report already high quality — stopping early.")
            break

        if search_fn:
            for query in critique.follow_up_queries()[:5]:
                if query not in follow_up_content:
                    print(f"  → Searching: {query}")
                    follow_up_content[query] = search_fn(query)

        augmented = current_report
        if follow_up_content:
            augmented += "\n\n## Additional Research\n"
            for q, content in follow_up_content.items():
                augmented += f"\n### {q}\n{content}\n"

        current_report = improve_report(augmented, topic, critique)

    return {
        "final_report": current_report,
        "critiques": all_critiques,
        "follow_up_content": follow_up_content,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python self_critique.py <report.md> <topic>")
        sys.exit(1)
    with open(sys.argv[1]) as f:
        md = f.read()
    topic = sys.argv[2]
    result = self_critique_loop(md, topic, max_rounds=1)
    out_path = sys.argv[1].replace(".md", "_improved.md")
    with open(out_path, "w") as f:
        f.write(result["final_report"])
    print(f"\nImproved report saved to: {out_path}")
