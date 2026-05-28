"""
Collaborative Multi-Agent Research — Aarushi Sharma
Two independent research agents produce reports, then cross-check each other's findings.
"""

from __future__ import annotations
import os
import re
from dataclasses import dataclass, field
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ── Agent prompts ─────────────────────────────────────────────────────────────

RESEARCHER_SYSTEM = """You are a rigorous autonomous research agent (Agent {agent_id}).
Given a topic and optional search results, write a comprehensive research report in Markdown.
Structure:
# <Title>
## Executive Summary
## Background
## Key Findings
## Analysis
## Conclusion
## References
[1] <url> — <brief description>
Be precise. Do not invent facts. If uncertain, say so. Use [1], [2] inline citations."""

CROSS_CHECK_SYSTEM = """You are a critical peer-reviewer.
You have two independent research reports on the same topic.
Your task:
1. Identify AGREEMENTS — facts/claims both reports share.
2. Identify CONFLICTS — facts/claims that contradict each other.
3. Identify UNIQUE CONTRIBUTIONS — important points in only one report.
4. Produce a SYNTHESISED REPORT merging both.

Return JSON with this schema:
{
  "agreements": ["<claim>"],
  "conflicts": [{"claim_A": "...", "claim_B": "...", "resolution": "unresolved|A_preferred|B_preferred", "reason": "..."}],
  "unique_to_A": ["<claim>"],
  "unique_to_B": ["<claim>"]
}
Return ONLY JSON, no markdown fences."""

SYNTHESIS_SYSTEM = """You are an expert research writer.
Given two research reports and a cross-check analysis, produce a single superior synthesised
research report in Markdown. Merge the best of both, resolve conflicts where evidence is clear,
mark unresolved conflicts with ⚠️ CONFLICT, and include a combined reference list."""


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class AgentReport:
    agent_id: str
    topic: str
    report: str
    search_results: str = ""


@dataclass
class CrossCheckResult:
    agreements: list[str] = field(default_factory=list)
    conflicts: list[dict] = field(default_factory=list)
    unique_to_A: list[str] = field(default_factory=list)
    unique_to_B: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = ["=== Cross-Check Analysis ==="]
        lines.append(f"✅ Agreements ({len(self.agreements)}): " +
                     (", ".join(self.agreements[:3]) or "none"))
        if self.conflicts:
            lines.append(f"⚠️  Conflicts ({len(self.conflicts)}):")
            for c in self.conflicts:
                lines.append(f"   A: {c['claim_A'][:80]}")
                lines.append(f"   B: {c['claim_B'][:80]}")
                lines.append(f"   → {c['resolution']}: {c.get('reason','')[:80]}")
        lines.append(f"🅰  Unique to Agent A: {len(self.unique_to_A)} claims")
        lines.append(f"🅱  Unique to Agent B: {len(self.unique_to_B)} claims")
        return "\n".join(lines)


@dataclass
class CollaborativeResult:
    report_A: AgentReport
    report_B: AgentReport
    cross_check: CrossCheckResult
    synthesised_report: str


# ── Core functions ────────────────────────────────────────────────────────────

def run_single_agent(
    agent_id: str,
    topic: str,
    search_results: str = "",
) -> AgentReport:
    """Run one research agent."""
    user_msg = f"Research topic: {topic}"
    if search_results:
        user_msg += f"\n\nSearch results available:\n{search_results}"

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=3000,
        messages=[
            {
                "role": "system",
                "content": RESEARCHER_SYSTEM.format(agent_id=agent_id)
            },
            {"role": "user", "content": user_msg}
        ]
    )
    report_text = response.choices[0].message.content.strip()

    return AgentReport(
        agent_id=agent_id,
        topic=topic,
        report=report_text,
        search_results=search_results,
    )


def cross_check_reports(report_A: AgentReport, report_B: AgentReport) -> CrossCheckResult:
    """Ask Groq to cross-check two reports and return structured analysis."""
    import json

    user_content = (
        f"TOPIC: {report_A.topic}\n\n"
        f"=== REPORT A (Agent {report_A.agent_id}) ===\n{report_A.report}\n\n"
        f"=== REPORT B (Agent {report_B.agent_id}) ===\n{report_B.report}"
    )

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=2000,
        messages=[
            {"role": "system", "content": CROSS_CHECK_SYSTEM},
            {"role": "user", "content": user_content}
        ]
    )
    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("`").strip()

    try:
        data = json.loads(raw)
        return CrossCheckResult(
            agreements=data.get("agreements", []),
            conflicts=data.get("conflicts", []),
            unique_to_A=data.get("unique_to_A", []),
            unique_to_B=data.get("unique_to_B", []),
        )
    except json.JSONDecodeError:
        return CrossCheckResult()


def synthesise_reports(
    report_A: AgentReport,
    report_B: AgentReport,
    cross_check: CrossCheckResult,
) -> str:
    """Produce a single superior synthesised report from both agents' work."""
    user_content = (
        f"TOPIC: {report_A.topic}\n\n"
        f"=== REPORT A ===\n{report_A.report}\n\n"
        f"=== REPORT B ===\n{report_B.report}\n\n"
        f"=== CROSS-CHECK ===\n{cross_check}"
    )
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=4000,
        messages=[
            {"role": "system", "content": SYNTHESIS_SYSTEM},
            {"role": "user", "content": user_content}
        ]
    )
    return response.choices[0].message.content.strip()


def collaborative_research(
    topic: str,
    search_results_A: str = "",
    search_results_B: str = "",
) -> CollaborativeResult:
    """
    Full collaborative research pipeline:
    Agent A researches → Agent B researches independently
    → Cross-check → Synthesise → Return everything.
    """
    print(f"[Collaborative] Agent A researching: {topic}")
    report_A = run_single_agent("A", topic, search_results_A)

    print(f"[Collaborative] Agent B researching: {topic}")
    report_B = run_single_agent("B", topic, search_results_B)

    print("[Collaborative] Cross-checking reports ...")
    cross_check = cross_check_reports(report_A, report_B)
    print(cross_check)

    print("[Collaborative] Synthesising final report ...")
    final = synthesise_reports(report_A, report_B, cross_check)

    return CollaborativeResult(
        report_A=report_A,
        report_B=report_B,
        cross_check=cross_check,
        synthesised_report=final,
    )


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    topic = " ".join(sys.argv[1:]) or "Impact of large language models on scientific research"
    result = collaborative_research(topic)
    out = f"synthesised_{topic[:40].replace(' ','_')}.md"
    with open(out, "w") as f:
        f.write(result.synthesised_report)
    print(f"\n✅ Synthesised report saved to: {out}")
