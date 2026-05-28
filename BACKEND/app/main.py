"""
Main API — Complete Merged File
Anushka (graph/memory) + Anisha (summarizer/report/citations) + Aarushi (critique/credibility/collaborative)
Run from BACKEND/ folder: uvicorn app.main:app --reload
"""

from __future__ import annotations
import uuid
import os
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

# ── Anushka's modules ─────────────────────────────────────────────────────────
from app.graph import research_graph
from app.memory import memory

# ── Anisha's modules ──────────────────────────────────────────────────────────
from app.summarizer import summarize_content
from app.report_generator import generate_report
from app.citation_manager import generate_citations

# ── Aarushi's modules ─────────────────────────────────────────────────────────
from app.citation_evaluator import evaluate_citations
from app.source_credibility import rank_sources
from app.self_critique import self_critique_loop, critique_report
from app.collaborative_agents import collaborative_research

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Research Agent API",
    version="1.0.0",
    description="Autonomous Research Agent — Anushka + Anisha + Aarushi",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory job store ───────────────────────────────────────────────────────
jobs: dict[str, dict] = {}


def _make_job(job_type: str) -> dict:
    jid = str(uuid.uuid4())
    jobs[jid] = {
        "id": jid,
        "type": job_type,
        "status": "pending",
        "created_at": datetime.utcnow().isoformat(),
        "result": None,
        "error": None,
    }
    return jobs[jid]


# ── Request models ────────────────────────────────────────────────────────────

class ResearchRequest(BaseModel):
    query: str
    mode: str = "single"
    enable_self_critique: bool = True
    max_critique_rounds: int = 1

class EvaluateCitationsRequest(BaseModel):
    report_markdown: str
    claim_map: Optional[dict[str, str]] = None

class ScoreSourcesRequest(BaseModel):
    urls: list[str]
    fetch_content: bool = False

class CritiqueRequest(BaseModel):
    report_markdown: str
    topic: str

class ImproveRequest(BaseModel):
    report_markdown: str
    topic: str
    max_rounds: int = 1


# ══════════════════════════════════════════════════════════════════════════════
# BASIC ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/")
def home():
    return {"message": "Autonomous Research Agent Running"}

@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0"}

@app.get("/memory")
def get_memory_original():
    return {"stored_facts": memory.get_facts()}

@app.get("/api/memory")
def get_memory():
    return {"stored_facts": memory.get_facts()}


# ══════════════════════════════════════════════════════════════════════════════
# ANUSHKA + ANISHA — SYNC RESEARCH ENDPOINT (original simple one)
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/research")
def research_sync(request: ResearchRequest):
    """Original sync research — Anushka graph + Anisha report."""
    result = research_graph.invoke({
        "query": request.query,
        "plan": "",
        "search_results": [],
        "extracted_text": [],
        "final_answer": ""
    })

    # Anisha's pipeline
    summary    = summarize_content(result["final_answer"])
    result["final_answer"] = summary
    report_md  = generate_report(result)
    citations  = generate_citations(result.get("search_results", []))

    return {
        "query"    : request.query,
        "plan"     : result["plan"],
        "answer"   : report_md,
        "citations": citations,
    }


# ══════════════════════════════════════════════════════════════════════════════
# AARUSHI — SOURCE CREDIBILITY
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/api/score-sources")
def score_sources(req: ScoreSourcesRequest):
    if not req.urls:
        raise HTTPException(400, "Provide at least one URL")
    results = rank_sources(req.urls, fetch_content=req.fetch_content)
    return {
        "ranked_sources": [
            {
                "url"   : s.url,
                "domain": s.domain,
                "score" : s.total_score,
                "tier"  : s.tier,
                "notes" : s.notes,
            }
            for s in results
        ]
    }


# ══════════════════════════════════════════════════════════════════════════════
# AARUSHI — CITATION EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/api/evaluate-citations")
def evaluate_citations_endpoint(req: EvaluateCitationsRequest):
    report = evaluate_citations(req.report_markdown, req.claim_map)
    return {
        "total"   : report.total_citations,
        "valid"   : report.valid_citations,
        "invalid" : report.invalid_citations,
        "accuracy": report.accuracy_score,
        "results" : [
            {
                "id"        : r.citation_id,
                "url"       : r.url,
                "valid"     : r.is_valid,
                "confidence": r.confidence,
                "reason"    : r.reason,
                "excerpt"   : r.matched_excerpt,
            }
            for r in report.results
        ],
    }


# ══════════════════════════════════════════════════════════════════════════════
# AARUSHI — CRITIQUE
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/api/critique")
def critique_endpoint(req: CritiqueRequest):
    result = critique_report(req.report_markdown, req.topic)
    return {
        "overall_quality"  : result.overall_quality,
        "overall_summary"  : result.overall_summary,
        "gaps"             : result.gaps,
        "unsupported_claims": result.unsupported_claims,
        "shallow_sections" : result.shallow_sections,
        "missing_perspectives": result.missing_perspectives,
        "follow_up_queries": result.follow_up_queries(),
    }


# ══════════════════════════════════════════════════════════════════════════════
# AARUSHI — IMPROVE REPORT (async job)
# ══════════════════════════════════════════════════════════════════════════════

def _run_improve(job: dict, req: ImproveRequest):
    try:
        job["status"] = "running"
        result = self_critique_loop(
            req.report_markdown, req.topic, max_rounds=req.max_rounds
        )
        job["result"] = {
            "final_report"    : result["final_report"],
            "num_critiques"   : len(result["critiques"]),
            "follow_up_content": {
                k: v[:500] for k, v in result["follow_up_content"].items()
            },
        }
        job["status"] = "done"
    except Exception as exc:
        job["status"] = "error"
        job["error"]  = str(exc)


@app.post("/api/improve-report")
def improve_report_endpoint(req: ImproveRequest, background_tasks: BackgroundTasks):
    job = _make_job("improve_report")
    background_tasks.add_task(_run_improve, job, req)
    return {"job_id": job["id"], "status": "pending"}


# ══════════════════════════════════════════════════════════════════════════════
# FULL PIPELINE — async research with all three teams integrated
# ══════════════════════════════════════════════════════════════════════════════

def _run_research(job: dict, req: ResearchRequest):
    try:
        job["status"] = "running"

        if req.mode == "collaborative":
            # ── Aarushi's two-agent pipeline ──────────────────────────────────
            result     = collaborative_research(req.query)
            report_md  = result.synthesised_report
            citations  = []
            extra = {
                "cross_check": {
                    "agreements": result.cross_check.agreements[:5],
                    "conflicts" : result.cross_check.conflicts[:5],
                }
            }

        else:
            # ── Anushka's LangGraph pipeline ──────────────────────────────────
            graph_result = research_graph.invoke({
                "query"         : req.query,
                "plan"          : "",
                "search_results": [],
                "extracted_text": [],
                "final_answer"  : ""
            })

            # ── Anisha's pipeline ─────────────────────────────────────────────
            summary = summarize_content(graph_result["final_answer"])
            graph_result["final_answer"] = summary
            report_md = generate_report(graph_result)
            citations = generate_citations(graph_result.get("search_results", []))

            extra = {
                "plan"     : graph_result.get("plan", ""),
                "citations": citations,
            }

        # ── Aarushi's self-critique (runs on both modes) ───────────────────────
        if req.enable_self_critique:
            critique_result = self_critique_loop(
                report_md, req.query, max_rounds=req.max_critique_rounds
            )
            report_md = critique_result["final_report"]
            extra["critique_summary"] = (
                critique_result["critiques"][-1].overall_summary
                if critique_result["critiques"] else ""
            )

        job["result"] = {"report": report_md, **extra}
        job["status"] = "done"

    except Exception as exc:
        job["status"] = "error"
        job["error"]  = str(exc)


@app.post("/api/research")
def research_async(req: ResearchRequest, background_tasks: BackgroundTasks):
    """Full async research — all three teams integrated."""
    job = _make_job("research")
    background_tasks.add_task(_run_research, job, req)
    return {"job_id": job["id"], "status": "pending"}


# ══════════════════════════════════════════════════════════════════════════════
# JOB MONITOR
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job

@app.get("/api/jobs")
def list_jobs():
    return {"jobs": list(jobs.values())}


# ══════════════════════════════════════════════════════════════════════════════
# PDF DOWNLOAD
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/download-report/{job_id}")
def download_report(job_id: str):
    job = jobs.get(job_id)
    if not job or job["status"] != "done":
        raise HTTPException(404, "Report not ready")
    try:
        from app.pdf_generator import generate_pdf
        pdf_path = generate_pdf(job["result"]["report"])
        return FileResponse(pdf_path, filename="research_report.pdf")
    except Exception as e:
        raise HTTPException(500, f"PDF generation failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# RUN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
