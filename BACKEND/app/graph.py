from typing import TypedDict, List
from langgraph.graph import StateGraph, END

from app.state import AgentState
from app.planner import create_research_plan
from app.tools.search_tool import search_web
from app.tools.fetch_tool import fetch_url_content
from app.tools.extract_tool import extract_clean_text
from app.reasoning import run_reasoning
from app.memory import memory

from app.summarizer import summarize_content
from app.deduplication import deduplicate_facts
from app.report_generator import generate_report
from app.pdf_generator import generate_pdf
from app.ambiguity_handler import is_ambiguous

def planning_node(state: AgentState):
    query = state["query"]

    if is_ambiguous(query):
        return {
            "final_answer": (
                "The research query appears ambiguous. "
                "Please provide a more specific topic."
            )
        }

    plan = create_research_plan(query)
    return {
        "plan": plan
    }


def search_node(state: AgentState):
    results = search_web(
        state["query"]
    )
    return {
        "search_results": results
    }


def extraction_node(state: AgentState):
    extracted_data = []
    for item in state["search_results"]:
        url = item["url"]
        html = fetch_url_content(url)
        if html:
            clean_text = extract_clean_text(html)
            extracted_data.append({
                "source": url,
                "text": clean_text[:3000]
            })
    return {
        "extracted_text": extracted_data
    }


def summarize_node(state: AgentState):
    research_data = []
    for item in state["extracted_text"]:
        summary = summarize_content(
            item["text"]
        )
        research_data.append({
            "summary": summary,
            "source": item["source"]
            })
    return {
        "research_data": research_data
    }


def deduplication_node(state: AgentState):
    unique_data = deduplicate_facts(
        state["research_data"]
    )
    return {
        "research_data": unique_data
    }


def reasoning_node(state: AgentState):
    combined_research = "\n\n".join(
        item["summary"]
        for item in state["research_data"]
    )

    answer = run_reasoning(
        state["query"],
        combined_research
    )
    memory.store_fact(answer)

    formatted_answer = answer + "\n\nReferences:\n"
    for index, item in enumerate(
        state["research_data"],
        start=1
    ):
        formatted_answer += (
            f"[Source {index}] "
            f"{item['source']}\n"
        )
    return {
        "final_answer": formatted_answer
    }


def report_node(state: AgentState):
    report = generate_report(state)
    pdf_path = generate_pdf(report)
    return {
        "report": report,
        "pdf_path": pdf_path
    }


def route_after_planning(state: AgentState):
    if "final_answer" in state and state["final_answer"]:
        return "end"
    return "search"


# --- Graph Construction ---
builder = StateGraph(AgentState)

# Registering processing nodes
builder.add_node(
    "planner",
    planning_node
)

builder.add_node(
    "search",
    search_node
)

builder.add_node(
    "extract",
    extraction_node
)

builder.add_node(
    "summarize",
    summarize_node
)

builder.add_node(
    "deduplicate",
    deduplication_node
)

builder.add_node(
    "reason",
    reasoning_node
)

# FIXED: Node key string changed to "generate_report_node" to avoid state.py conflicts
builder.add_node(
    "generate_report_node",
    report_node
)

# Graph Routing Configuration
builder.set_entry_point(
    "planner"
)

builder.add_conditional_edges(
    "planner",
    route_after_planning,
    {
        "search": "search",
        "end": END
    }
)

builder.add_edge(
    "search",
    "extract"
)

builder.add_edge(
    "extract",
    "summarize"
)

builder.add_edge(
    "summarize",
    "deduplicate"
)

builder.add_edge(
    "deduplicate",
    "reason"
)

# FIXED: Pointing edges seamlessly into and out of the renamed node
builder.add_edge(
    "reason",
    "generate_report_node"
)

builder.add_edge(
    "generate_report_node",
    END
)

research_graph = builder.compile()
