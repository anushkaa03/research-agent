from langgraph.graph import StateGraph, END

from BACKEND.app.state import AgentState
from BACKEND.app.planner import create_research_plan
from BACKEND.app.tools.search_tool import search_web
from BACKEND.app.tools.fetch_tool import fetch_url_content
from BACKEND.app.tools.extract_tool import extract_clean_text
from BACKEND.app.reasoning import run_reasoning
from BACKEND.app.memory import memory

from BACKEND.app.summarizer import summarize_content
from BACKEND.app.deduplication import deduplicate_facts
from BACKEND.app.report_generator import generate_report
from BACKEND.app.pdf_generator import generate_pdf

def planning_node(state: AgentState):

    plan = create_research_plan(
        state["query"]
    )

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

builder = StateGraph(AgentState)

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

builder.add_node(
    "report",
    report_node
)

builder.set_entry_point(
    "planner"
)

builder.add_edge(
    "planner",
    "search"
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

builder.add_edge(
    "reason",
    "report"
)

builder.add_edge(
    "report",
    END
)

research_graph = builder.compile()