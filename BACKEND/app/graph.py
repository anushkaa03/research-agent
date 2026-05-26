from langgraph.graph import StateGraph, END

from BACKEND.app.state import AgentState
from BACKEND.app.planner import create_research_plan
from BACKEND.app.tools.search_tool import search_web
from BACKEND.app.tools.fetch_tool import fetch_url_content
from BACKEND.app.tools.extract_tool import extract_clean_text
from BACKEND.app.reasoning import run_reasoning
from BACKEND.app.memory import memory
from BACKEND.app.summarizer import summarize_content
from BACKEND.app.citation_manager import generate_citations

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

            extracted_data.append(
                clean_text[:3000]
            )

    return {
        "extracted_text": extracted_data
    }

def summarize_node(state: AgentState):

    summaries = []

    for text in state["extracted_text"]:

        summary = summarize_content(text)

        summaries.append(summary)

    return {
        "summaries": summaries
    }
def citation_node(state: AgentState):

    citations = generate_citations(
        state["search_results"]
    )

    return {
        "citations": citations
    }

def reasoning_node(state: AgentState):

    combined_research = "\n\n".join(
        state["summaries"]
    )

    answer = run_reasoning(
        state["query"],
        combined_research
    )

    memory.store_fact(answer)

    formatted_answer = answer + "\n\nReferences:\n"

    for citation in state["citations"]:

        formatted_answer += (
            f"{citation['id']} "
            f"{citation['url']}\n"
        )

    return {
        "final_answer": formatted_answer
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
    "citation",
    citation_node
)

builder.add_node(
    "reason",
    reasoning_node
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
    "citation"
)

builder.add_edge(
    "citation",
    "reason"
)
builder.add_edge(
    "reason",
    END
)

research_graph = builder.compile()