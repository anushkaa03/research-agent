from langgraph.graph import StateGraph, END

from BACKEND.app.state import AgentState
from BACKEND.app.planner import create_research_plan
from BACKEND.app.tools.search_tool import search_web
from BACKEND.app.tools.fetch_tool import fetch_url_content
from BACKEND.app.tools.extract_tool import extract_clean_text
from BACKEND.app.reasoning import run_reasoning
from BACKEND.app.memory import memory


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


def reasoning_node(state: AgentState):

    combined_research = "\n\n".join(
        state["extracted_text"]
    )

    answer = run_reasoning(
        state["query"],
        combined_research
    )

    memory.store_fact(answer)

    return {
        "final_answer": answer
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
    "reason"
)

builder.add_edge(
    "reason",
    END
)

research_graph = builder.compile()