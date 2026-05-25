from typing import TypedDict, List


class AgentState(TypedDict):

    query: str
    plan: str
    search_results: List
    extracted_text: List[str]
    final_answer: str