from dotenv import load_dotenv
import os

load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant",
    temperature=0
)

planning_prompt = ChatPromptTemplate.from_template(
    """
    You are a research planning agent.

    User Question:
    {query}

    Create a step-by-step research plan.
    """
)


def create_research_plan(query: str):

    chain = planning_prompt | llm

    response = chain.invoke(
        {
            "query": query
        }
    )

    return response.content