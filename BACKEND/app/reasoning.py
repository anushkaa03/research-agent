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

reasoning_prompt = ChatPromptTemplate.from_template(
    """
    You are an autonomous reasoning agent.

    Question:
    {query}

    Research Data:
    {research}

    Analyze carefully and provide insights.
    """
)


def run_reasoning(query: str, research: str):

    chain = reasoning_prompt | llm

    response = chain.invoke(
        {
            "query": query,
            "research": research
        }
    )

    return response.content