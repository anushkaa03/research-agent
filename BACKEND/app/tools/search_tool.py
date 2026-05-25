from tavily import TavilyClient
import os
from dotenv import load_dotenv

load_dotenv()

client = TavilyClient(
    api_key=os.getenv("TAVILY_API_KEY")
)


def search_web(query: str):

    response = client.search(
        query=query,
        search_depth="advanced",
        max_results=5
    )

    return response["results"]