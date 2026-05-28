from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant"
)

def summarize_content(extracted_text):

    prompt = f"""
    Summarize the following research content clearly.
    Focus on:
    - important technical insights
    - recent developments
    - concise explanation

    Content:
    {extracted_text}
    """

    response = llm.invoke([HumanMessage(content=prompt)])

    return response.content