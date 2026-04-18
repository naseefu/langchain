from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from tavily import TavilyClient

load_dotenv()

tavily = TavilyClient()


@tool
def search_tool(query: str):
    """Search the internet"""
    print(f"Searching for {query}")
    return tavily.search(query)


# ✅ modern chat model
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

tools = [search_tool]

agent = create_agent(model=llm, tools=tools)

def main():
    query = HumanMessage(content="Search for 3 job posting in linkedin for java developer")
    response = agent.invoke({"messages": [query]})
    print(response)


if __name__ == "__main__":
    main()
