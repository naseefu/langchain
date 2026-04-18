from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, ToolMessage, BaseMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch

load_dotenv()

search_tool = TavilySearch()

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

tools = [search_tool]

llm_with_tools = llm.bind_tools(tools=tools)

tool_map = {t.name: t for t in tools}


def main():

    messages: list[BaseMessage] = [
        HumanMessage(content="Search for 3 job posting in linkedin for java developer")
    ]

    while True:
        response = llm_with_tools.invoke(messages)
        messages.append(response)
        if not response.tool_calls:
            break
        for tool_call in response.tool_calls:
            result = tool_map[tool_call["name"]].invoke(tool_call["args"])
            messages.append(ToolMessage(content=result, tool_call_id=tool_call["id"]))
    print(response.content)


if __name__ == "__main__":
    main()
