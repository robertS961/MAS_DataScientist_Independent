from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from tools import WebSearch
from classes import State
from langgraph.types import Command
from typing import Literal


def search_node():
    search_agent = create_react_agent(
        model="openai:gpt-4o", 
        tools=[WebSearch],
        name = "sn",
        prompt = (
            "You are a research agent for data science. Use the newest data science practices and search for them on the web \n"
        )
        )
    return search_agent

'''
def search_node(state: State) -> Command[Literal["supervisor"]]:
    result = search_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="search")
            ]
        },
        # We want our workers to ALWAYS "report back" to the supervisor when done
        goto="supervisor",
    )
'''