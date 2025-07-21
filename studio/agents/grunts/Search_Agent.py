from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from tools import WebSearch
from classes import State
from langgraph.types import Command
from typing import Literal



def search_node(state: State) -> Command[Literal["supervisor"]]:
    print("\n Made it to the search node \n")
    search_agent = create_react_agent(model="openai:gpt-4o", tools=[WebSearch])
    print("\nSearch Agent created!\n")
    result = search_agent.invoke(state)
    print(f"\n\n These are the results from invoking search {result} \n\n")
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="search")
            ]
        },
        # We want our workers to ALWAYS "report back" to the supervisor when done
        goto="supervisor",
    )