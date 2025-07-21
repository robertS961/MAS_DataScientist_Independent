from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from tools import Web_Search
from classes import State
from langgraph.types import Command
from typing import Literal
from tools import scrape_webpages


def web_scraper_node(state: State) -> Command[Literal["supervisor"]]:
    web_scraper_agent = create_react_agent(model="openai:gpt-4o", tools=[scrape_webpages])
    result = web_scraper_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="web_scraper")
            ]
        },
        # We want our workers to ALWAYS "report back" to the supervisor when done
        goto="supervisor",
    )