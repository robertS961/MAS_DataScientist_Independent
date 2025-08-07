from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from tools import Web_Search
from classes import State
from langgraph.types import Command
from typing import Literal
from tools import scrape_webpages
from helper_functions import get_llm

def web_scraper_node(state:State):
    llm = get_llm()
    data = state['dataset_info']
    prompt=(
            "You are a web scraper data science agent.\n\n"
            "INSTRUCTIONS:\n"
            "- Assist ONLY with wep scrapping research-related tasks, your task is to discover novel data science strategies. \n"
            f"Apply the learned knowledge to the following data set columns {data}\n"
            "- After you're done with your tasks, respond to the supervisor directly\n"
            "- Respond ONLY with the results of your work, do NOT include ANY other text.\n"
            f"The results should be data science ideas applied to the {data} \n"
    )
    web_scraper_node = create_react_agent(
        model=llm, 
        tools=[scrape_webpages],
        name = "web_scrape_node",
        prompt = prompt 
        )
    return web_scraper_node

'''
def web_scraper_node(state: State) -> Command[Literal["supervisor"]]:
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
'''