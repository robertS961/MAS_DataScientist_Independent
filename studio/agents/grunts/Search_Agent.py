from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from tools import WebSearch
from classes import State
from langgraph.types import Command
from typing import Literal


def search_node(state: State):
    data = state['dataset_info']
    prompt=(
            "You are a research data science agent.\n\n"
            "INSTRUCTIONS:\n"
            "- Assist ONLY with research-related tasks, your task is to discover novel data science strategies for tabular data. \n"
            "You have access to an internet websearch tool. Use it to find novel data science straegies online \n!"
            "Also feel free to use your own data science ideas.\n"
            f"You MUST apply these ideas to tabular data specifically the data set at hand! The data colums are {data}\n"
            "- After you're done with your tasks, respond to the supervisor directly\n"
            "- Respond ONLY with the results of your work, do NOT include ANY other text. \n"
            f"The results should be ideas from the web or yourself that involve data science ideas using the {data} columns\n"
    )
    websearch = WebSearch()
    search_agent = create_react_agent(
        model="openai:gpt-4o", 
        tools=[websearch],
        name = "search_node",
        prompt = prompt
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