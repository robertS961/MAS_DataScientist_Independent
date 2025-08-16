from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from tools import WebSearch
from classes import State
from langgraph.types import Command
from typing import Literal
from helper_functions import get_llm


def prompt(expert_intro, message, data):
    return (
        f"{expert_intro}.\n\n"
        "INSTRUCTIONS:\n"
        f"Here is the current state of the analysis:\n{message}\n\n"
        f"The dataset description is:\n{data}\n\n"
        "Come up with data science ideas to perform on the tabular columns\n\n"
        "Return at least 4 ideas that are novel and involve the exact column name.\n\n "
        "If you are passed ideas feel free to improve them or use them to create better ideas\n\n"
    )
def create_search_nodes(count: int, state: State, name = "basic"):
    llm = get_llm(model = 'gpt-5-mini')
    agents = []
    for cnt in range(count):
        data = state['dataset_info']
        message = state['messages']
        expert_intro = {
            0: "You are an expert data scientist specializing in exploratory data analysis. Your job is to identify impactful trends from datasets.",
            1: "You are a senior analyst reviewing the earlier results and adding deeper statistical insights and visualizations.",
            2: "You are a final reviewer, refining the insights and making sure the Python visualizations are well-structured and insightful.",
        }[(cnt % 3)]
        #websearch = WebSearch()
        search_agent = create_react_agent(
            model=llm, 
            tools=[],
            name = f"search_node_{cnt}_{name}",
            prompt = prompt(expert_intro, message, data)
            )
        agents.append(search_agent)
    return agents

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