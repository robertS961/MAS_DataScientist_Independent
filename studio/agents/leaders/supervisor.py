from helper_functions import make_supervisor_node
from langchain_openai import ChatOpenAI
from classes import TempState
from langgraph.graph import StateGraph, START
from agents import search_node, web_scraper_node, vis_a, Research_Stat_Agent, Research_DataScience_Agent
from dotenv import load_dotenv
from classes import State
load_dotenv()

from langgraph_supervisor import create_supervisor
llm =  ChatOpenAI(model="gpt-4o")
def supervisor(state:State):
    stat_agent = Research_Stat_Agent(State)
    datascience_agent = Research_DataScience_Agent(State)

    prompt=(
        "You are a supervisor managing two agents:\n"
        f"You have access to the following columns of a dataset {state['dataset_info']} \n"
        f"Your goal is to assign taks to your worker agents and discover meaningful data science ideas to perform on this {state['dataset_info']}"
        "- a research data science agent. Assign research-related data science tasks to this agent\n"
        "- a research statistical learning agent. Assign research-related statistical tasks to this agent\n"
        "Assign work to one agent at a time, do not call agents in parallel.\n"
        "Do not do any work yourself."
        "Please return a neatly labeled list of the ideas from both the statistical learning agent and the data science agent! \n\n"
    )
    workflow = create_supervisor(
        agents = [stat_agent, datascience_agent],
        model = llm,
        prompt = prompt,
    )
    return workflow.compile()

'''
llm = ChatOpenAI(model="gpt-4o")
research_supervisor_node = make_supervisor_node(llm, ["search", "web_scraper"])

def create_supervisor():
    return (
        StateGraph(State)
        .add_node("supervisor", research_supervisor_node)
        .add_node("search", search_node)
        .add_node("web_scraper", web_scraper_node)
        .add_edge(START, "supervisor")
        .compile()
    )
'''


