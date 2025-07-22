from helper_functions import make_supervisor_node
from langchain_openai import ChatOpenAI
from classes import TempState
from langgraph.graph import StateGraph, START
from agents import web_scraper_node, vis_a, Research_Stat_Agent, Research_DataScience_Agent, create_search_nodes
from dotenv import load_dotenv
from classes import State
load_dotenv()

from langgraph_supervisor import create_supervisor
llm =  ChatOpenAI(model="gpt-4o")
def supervisor(state:State, agent: int = 0 ):
    if agent == 0:
        agents = [Research_Stat_Agent(State), Research_DataScience_Agent(State)]
        prompt=(
            "You are a supervisor managing two agents:\n"
            f"You have access to the following columns of a dataset {state['dataset_info']} \n"
            f"Your goal is to assign tasks to your worker agents and discover meaningful data science ideas to perform on this {state['dataset_info']}"
            "- a research data science agent. Assign research-related data science tasks to this agent\n"
            "- a research statistical learning agent. Assign research-related statistical tasks to this agent\n"
            "Assign work to one agent at a time, do not call agents in parallel.\n"
            "Do not do any work yourself."
            "Please return a neatly labeled list of the ideas from both the statistical learning agent and the data science agent! \n\n"
        )
    elif agent == 1:
        agents = create_search_nodes(3, State)
        prompt = (
            "You are a supervisor managing three agents:\n"
            f"You have access to the following columns of a dataset {state['dataset_info']} \n"
            f"Your goal is to assign tasks to your worker agents and discover meaningful data science ideas to perform on this {state['dataset_info']}"
            "You have search_node_0, search_node_1, and search_node2! \n"
            "Each Agent is a slightly different data science expert. Make sure to call each agent at least once. No agent must go uncalled!\n"
            "Assign work to one agent at a time, do not call agents in parallel.\n"
            "Please return a neatly labeled list of the ideas from all the agents! Feel free to edit any of the ideas you see and make them better! \n\n"
        )
    workflow = create_supervisor(
        agents = agents,
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


