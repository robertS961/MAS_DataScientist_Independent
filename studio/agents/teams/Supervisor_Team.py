

from helper_functions import make_supervisor_node
from langchain_openai import ChatOpenAI
from classes import TempState
from langgraph.graph import StateGraph, START
from agents import web_scraper_node, vis_a, Research_Stat_Agent, Research_DataScience_Agent, create_search_nodes, chain, supervisor
from dotenv import load_dotenv
from classes import State
load_dotenv()

from langgraph_supervisor import create_supervisor
llm =  ChatOpenAI(model="gpt-4o")
def supervisor_team(state:State ):
    supervisor_graph = supervisor(state, agent = 1)
    chain_graph = chain(3, state)
    agents = [supervisor_graph, chain_graph]
    prompt = (
        "You are a supervisor managing two agents:\n"
        f"You have access to the following columns of a dataset {state['dataset_info']} \n"
        f"Your goal is to assign tasks to your worker agents and discover meaningful data science ideas to perform on this {state['dataset_info']}"
        "Each agent will come to you with data science and statistical learning ideas to perform on the data. \n"
        "Call both agents and pick the best ideas from both! Feel free to revise any ideas too or add your own if they are better!\n"
        "Assign work to one agent at a time, do not call agents in parallel.\n"
        "Please return a neatly labeled list of at LEAST 10 ideas! Thank you! \n\n"
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


