

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
def supervisor_team(state:State, name = "team_supervisor" ):
    supervisor_graph = supervisor(state, agent = 1)
    print("\n supervisor_Graph created\n ")
    chain_graph = chain(3, state)
    print("\n Chain graph created \n")

    agents = [supervisor_graph, chain_graph]
    print("agents loaded in\n ")
    prompt = (
        "You are a supervisor managing two agents:\n"
        f"You have access to the following columns of a dataset {state['dataset_info']} \n"
        f"Your goal is to assign tasks to your worker agents and discover meaningful data science ideas to perform on this {state['dataset_info']}"
        "Each agent will come to you with data science and statistical learning ideas to perform on the data. \n"
        "Feel free to revise any ideas too or add your own if they are better!\n"
        "Assign work to one agent at a time, do not call agents in parallel!Make sure both agents are called before you return!\n"
        "Please return a neatly labeled list of at LEAST 10 ideas! Thank you! \n\n"
    )
    workflow = create_supervisor(
        agents = agents,
        model = llm,
        prompt = prompt,
        supervisor_name = name,
    )
    return workflow.compile(name = "supervisor_team")

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


