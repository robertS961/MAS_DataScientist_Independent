from helper_functions import make_supervisor_node
from langchain_openai import ChatOpenAI
from classes import TempState
from langgraph.graph import StateGraph, START
from agents import search_node, web_scraper_node
from dotenv import load_dotenv
load_dotenv()

from langgraph_supervisor import create_supervisor
llm =  ChatOpenAI(model="gpt-4o")
def supervisor():
    sn = search_node()
    wsn = web_scraper_node()
    prompt = "You a supervisor. Call each agent once. Do not call in parallel. Return meaningful results!"
    workflow = create_supervisor(
        agents = [sn, wsn],
        model = llm,
        state_schema = TempState,
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


