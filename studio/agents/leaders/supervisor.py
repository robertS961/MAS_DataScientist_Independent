from helper_functions import make_supervisor_node
from langchain_openai import ChatOpenAI
from classes import State
from langgraph.graph import StateGraph, START
from agents import search_node, web_scraper_node
from dotenv import load_dotenv
load_dotenv()

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



