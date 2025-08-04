from classes import State
from langgraph.graph import StateGraph, START, END
from agents import vis_a, code_agent, pdf_checker_agent
from helper_functions import get_last_ai_message

   
def code_enhancer(state:State):
    return (
        StateGraph(State)
        .add_node('pdf_checker_agent', pdf_checker_agent )
        .add_node('vis_agent', vis_a)
        .add_node('code_agent', code_agent, destinations=('vis_agent', END))
        .add_edge(START, 'pdf_checker_agent')
        .add_edge('pdf_checker_agent', 'code_agent')
        .add_edge('vis_agent', 'code_agent')
        .compile(name = 'lead_revise_code')
    )