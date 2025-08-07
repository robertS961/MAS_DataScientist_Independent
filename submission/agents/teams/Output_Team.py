
from classes import State, Configurable
from agents import create_code, code_enhancer
from helper_functions import create_reflection_graph
from langgraph.graph import StateGraph, START, END

def create_output_team(state:State):
    created_code= create_code(State)
    revised_code = code_enhancer(State)
    return (
        StateGraph(State)
        .add_node('created_code', created_code )
        .add_node('revised_code', revised_code)
        .add_edge(START, 'created_code')
        .add_edge('created_code', 'revised_code')
        .add_edge('revised_code', END)
        .compile()
    )



