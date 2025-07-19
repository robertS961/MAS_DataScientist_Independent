
from classes import State
from langgraph.graph import StateGraph, START, END
from agents import vis_a, code_agent


def create_vis():
    return (
        StateGraph(State)
        .add_node('vis_agent', vis_a)
        .add_node('code_agent', code_agent, destinations=('vis_agent', END))
        .add_edge(START, 'vis_agent')
        .add_edge('vis_agent', 'code_agent')
        .compile()
    )