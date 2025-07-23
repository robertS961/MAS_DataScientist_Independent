
from classes import State
from langgraph.graph import StateGraph, START, END
from agents import vis_a, code_agent, reducer_agent


def create_code(state:State):
    return (
        StateGraph(State)
        .add_node('reducer_agent', reducer_agent )
        .add_node('vis_agent', vis_a)
        .add_node('code_agent', code_agent, destinations=('vis_agent', END))
        .add_edge(START, 'reducer_agent')
        .add_edge('reducer_agent', 'vis_agent')
        .add_edge('vis_agent', 'code_agent')
        .compile()
    )