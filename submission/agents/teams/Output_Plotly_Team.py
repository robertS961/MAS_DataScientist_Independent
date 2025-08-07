
from classes import State, Configurable
from agents import plotly_leader, plotly_enhancer_leader
from langgraph.graph import StateGraph, START, END

def create_output_plotly_team(state:State):
    created_code= plotly_leader(State)
    revised_code = plotly_enhancer_leader(State)
    return (
        StateGraph(State)
        .add_node('plotly_leader', plotly_leader )
        .add_node('plotly_enhancer_leader', plotly_enhancer_leader)
        .add_edge(START, 'plotly_leader')
        .add_edge('plotly_leader', 'plotly_enhancer_leader')
        .add_edge('plotly_enhancer_leader', END)
        .compile()
    )