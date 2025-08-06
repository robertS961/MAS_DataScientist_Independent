from classes import State
from langgraph.graph import StateGraph, START, END
from agents import vis_a, code_plotly, ploty_enhancer_agent

   
def plotly_enhancer_leader(state:State):
    return (
        StateGraph(State)
        .add_node('plotly_enhancer_agent', ploty_enhancer_agent)
        .add_node('vis_agent', vis_a)
        .add_node('code_plotly', code_plotly, destinations=('vis_agent', END))
        .add_edge(START, 'plotly_enhancer_agent')
        .add_edge('plotly_enhancer_agent', 'code_plotly')
        .add_edge('vis_agent', 'code_plotly')
        .compile(name = 'Plotly_Enhancer_Leader')
    )