from classes import State
from langgraph.graph import StateGraph, START, END
from agents import vis_a, code_plotly, ploty_agent, reducer_agent

   
def vegalite_leader(state:State):
    return (
        StateGraph(State)
        .add_node('reducer_agent', reducer_agent)
        .add_node('plotly_agent', ploty_agent)
        .add_node('vis_agent', vis_a)
        .add_node('code_plotly', code_plotly, destinations=('vis_agent', END))
        .add_edge(START, 'reducer_agent')
        .add_edge('reducer_agent', 'plotly_agent')
        .add_edge('plotly_agent', 'code_plotly')
        .add_edge('vis_agent', 'code_plotly')
        .compile(name = 'Plotly_Leader')
    )