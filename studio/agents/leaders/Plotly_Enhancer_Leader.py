from classes import State
from langgraph.graph import StateGraph, START, END
from agents import vis_a, ploty_enhancer_agent, code_plotly_final, narrative_agent, code_narrative, vis_narrative

   
def plotly_enhancer_leader(state:State):
    return (
        StateGraph(State)
        .add_node('plotly_enhancer_agent', ploty_enhancer_agent)
        .add_node('vis_agent', vis_a)
        .add_node('narrative_agent', narrative_agent)
        .add_node('code_narrative', code_narrative, destinations = (END, 'vis_narrative'))
        .add_node('code_plotly_final', code_plotly_final, destinations=('vis_agent', 'narrative_agent'))
        .add_node('vis_narrative', vis_narrative)
        .add_edge(START, 'plotly_enhancer_agent')
        .add_edge('plotly_enhancer_agent', 'code_plotly_final')
        .add_edge('narrative_agent', 'code_narrative')
        .add_edge('vis_agent', 'code_plotly_final')
        .add_edge("vis_narrative", "code_narrative" )
        .compile(name = 'Plotly_Enhancer_Leader')
    )