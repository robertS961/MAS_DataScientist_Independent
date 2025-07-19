from classes import State
from langgraph.graph import StateGraph, START, END
from agents import make_judge

def create_judge():
    return(
        StateGraph(State)
        .add_node('judge_agent', make_judge)
        .add_edge(START, 'judge_agent')
        .add_edge('judge_agent', END)
        .compile()
    )