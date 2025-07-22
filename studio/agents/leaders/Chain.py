from agents import create_search_nodes
from classes import State
from langgraph.graph import StateGraph, START, END

def chain(count: int, state:State):
    agents = create_search_nodes(count, State)
    graph = StateGraph(State)
    for node_cnt in range(count):
        curr = f"search_node_{node_cnt}"
        prev = f"search_node_{node_cnt -1}" if node_cnt > 0 else START
        graph.add_node(curr, agents[node_cnt])
        graph.add_edge(prev, curr)
    graph.add_edge(f"search_node_{node_cnt}", END)
    return graph.compile()

