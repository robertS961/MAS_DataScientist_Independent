from agents import create_search_nodes
from classes import State
from langgraph.graph import StateGraph, START, END

def chain(count: int, state:State, name = "chain_gang"):
    agents = create_search_nodes(count, State, "chain")
    agents_name = [tuple([f"chain_agent_{cnt}", agent]) for cnt, agent in enumerate(agents)]
    graph = StateGraph(State)
    graph.add_sequence(agents_name)
    graph.add_edge(START, "chain_agent_0")
    graph.add_edge("chain_agent_2", END)
    return graph.compile(name = name)
    '''
    for node_cnt in range(count):
        curr = f"search_node_{node_cnt}"
        prev = f"search_node_{node_cnt -1}" if node_cnt > 0 else START
        graph.add_node(curr, agents[node_cnt])
        graph.add_edge(prev, curr)
    graph.add_edge(f"search_node_{node_cnt}", END)
    
    return graph.compile(name = name)
    '''

