from classes import State
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from typing import Optional, Type, Any, get_type_hints, Literal
from langgraph.managed import RemainingSteps
from classes import State


class MessagesWithSteps(State):
    remaining_steps: RemainingSteps

def end_or_reflect(state: MessagesWithSteps) -> Literal["__end__", "graph"]:
    print(state["remaining_steps"], len(state["messages"]))
    if state["remaining_steps"] <= 2:
        return "__end__"
    if len(state["messages"]) <= 0:
        return "__end__"
    return "graph"
    


def create_reflection_graph(
    graph: CompiledStateGraph,
    reflection: CompiledStateGraph,
    visualization: CompiledStateGraph,
    state_schema: Optional[Type[Any]] = None,
    config_schema: Optional[Type[Any]] = None,
) -> StateGraph:
    _state_schema = state_schema or graph.builder.schema


    if "remaining_steps" in _state_schema.__annotations__:
        raise ValueError(
            "Has key 'remaining_steps' in state_schema, this shadows a built in key"
        )

    if "messages" not in _state_schema.__annotations__:
        raise ValueError("Missing required key 'messages' in state_schema")

    class StateSchema(_state_schema):
        remaining_steps: RemainingSteps

    rgraph = StateGraph(StateSchema, config_schema=config_schema)
    rgraph.add_node("graph", graph)
    rgraph.add_node("reflection", reflection)
    rgraph.add_node('visualization', visualization)
    rgraph.add_edge(START, "graph")
    rgraph.add_edge("graph", "reflection")
    rgraph.add_conditional_edges("reflection", end_or_reflect)
    return rgraph