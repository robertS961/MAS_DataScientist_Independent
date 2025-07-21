
import operator
from typing import List
from typing_extensions import TypedDict, Annotated
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages] # Might need to change to str
    dataset_info: str
    error: str
    ideas: str
    flag: bool
    last_ai_message_content: str
