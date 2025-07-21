
from typing import List
from typing_extensions import TypedDict

class State(TypedDict):
    messages: List[str] # Might need to change to str
    feedback: List[str]
    dataset_info: str
    error: str
    ideas: str
    flag: bool
    next: str