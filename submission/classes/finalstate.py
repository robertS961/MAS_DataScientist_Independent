
import operator
from typing import List, Annotated
from typing_extensions import TypedDict

class FinalState(TypedDict):
    messages: List[str] # Might need to change to str
    dataset_info: str
    error: str
    revise: bool
