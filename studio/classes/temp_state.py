

from typing import List
from typing_extensions import TypedDict

class TempState(TypedDict):
    messages: List[str] # Might need to change to str
    remaining_steps: int
