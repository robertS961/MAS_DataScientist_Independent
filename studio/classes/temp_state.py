
import operator
from typing import List, Annotated
from typing_extensions import TypedDict

class TempState(TypedDict):
    messages: Annotated[list, operator.add ] # Might need to change to str
    remaining_steps: int
    dataset_info: str
