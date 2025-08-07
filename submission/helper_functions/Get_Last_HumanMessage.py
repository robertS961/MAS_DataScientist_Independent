
from langchain.schema import HumanMessage

def get_last_human_message(messages):
    # Iterate backwards to find the last AIMessage
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content  # or return the whole msg if you want more than content
    return None  # if no AIMessage found