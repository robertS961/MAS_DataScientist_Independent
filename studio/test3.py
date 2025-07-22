from helper_functions import initialize_state_from_csv, define_variables, pretty_print_messages
from agents import chain, supervisor, create_search_nodes, supervisor_team
from classes import State, Configurable
from langchain.schema import AIMessage

state = initialize_state_from_csv()
data = state['dataset_info']
dic, config = define_variables(thread = 1, loop_limit = 6, data = data)

new_graph = supervisor(state, agent = 1)

for chunk in new_graph.stream(input = dic, config= config):
    pretty_print_messages(chunk)

def get_last_ai_message(messages):
            # Iterate backwards to find the last AIMessage
            for msg in reversed(messages):
                if isinstance(msg, AIMessage):
                    return msg.content  # or return the whole msg if you want more than content
            return None  # if no AIMessage found

        # Example usage
print(f"\n\n This is the last chunk : {chunk} \n\n")
last_ai_content = get_last_ai_message(chunk['supervisor']['messages'])
print(f"\n\n This is the last_ai_content {last_ai_content} \n\n")

