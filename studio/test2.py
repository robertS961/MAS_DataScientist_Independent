
from helper_functions import initialize_state_from_csv, define_variables, pretty_print_messages
from agents import chain
from classes import State, Configurable
from langchain.schema import AIMessage

state = initialize_state_from_csv()
data = state['dataset_info']
dic, config = define_variables(thread = 1, loop_limit = 6, data = data)


compiled_graph = chain(3, state)
print(dic)
for chunk in compiled_graph.stream(input = dic, config= config):
    pretty_print_messages(chunk)

print(f"\n\n This is the final chunk: {chunk}\n\n")

def get_last_ai_message(messages):
            # Iterate backwards to find the last AIMessage
            for msg in reversed(messages):
                if isinstance(msg, AIMessage):
                    return msg.content  # or return the whole msg if you want more than content
            return None  # if no AIMessage found

        # Example usage
last_ai_content = get_last_ai_message(chunk['search_node_2']['messages'])
print(f"\n\n This is the last_ai_content {last_ai_content} \n\n")




