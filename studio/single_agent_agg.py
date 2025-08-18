

from helper_functions import initialize_state_from_csv, define_variables, pretty_print_messages, get_datainfo, data_describe, get_last_ai_message, run_code
from agents import create_output_plotly_team, test_single_agent, aggregate_agent
from classes import State, Configurable
import re


state = initialize_state_from_csv()
data = state['dataset_info']
data_info = get_datainfo("dataset.csv")
data_description = data_describe("dataset.csv")
with open("single_agent_ideas.txt", "r", encoding = "utf-8") as f:
    text_data = f.read()
dic, config = define_variables(thread = 1, loop_limit = 10, data = data,name = "aggregate", data_info = data_info, data_description= data_description, input = text_data)

ans = aggregate_agent(State).invoke(input = dic, config = config)
ideas = get_last_ai_message(ans['messages'])

with open("single_agent_agg_ideas.txt", "a", encoding="utf-8") as f:
        f.write(f" \n------- Next set of Ideas -------\n")
        f.write(ideas)


