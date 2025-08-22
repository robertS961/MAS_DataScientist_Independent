from helper_functions import initialize_state_from_csv, define_variables, pretty_print_messages, get_datainfo, data_describe, get_last_ai_message, run_code
from agents import create_research_team
from classes import State, Configurable
import re


state = initialize_state_from_csv()
data = state['dataset_info']
data_info = get_datainfo("dataset.csv")
data_description = data_describe("dataset.csv")
dic, config = define_variables(thread = 1, loop_limit = 10, data = data,name = "single_agent", data_info = data_info, data_description= data_description)
dic['revise'] = True

for i in range(48, 50):
    ans = create_research_team(State).compile().invoke(input = dic, config =config)
    ideas = get_last_ai_message(ans['messages'])
    with open("sresearch_team.txt", "a", encoding="utf-8") as f:
        f.write(f" \n------- Next set of Ideas for {i}-------\n")
        f.write(ideas)
