

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

with open("schain_agent_ideas.txt", "r", encoding = "utf-8") as f:
      text_data_1 = f.read()

with open("research_agent_ideas.txt", "r", encoding = "utf-8") as f:
      text_data_2 = f.read()

with open("sml_agent_ideas.txt", "r", encoding = "utf-8") as f:
      text_data_3 = f.read()

with open('supervisor_agent_ideas.txt', "r", encoding = "utf-8") as f:
      text_data_4 = f.read()

with open('schain_agent_6ideas.txt', "r", encoding = "utf-8") as f:
      text_data_5 = f.read()

with open('schain_agent_9ideas.txt', "r", encoding = "utf-8") as f:
      text_data_6 = f.read()

with open('sresearch_team.txt', "r", encoding = "utf-8") as f:
      text_data_7 = f.read()







'''
state, config = define_variables(thread = 1, loop_limit = 10, data = data,name = "aggregate", data_info = data_info, data_description= data_description, input = text_data)
ans = aggregate_agent(State).invoke(input = state, config = config)
ideas = get_last_ai_message(ans['messages'])

with open("single_agent_agg_ideas.txt", "a", encoding="utf-8") as f:
        f.write(f" \n------- Next set of Ideas for Base Model -------\n")
        f.write(ideas)


state, config = define_variables(thread = 1, loop_limit = 10, data = data,name = "aggregate", data_info = data_info, data_description= data_description, input = text_data_1)
ans = aggregate_agent(State).invoke(input = state, config = config)
ideas = get_last_ai_message(ans['messages'])

with open("single_agent_agg_ideas.txt", "a", encoding="utf-8") as f:
        f.write(f" \n------- Next set of Ideas for Chain -------\n")
        f.write(ideas)

state, config = define_variables(thread = 1, loop_limit = 10, data = data,name = "aggregate", data_info = data_info, data_description= data_description, input = text_data_2)
ans = aggregate_agent(State).invoke(input = state, config = config)
ideas = get_last_ai_message(ans['messages'])

with open("single_agent_agg_ideas.txt", "a", encoding="utf-8") as f:
        f.write(f" \n------- Next set of Ideas for Research Team -------\n")
        f.write(ideas)

state, config = define_variables(thread = 1, loop_limit = 10, data = data,name = "aggregate", data_info = data_info, data_description= data_description, input = text_data_3)
ans = aggregate_agent(State).invoke(input = state, config = config)
ideas = get_last_ai_message(ans['messages'])

with open("single_agent_agg_ideas.txt", "a", encoding="utf-8") as f:
        f.write(f" \n------- Next set of Ideas for Machine Learning Team -------\n")
        f.write(ideas)

state, config = define_variables(thread = 1, loop_limit = 10, data = data,name = "aggregate", data_info = data_info, data_description= data_description, input = text_data_4)
ans = aggregate_agent(State).invoke(input = state, config = config)
ideas = get_last_ai_message(ans['messages'])

with open("single_agent_agg_ideas.txt", "a", encoding="utf-8") as f:
        f.write(f" \n------- Next set of Ideas for Supervisor Independent Team -------\n")
        f.write(ideas)
'''
state, config = define_variables(thread = 1, loop_limit = 10, data = data,name = "aggregate", data_info = data_info, data_description= data_description, input = text_data_7)
ans = aggregate_agent(State).invoke(input = state, config = config)
ideas = get_last_ai_message(ans['messages'])

with open("single_agent_agg_ideas.txt", "a", encoding="utf-8") as f:
        f.write(f" \n------- Next set of Ideas for Research_Team agents -------\n")
        f.write(ideas)