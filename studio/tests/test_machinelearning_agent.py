import re
from helper_functions import initialize_state_from_csv, define_variables, pretty_print_messages, get_last_ai_message, generate_pdf_report, get_datainfo
from agents import machinelearning_agent, create_code
from classes import State, Configurable
from langchain.schema import AIMessage


state = initialize_state_from_csv()
data = state['dataset_info']
data_info = get_datainfo("dataset.csv")
dic, config = define_variables(thread = 1, loop_limit = 10, data = data, data_info = data_info, name = "ml")

agent = machinelearning_agent(State)

result = agent.invoke(dic, config)
return_message = get_last_ai_message(result['messages'])
print(return_message)
'''
dic, config = define_variables(thread = 1, loop_limit = 12, data = data, data_info = data_info, name = "code", input = return_message)

graph = create_code(State)
print("Graph for Visualization Created! \n")

for chunk in graph.stream(input = dic, config = config):
    pretty_print_messages(chunk)


print(f"This is the chunk \n {chunk} \n")
result = chunk['code_agent']['messages'][-1]['content']

code = re.findall(r"```python\n(.*?)\n```", result, re.DOTALL)
print(f"This is the code ! \n {code} \n")
with open("extracted_code.py", "a", encoding="utf-8") as f:
    f.write("# ---- NEW BLOCK ---- # \n")
    for block in code:
        f.write(block + "\n\n")

generate_pdf_report(result, 'output.pdf')


'''

