import csv
from agents import supervisor
from dotenv import load_dotenv
from helper_functions import pretty_print_messages
from classes import State, TempState, Configurable

load_dotenv()

def initialize_state_from_csv() -> dict:
        # The dataset should be first input to the agentic configuration, and it should be generalizable to any dataset
        path = "./dataset.csv"
        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=',')
            header = next(reader)
            first_row = next(reader)

        attributes = ", ".join(header)
        example_values = "\t".join(first_row)

        # if the final output contains Vega-Lite codes, then use the github hosted dataset rather than the local dataset
        # file_name = "https://raw.githubusercontent.com/demoPlz/mini-template/main/studio/dataset.csv"
        file_name = "dataset.csv"
        example_input = f"""
            There is a dataset, there are the following {len(header)} attributes:
            {attributes}
            Name of csv file is {file_name}
        """
        state = {
            "dataset_info": str(example_input)
        }

        return state
supervisor = supervisor(State)
state = initialize_state_from_csv()

print("Supervisor is compiled \n\n")
data = state['dataset_info']
print(f"\n\nhis is the data {data} \n\n")

for chunk in supervisor.stream(
    input = {
    "messages": [
        {
            "role": "user",
            "content": (f"Please come up with at least 6 unique and novel data science and statistical learning ideas to apply on tabular data based on these columns {data}!\n"
            "Make sure a few ideas come from the statistical learning research agent and a few from the data science research agent \n"
        )}],
    "dataset_info": data,   
}, config = {"thread_id": 1, "recursion_limit": 6} 
):
    pretty_print_messages(chunk)


