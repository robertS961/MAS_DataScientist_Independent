from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

import csv
from helpers import get_llm
from report_html import generate_html_report
from report_pdf import generate_pdf_report

class State(TypedDict):
    message: str
    dataset_info: str

def generate_msg_node(version: int):
    def _generate(state: State):
        dataset_info = state["dataset_info"]
        previous_message = state.get("message", "")
        
        expert_intro = {
            1: "You are an expert data scientist specializing in exploratory data analysis. Your job is to identify impactful trends from datasets.",
            2: "You are a senior analyst reviewing the earlier results and adding deeper statistical insights and visualizations.",
            3: "You are a final reviewer, refining the insights and making sure the Python visualizations are well-structured and insightful.",
        }[version]

        prompt = (
            f"{expert_intro}\n\n"
            f"Here is the current state of the analysis:\n{previous_message}\n\n"
            f"The dataset description is:\n{dataset_info}\n\n"
            "Please improve and extend the Python code and narrative to make the analysis deeper and more impactful. Output should be Python code with graphs and written insights."
        )

        llm = get_llm(temperature=0, max_tokens=4096)
        response = llm.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content="Continue and refine the analysis.")
        ])
        return {"message": response}
    return _generate


def create_workflow():
    # create the agentic workflow using LangGraph
    builder = StateGraph(State)

    #Add Each Expert Node in a Chain
    builder.add_node("generate_msg_1", generate_msg_node(1))
    builder.add_node("generate_msg_2", generate_msg_node(2))
    builder.add_node("generate_msg_3", generate_msg_node(3))

    #Structure the nodes together
    builder.add_edge(START, 'generate_msg_1')
    builder.add_edge('generate_msg_1', 'generate_msg_2')
    builder.add_edge('generate_msg_2', 'generate_msg_3')
    builder.add_edge('generate_msg_3', END)

    return builder.compile()

class Agent:
    def __init__(self):
        self.workflow = None

    def initialize(self):
        self.workflow = create_workflow()

    def initialize_state_from_csv(self) -> dict:
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
    def decode_output(self, output: dict):
        # if the final output contains Vega-Lite codes, then use generate_html_report
        # if the final output contains Python codes, then use generate_pdf_report

        generate_pdf_report(output, "output.pdf")
        # generate_html_report(output, "output.html")
    def process(self):

        if self.workflow is None:
            raise RuntimeError("Agent not initialised. Call initialize() first.")
        
        # initialize the state & read the dataset
        state = self.initialize_state_from_csv()
        

        # invoke the workflow
        output_state = self.workflow.invoke(state)

        # flatten the output
        """
        def _flatten(value):
            return getattr(value, "content", value)
        result = {k: _flatten(v) for k, v in output_state.items()}
        """
        # Flatten the output since Langraph outputs AIMessage(content = " content here")
        def _flatten(value):
            return getattr(value, "content", value)

        result = {key: _flatten(field_value) for key, field_value in output_state.items()}
        print("----- Generated Code Output -----")
        print(result['message'])  
        print("---------------------------------")

        # decode the output
        self.decode_output(result)

        # return the result
        return result