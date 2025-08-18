
from langgraph.prebuilt import create_react_agent
from classes import State
from helper_functions import get_llm

def test_single_agent(state: State):
    llm = get_llm(model = 'o4-mini')
    code = state['code']
    dataset_info = state['dataset_info']
    prompt = f"Please generate 5 ideas to visualize insights from the dataset by using data science, statistical learning, or machine learning, use the column names in the output: {dataset_info} \n"
    
    t_s_a = create_react_agent(
        model=llm,
        tools=[],
        name = "Test_Single_Agent",
        prompt = prompt
    )
    return t_s_a