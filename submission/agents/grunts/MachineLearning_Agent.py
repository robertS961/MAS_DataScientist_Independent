
from langgraph.prebuilt import create_react_agent
from classes import State
from helper_functions import get_llm

def machinelearning_agent(state: State):
    llm = get_llm(model = 'gpt-5')
    data = state['dataset_info']
    data_info = state['data_info']
    prompt = (
        "You are a machine learning expert with over 25 years of experience! \n" 
        f"Your goal is use your own knowledge to generate machine learning algorithms on this tabular data \n {data} \n "
        f"The information from the dataframe reveals this about the above columns \n {data_info} \n"
        "Use the above information on methods such as linear regression, logistic regression, boosting, random forest, clustering , k nearest neighbors, support vector machine, etc. \n"
        "Use the above ideas with your knowledge on the above data set. Return a list of ideas to apply. Label each idea!\n"
        "Make sure the ideas corretly use the columns! Like don't use linear regression on strings etc. \n"
        "Thank you! \n"
    )
    #websearch = WebSearch()
    ml_agent = create_react_agent(
        model=llm, 
        tools=[],
        name = "MachineLearning_Agent",
        prompt = prompt
    )
    return ml_agent