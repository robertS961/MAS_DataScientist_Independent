from langgraph.prebuilt import create_react_agent
from classes import Temp, State
from helper_functions import get_llm

def aggregate_agent(state: State):
    llm = get_llm(model = 'o4-mini')
    ideas = state['ideas']
    prompt = (
        f"You will be given a text file in the form of a string of statistic and data science ideas \n {ideas} \n"
        "Aggregate the most common concepts lets say in the range of 5-10. You can be liberal with the ideas like linear regression, heatmap, correlation plot (the use of the columns doesn't need to be the same)\n"
        "Return the list of the most common ideas and the count of each of them \n"
    )
    #websearch = WebSearch()
    aggregate_a = create_react_agent(
        model=llm, # Change it back to "openai:o4-mini",
        tools=[],
        name = "Aggregate_Agent",
        prompt = prompt
    )
    return aggregate_a