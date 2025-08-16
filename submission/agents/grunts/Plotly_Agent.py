
from langgraph.prebuilt import create_react_agent
from classes import State
from helper_functions import get_llm

def ploty_agent(state: State):
    llm = get_llm(model = 'gpt-5-mini')
    prompt = (
        "You are a python plotly expert with over 10 years of experience! \n"
        "You will be given ideas to be executed on a tabular data set. It will be ideas on various statistical, machine learning, and data science ideas. \n"
        "Take these ideas and turn them into interactive and beautiful plotly graphs!\n"
        "Make sure to include a narrative description of the graphs and what they represent! This narrative should be at least a pharagraph long and should explain the importance of the graph for other data scientist!\n"
        "Make sure the narrative flows from each graph unto the next. Like reading a paper! \n"
        "Make the colors of the graphs vibrant and clear, avoid using dull and similar colors! \n"
        "Once Plotly Code is written add Html/Javascript code to display all the figures so it can be ran on a server!\n"
        "For the HTML/Javascript code, make sure to include the Plotly JS via CDN directly inside each div for better compatibility! Thus include_plotlyjs='cdn' for each div, NOT FALSE(This is outdated!) \n"
        "Do NOT use '<script src=https://cdn.plot.ly/plotly-latest.min.js></script>' in the HTML code, instead use include_plotlyjs='cdn' in the plot function! \n"
        "Make sure the code runs without errors and bugs! Return the code like ```python #CODE HERE ```\n"
        "Thank you! \n"
    )
    #websearch = WebSearch()
    plotly_ = create_react_agent(
        model= llm, # Change it back to "openai:o4-mini",
        tools=[],
        name = "Plotly_Agent",
        prompt = prompt
    )
    return plotly_