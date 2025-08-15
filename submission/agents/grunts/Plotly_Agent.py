
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
        "Make sure the code runs without errors and bugs! \n"
        "Thank you! \n"
    )
    '''
    prompt = (
        "You are a python plotly expert with over 10 years of experience! \n"
        "You will be given code to be executed on a tabular data set. It will display graphs on various statistical, machine learning, and data science ideas. \n" 
        "You will take this python code and turn it into interactive and beautiful plotly graphs! \n"
        "Keep the same graphs and displays, just turn them into plotly graphs \n"
        "Then add HTML/Javascript Code at the end to display all the figures and print it to output.html file so it can be run on online servers! \n"
        "There could be double pairs of columns that are referenced in the code, so make sure Aggregate them together for line plots,bar plots, etc. to display clear figures! \n"
        "Please respond with the full python code with plotly and the html/javascript code at the end! \n"
        "Make sure the code runs without errors and bugs! \n"
        "Thank you! \n"
    )
    '''
    #websearch = WebSearch()
    plotly_ = create_react_agent(
        model= llm, # Change it back to "openai:o4-mini",
        tools=[],
        name = "Plotly_Agent",
        prompt = prompt
    )
    return plotly_