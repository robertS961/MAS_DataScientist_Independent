
from langgraph.prebuilt import create_react_agent
from classes import State

def ploty_agent(state: State):
    prompt = (
        "You are a python plotly expert with over 10 years of experience! \n"
        "You will be given ideas to be executed on a tabular data set. It will be ideas on various statistical, machine learning, and data science ideas. \n"
        "Take these ideas and turn them into interactive and beautiful plotly graphs!\n"
        "Make the colors of the graphs vibrant and clear, avoid using dull and similar colors! \n"
        "Once Plotly Code is written add Html/Javascript code to display all the figures so it can be ran on a server!\n"
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
        model="openai:o4-mini", 
        tools=[],
        name = "Plotly_Agent",
        prompt = prompt
    )
    return plotly_