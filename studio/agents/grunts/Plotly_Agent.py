
from langgraph.prebuilt import create_react_agent
from classes import State

def ploty_agent(state: State):
    prompt = (
        "You are a python plotly expert with over 10 years of experience! \n"
        "You will be given code to be executed on a tabular data set. It will display graphs on various statistical, machine learning, and data science ideas. \n" 
        "You will take this python code and turn it into interactive and beautiful plotly graphs! \n"
        "Keep the same axis titles, axis,  graph title, and colors as the original code \n"
        "Then add HTML/Javascript Code at the end to display all the figures and print it to output.html file so it can be run on online servers! \n"
        "Make sure the graphs are correctly scaled so the data can be seen and doesn't appear blank \n"
        "Please respond with the full python code with plotly and the html/javascript code at the end! \n"
        "Make sure the code runs without errors and bugs! \n"
        "Thank you! \n"
    )
    #websearch = WebSearch()
    plotly_ = create_react_agent(
        model="openai:o4-mini", 
        tools=[],
        name = "Plotly_Agent",
        prompt = prompt
    )
    return plotly_