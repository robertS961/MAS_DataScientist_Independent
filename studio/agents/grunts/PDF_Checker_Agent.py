
from langgraph.prebuilt import create_react_agent
from classes import State

def pdf_checker_agent(state: State):
    prompt = (
        "You are a data visualization expert with over 20 years of experience! \n" 
        "Your goal is to imporve data visualizations that are written in python code! \n "
        "You want to check the code and see if there is anyway to improve the visualizations!\n "
        "Check all the to be created images produced by the code especially the titles, axis, and clarity of the images \n"
        "If any of the above is unclear or the figure itself then fix the code.\n"
        "Return all the previous code except where you fixed it. Just edit that into the code. \n"
        "Thank you! \n"
    )
    #websearch = WebSearch()
    ml_agent = create_react_agent(
        model="openai:o4-mini", 
        tools=[],
        name = "PDF_Checker_Agent",
        prompt = prompt
    )
    return ml_agent