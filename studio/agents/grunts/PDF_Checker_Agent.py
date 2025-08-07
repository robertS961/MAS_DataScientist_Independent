
from langgraph.prebuilt import create_react_agent
from classes import State
from helper_functions import get_llm

def pdf_checker_agent(state: State):
    llm = get_llm()
    prompt = (
        "You are a data visualization expert with over 20 years of experience! \n" 
        "Your goal is to improve data visualizations that are written in python code! \n "
        "You want to check the code and see if there is anyway to improve the visualizations!\n "
        "Check all the to be created images produced by the code especially the titles, axis, colors, keys, blank plots, and clarity of the images \n"
        "If any of the above is unclear or the figure itself then fix the code. Also if any axis have long names then shorter them as each figure and axis/titles must fit on a 11 inch screen!\n"
        "If there is any print statements then incorporate them into the figure. This will add be turned into a pdf and the print statements won't show!"
        "Return all the previous code except where you fixed it. Just edit that into the code. \n"
        "Thank you! \n"
    )
    #websearch = WebSearch()
    ml_agent = create_react_agent(
        model=llm, #Might need to change this to gpt-4o mini
        tools=[],
        name = "PDF_Checker_Agent",
        prompt = prompt
    )
    return ml_agent