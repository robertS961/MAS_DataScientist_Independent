from classes import State
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from helper_functions import get_last_ai_message, get_llm


def vis_narrative(state:State):
    llm = get_llm(model = 'gpt-5-mini')
    data = state['dataset_info']
    data_info = state['data_info']
    code = state['code'] 
    ideas = state['ideas']
    error = state['errors']
    prompt = (
        "You are a Python Coding Expert with 20 years of experience coding in python! Your job is to fix broken code! \n"
        f"Code has already been generated for these several ideas \n {ideas} \n\n "
        f"The following code was created for those above ideas \n {code}\n  : based on these data columns \n {data}\n\n"
        f"However there were errors in the code. Here is the error: \n {error}\n\n"
        f" The code is going to be run on a dataset with this information \n {data_info} \n"
        "Please fix the code and make sure it runs correctly!\n\n"
        "Return the fixed code in place of the faulty code in what was given to you along with the previous code that was correct \n"
        "You should return the code with the bugs fixed! Do not return just fixed code! Return all the old code with the fixes made!\n"
        "The return code should ALL be sandwhiched between ONE SET of triple single quotes like ''' CODE ''' \n"
        "Lastly make sure the number of code ideas/blocks given to you is returned with the same amount. 5 code block ideas in then 5 out! \n"
        "Thank you \n"
    )
    
    vis_nar = create_react_agent(
        model = llm,
        tools = [],
        prompt = prompt,
        name = 'vis_narrative',
        checkpointer=InMemorySaver()
    )
    return vis_nar