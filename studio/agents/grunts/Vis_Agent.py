from classes import State
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from helper_functions import get_last_ai_message


def vis_a(state:State):
    data = state['dataset_info']
    data_info = state['data_info']
    code = state['code'] 
    if not state['revise']: 
        state['ideas'] = get_last_ai_message(state['messages'])
        state['revise'] = True
        ideas = state['ideas']
        prompt = (
            "You are a Python visualization expert. You generate stunning visualizations using matplotlib, seaborn, plotly, or any libraries you can think of!.\n\n"
            "We have already created several ideas. Your objective is to take the ideas and code them into meaningful and beauitful data visualizations! \n\n"
            f"The dataset columns are as follows: \n{data}\n\n"
            f"The ideas are as follows: \n{ideas}\n\n"
            f"The data columns are the following Dtype and contain the following Non-Null Count \n{data_info}\n"
            "Please first fill in all the NaN values. Do this using the most modern methods. Do not drop them! \n"
            "Next write PYthon code that creates meaningful and aesthetic data visualizations.\n"
            "Check the code to make sure the dtypes are correct, and change any categorical variables if needed so that the code runs smoothly! \n"
            "Make sure the number of plots you create matches the number of ideas, if it is impossible then create more data science ideas from the columns! \n\n"
            "Make sure the Python code runs and there isn't any bugs. Also write it in simple python code so the users can easily understand it \n\n"
            "Also Sklearn is depreciated use scikit-learn! \n\n"
        )  
    else:
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
            "You should return the code with the bugs fixed! Do not return just fixed code! Return all the old code with the fixes made! Thank you \n"
        )
    
    vis_agent = create_react_agent(
        model = "openai:gpt-4o",
        tools = [],
        prompt = prompt,
        name = 'vis_agent',
        checkpointer=InMemorySaver()
    )
    return vis_agent