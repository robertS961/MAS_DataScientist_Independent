from classes import State, Configurable

def define_variables(thread: int, loop_limit: int, data:str, data_info:str , name:str = "research", input:str = None ):
    if name == "research":
    #Define the Prompt to insert into the LLM
        prompt= (f"You are given a tabular dataset. Here is the data {data} \n"
                "Your goal to research novel data science and statistic ideas to perform on the data \n"
                f"Please come up with at least 6 unique and novel data science and statistical learning ideas to apply on tabular data based on these columns {data}!\n"
                "Make sure a few ideas come from the statistical learning research agent and a few from the data science research agent \n"
                "The ideas should be clearly labeled and readable \n\n"
                "This data will be passed on for evulation so please make it the best you can! Thank you \n\n"
        )
    elif name == "code":
        prompt = ("You are give previous researched data science, statistical learning, and other ideas from previous research agents! \n"
        f"The ideas are listed here \n {input} \n"
        f"These ideas use column names from the data set which has the following columns \n {data} \n"
        "You should take these ideas and generate python code for each ideas \n"
        "The code should be simple and clean \n"
        "It should contain zero bugs and run without issues. Please test you code before returning it\n"
        "Thank you! Lets COOK, return this python Code! \n"
        )
    elif name == "ml":
        prompt = (
            f"You are given a tabular dataset. Here is the data \n {data} \n"
            "Your goal is to come up with novel machine learning techniques to apply to the above columns \n"
            "Return a list of ideas in a neat format! \n"
            "Thank you !"
        )

    dic: State = {'messages': [
        {
            "role": "user",
            "content": f"{prompt}"
        }
    ],
        "dataset_info": data,
        "data_info": data_info,
        'revise': False,
        "ideas": "",
        "code": "",
        "errors": "",
    }
    #Orginal State Variables for Invoke
    
    #Orginal Configurations for our graph
    config :Configurable = {"thread_id": thread, "recursion_limit": loop_limit}
    return dic, config