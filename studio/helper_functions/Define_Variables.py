from classes import State, Configurable

def define_variables(thread: int, loop_limit: int, data:str, data_info:str , name:str = "research", input:str = "", code:str= "" ):
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
        "You should take these ideas and generate python code for each idea that creates beautiful graphs and visualizations \n"
        "Even if the result can be printed, please instead make a visualization to show the result!\n "
        "The code should be simple and clean \n"
        "It should contain zero bugs and run without issues. Please test you code before returning it\n"
        "Make sure in the code you replace NaN values first based on the data info\n"
        "Thank you! Lets COOK, return this python Code! \n"
        )
    elif name == "ml":
        prompt = (
            f"You are given a tabular dataset. Here is the data \n {data} \n"
            "Your goal is to come up with novel machine learning techniques to apply to the above columns \n"
            "Return a list of ideas in a neat format! \n"
            "Thank you !"
        )
    elif name == "fix-vis":
        prompt = (
            f"You are given code for a pdf.  Here it is \n {code} \n"
            "Your goal is to evulate this pdf and make sure each visualization is clear to understand!\n"
            "Focus on clear titles, axis, correct sizes figures/designs, colors, and keys for uncertain figures \n"
            "Thank you !"
        )
    elif name == "plotly":
        prompt = (
            f"You are given ideas to perform on a tabular data set. Here are the ideas \n {input} \n"
            "Pick the best 5 ideas from the list! \n"
            f"These ideas use column names from the data set which has the following columns \n {data} \n"
            f"The column types are \n {data_info} \n, so make sure to take out NaN values and fill them with the mean or median \n"
            "You shall take these ideas and turn them into python code that creates beautiful graphs and visualizations \n"
            "These results hould be displayed with plotly and should be interactive \n"
            "Lastly take the code and add HTML/Javascript Code at the end to display all the figures and print it to output.html file so it can be run on online servers! \n"
            "Please test the code and make sure it runs without errors and bugs! \n"
            "Return all the code only! Thank you!"
        )
    '''
    elif name == "plotly":
        prompt = (
            f"Here is the code for a pdf.  Here it is \n {code} \n"
            f"The code is based on a tabular data set. Here are the data columns \n{data} \n"
            f"The columns are \n {data_info} \n"
            "Your goal is to turn this code into interactive and beautiful plotly graphs! \n"
            "Keep all the orginal graphs and displays, just turn them into plotly graphs \n"
            "Also add html/javascript code at the end to display all the figures and print it to output.html file so it can be run on online servers! \n"
            "Make sure the code runs without errors and bugs! \n"
            "Thank you! \n"
        )
    '''


    dic: State = {'messages': [
        {
            "role": "user",
            "content": f"{prompt}"
        }
    ],
        "dataset_info": data,
        "data_info": data_info,
        'revise': False,
        "ideas": input,
        "code": code,
        "errors": "",
    }
    #Orginal State Variables for Invoke
    
    #Orginal Configurations for our graph
    config :Configurable = {"thread_id": thread, "recursion_limit": loop_limit}
    return dic, config