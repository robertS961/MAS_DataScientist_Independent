from classes import State, Configurable

def define_variables(thread: int, loop_limit: int, data:str):
        #Define the Prompt to insert into the LLM
        prompt= (f"You are given a tabular dataset. Here is the data {data} \n"
                "Your goal to research novel data science and statistic ideas to perform on the data \n"
                f"Please come up with at least 6 unique and novel data science and statistical learning ideas to apply on tabular data based on these columns {data}!\n"
                "Make sure a few ideas come from the statistical learning research agent and a few from the data science research agent \n"
                "The ideas should be clearly labeled and readable \n\n"
                "This data will be passed on for evulation so please make it the best you can! Thank you \n\n"
        )
        #Orginal State Variables for Invoke
        dic: State = {"messages": [
            {
                "role": "user",
                "content": f"{prompt}"
            }
            ],
            "dataset_info": data,
            "error": "",
            "ideas": "",
            'flag': True,
            'revise':False,
            "last_ai_message_content": ""
        }
        #Orginal Configurations for our graph
        config :Configurable = {"thread_id": thread, "recursion_limit": loop_limit}
        return dic, config