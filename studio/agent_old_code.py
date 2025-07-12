"""
def generate_team(builder):
    team_size = 3
    # Create the Team and connections between players
    for player in range(team_size):
        curr_player = f"generate_msg_{player}"
        func = generate_msg_node(player)
        builder.add_node(curr_player, func )
        builder.add_edge(START, curr_player)
    
    #Creat the Assistant - dummy variable, does nothing
    assistant, coach = "generate_msg_assistant", 'generate_msg_coach'
    builder.add_node( assistant, generate_msg_assistant) # Add this function
    builder.add_node(coach, generate_msg_assistant) # Add this function
    for player in range(team_size):
        curr_player = f"generate_msg_{player}"
        builder.add_edge(curr_player, assistant)
    builder.add_edge(assistant, coach)
    builder.add_edge(coach, END)


    # Leys the lads practice. Train them up with great drills
    practice_length = 4
    for quarter in range(practice_length):
        drill = np.random.binomial(1, p = 0.5, size = team_size -1)
        for player_num, choice in enumerate(drill):
            # Choice is either 0 or 1. 1 means the off player comes to drill. 0 means the off player doesn't come
            curr_player, rotating_player = f"generate_msg_{player_num}", f"generate_msg_{team_size - 1}"
            if choice: 
                builder.add_edge(curr_player , rotating_player)
                builder.add_edge(rotating_player ,curr_player )
            else:
                builder.add_edge(curr_player, curr_player)
    
    # Let the Coach see the results
    
    builder.add_edge(coach, END)

"""

'''
def generate_msg_supervisor(state: State):
    dataset_info = state["dataset_info"]
    previous_message = state.get("message", "")

    supervisor_prompt = (
        "You are a senior data science supervisor tasked with reviewing a draft Python report "
        "that includes code and narrative insights based on a dataset. Your goal is to review the report "
        "carefully and improve it by:\n"
        "- Clarifying vague statements\n"
        "- Filling in missing explanations for code\n"
        "- Adding any relevant insights that were missed\n"
        "- Make sure there isn't any reindexing on an axis with duplicate labels \n"
        "- Make sure no objects have attriute dtype for numpy \n"
        "- Make sure there isn't any errors in the code and it is runable, especially after you edit it \n"
        "- Ensuring the Python code is complete, clean, and runs correctly\n\n"
        "Here is the current draft:\n"
        f"{previous_message}\n\n"
        "Here is the dataset description:\n"
        f"{dataset_info}\n\n"
        "Please return the improved version of the full Python report with narrative and visualizations."
    )

    llm = get_llm(temperature=0, max_tokens=4096)
    response = llm.invoke([
        SystemMessage(content=supervisor_prompt),
        HumanMessage(content="Please review and revise the report. Make sure the code runs clean. They should be zero bugs! Thank you")
    ])
    return {"message": response}
'''


def generate_msg_node(version: int):
    def _generate(state: State):
        #Safety Check for Infinite Loops
        '''iteration = state.get("iteration", 0)
        max_iterations = 1
        if iteration >= max_iterations:
            return state'''
        
        # if the prompt is to generate Vega-Lite charts, then specify in sys_prompt and use generate_html_report()
        # sys_prompt = f"Please generate Vega-Lite graphs to visualize insights from the dataset, output should be graphs and narrative: {dataset_info}"
        dataset_info = state["dataset_info"]
        previous_message = state.get("message", "")
        
        expert_intro = {
            1: "You are an expert data scientist specializing in exploratory data analysis. Your job is to identify impactful trends from datasets. Focus on modern day statistical learning techniques like regression, clustering, correlation, p values, t test etc. Look for variable relationships",
            2: "You are a senior analyst reviewing the earlier results and adding deeper statistical insights and visualizations. Feel free to also dive into individual variables, discover distributions or connections to compounding other variables",
            3: "You are a statistician, find unique and interesting ideas backed by statistics in the data",
        }[(version % 3) + 1]

        prompt = (
            f"{expert_intro}\n\n"
            f"Here is the current state of the analysis:\n{previous_message}\n\n"
            f"The dataset description is:\n{dataset_info}\n\n"
            "Please come up with several unique ideas to perform on the data. The ideas should be codeable and produce visualization. Do not code them. Just state them. \n"
            "You should return a list of 5 - 10 unique narratives about the data that can be tested with data visualizations. \n\n "
        )

        llm = get_llm(temperature=0, max_tokens=4096)
        response = llm.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content="Please generate and show the ideas to me. The output must be very specific. I want each ideas on its own line. So '\n' in between them. Nothing else, thank you")
        ])
        
        # ans = getattr(response, "content", response)
        return {
        "messages":  [response.content]
         #"iterations": iteration + 1
        }
    return _generate


def generate_msg_coach(state: State) -> State:
    # Get the list of messages from all player nodes
    messages = state["messages"]  # This is now a list of strings
    dataset_info = state["dataset_info"]

    # Collect all ideas from each message
    all_ideas = []
    for msg in messages:
        ideas = msg.split("\n")
        cleaned = [idea.strip() for idea in ideas if idea.strip()]
        all_ideas.extend(cleaned)

    prompt = (
        "You are a senior data scientist leading a panel of experts. "
        "Given the following list of ideas generated by your team, select the top 8 ideas that are:\n"
        "- Novel and unique\n"
        "- Testable with data\n"
        "- Capable of producing beautiful data visualizations\n\n"
        "Ideas:\n"
        + "\n".join(f"{i+1}. {idea}" for i, idea in enumerate(all_ideas)) +
        "\n\nPlease return the selected top 8 ideas that a panel of data scientists would consistently pick."
    )
    human_prompt = (
        "Select the best 8 ideas as requested. Please list one idea per line, with no extra blank lines. Only Select 8 thank you\n"
        " In addition the file should only contain python code. Nothing else. So if there is weird titles or extra content, please remove that. Thank you \n"
    )

    llm = get_llm(temperature=0, max_tokens=4096)
    response = llm.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content= human_prompt),
    ])
    # "final_message": getattr(response, "content", response),
    return {"final_message": response.content}

def generate_msg_GM(state: State) -> State:
    dataset_info = state.get("dataset_info", "")
    ideas_text = state.get("final_message", "")

    prompt = (
        "You are a senior data scientist writing Python code for a report. "
        "Below is a list of 8 data visualization ideas that were selected by a panel of top data scientist.\n\n"
        "Dataset Info:\n"
        f"{dataset_info}\n\n"
        "Ideas:\n"
        f"{ideas_text}\n\n"
        "Your job is to write Python code that implements each idea. For each idea:\n"
        "- Write clean, well-documented Python code\n"
        "- Use libraries like pandas, matplotlib, seaborn, or plotly\n"
        "- Use beautiful and effective visualizations\n"
        "- Structure your answer as:\n"
        "- Test your code for each idea and make sure it runs without bugs"
        "Make sure that visualizations are clear and insightful"
    )

    llm = get_llm(temperature=0.3, max_tokens=4096)
    response = llm.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content="Please generate the code for each idea with explanations. Make sure you test the code and it runs without bugs or errors. Double, triple check that please! Thank you")
    ])
    # "message": getattr(response, "content", response),
    return {
        "report": response.content
        # "iteration": state.get("iteration", 0) + 1
    }

def generate_team(builder):
    team_size = 3

    # Create the player connections
    for player in range(team_size):
        curr_player = f"generate_msg_{player}"
        func = generate_msg_node(player)
        builder.add_node(curr_player, func)
        builder.add_edge(START, curr_player)
    
    #Create the Coach to collect all the players and pick the best ones
    coach = "generate_msg_coach"
    builder.add_node(coach, generate_msg_coach)
    for player in range(team_size):
        builder.add_edge(f"generate_msg_{player}", coach)
    
    #Create the General Manager to Aggregate the Final Players into a Cohesive Team
    general_manager = "generate_msg_GM"
    builder.add_node(general_manager, generate_msg_GM)
    builder.add_edge(coach, general_manager)
    builder.add_edge(general_manager, END)
    return builder