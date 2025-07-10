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