from classes import State
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage



def make_judge(state:State):
    current_message = state['messages']
    data = state['dataset_info']
    critique_prompt = (
        
        "-You are a senior data scientist. You have over 20 years of experience! \n\n"
        f"- You are going to be given data science and statistical learning ideas that can be applied to tabular data shown here {data}! \n\n"
        "Use your experience to state if these ideas are sufficient for the data \n\n"
        "If any of them aren't sufficient then change the idea to make it sufficient and suggest it back\n\n"
        "Only suggest back new ideas based off the columns of the dataset , or modified original ideas from the message given to you \n\n"
        "You should return a list of data science and statistical learning ideas like it was given to you except some of them are new or modified! \n\n"
    )

    """Evaluate the assistant's response using a separate judge model."""
    model = ChatOpenAI(model = 'gpt-4o', temperature = 0, max_tokens = 4096)
    response = model.invoke([
        SystemMessage(content=critique_prompt),
        HumanMessage(content=f"Here is the current message{current_message} \n\n. State wether the message meets the protocol. If false return a message so a string with all the improvements. If True return the string True \n\n")
    ])
    print(f"\n\nThis is the content {response.content} \n\n")

    if response.content == "True":
        print("✅ Response approved by judge")
        return {'revise': True}
    else:
        # Otherwise, return the judge's critique as a new user message
        print("⚠️ Judge requested improvements.")
        return {"messages": [{"role": "system", "content": str(response.content)}]}