from classes import State
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage



def make_judge(state:State):
    current_message = state['messages']
    critique_prompt = (
        
        "-You are a senior data scientist. You have over 20 years of experience! \n\n"
        "- You are going to be given data science and statistical learning ideas that can be applied to tabular data! \n\n"
        "Use your experience to state if it uses modern day data science practices \n\n"
        "If there is any room for improvement please suggest it back to the team of data scientist! \n\n"
        "If the response meets the ALL the above criteria, return an empty string"
        "If you find ANY issues with the response, do not return an empty string. Instead, provide specific and constructive feedback which should return as a string."
        "Be detailed in your critique so the team can understand exactly how to improve."

    )

    """Evaluate the assistant's response using a separate judge model."""
    model = ChatOpenAI(model = 'gpt-4o', temperature = 0, max_tokens = 4096)
    response = model.invoke([
        SystemMessage(content=critique_prompt),
        HumanMessage(content=f"Here is the current message{current_message} \n\n. State wether this code meets the protocol. If false return a message so a string with all the improvements. If True return an empty string \n\n")
    ])
    print(f" \n\n These are the eval_result : {response} \n\n")

    if response == "":
        print("✅ Response approved by judge")
        return
    else:
        # Otherwise, return the judge's critique as a new user message
        print("⚠️ Judge requested improvements")
        return {"feedback": [{"role": "user", "content": str(response.content)}]}