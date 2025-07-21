from agents import supervisor
from dotenv import load_dotenv

load_dotenv()


supervisor = supervisor()
print("Supervisor is compiled \n\n")

for chunk in supervisor.stream(
    {
    "messages": [
        {
            "role": "user",
            "content": "what's the combined headcount of the FAANG companies in 2024?"
        }
    ]
}
):
    print(chunk)
    print("\n ------ \n")


