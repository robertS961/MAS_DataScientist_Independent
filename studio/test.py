from agents import create_supervisor


supervisor = create_supervisor()
print("Supervisor is compiled \n\n")

for chunk in supervisor.stream(
    {"messages": [("user", "when is Taylor Swift's next tour?")]}
):
    print(chunk)


