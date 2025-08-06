import pandas as pd
import io

def describe_to_string(data:str) -> str:
    """
    Returns the output of df.describe() as a formatted string (like print would show).
    """
    df = pd.read_csv(data)
    # Use StringIO to capture the output of describe() as a string
    buffer = io.StringIO()
    df.describe().to_string(buf=buffer)
    return buffer.getvalue()

info = describe_to_string('dataset.csv')
print(info)

