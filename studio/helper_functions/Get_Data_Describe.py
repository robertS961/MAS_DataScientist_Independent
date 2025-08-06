import pandas as pd
import io

def data_describe(data:str) -> str:
    """
    Returns the output of df.describe() as a formatted string (like print would show).
    """
    df = pd.read_csv(data)
    # Use StringIO to capture the output of describe() as a string
    buffer = io.StringIO()
    df.describe().to_string(buf=buffer)
    return buffer.getvalue()


