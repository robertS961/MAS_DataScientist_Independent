import pandas as pd
from io import StringIO

def get_datainfo(data: str) -> str:
    buffer = StringIO()
    df = pd.read_csv(data)
    df.info(buf = buffer)
    df_info_str =buffer.getvalue()
    return df_info_str

ans = get_datainfo('dataset.csv')
print(ans)
