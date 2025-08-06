
import pandas as pd
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import statsmodels.api as sm
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot

# 1. Load and preprocess data
df = pd.read_csv("dataset.csv")

# Numeric columns: fill NaN with median
num_cols = [
    "FirstPage", "LastPage", "AminerCitationCount", "CitationCount_CrossRef",
    "PubsCited_CrossRef", "Downloads_Xplore"
]
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

# Categorical columns: fill NaN with "Unknown"
cat_cols = ["Conference", "PaperType", "AuthorKeywords", "Award", "Abstract"]
for col in cat_cols:
    df[col] = df[col].fillna("Unknown")

# Binary flag for Award
df["AwardFlag"] = np.where(df["Award"] != "Unknown", 1, 0)

# Year should be int
df["Year"] = df["Year"].astype(int)
min_year, max_year = df["Year"].min(), df["Year"].max()

# 2. Conference Impact Analysis: top 5 by citations
conf_agg = (
    df.groupby(["Year", "Conference"])
      .agg(Citations=("CitationCount_CrossRef", "sum"),
           Downloads=("Downloads_Xplore", "sum"))
      .reset_index()
)
top5 = (
    conf_agg.groupby("Conference")["Citations"]
            .sum()
            .nlargest(5)
            .index
            .tolist()
)

print("Top 5 Conferences by Total Citations:")
print(top5)
conf_plot = conf_agg[conf_agg["Conference"].isin(top5)]
print("conf_plot head:\n", conf_plot.head())
print("conf_plot shape:", conf_plot.shape)
print("conf_plot columns:", conf_plot.columns.tolist())

# Figure 1: Citations over time with range slider & selector
fig1 = px.line(
    conf_plot,
    x="Year", y="Citations", color="Conference",
    title="Top 5 Conferences: Citation Trends Over Years",
    markers=True,
    color_discrete_sequence=px.colors.qualitative.Bold
)
print("Number of traces in fig1:", len(fig1.data))
for trace in fig1.data:
     print(f"Trace name: {trace.name}, Points: {len(trace.x)}")
fig1.update_traces(
    hovertemplate="Conference: %{name}<br>Year: %{x}<br>Citations: %{y}<extra></extra>"
)
fig1.update_layout(
    template="plotly_dark",
    xaxis=dict(
        title="Year",
        range=[min_year - 1, max_year + 1],
        rangeselector=dict(
            buttons=[
                dict(count=5, label="5Y", step="year", stepmode="backward"),
                dict(count=10, label="10Y", step="year", stepmode="backward"),
                dict(step="all")
            ]
        ),
        rangeslider=dict(visible=True)
    ),
    yaxis=dict(title="Total Citations", rangemode="tozero")
)

print("Top 5 Conferences by Total Citations:")
print(top5)
print("Number of rows in conf_plot:", len(conf_plot))

plot(fig1, filename="test.html", auto_open=True)