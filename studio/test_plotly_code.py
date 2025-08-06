# ---- NEW BLOCK ---- # 
# enhanced_plotly_dashboard.py

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
conf_plot = conf_agg[conf_agg["Conference"].isin(top5)]

# Figure 1: Citations over time with range slider & selector
fig1 = px.line(
    conf_plot,
    x="Year", y="Citations", color="Conference",
    title="Top 5 Conferences: Citation Trends Over Years",
    markers=True,
    color_discrete_sequence=px.colors.qualitative.Bold
)
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

# Figure 2: Downloads over time with same controls
fig2 = px.line(
    conf_plot,
    x="Year", y="Downloads", color="Conference",
    title="Top 5 Conferences: Download Trends Over Years",
    markers=True,
    color_discrete_sequence=px.colors.qualitative.Vivid
)
fig2.update_traces(
    hovertemplate="Conference: %{name}<br>Year: %{x}<br>Downloads: %{y}<extra></extra>"
)
fig2.update_layout(
    template="plotly_dark",
    xaxis=fig1.layout.xaxis,
    yaxis=dict(title="Total Downloads", rangemode="tozero")
)

# 3. Trend Analysis of Keywords via LDA
docs = df["AuthorKeywords"].str.replace(";", " ").tolist()
vectorizer = CountVectorizer(max_df=0.9, min_df=5, stop_words="english")
tf_matrix = vectorizer.fit_transform(docs)

lda = LatentDirichletAllocation(n_components=4, random_state=0)
topic_matrix = lda.fit_transform(tf_matrix)

lda_df = pd.DataFrame(
    topic_matrix,
    columns=[f"Topic {i+1}" for i in range(4)]
)
lda_df["Year"] = df["Year"].values
topic_year = lda_df.groupby("Year").mean().reset_index()

fig3 = go.Figure()
colors = px.colors.qualitative.Safe
for i, topic in enumerate(topic_year.columns[1:]):
    fig3.add_trace(go.Scatter(
        x=topic_year["Year"],
        y=topic_year[topic],
        mode="lines+markers",
        name=topic,
        line=dict(color=colors[i], width=2),
        hovertemplate=f"Year: %{{x}}<br>{topic}: %{{y:.3f}}<extra></extra>"
    ))
fig3.update_layout(
    title="Evolution of Keyword Topics Over Years",
    template="plotly_dark",
    xaxis=dict(
        title="Year",
        range=[min_year - 1, max_year + 1],
        rangeselector=fig1.layout.xaxis.rangeselector,
        rangeslider=dict(visible=True)
    ),
    yaxis=dict(title="Avg Topic Proportion", rangemode="tozero")
)

# 4. Predictive Modeling of Paper Downloads: feature importance
mod_df = df[["Downloads_Xplore", "PaperType", "Conference", "Year"]].copy()
mod_df = pd.get_dummies(mod_df, columns=["PaperType", "Conference"], drop_first=True)
X = mod_df.drop("Downloads_Xplore", axis=1)
y = mod_df["Downloads_Xplore"]

rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X, y)
imp = pd.Series(rf.feature_importances_, index=X.columns)
feat_imp = imp.sort_values(ascending=False).head(10).reset_index()
feat_imp.columns = ["Feature", "Importance"]

fig4 = px.bar(
    feat_imp,
    x="Importance", y="Feature",
    orientation="h",
    title="Top 10 Features for Predicting Downloads",
    color="Importance",
    color_continuous_scale=px.colors.sequential.Plasma
)
fig4.update_traces(
    hovertemplate="Feature: %{y}<br>Importance: %{x:.4f}<extra></extra>"
)
fig4.update_layout(
    template="plotly_dark",
    yaxis=dict(autorange="reversed", title="Feature"),
    xaxis=dict(title="Importance")
)

# 5. Regression Analysis on Research Impact
reg_df = df[[
    "CitationCount_CrossRef", "Downloads_Xplore", "AwardFlag", "PubsCited_CrossRef"
]].dropna()
Xr = sm.add_constant(
    reg_df[["Downloads_Xplore", "AwardFlag", "PubsCited_CrossRef"]]
)
yr = reg_df["CitationCount_CrossRef"]
ols = sm.OLS(yr, Xr).fit()

coef = ols.params.drop("const").sort_values()
coef_df = coef.reset_index()
coef_df.columns = ["Feature", "Coefficient"]

fig5 = px.bar(
    coef_df,
    x="Coefficient", y="Feature",
    orientation="h",
    title="Regression Coefficients for Citation Count",
    color="Coefficient",
    color_continuous_scale=px.colors.sequential.Viridis
)
fig5.update_traces(
    hovertemplate="Feature: %{y}<br>Coefficient: %{x:.4f}<extra></extra>"
)
fig5.update_layout(
    template="plotly_dark",
    yaxis=dict(autorange="reversed", title="Feature"),
    xaxis=dict(title="Coefficient")
)

# 6. Sentiment Analysis of Abstracts
nltk.download("vader_lexicon", quiet=True)
sia = SentimentIntensityAnalyzer()
df["Sentiment"] = df["Abstract"].apply(
    lambda txt: sia.polarity_scores(txt)["compound"]
)

# Fig6: Histogram with marginal rug
fig6 = px.histogram(
    df,
    x="Sentiment",
    nbins=50,
    title="Distribution of Abstract Sentiment Scores",
    marginal="rug",
    color_discrete_sequence=["#EF553B"]
)
fig6.update_traces(
    hovertemplate="Sentiment: %{x:.3f}<br>Count: %{y}<extra></extra>"
)
fig6.update_layout(template="plotly_dark", xaxis=dict(title="Sentiment Score"))

# Fig7: Avg sentiment over years with slider
sent_year = df.groupby("Year")["Sentiment"].mean().reset_index()
fig7 = px.line(
    sent_year,
    x="Year", y="Sentiment",
    title="Average Abstract Sentiment Over Years",
    markers=True,
    color_discrete_sequence=["#00CC96"]
)
fig7.update_traces(
    hovertemplate="Year: %{x}<br>Avg Sentiment: %{y:.3f}<extra></extra>"
)
fig7.update_layout(
    template="plotly_dark",
    xaxis=dict(
        title="Year",
        range=[min_year - 1, max_year + 1],
        rangeselector=fig1.layout.xaxis.rangeselector,
        rangeslider=dict(visible=True)
    ),
    yaxis=dict(title="Avg Sentiment", rangemode="tozero")
)

# 7. Export all to a standalone HTML
all_figs = [fig1, fig2, fig3, fig4, fig5, fig6, fig7]
divs = [plot(fig, include_plotlyjs=False, output_type="div") for fig in all_figs]

html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8" />
    <title>Enhanced Interactive Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ background-color: #1e1e1e; color: #fff; margin:0; padding:0; }}
        h1 {{ text-align:center; padding:20px 0; }}
        .plot-container {{ width:90%; margin:20px auto; }}
    </style>
</head>
<body>
    <h1>Enhanced Interactive Data Science Dashboard</h1>
    {"".join(f'<div class="plot-container">{d}</div>' for d in divs)}
</body>
</html>
"""

with open("output.html", "w", encoding="utf-8") as f:
    f.write(html_template)

print("Enhanced dashboard generated in output.html")

