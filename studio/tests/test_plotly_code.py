# ---- NEW BLOCK ---- # 
# Filename: generate_visuals.py

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_curve, auc
import warnings
warnings.filterwarnings("ignore")

# Set default dark template
px.defaults.template = "plotly_dark"

# 1. LOAD AND PREPROCESS DATA
df = pd.read_csv("dataset.csv")

# Fill numeric NaNs with median
for col in ["AminerCitationCount", "CitationCount_CrossRef",
            "PubsCited_CrossRef", "Downloads_Xplore"]:
    df[col] = df[col].fillna(df[col].median())

# Fill object NaNs
df["PaperType"] = df["PaperType"].fillna("Unknown")
df["Conference"] = df["Conference"].fillna("Unknown")
df["AuthorKeywords"] = df["AuthorKeywords"].fillna("")
df["InternalReferences"] = df["InternalReferences"].fillna("")
df["GraphicsReplicabilityStamp"] = df["GraphicsReplicabilityStamp"].fillna("None")
df["Award"] = df["Award"].fillna("")

# ---------------------------------------------------------------------
# 2. Enhanced Trend Analysis on Citation Counts
df_trend = df.groupby("Year")[["AminerCitationCount", "CitationCount_CrossRef"]].mean().reset_index()

fig1 = px.line(
    df_trend,
    x="Year",
    y=["AminerCitationCount", "CitationCount_CrossRef"],
    markers=True,
    color_discrete_sequence=["#636EFA", "#EF553B"],
    labels={"value": "Average Count", "variable": "Citation Type"},
    title="Average Aminer vs CrossRef Citation Trends Over Years",
    template="plotly_dark"
)
fig1.update_layout(
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=5, label="5y", step="year", stepmode="backward"),
                dict(count=10, label="10y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        ),
        rangeslider=dict(visible=True),
        type="date"
    ),
    yaxis=dict(title="Average Citations", rangemode="tozero"),
    hovermode="x unified"
)

# ---------------------------------------------------------------------
# 3. Machine Learning to Predict Awards (Logistic Regression + ROC)
df2 = df[[
    "PaperType", "AuthorKeywords", "InternalReferences",
    "Downloads_Xplore", "Year", "Conference", "Award"
]].copy()
df2["Award_Flag"] = (df2["Award"] != "").astype(int)
df2["num_keywords"] = df2["AuthorKeywords"].apply(lambda x: len(x.split(";")) if x else 0)
df2["num_references"] = df2["InternalReferences"].apply(lambda x: len(x.split(";")) if x else 0)

X2 = pd.get_dummies(
    df2[["PaperType", "Downloads_Xplore", "Year", "num_keywords", "num_references", "Conference"]],
    drop_first=True
)
y2 = df2["Award_Flag"]
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.3, random_state=42)

clf = LogisticRegression(max_iter=1000)
clf.fit(X2_train, y2_train)
y2_proba = clf.predict_proba(X2_test)[:, 1]
fpr, tpr, _ = roc_curve(y2_test, y2_proba)
roc_auc = auc(fpr, tpr)

fig2 = go.Figure([
    go.Scatter(
        x=fpr, y=tpr,
        mode="lines",
        line=dict(color="#00CC96", width=3),
        name=f"ROC curve (AUC = {roc_auc:.2f})",
        hovertemplate="FPR: %{x:.2f}<br>TPR: %{y:.2f}<extra></extra>"
    ),
    go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        line=dict(color="gray", dash="dash"),
        showlegend=False
    )
])
fig2.update_layout(
    title="ROC Curve for Award Prediction",
    xaxis=dict(title="False Positive Rate", range=[0, 1]),
    yaxis=dict(title="True Positive Rate", range=[0, 1]),
    hovermode="closest",
    template="plotly_dark"
)

# ---------------------------------------------------------------------
# 4. Regression Analysis for Download Prediction (Linear Regression)
df3 = df[[
    "Downloads_Xplore", "AminerCitationCount",
    "GraphicsReplicabilityStamp", "InternalReferences",
    "Year", "Conference"
]].copy()
df3["replicable"] = (df3["GraphicsReplicabilityStamp"] != "None").astype(int)
df3["num_references"] = df3["InternalReferences"].apply(lambda x: len(x.split(";")) if x else 0)

X3 = pd.get_dummies(
    df3[["AminerCitationCount", "replicable", "num_references", "Year", "Conference"]],
    drop_first=True
)
y3 = df3["Downloads_Xplore"]
X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size=0.3, random_state=42)

reg = LinearRegression()
reg.fit(X3_train, y3_train)
y3_pred = reg.predict(X3_test)

# Scatter plot with marginal histograms
fig3 = px.scatter(
    x=y3_test, y=y3_pred,
    labels={"x": "Actual Downloads", "y": "Predicted Downloads"},
    marginal_x="histogram",
    marginal_y="histogram",
    color_discrete_sequence=["#AB63FA"],
    title="Actual vs Predicted Downloads with Marginal Distributions",
    template="plotly_dark"
)
fig3.add_shape(
    type="line",
    x0=y3_test.min(), x1=y3_test.max(),
    y0=y3_test.min(), y1=y3_test.max(),
    line=dict(color="white", dash="dash"),
    xref="x", yref="y"
)
fig3.update_layout(hovermode="closest")

# Bar chart of coefficients
coef_df = pd.DataFrame({
    "feature": X3.columns,
    "coefficient": reg.coef_
}).assign(abs_coef=lambda d: d["coefficient"].abs()).sort_values("abs_coef", ascending=False)

fig3_coef = px.bar(
    coef_df,
    x="coefficient", y="feature",
    orientation="h",
    color="coefficient",
    color_continuous_scale="Viridis",
    title="Regression Coefficients for Download Prediction",
    template="plotly_dark"
)
fig3_coef.update_layout(yaxis=dict(categoryorder="total ascending"))

# ---------------------------------------------------------------------
# 5. Keyword Trend Analysis
df4 = df[["Year", "AuthorKeywords"]].copy()
df4 = df4[df4["AuthorKeywords"] != ""]
df4["AuthorKeywords"] = df4["AuthorKeywords"].str.split(";")
df4 = df4.explode("AuthorKeywords")
df4["AuthorKeywords"] = df4["AuthorKeywords"].str.strip()
top5 = df4["AuthorKeywords"].value_counts().nlargest(5).index.tolist()
counts = (
    df4[df4["AuthorKeywords"].isin(top5)]
    .groupby(["Year", "AuthorKeywords"])
    .size()
    .reset_index(name="count")
)

fig4 = px.line(
    counts,
    x="Year", y="count", color="AuthorKeywords",
    markers=True,
    color_discrete_sequence=px.colors.qualitative.Safe,
    title="Trend of Top 5 Author Keywords Over Years",
    template="plotly_dark"
)
fig4.update_layout(
    xaxis=dict(rangeslider=dict(visible=True)),
    hovermode="x unified"
)

# ---------------------------------------------------------------------
# 6. Impact of Awards on Citations and Downloads
df5 = df[["Award", "CitationCount_CrossRef", "Downloads_Xplore"]].copy()
df5["Awarded"] = np.where(df5["Award"] == "", "Not Awarded", "Awarded")
grouped = df5.groupby("Awarded")[["CitationCount_CrossRef", "Downloads_Xplore"]].mean().reset_index()

fig5 = go.Figure()
fig5.add_trace(go.Bar(
    x=grouped["Awarded"],
    y=grouped["CitationCount_CrossRef"],
    name="Avg Citations",
    marker_color="#FFA15A",
    hovertemplate="%{y:.2f} citations<extra></extra>"
))
fig5.add_trace(go.Bar(
    x=grouped["Awarded"],
    y=grouped["Downloads_Xplore"],
    name="Avg Downloads",
    marker_color="#19D3F3",
    hovertemplate="%{y:.2f} downloads<extra></extra>"
))
fig5.update_layout(
    barmode="group",
    title="Impact of Awards on Citations and Downloads",
    xaxis=dict(title=""),
    yaxis=dict(title="Average Count", rangemode="tozero"),
    hovermode="x unified",
    template="plotly_dark"
)

# ---------------------------------------------------------------------
# 7. EXPORT TO HTML
figs = [fig1, fig2, fig3, fig3_coef, fig4, fig5]
narratives = [
    "Line chart of average Aminer vs CrossRef citations per year, showing evolving citation trends.",
    "ROC curve for the logistic regression model predicting paper awards. AUC indicates predictive power.",
    "Scatter of actual vs predicted downloads with marginal histograms to show distribution.",
    "Horizontal bar chart of regression coefficients revealing factors influencing download counts.",
    "Trends of the top 5 most frequent author keywords over time, highlighting topic shifts.",
    "Grouped bar chart comparing average citations and downloads for awarded vs non-awarded papers."
]

html_parts = []
for idx, fig in enumerate(figs, start=1):
    snippet = fig.to_html(
        include_plotlyjs="cdn",
        full_html=False,
        div_id=f"fig{idx}"
    )
    html_parts.append(
        f"""
        <div style="margin-bottom:40px;">
          {snippet}
          <button onclick="var p=document.getElementById('narr{idx}');
                           p.style.display=(p.style.display=='none'?'block':'none');"
                  style="margin-top:10px;">
            Toggle Narrative
          </button>
          <p id="narr{idx}" style="display:none; max-width:700px; font-style:italic; color:#DDD;">
            {narratives[idx-1]}
          </p>
        </div>
        <hr style="border-color:#444;">
        """
    )

html_template = f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Interactive Plotly Visualizations</title>
</head>
<body style="background-color:#111; color:#EEE; font-family:Arial, sans-serif; padding:20px;">
  <h1 style="text-align:center;">Interactive Plotly Visualizations</h1>
  {' '.join(html_parts)}
</body>
</html>
"""

with open("output.html", "w", encoding="utf-8") as f:
    f.write(html_template)

print("✅ output.html generated successfully in dark mode with enhanced interactivity.")

