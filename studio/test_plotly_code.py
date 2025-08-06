# ---- NEW BLOCK ---- # 
# plotly_visualizations_enhanced.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 1. Load & clean
df = pd.read_csv("dataset.csv")

# Numeric fill
for col in ["AminerCitationCount", "CitationCount_CrossRef", "Downloads_Xplore"]:
    df[col] = df[col].fillna(df[col].median())

# Categorical/text fill
df["PaperType"] = df["PaperType"].fillna("Unknown")
df["Conference"] = df["Conference"].fillna("Unknown")
df["InternalReferences"] = df["InternalReferences"].fillna("")
df["GraphicsReplicabilityStamp"] = df["GraphicsReplicabilityStamp"].fillna("")

# Derived features
df["NumInternalRefs"] = df["InternalReferences"].apply(lambda x: len(x.split(";")) if x else 0)
df["GraphicsFlag"] = (df["GraphicsReplicabilityStamp"] != "").astype(int)

# 2. Enhanced Trend Analysis (dark template, slider, dropdown)
trend_df = df.groupby(["Year", "PaperType"]) \
             .agg(Aminer=("AminerCitationCount","mean"),
                  CrossRef=("CitationCount_CrossRef","mean")) \
             .reset_index()

# Base "All" year-aggregated
all_df = df.groupby("Year").agg(
    Aminer=("AminerCitationCount","mean"),
    CrossRef=("CitationCount_CrossRef","mean")
).reset_index()

fig_trend = go.Figure()
# All traces
fig_trend.add_trace(go.Scatter(
    x=all_df.Year, y=all_df.Aminer,
    mode="lines+markers", name="Aminer Citations",
    line=dict(color=px.colors.qualitative.Dark24[0], width=3)))
fig_trend.add_trace(go.Scatter(
    x=all_df.Year, y=all_df.CrossRef,
    mode="lines+markers", name="CrossRef Citations",
    line=dict(color=px.colors.qualitative.Dark24[2], width=3)))

# Per-paper-type hidden by default
paper_types = sorted(trend_df.PaperType.unique())
for i, pt in enumerate(paper_types):
    sub = trend_df[trend_df.PaperType == pt]
    fig_trend.add_trace(go.Scatter(
        x=sub.Year, y=sub.Aminer,
        mode="lines", name=f"Aminer ({pt})",
        line=dict(dash="dash", color=px.colors.qualitative.Dark24[0]),
        visible=False))
    fig_trend.add_trace(go.Scatter(
        x=sub.Year, y=sub.CrossRef,
        mode="lines", name=f"CrossRef ({pt})",
        line=dict(dash="dash", color=px.colors.qualitative.Dark24[2]),
        visible=False))

# Dropdown menus
buttons = []
# button for "All"
vis0 = [True, True] + [False] * (2 * len(paper_types))
buttons.append(dict(label="All", method="update",
                    args=[{"visible": vis0},
                          {"title":"Citation Trend: All PaperTypes"}]))
# buttons for each paper type
for idx, pt in enumerate(paper_types):
    vis = [False] * (2 + 2*len(paper_types))
    base = 2 + idx*2
    vis[base] = vis[base+1] = True
    buttons.append(dict(label=pt, method="update",
                        args=[{"visible": vis},
                              {"title":f"Citation Trend: {pt}"}]))

fig_trend.update_layout(
    template="plotly_dark",
    updatemenus=[dict(active=0, buttons=buttons, x=1.15, y=1.05)],
    title="Citation Trend: All PaperTypes",
    xaxis=dict(title="Year", rangeslider=dict(visible=True)),
    yaxis=dict(title="Mean Citations"),
    margin=dict(r=150)
)

# 3. Award Prediction ROC (dark, hover & legend)
df["AwardBinary"] = df["Award"].notna().astype(int)
X = pd.get_dummies(df[["PaperType","Conference","Year","Downloads_Xplore","NumInternalRefs"]],
                   drop_first=True)
y = df["AwardBinary"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.3, random_state=42)

# Models
lr = LogisticRegression(max_iter=500).fit(X_train, y_train)
rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)

fpr_lr, tpr_lr, _ = roc_curve(y_test, lr.predict_proba(X_test)[:,1])
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf.predict_proba(X_test)[:,1])
auc_lr, auc_rf = auc(fpr_lr, tpr_lr), auc(fpr_rf, tpr_rf)

fig_ml = go.Figure()
fig_ml.add_trace(go.Scatter(
    x=fpr_lr, y=tpr_lr, mode="lines",
    name=f"Logistic (AUC={auc_lr:.2f})",
    line=dict(color=px.colors.qualitative.Dark24[4], width=3)))
fig_ml.add_trace(go.Scatter(
    x=fpr_rf, y=tpr_rf, mode="lines",
    name=f"RandomForest (AUC={auc_rf:.2f})",
    line=dict(color=px.colors.qualitative.Dark24[6], width=3)))
fig_ml.add_trace(go.Scatter(
    x=[0,1], y=[0,1], mode="lines",
    name="Chance", line=dict(color="gray", dash="dash")))

fig_ml.update_layout(
    template="plotly_dark",
    title="ROC Curves for Award Prediction",
    xaxis=dict(title="False Positive Rate"),
    yaxis=dict(title="True Positive Rate")
)

# 4. Regression: Actual vs Predicted Downloads (dark, residual coloring)
reg_df = df[["Downloads_Xplore","AminerCitationCount","GraphicsFlag","NumInternalRefs","Year","Conference"]]
reg_df = pd.get_dummies(reg_df, drop_first=True)
Xr, yr = reg_df.drop(columns="Downloads_Xplore"), reg_df["Downloads_Xplore"]
model_reg = LinearRegression().fit(Xr, yr)
yp = model_reg.predict(Xr)
resid = yp - yr

fig_reg = go.Figure(data=go.Scatter(
    x=yr, y=yp,
    mode="markers",
    marker=dict(
        size=7,
        color=resid,
        colorscale="RdBu",
        showscale=True,
        colorbar=dict(title="Residual")
    ),
    hovertemplate=
        "Actual: %{x:.0f}<br>" +
        "Pred: %{y:.0f}<br>" +
        "Resid: %{marker.color:.1f}<extra></extra>"
))
fig_reg.add_shape(
    type="line", x0=yr.min(), y0=yr.min(),
    x1=yr.max(), y1=yr.max(),
    line=dict(color="white", dash="dash")
)
fig_reg.update_layout(
    template="plotly_dark",
    title="Actual vs Predicted Downloads",
    xaxis=dict(title="Actual Downloads"),
    yaxis=dict(title="Predicted Downloads")
)

# 5. Conference Impact (top 15, horizontal), dark
conf_stats = df.groupby("Conference").agg(
    Aminer=("AminerCitationCount","mean"),
    CrossRef=("CitationCount_CrossRef","mean")
).reset_index().sort_values("Aminer", ascending=False).head(15)

conf_m = conf_stats.melt(id_vars="Conference", var_name="Metric", value_name="Mean")
fig_conf = px.bar(
    conf_m, x="Mean", y="Conference", color="Metric",
    orientation="h", barmode="group",
    color_discrete_sequence=px.colors.qualitative.Dark24,
    title="Top 15 Conferences by Mean Citations"
)
fig_conf.update_layout(template="plotly_dark", yaxis=dict(autorange="reversed"))

# 6. Award Impact (violin), dark
df_aw = df.copy()
df_aw["AwardStr"] = df_aw["AwardBinary"].map({0:"No Award",1:"Award"})
fig_award = make_subplots(rows=1, cols=2, subplot_titles=("Citations","Downloads"))
fig_award.add_trace(go.Violin(
    x=df_aw["AwardStr"], y=df_aw["CitationCount_CrossRef"],
    name="Citations", spanmode="hard",
    line_color=px.colors.qualitative.Dark24[8],
    fillcolor="rgba(50,50,200,0.6)"
), row=1, col=1)
fig_award.add_trace(go.Violin(
    x=df_aw["AwardStr"], y=df_aw["Downloads_Xplore"],
    name="Downloads", spanmode="hard",
    line_color=px.colors.qualitative.Dark24[10],
    fillcolor="rgba(200,50,50,0.6)"
), row=1, col=2)
fig_award.update_layout(template="plotly_dark", title="Award Impact on Citations & Downloads")

# Collect figures
figs = [
    ("Citation Trend Analysis", fig_trend),
    ("Award Prediction ROC", fig_ml),
    ("Download Regression Analysis", fig_reg),
    ("Conference Impact Assessment", fig_conf),
    ("Award Impact Violin Plots", fig_award)
]

# 7. Build HTML with CDN in each div
html_parts = []
for title, fig in figs:
    div = fig.to_html(include_plotlyjs='cdn', full_html=False)
    html_parts.append(f"<h2>{title}</h2>\n{div}\n")

html_page = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Dark Mode Plotly Dashboard</title>
<style>
  body {{ background:#111; color:#ddd; font-family:sans-serif; padding:20px; }}
  h1 {{ text-align:center; }}
  h2 {{ margin-top:50px; }}
</style>
</head><body>
<h1>Interactive Dark-Mode Visualizations</h1>
{''.join(html_parts)}
</body></html>"""

with open("output.html","w",encoding="utf-8") as f:
    f.write(html_page)

print("output.html generated successfully in dark mode.")

