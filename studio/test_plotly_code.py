# ---- NEW BLOCK ---- # 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_curve, auc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# Set default template to dark
pio.templates.default = "plotly_dark"

# 1. LOAD & BASIC CLEAN
df = pd.read_csv('dataset.csv')

# Fill numeric NaNs with median
num_cols = ['AminerCitationCount','CitationCount_CrossRef','PubsCited_CrossRef','Downloads_Xplore']
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

# Fill categorical NaNs
df['PaperType'] = df['PaperType'].fillna('Unknown')
df['Conference'] = df['Conference'].fillna('Unknown')
df['InternalReferences'] = df['InternalReferences'].fillna('')
df['AuthorKeywords'] = df['AuthorKeywords'].fillna('')
df['GraphicsReplicabilityStamp'] = df['GraphicsReplicabilityStamp'].fillna('None')
df['Award'] = df['Award'].fillna('No')

# DERIVE FEATURES
df['num_internal_refs'] = df['InternalReferences'].apply(lambda x: len(x.split(';')) if x else 0)
df['num_keywords']      = df['AuthorKeywords'].apply(lambda x: len(x.split(';')) if x else 0)
df['has_stamp']         = (df['GraphicsReplicabilityStamp'] != 'None').astype(int)
df['Award_label']       = (df['Award'] != 'No').astype(int)

# 2. ENHANCED TREND ANALYSIS: CITATION COUNTS BY CONFERENCE & METRIC
top_confs = df['Conference'].value_counts().nlargest(5).index.tolist()
df_trend = (
    df[df['Conference'].isin(top_confs)]
    .groupby(['Year','Conference'])[['AminerCitationCount','CitationCount_CrossRef']]
    .mean()
    .reset_index()
    .melt(id_vars=['Year','Conference'],
          value_vars=['AminerCitationCount','CitationCount_CrossRef'],
          var_name='Metric',
          value_name='AvgCitations')
)

fig1 = px.line(
    df_trend,
    x='Year', y='AvgCitations',
    color='Conference', line_dash='Metric',
    markers=True,
    title="Yearly Avg. Aminer vs CrossRef Citations for Top 5 Conferences",
    color_discrete_sequence=px.colors.qualitative.Bold
)
fig1.update_traces(
    hovertemplate="<b>%{fullData.name}</b><br>Year: %{x}<br>Avg. Citations: %{y:.1f}<extra></extra>"
)
fig1.update_xaxes(dtick=1, rangeslider_visible=True)
fig1.update_layout(
    xaxis_title="Year",
    yaxis_title="Average Citations",
    legend_title="Conference & Metric",
    margin=dict(l=60, r=20, t=60, b=40)
)

# 3. TREND ANALYSIS BY PAPER TYPE (Top 5 types)
top_pt = df['PaperType'].value_counts().nlargest(5).index.tolist()
df_pt = (
    df[df['PaperType'].isin(top_pt)]
    .groupby(['Year','PaperType'])['CitationCount_CrossRef']
    .mean().reset_index()
)
fig2 = px.line(
    df_pt,
    x='Year', y='CitationCount_CrossRef',
    color='PaperType',
    markers=True,
    title="CrossRef Citation Trends by Top 5 Paper Types",
    color_discrete_sequence=px.colors.qualitative.Set2
)
fig2.update_traces(
    hovertemplate="Paper Type: %{fullData.name}<br>Year: %{x}<br>Avg. Citations: %{y:.1f}<extra></extra>"
)
fig2.update_xaxes(dtick=1, rangeslider_visible=True)
fig2.update_layout(
    xaxis_title="Year",
    yaxis_title="Avg. CrossRef Citations",
    legend_title="Paper Type",
    margin=dict(l=60, r=20, t=60, b=40)
)

# 4. MACHINE LEARNING: PREDICTING AWARDS
# Prepare features
X = pd.get_dummies(df[['PaperType','Conference']], drop_first=True)
X['num_internal_refs'] = df['num_internal_refs']
X['num_keywords']      = df['num_keywords']
X['Downloads_Xplore']  = df['Downloads_Xplore']
X['Year']              = df['Year']
y = df['Award_label']

# Split & train
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_score = clf.predict_proba(X_test)[:,1]

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)
fig3 = go.Figure([
    go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC (AUC={roc_auc:.2f})',
               line=dict(color='cyan', width=3)),
    go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Chance',
               line=dict(color='magenta', dash='dash'))
])
fig3.update_layout(
    title="ROC Curve: Award Prediction",
    xaxis_title="False Positive Rate",
    yaxis_title="True Positive Rate",
    hovermode="x unified",
    margin=dict(l=60, r=20, t=60, b=40)
)
fig3.update_traces(
    hovertemplate="FPR: %{x:.2f}<br>TPR: %{y:.2f}<extra></extra>"
)

# Feature Importances
imp = pd.Series(clf.feature_importances_, index=X.columns).nlargest(10)
fig4 = px.bar(
    x=imp.values, y=imp.index,
    orientation='h',
    title="Top 10 Feature Importances for Award Model",
    color=imp.values,
    color_continuous_scale='Viridis'
)
fig4.update_layout(
    xaxis_title="Importance",
    yaxis_title="Feature",
    margin=dict(l=150, r=20, t=60, b=40)
)
fig4.update_traces(
    hovertemplate="Feature: %{y}<br>Importance: %{x:.3f}<extra></extra>"
)

# 5. REGRESSION: DOWNLOAD PREDICTION
reg_feats = df[['AminerCitationCount','num_internal_refs','has_stamp','Year']]
reg_target = df['Downloads_Xplore']
Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    reg_feats, reg_target,
    test_size=0.3, random_state=42
)
lr = LinearRegression()
lr.fit(Xr_train, yr_train)
y_pred = lr.predict(Xr_test)

fig5 = go.Figure([
    go.Scatter(
        x=yr_test, y=y_pred,
        mode='markers',
        marker=dict(color='orange', size=6, opacity=0.7),
        name='Predicted vs Actual',
        hovertemplate="Actual: %{x:.0f}<br>Pred: %{y:.0f}<extra></extra>"
    ),
    go.Scatter(
        x=[yr_test.min(), yr_test.max()],
        y=[yr_test.min(), yr_test.max()],
        mode='lines',
        line=dict(color='white', dash='dash'),
        name='Ideal'
    )
])
fig5.update_layout(
    title="Predicted vs Actual Downloads_Xplore",
    xaxis_title="Actual Downloads",
    yaxis_title="Predicted Downloads",
    margin=dict(l=60, r=20, t=60, b=40)
)

# 6. KEYWORD TREND ANALYSIS
kw_df = df.loc[df['AuthorKeywords'] != '', ['Year','AuthorKeywords']].copy()
kw_df['keyword_list'] = kw_df['AuthorKeywords'].str.split(';')
kw_exploded = kw_df.explode('keyword_list')
kw_exploded['keyword_list'] = kw_exploded['keyword_list'].str.strip().str.lower()
top_kws = kw_exploded['keyword_list'].value_counts().nlargest(10).index.tolist()
kw_trend = (
    kw_exploded[kw_exploded['keyword_list'].isin(top_kws)]
    .groupby(['Year','keyword_list']).size()
    .reset_index(name='Count')
)

fig6 = px.line(
    kw_trend,
    x='Year', y='Count',
    color='keyword_list',
    markers=True,
    title="Yearly Trends of Top 10 Author Keywords",
    color_discrete_sequence=px.colors.qualitative.Plotly
)
fig6.update_traces(
    hovertemplate="Keyword: %{fullData.name}<br>Year: %{x}<br>Count: %{y}<extra></extra>"
)
fig6.update_xaxes(dtick=1, rangeslider_visible=True)
fig6.update_layout(
    xaxis_title="Year",
    yaxis_title="Keyword Frequency",
    legend_title="Keyword",
    margin=dict(l=60, r=20, t=60, b=40)
)

# 7. IMPACT OF AWARDS ON CITATIONS & DOWNLOADS
df_aw = df.copy()
df_aw['AwardStr'] = df_aw['Award_label'].map({0:'No Award',1:'Award'})
fig7 = make_subplots(
    rows=1, cols=2,
    subplot_titles=("CrossRef Citations by Award","Downloads by Award")
)

fig7.add_trace(
    go.Box(
        x=df_aw['AwardStr'], y=df_aw['CitationCount_CrossRef'],
        name="Citations", marker_color='gold',
        boxmean='sd', hovertemplate="Award: %{x}<br>Citations: %{y}<extra></extra>"
    ), row=1, col=1
)
fig7.add_trace(
    go.Box(
        x=df_aw['AwardStr'], y=df_aw['Downloads_Xplore'],
        name="Downloads", marker_color='lightseagreen',
        boxmean='sd', hovertemplate="Award: %{x}<br>Downloads: %{y}<extra></extra>"
    ), row=1, col=2
)
fig7.update_layout(
    title="Effect of Awards on Citations & Downloads",
    margin=dict(l=60, r=20, t=60, b=40),
    showlegend=False
)

# 8. EXPORT TO HTML WITH NARRATIVE TOGGLES
figs = [fig1, fig2, fig3, fig4, fig5, fig6, fig7]
descs = [
    "Average Aminer vs CrossRef citation trends per year across the five most active conferences.",
    "Yearly CrossRef citation evolution for the top five paper types by frequency.",
    "ROC curve and AUC for RandomForest model predicting whether a paper wins an award.",
    "Top ten model features driving award predictions, sorted by importance score.",
    "Scatter of actual vs predicted download counts with ideal reference line.",
    "Trends over time for the ten most frequent author-supplied keywords.",
    "Comparison of citation and download distributions for award‐winning vs non-winning papers."
]

html_blocks = []
for i, fig in enumerate(figs):
    div = pio.to_html(fig, include_plotlyjs='cdn', full_html=False)
    html_blocks.append(f"""
<div style="margin-bottom:50px;">
  {div}
  <button onclick="let d=document.getElementById('desc{i}'); 
                  d.style.display = d.style.display==='none'?'block':'none';"
          style="margin-top:10px; padding:6px 12px; background:#444; color:#fff; border:none; cursor:pointer;">
    Toggle Description
  </button>
  <div id="desc{i}" style="display:none; margin-top:8px; padding:10px; 
                           background:#222; color:#ddd; border:1px solid #555;">
    {descs[i]}
  </div>
</div>
""")

html_page = f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Interactive Plotly Dashboard</title>
  <style>
    body {{ background:#111; color:#eee; font-family:sans-serif; padding:20px; }}
    h1 {{ text-align:center; }}
  </style>
</head>
<body>
  <h1>Research Paper Insights Dashboard</h1>
  {''.join(html_blocks)}
</body>
</html>
"""

with open('output.html', 'w', encoding='utf-8') as f:
    f.write(html_page)

print("Generated output.html with enhanced interactive Plotly figures.")

