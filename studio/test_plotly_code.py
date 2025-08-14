# ---- NEW BLOCK ---- # 
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots

# Set dark theme for all figures
pio.templates.default = "plotly_dark"

# 1. Load and clean data
df = pd.read_csv('dataset.csv')

# Fill numeric NaNs with median
num_cols = ['AminerCitationCount', 'CitationCount_CrossRef',
            'PubsCited_CrossRef', 'Downloads_Xplore']
for c in num_cols:
    df[c] = df[c].fillna(df[c].median())

# Fill categoricals
df['GraphicsReplicabilityStamp'] = df['GraphicsReplicabilityStamp'].fillna('No')
df['Award'] = df['Award'].fillna('NoAward')

# Feature engineering
df['InternalReferencesCount'] = (
    df['InternalReferences']
      .fillna('')
      .apply(lambda x: len(str(x).split(';')) if x else 0)
)
df['AuthorKeywordsCount'] = (
    df['AuthorKeywords']
      .fillna('')
      .apply(lambda x: len(str(x).split(';')) if x else 0)
)

# -----------------------------------------------------------------------------
# 2. Enhanced Trend Analysis on Citation Counts
trend = df.groupby('Year', as_index=False)[
    ['AminerCitationCount','CitationCount_CrossRef']
].mean()

fig1 = go.Figure()
fig1.add_trace(go.Scatter(
    x=trend['Year'], y=trend['AminerCitationCount'],
    mode='lines+markers', name='AMiner Citations',
    hovertemplate="Year: %{x}<br>AMiner Avg: %{y:.1f}",
    line=dict(color='crimson', width=3)))
fig1.add_trace(go.Scatter(
    x=trend['Year'], y=trend['CitationCount_CrossRef'],
    mode='lines+markers', name='CrossRef Citations',
    hovertemplate="Year: %{x}<br>CrossRef Avg: %{y:.1f}",
    line=dict(color='royalblue', width=3)))

# add range slider and selector
fig1.update_layout(
    title="Average Citation Trends Over Years",
    xaxis=dict(
        title="Year",
        rangeslider=dict(visible=True),
        rangeselector=dict(
            buttons=list([
                dict(count=5, label="5y", step="year", stepmode="backward"),
                dict(count=10, label="10y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
    ),
    yaxis=dict(title="Average Citation Count"),
    legend=dict(title="Citation Source"),
    margin=dict(t=80, b=60)
)

# -----------------------------------------------------------------------------
# 3. Regression Analysis for Download Prediction
X_reg = df[['AminerCitationCount','InternalReferencesCount',
            'Year']].copy()
X_reg['GraphicsReplicabilityStamp_enc'] = (
    df['GraphicsReplicabilityStamp'] != 'No'
).astype(int)
y_reg = df['Downloads_Xplore']

lr = LinearRegression()
lr.fit(X_reg, y_reg)
y_pred = lr.predict(X_reg)

# Combine scatter and coefficients into one subplot
fig2 = make_subplots(
    rows=1, cols=2,
    subplot_titles=("Predicted vs Actual Downloads","Regression Coefficients"),
    column_widths=[0.6,0.4], horizontal_spacing=0.15
)

# Scatter
fig2.add_trace(go.Scatter(
    x=y_reg, y=y_pred,
    mode='markers',
    marker=dict(color='mediumseagreen', size=6, opacity=0.7),
    customdata=np.stack([
        df['Year'], df['AminerCitationCount'], df['InternalReferencesCount']
    ], axis=-1),
    hovertemplate=
        "Year: %{customdata[0]}<br>"
        "Actual: %{x:.0f}<br>"
        "Predicted: %{y:.0f}<br>"
        "AMiner Cit: %{customdata[1]:.0f}<br>"
        "Refs Count: %{customdata[2]}",
    name='Data Points'
), row=1, col=1)

fig2.add_trace(go.Line(
    x=[0, y_reg.max()*1.05],
    y=[0, y_reg.max()*1.05],
    line=dict(color='lightgray', dash='dash'),
    showlegend=False
), row=1, col=1)

# Coefficients bar
coef_df = pd.DataFrame({
    'feature': X_reg.columns,
    'coefficient': lr.coef_
})
fig2.add_trace(go.Bar(
    x=coef_df['feature'], y=coef_df['coefficient'],
    marker=dict(color=px.colors.sequential.Viridis),
    customdata=coef_df['coefficient'],
    hovertemplate="Feature: %{x}<br>Coef: %{customdata:.3f}",
    name='Coef'
), row=1, col=2)

fig2.update_layout(
    title="Download Prediction: Regression Results",
    showlegend=False,
    margin=dict(t=80, b=40)
)
fig2.update_xaxes(title_text="Actual Downloads", row=1, col=1, range=[0, y_reg.max()*1.05])
fig2.update_yaxes(title_text="Predicted Downloads", row=1, col=1, range=[0, y_reg.max()*1.05])
fig2.update_xaxes(title_text="Feature", row=1, col=2)
fig2.update_yaxes(title_text="Coefficient", row=1, col=2)

# -----------------------------------------------------------------------------
# 4. Machine Learning to Predict Awards
df_ml = df.copy()
df_ml['AwardBinary'] = (df_ml['Award'] != 'NoAward').astype(int)
le_pt = LabelEncoder()
le_co = LabelEncoder()
df_ml['PaperType_enc'] = le_pt.fit_transform(df_ml['PaperType'])
df_ml['Conference_enc'] = le_co.fit_transform(df_ml['Conference'])

features = [
    'PaperType_enc','Conference_enc','AuthorKeywordsCount',
    'InternalReferencesCount','Downloads_Xplore','Year'
]
X_clf = df_ml[features]
y_clf = df_ml['AwardBinary']

X_train, X_test, y_train, y_test = train_test_split(
    X_clf, y_clf, test_size=0.3, random_state=42, stratify=y_clf
)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_prob = clf.predict_proba(X_test)[:,1]
y_pred = clf.predict(X_test)

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
fig3_roc = go.Figure()
fig3_roc.add_trace(go.Scatter(
    x=fpr, y=tpr, mode='lines',
    line=dict(color='darkorange', width=3),
    name=f'AUC = {roc_auc:.2f}'
))
fig3_roc.add_trace(go.Line(
    x=[0,1], y=[0,1],
    line=dict(color='gray', dash='dash'),
    showlegend=False
))
fig3_roc.update_layout(
    title="ROC Curve: Award Prediction",
    xaxis_title="False Positive Rate",
    yaxis_title="True Positive Rate",
    margin=dict(t=60)
)

# Confusion matrix heatmap
cm = confusion_matrix(y_test, y_pred)
fig3_cm = go.Figure(data=go.Heatmap(
    z=cm,
    x=['Pred NoAward','Pred Award'],
    y=['True NoAward','True Award'],
    colorscale='Blues',
    hovertemplate="Predicted %{x}<br>Actual %{y}<br>Count %{z}<extra></extra>"
))
fig3_cm.update_layout(
    title="Confusion Matrix",
    margin=dict(t=60)
)

# -----------------------------------------------------------------------------
# 5. Keyword Trend Analysis
kw = df[['Year','AuthorKeywords']].dropna()
kw = kw.assign(kw_list=kw['AuthorKeywords'].str.split(';')).explode('kw_list')
kw['kw_list'] = kw['kw_list'].str.strip().str.lower()
top5 = kw['kw_list'].value_counts().nlargest(5).index.tolist()
kw_top = kw[kw['kw_list'].isin(top5)]
kw_trend = kw_top.groupby(['Year','kw_list']).size().reset_index(name='count')

fig4 = px.line(
    kw_trend, x='Year', y='count', color='kw_list',
    title="Top 5 Keywords Over Time",
    labels={'kw_list':'Keyword','count':'Frequency'}
)
fig4.update_traces(mode='lines+markers')
fig4.update_layout(
    xaxis=dict(rangeslider=dict(visible=True)),
    legend_title="Keyword",
    margin=dict(t=60, b=40)
)

# -----------------------------------------------------------------------------
# 6. Impact of Awards on Citations & Downloads (Violin plots)
impact = df.copy()
impact['AwardFlag'] = np.where(impact['Award']!='NoAward','Awarded','Not Awarded')

fig5 = make_subplots(rows=1, cols=2,
                     subplot_titles=("Citations by Award","Downloads by Award"))
# Citations violin
fig5.add_trace(go.Violin(
    x=impact['AwardFlag'], y=impact['CitationCount_CrossRef'],
    box_visible=True, meanline_visible=True,
    jitter=0.3, points='all',
    marker=dict(color='orchid'),
    hovertemplate="Group: %{x}<br>Citations: %{y}"
), row=1, col=1)
# Downloads violin
fig5.add_trace(go.Violin(
    x=impact['AwardFlag'], y=impact['Downloads_Xplore'],
    box_visible=True, meanline_visible=True,
    jitter=0.3, points='all',
    marker=dict(color='teal'),
    hovertemplate="Group: %{x}<br>Downloads: %{y}"
), row=1, col=2)

fig5.update_layout(
    title="Award Impact on Citations & Downloads",
    showlegend=False,
    margin=dict(t=80, b=40)
)

# -----------------------------------------------------------------------------
# 7. Export all figures to a single HTML with CDN include
figs = [fig1, fig2, fig3_roc, fig3_cm, fig4, fig5]
html_parts = []

for i, fig in enumerate(figs, start=1):
    # include plotly.js only once via CDN on the first figure
    include_js = 'cdn' if i == 1 else False
    html_div = pio.to_html(
        fig,
        include_plotlyjs=include_js,
        full_html=False,
        div_id=f"fig{i}"
    )
    html_parts.append(html_div)

html_page = f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Interactive Data Science & StatLearning Visualizations</title>
</head>
<body>
  {'<br>'.join(html_parts)}
</body>
</html>
"""

with open("output.html", "w", encoding="utf-8") as f:
    f.write(html_page)

print("All interactive figures written to output.html")

