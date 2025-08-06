# ---- NEW BLOCK ---- # 
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_curve, auc
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# 1. Load and clean data
# -----------------------------------------------------------------------------
df = pd.read_csv('dataset.csv')

# Fill numeric NaNs with median
for col in ['AminerCitationCount', 'CitationCount_CrossRef',
            'Downloads_Xplore', 'PubsCited_CrossRef']:
    df[col] = df[col].fillna(df[col].median())

# Award flag
df['AwardFlag'] = df['Award'].notnull().astype(int)

# GraphicsReplicabilityStamp fill
df['GraphicsReplicabilityStamp'] = df['GraphicsReplicabilityStamp'].fillna('None')

# Helper functions to count keywords and references
def count_items(s):
    if pd.isna(s) or not isinstance(s, str) or s.strip() == '':
        return 0
    parts = re.split(r'[;,]', s)
    return sum(1 for p in parts if p.strip())

df['num_keywords']      = df['AuthorKeywords'].apply(count_items)
df['num_internal_refs'] = df['InternalReferences'].apply(count_items)

# -----------------------------------------------------------------------------
# 2. Enhanced Trend Analysis on Citation Counts
#    - Top 5 Conferences over Years
#    - All PaperTypes over Years
# -----------------------------------------------------------------------------
# Top 5 conferences
top5_conf = df['Conference'].value_counts().nlargest(5).index.tolist()
trend_conf = (df[df['Conference'].isin(top5_conf)]
              .groupby(['Year', 'Conference'])
              .agg(mean_citation=('CitationCount_CrossRef','mean'))
              .reset_index())

fig1_conf = px.line(
    trend_conf,
    x='Year', y='mean_citation', color='Conference',
    title='Mean CrossRef Citations by Year for Top 5 Conferences',
    line_shape='spline', markers=True
)
fig1_conf.update_layout(
    template='plotly_white',
    hovermode='x unified',
    xaxis=dict(title='Year', dtick=1, range=[df['Year'].min()-1, df['Year'].max()+1]),
    yaxis=dict(title='Average CrossRef Citations', rangemode='tozero'),
    legend=dict(title='Conference')
)

# By PaperType
trend_pt = (df.groupby(['Year','PaperType'])
            .agg(mean_cit=('CitationCount_CrossRef','mean'))
            .reset_index())
fig1_pt = px.line(
    trend_pt,
    x='Year', y='mean_cit', color='PaperType',
    title='Mean CrossRef Citations by Year and Paper Type',
    line_shape='spline', markers=True
)
fig1_pt.update_layout(
    template='plotly_white',
    hovermode='x unified',
    xaxis=dict(title='Year', dtick=1, range=[df['Year'].min()-1, df['Year'].max()+1]),
    yaxis=dict(title='Average CrossRef Citations', rangemode='tozero'),
    legend=dict(title='Paper Type')
)

# -----------------------------------------------------------------------------
# 3. Machine Learning: Predict Awards (ROC & Feature Importance)
# -----------------------------------------------------------------------------
X_award = df[['Year','Downloads_Xplore','num_keywords','num_internal_refs',
             'PaperType','Conference']].copy()
X_award = pd.get_dummies(X_award, columns=['PaperType','Conference'], drop_first=True)
y_award = df['AwardFlag']

X_tr, X_te, y_tr, y_te = train_test_split(
    X_award, y_award,
    test_size=0.3, random_state=42, stratify=y_award
)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_tr, y_tr)
y_score = clf.predict_proba(X_te)[:,1]

fpr, tpr, _ = roc_curve(y_te, y_score)
roc_auc = auc(fpr, tpr)

fig2_roc = go.Figure()
fig2_roc.add_trace(go.Scatter(
    x=fpr, y=tpr, mode='lines',
    line=dict(color='royalblue', width=3),
    name=f'ROC Curve (AUC = {roc_auc:.2f})',
    hovertemplate='FPR: %{x:.2f}<br>TPR: %{y:.2f}<extra></extra>'
))
fig2_roc.add_trace(go.Scatter(
    x=[0,1], y=[0,1], mode='lines',
    line=dict(color='gray', dash='dash'),
    name='Random Classifier'
))
fig2_roc.update_layout(
    title='ROC Curve for Award Prediction',
    template='plotly_dark',
    xaxis_title='False Positive Rate',
    yaxis_title='True Positive Rate',
    hovermode='closest'
)

# Feature importances
imps = pd.Series(clf.feature_importances_, index=X_award.columns)
imp_top = imps.sort_values(ascending=False).head(15).reset_index()
imp_top.columns = ['feature','importance']
fig2_imp = px.bar(
    imp_top, x='importance', y='feature',
    orientation='h',
    title='Top 15 Features for Award Prediction',
    color='importance', color_continuous_scale='Blues'
)
fig2_imp.update_layout(
    template='plotly_white',
    xaxis_title='Importance',
    yaxis_title='Feature',
    yaxis={'categoryorder':'total ascending'}
)

# -----------------------------------------------------------------------------
# 4. Regression Analysis: Predict Downloads
# -----------------------------------------------------------------------------
X_reg = df[['Year','AminerCitationCount','num_internal_refs',
            'GraphicsReplicabilityStamp','Conference']].copy()
X_reg = pd.get_dummies(X_reg,
                       columns=['GraphicsReplicabilityStamp','Conference'],
                       drop_first=True)
y_reg = df['Downloads_Xplore']

Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(
    X_reg, y_reg, test_size=0.3, random_state=42
)
regr = RandomForestRegressor(n_estimators=100, random_state=42)
regr.fit(Xr_tr, yr_tr)
yr_pred = regr.predict(Xr_te)

fig3 = go.Figure()
fig3.add_trace(go.Scatter(
    x=yr_te, y=yr_pred, mode='markers',
    marker=dict(size=6, color='darkorange', opacity=0.7),
    hovertemplate='Actual: %{x:.1f}<br>Predicted: %{y:.1f}<extra></extra>'
))
# Identity line
minv = min(yr_te.min(), yr_pred.min())
maxv = max(yr_te.max(), yr_pred.max())
fig3.add_trace(go.Line(
    x=[minv, maxv], y=[minv, maxv],
    line=dict(color='red', dash='dash'),
    name='Ideal Fit'
))
fig3.update_layout(
    title='Actual vs Predicted Downloads',
    template='plotly_white',
    xaxis_title='Actual Downloads',
    yaxis_title='Predicted Downloads',
    hovermode='closest'
)

# -----------------------------------------------------------------------------
# 5. Keyword Trend Analysis (Top 10 Keywords)
# -----------------------------------------------------------------------------
kw = df[['Year','AuthorKeywords']].dropna().copy()
kw = kw.assign(
    kwlist=kw['AuthorKeywords'].str.split(r'[;,]')
).explode('kwlist')
kw['kwlist'] = kw['kwlist'].str.strip().replace('', np.nan)
kw = kw.dropna(subset=['kwlist'])

top10kw = kw['kwlist'].value_counts().nlargest(10).index.tolist()
trend_kw = (kw[kw['kwlist'].isin(top10kw)]
            .groupby(['Year','kwlist'])
            .size().reset_index(name='count'))

fig4 = px.line(
    trend_kw, x='Year', y='count', color='kwlist',
    title='Yearly Trends of Top 10 Keywords',
    line_shape='spline', markers=True
)
fig4.update_layout(
    template='plotly_white',
    hovermode='x unified',
    xaxis=dict(title='Year', dtick=1, range=[df['Year'].min()-1, df['Year'].max()+1]),
    yaxis=dict(title='Count of Papers', rangemode='tozero'),
    legend_title='Keyword'
)

# -----------------------------------------------------------------------------
# 6. Impact of Awards on Citations & Downloads (Box + Jitter)
# -----------------------------------------------------------------------------
df['AwardLabel'] = df['AwardFlag'].map({0: 'No Award', 1: 'Award'})

# Citations
fig5_cit = go.Figure()
fig5_cit.add_trace(go.Box(
    x=df['AwardLabel'], y=df['CitationCount_CrossRef'],
    name='Citations',
    boxpoints='all', jitter=0.3, pointpos=-1.8,
    marker_color='lightseagreen',
    hovertemplate='Award: %{x}<br>Citations: %{y}<extra></extra>'
))
fig5_cit.update_layout(
    title='Citation Count by Award Status',
    template='plotly_white',
    yaxis=dict(title='CitationCount_CrossRef', rangemode='tozero'),
    xaxis=dict(title='Award Status')
)

# Downloads
fig5_dwn = go.Figure()
fig5_dwn.add_trace(go.Box(
    x=df['AwardLabel'], y=df['Downloads_Xplore'],
    name='Downloads',
    boxpoints='all', jitter=0.3, pointpos=-1.8,
    marker_color='mediumpurple',
    hovertemplate='Award: %{x}<br>Downloads: %{y}<extra></extra>'
))
fig5_dwn.update_layout(
    title='Download Count by Award Status',
    template='plotly_white',
    yaxis=dict(title='Downloads_Xplore', rangemode='tozero'),
    xaxis=dict(title='Award Status')
)

# -----------------------------------------------------------------------------
# 7. Write all figures to an HTML file for online display
# -----------------------------------------------------------------------------
html_parts = []

# Include plotly.js once
html_parts.append(fig1_conf.to_html(full_html=False, include_plotlyjs='cdn'))
html_parts.append("<hr>")
html_parts.append(fig1_pt.to_html(full_html=False, include_plotlyjs=False))
html_parts.append("<hr>")
html_parts.append(fig2_roc.to_html(full_html=False, include_plotlyjs=False))
html_parts.append("<hr>")
html_parts.append(fig2_imp.to_html(full_html=False, include_plotlyjs=False))
html_parts.append("<hr>")
html_parts.append(fig3.to_html(full_html=False, include_plotlyjs=False))
html_parts.append("<hr>")
html_parts.append(fig4.to_html(full_html=False, include_plotlyjs=False))
html_parts.append("<hr>")
html_parts.append(fig5_cit.to_html(full_html=False, include_plotlyjs=False))
html_parts.append("<hr>")
html_parts.append(fig5_dwn.to_html(full_html=False, include_plotlyjs=False))

html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8" />
    <title>Interactive Plotly Visualizations</title>
    <style>
      body {{ font-family: Arial, sans-serif; margin: 0 40px; }}
      h1 {{ text-align: center; margin-top: 20px; }}
      hr {{ margin: 50px 0; border: none; border-top: 1px solid #eee; }}
    </style>
</head>
<body>
    <h1>Dataset Interactive Visualizations</h1>
    {' '.join(html_parts)}
</body>
</html>
"""

with open('output.html', 'w', encoding='utf-8') as f:
    f.write(html_content)

print("output.html has been generated with all interactive figures.")

