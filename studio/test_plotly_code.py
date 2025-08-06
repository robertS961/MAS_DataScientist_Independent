# ---- NEW BLOCK ---- # 
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.linear_model import LinearRegression

# -----------------------------------------------------------------------------
# 0. Set default Plotly theme to dark
# -----------------------------------------------------------------------------
pio.templates.default = "plotly_dark"

# -----------------------------------------------------------------------------
# 1. Load and preprocess data
# -----------------------------------------------------------------------------
df = pd.read_csv('dataset.csv')

# Fill numeric NaNs with median
for col in ['AminerCitationCount', 'CitationCount_CrossRef', 'Downloads_Xplore']:
    df[col] = df[col].fillna(df[col].median())

# -----------------------------------------------------------------------------
# 2. Enhanced Trend Analysis on Citation Counts
#    (a) By top 5 Conferences
# -----------------------------------------------------------------------------
top_confs = df['Conference'].value_counts().nlargest(5).index.to_list()
df_conf = df[df['Conference'].isin(top_confs)]
df_conf_melt = (
    df_conf
    .groupby(['Year','Conference'])[['AminerCitationCount','CitationCount_CrossRef']]
    .mean()
    .reset_index()
    .melt(id_vars=['Year','Conference'],
          value_vars=['AminerCitationCount','CitationCount_CrossRef'],
          var_name='Metric', value_name='AvgCitation')
)

fig1 = px.line(
    df_conf_melt,
    x='Year', y='AvgCitation',
    color='Conference',
    line_dash='Metric',
    markers=True,
    color_discrete_sequence=px.colors.qualitative.Vivid,
    title='Average Citation Trends by Conference (Top 5)',
    labels={'AvgCitation':'Average Citation', 'Metric':'Citation Type'}
)
fig1.update_layout(
    template='plotly_dark',
    xaxis=dict(
        title='Year',
        range=[df.Year.min(), df.Year.max()],
        rangeslider=dict(visible=True),
        showgrid=False
    ),
    yaxis=dict(title='Avg Citation', showgrid=False),
    legend_title_text='Conference / Metric'
)
fig1.update_traces(hovertemplate='Year: %{x}<br>%{fullData.name}<br>Citation: %{y:.1f}')

# -----------------------------------------------------------------------------
#    (b) By top 5 Paper Types
# -----------------------------------------------------------------------------
top_ptypes = df['PaperType'].value_counts().nlargest(5).index.to_list()
df_pt = df[df['PaperType'].isin(top_ptypes)]
df_pt_melt = (
    df_pt
    .groupby(['Year','PaperType'])[['AminerCitationCount','CitationCount_CrossRef']]
    .mean()
    .reset_index()
    .melt(id_vars=['Year','PaperType'],
          value_vars=['AminerCitationCount','CitationCount_CrossRef'],
          var_name='Metric', value_name='AvgCitation')
)

fig2 = px.line(
    df_pt_melt,
    x='Year', y='AvgCitation',
    color='PaperType',
    line_dash='Metric',
    markers=True,
    color_discrete_sequence=px.colors.qualitative.Bold,
    title='Average Citation Trends by Paper Type (Top 5)',
    labels={'AvgCitation':'Average Citation', 'Metric':'Citation Type'}
)
fig2.update_layout(
    template='plotly_dark',
    xaxis=dict(
        title='Year',
        range=[df.Year.min(), df.Year.max()],
        rangeslider=dict(visible=True),
        showgrid=False
    ),
    yaxis=dict(title='Avg Citation', showgrid=False),
    legend_title_text='PaperType / Metric'
)
fig2.update_traces(hovertemplate='Year: %{x}<br>%{fullData.name}<br>Citation: %{y:.1f}')

# -----------------------------------------------------------------------------
# 3. Machine Learning to Predict Awards
# -----------------------------------------------------------------------------
df_ml = df.copy()
df_ml['Award_flag'] = df_ml['Award'].notna().astype(int)

# features: number of keywords & internal references
df_ml['n_keywords'] = (
    df_ml['AuthorKeywords']
      .fillna('')
      .apply(lambda x: len([kw.strip() for kw in x.replace(';',',').split(',') if kw.strip()]))
)
df_ml['n_int_refs'] = (
    df_ml['InternalReferences']
      .fillna('')
      .apply(lambda x: len([r.strip() for r in x.replace(';',',').split(',') if r.strip()]))
)

# one-hot encode PaperType & Conference
df_ml = pd.get_dummies(df_ml, columns=['PaperType','Conference'], drop_first=True)

features_ml = (
    ['Year','Downloads_Xplore','n_keywords','n_int_refs'] +
    [c for c in df_ml.columns if c.startswith('PaperType_') or c.startswith('Conference_')]
)
X = df_ml[features_ml]
y = df_ml['Award_flag']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_prob = clf.predict_proba(X_test)[:,1]
y_pred = clf.predict(X_test)

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
fig3 = go.Figure()
fig3.add_trace(go.Scatter(
    x=fpr, y=tpr,
    mode='lines+markers',
    name=f'ROC (AUC={roc_auc:.2f})',
    line=dict(color='lime', width=3)
))
fig3.add_trace(go.Scatter(
    x=[0,1], y=[0,1],
    mode='lines',
    name='Chance',
    line=dict(color='gray', dash='dash')
))
fig3.update_layout(
    template='plotly_dark',
    title='ROC Curve for Award Prediction',
    xaxis=dict(title='False Positive Rate', range=[0,1]),
    yaxis=dict(title='True Positive Rate', range=[0,1]),
    legend=dict(x=0.7, y=0.2)
)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
fig4 = go.Figure(data=go.Heatmap(
    z=cm,
    x=['Predicted 0','Predicted 1'],
    y=['Actual 0','Actual 1'],
    text=cm,
    texttemplate="%{text}",
    colorscale='Viridis',
    showscale=True
))
fig4.update_layout(
    template='plotly_dark',
    title='Confusion Matrix for Award Prediction',
    xaxis=dict(title='Predicted'),
    yaxis=dict(title='Actual')
)

# -----------------------------------------------------------------------------
# 4. Regression Analysis for Download Prediction
# -----------------------------------------------------------------------------
df_reg = df.copy()
df_reg['GraphicsReplicabilityStamp'] = df_reg['GraphicsReplicabilityStamp'].fillna('None')
df_reg['n_int_refs'] = df_reg['InternalReferences'].fillna('').apply(
    lambda x: len([r.strip() for r in x.replace(';',',').split(',') if r.strip()])
)

# one-hot encode GraphicsReplicabilityStamp & Conference
df_reg = pd.get_dummies(
    df_reg,
    columns=['GraphicsReplicabilityStamp','Conference'],
    drop_first=True
)

features_reg = (
    ['AminerCitationCount','n_int_refs','Year'] +
    [c for c in df_reg.columns if c.startswith('GraphicsReplicabilityStamp_') or c.startswith('Conference_')]
)
Xr = df_reg[features_reg]
yr = df_reg['Downloads_Xplore']

Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    Xr, yr, test_size=0.3, random_state=42
)
lr = LinearRegression()
lr.fit(Xr_train, yr_train)
yr_pred = lr.predict(Xr_test)

# Coefficients Bar (horizontal)
coef_df = pd.DataFrame({
    'Feature': features_reg,
    'Coefficient': lr.coef_
})
coef_df['abs_coef'] = coef_df['Coefficient'].abs()
coef_df = coef_df.sort_values('abs_coef', ascending=True)

fig5 = px.bar(
    coef_df,
    x='Coefficient', y='Feature',
    orientation='h',
    color='Coefficient',
    color_continuous_scale='Plasma',
    title='Regression Coefficients for Download Prediction',
    labels={'Coefficient':'Coef Value','Feature':'Feature'}
)
fig5.update_layout(template='plotly_dark', margin=dict(l=200))

# Predicted vs Actual Scatter
fig6 = go.Figure()
fig6.add_trace(go.Scatter(
    x=yr_test, y=yr_pred,
    mode='markers',
    marker=dict(color='cyan', size=6),
    name='Predicted vs Actual'
))
fig6.add_trace(go.Scatter(
    x=[yr_test.min(), yr_test.max()],
    y=[yr_test.min(), yr_test.max()],
    mode='lines',
    line=dict(color='magenta', dash='dash'),
    name='Ideal'
))
fig6.update_layout(
    template='plotly_dark',
    title='Predicted vs Actual Downloads',
    xaxis=dict(title='Actual Downloads'),
    yaxis=dict(title='Predicted Downloads')
)

# -----------------------------------------------------------------------------
# 5. Keyword Trend Analysis
# -----------------------------------------------------------------------------
kw_df = df[['Year','AuthorKeywords']].copy()
kw_df['AuthorKeywords'] = kw_df['AuthorKeywords'].fillna('')
kw_df['KeywordList'] = kw_df['AuthorKeywords'].apply(
    lambda x: [kw.strip() for kw in x.replace(';',',').split(',') if kw.strip()]
)
kw_exp = kw_df.explode('KeywordList')
kw_exp = kw_exp[kw_exp['KeywordList'] != '']

top_keywords = kw_exp['KeywordList'].value_counts().nlargest(10).index.to_list()
kw_top = kw_exp[kw_exp['KeywordList'].isin(top_keywords)]
kw_trend = kw_top.groupby(['Year','KeywordList']).size().reset_index(name='Count')

fig7 = px.line(
    kw_trend,
    x='Year', y='Count',
    color='KeywordList',
    markers=True,
    title='Top 10 Keyword Trends Over Years',
    labels={'Count':'Frequency','KeywordList':'Keyword'},
    color_discrete_sequence=px.colors.qualitative.Safe
)
fig7.update_layout(
    template='plotly_dark',
    xaxis=dict(
        title='Year',
        range=[df.Year.min(), df.Year.max()],
        rangeslider=dict(visible=True),
        showgrid=False
    ),
    yaxis=dict(title='Count', showgrid=False),
    legend_title_text='Keyword'
)
fig7.update_traces(hovertemplate='%{fullData.name}<br>Year: %{x}<br>Count: %{y}')

# -----------------------------------------------------------------------------
# 6. Impact of Awards on Citations and Downloads
# -----------------------------------------------------------------------------
df_aw = df.copy()
df_aw['Award_flag'] = df_aw['Award'].notna().astype(int)
stats = df_aw.groupby('Award_flag')[['CitationCount_CrossRef','Downloads_Xplore']].mean().reset_index()
stats_melt = stats.melt(
    id_vars='Award_flag',
    value_vars=['CitationCount_CrossRef','Downloads_Xplore'],
    var_name='Metric',
    value_name='Average'
)
stats_melt['Status'] = stats_melt['Award_flag'].map({0:'Non-Awarded',1:'Awarded'})

fig8 = px.bar(
    stats_melt,
    x='Metric', y='Average',
    color='Status',
    barmode='group',
    title='Average Citations & Downloads: Awarded vs Non-Awarded',
    labels={'Metric':'Metric','Average':'Mean Value'}
)
fig8.update_layout(
    template='plotly_dark',
    xaxis=dict(title='Metric'),
    yaxis=dict(title='Mean'),
    legend_title_text='Paper Status'
)

# -----------------------------------------------------------------------------
# 7. Export all figures to a single HTML file
# -----------------------------------------------------------------------------
figs = [fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8]
html_parts = []
for i, fig in enumerate(figs, start=1):
    html_parts.append(
        fig.to_html(
            include_plotlyjs='cdn',
            full_html=False,
            div_id=f'fig{i}'
        )
    )

html_page = (
    '<!DOCTYPE html>'
    '<html><head><meta charset="utf-8"><title>Interactive Plots</title></head>'
    '<body style="background-color:#1e1e1e; margin:0; padding:20px;">\n'
    + "\n".join(html_parts) +
    '\n</body></html>'
)

with open('output.html', 'w', encoding='utf-8') as f:
    f.write(html_page)

print("✅ output.html generated with enhanced interactive figures.")

