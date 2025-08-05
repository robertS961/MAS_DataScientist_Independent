# ---- NEW BLOCK ---- # 
import pandas as pd
import numpy as np
import networkx as nx
from itertools import combinations
from sklearn.linear_model import LinearRegression
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# 1. Load and clean data
df = pd.read_csv('dataset.csv')

# Fill numeric NaNs with median
num_cols = ['AminerCitationCount', 'CitationCount_CrossRef', 'PubsCited_CrossRef', 'Downloads_Xplore']
for c in num_cols:
    df[c] = df[c].fillna(df[c].median())

# Fill Award missing with 'No Award'
df['Award'] = df['Award'].fillna('No Award')

# -------------------------------------------------
# 2. Collaborative Network Analysis
#    Build a co‐affiliation network: authors sharing same affiliation get connected
# -------------------------------------------------
# Explode authors and affiliations
a = df[['AuthorNames-Deduped', 'AuthorAffiliation']].dropna()
a['AuthorList'] = a['AuthorNames-Deduped'].str.split(';')
a['AffilList'] = a['AuthorAffiliation'].str.split(';')
rows = []
for _, row in a.iterrows():
    authors = [x.strip() for x in row['AuthorList'] if x.strip()]
    affils  = [x.strip() for x in row['AffilList'] if x.strip()]
    for auth in authors:
        for aff in affils:
            rows.append((auth, aff))
aa = pd.DataFrame(rows, columns=['Author','Affiliation'])

# Build graph
G = nx.Graph()
for affil, grp in aa.groupby('Affiliation'):
    auths = grp['Author'].unique().tolist()
    for u, v in combinations(auths, 2):
        if G.has_edge(u,v):
            G[u][v]['weight'] += 1
        else:
            G.add_edge(u, v, weight=1)

# Take largest connected component
components = sorted(nx.connected_components(G), key=len, reverse=True)
main_nodes = components[0]
G_sub = G.subgraph(main_nodes).copy()

# Position nodes
pos = nx.spring_layout(G_sub, k=0.5, iterations=50)
edge_x = []
edge_y = []
for u, v in G_sub.edges():
    x0, y0 = pos[u]
    x1, y1 = pos[v]
    edge_x += [x0, x1, None]
    edge_y += [y0, y1, None]
edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=0.5, color='#888'),
    hoverinfo='none',
    mode='lines'
)
node_x, node_y, node_text = [], [], []
for n in G_sub.nodes():
    x, y = pos[n]
    node_x.append(x)
    node_y.append(y)
    node_text.append(n)
node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode='markers+text',
    text=node_text,
    textposition='top center',
    hoverinfo='text',
    marker=dict(size=8, color='skyblue', line_width=1)
)
fig1 = go.Figure(data=[edge_trace, node_trace])
fig1.update_layout(title='Author Collaboration Network (by Shared Affiliation)',
                   showlegend=False, margin=dict(l=20,r=20,t=40,b=20))

# -------------------------------------------------
# 3. Conference Impact Assessment
#    Avg citation counts per year for top 5 conferences
# -------------------------------------------------
top5 = df['Conference'].value_counts().nlargest(5).index.tolist()
df5 = df[df['Conference'].isin(top5)]
df_grp = df5.groupby(['Year','Conference'])[['AminerCitationCount','CitationCount_CrossRef']].mean().reset_index()
fig2 = px.line(df_grp, x='Year', y=['AminerCitationCount','CitationCount_CrossRef'],
               color='Conference', line_dash='variable',
               labels={'value':'Average Citations','variable':'Citation Source'},
               title='Avg Citation Counts over Years for Top 5 Conferences')

# -------------------------------------------------
# 4. Keyword Trend Analysis
#    Top 10 keywords frequency per year (stacked area)
# -------------------------------------------------
# explode keywords
kw = df[['Year','AuthorKeywords']].dropna()
kw = kw.assign(Keyword=kw['AuthorKeywords'].str.split(';')).explode('Keyword')
kw['Keyword'] = kw['Keyword'].str.strip().str.lower()
# filter top 10
top10 = kw['Keyword'].value_counts().nlargest(10).index.tolist()
kw10 = kw[kw['Keyword'].isin(top10)]
kw_grp = kw10.groupby(['Year','Keyword']).size().reset_index(name='Count')
fig3 = px.area(kw_grp, x='Year', y='Count', color='Keyword',
               title='Trends of Top 10 Author Keywords Over Years')

# -------------------------------------------------
# 5. Regression Analysis for Downloads Prediction
#    Scatter and regression line: Downloads vs AminerCitationCount
# -------------------------------------------------
X = df[['AminerCitationCount']].values
y = df['Downloads_Xplore'].values
model = LinearRegression().fit(X, y)
x_range = np.linspace(X.min(), X.max(), 100).reshape(-1,1)
y_pred = model.predict(x_range)
fig4 = go.Figure()
fig4.add_trace(go.Scatter(x=df['AminerCitationCount'], y=df['Downloads_Xplore'],
                          mode='markers', name='Data',
                          marker=dict(color='darkblue', opacity=0.6)))
fig4.add_trace(go.Line(x=x_range.flatten(), y=y_pred,
                       line=dict(color='red'), name='Fit: y={:.2f}x+{:.2f}'.format(model.coef_[0], model.intercept_)))
fig4.update_layout(title='Downloads_Xplore vs AminerCitationCount with Regression Line',
                   xaxis_title='AminerCitationCount', yaxis_title='Downloads_Xplore')

# -------------------------------------------------
# 6. Impact of Awards on Citations & Downloads
#    Box plots comparing Award vs No Award
# -------------------------------------------------
fig5 = make_subplots(rows=1, cols=2, subplot_titles=('Citations by Award','Downloads by Award'))
fig5.add_trace(
    go.Box(x=df['Award'], y=df['CitationCount_CrossRef'], name='Citations'),
    row=1, col=1
)
fig5.add_trace(
    go.Box(x=df['Award'], y=df['Downloads_Xplore'], name='Downloads'),
    row=1, col=2
)
fig5.update_layout(title_text='Impact of Awards on Citation & Download Counts', showlegend=False)

# -------------------------------------------------
# 7. Write to HTML (output.html)
# -------------------------------------------------
from plotly.offline import plot

plots = []
for fig in [fig1, fig2, fig3, fig4, fig5]:
    plots.append(fig.to_html(full_html=False, include_plotlyjs='cdn'))

html_page = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8" />
    <title>Interactive Visualizations</title>
</head>
<body>
    <h1>Interactive Data Science Visualizations</h1>
    {''.join(plots)}
</body>
</html>
"""

with open('output.html', 'w', encoding='utf-8') as f:
    f.write(html_page)

print("output.html generated successfully.")

