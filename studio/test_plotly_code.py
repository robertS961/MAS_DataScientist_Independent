# ---- NEW BLOCK ---- # 
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score

# 0. LOAD & PREPROCESS
df = pd.read_csv('dataset.csv')

# Fill NaNs in numeric columns
for col in ['FirstPage','LastPage','AminerCitationCount','Downloads_Xplore']:
    df[col].fillna(df[col].mean(), inplace=True)
# Fill NaNs in text columns
df['Abstract'].fillna('', inplace=True)
df['AuthorKeywords'].fillna('', inplace=True)
# GraphicsReplicabilityStamp: assume 'Yes'/'No'
df['GraphicsReplicabilityStamp'].fillna('No', inplace=True)

# Convert Award and GraphicsReplicabilityStamp to binary
df['Award'] = df['Award'].notna().astype(int)
df['GraphicsReplicabilityStamp'] = (df['GraphicsReplicabilityStamp']=='Yes').astype(int)

# Imputer
imputer = SimpleImputer(strategy='mean')

# 1. TREND OF AMINER CITATION COUNTS BY CONFERENCE
fig1 = px.line(
    df, x='Year', y='AminerCitationCount', color='Conference',
    title='Citation Trend Over Time by Conference',
    labels={'Year':'Year','AminerCitationCount':'Aminer Citation Count'},
    color_discrete_sequence=px.colors.qualitative.Tab10
)
fig1.update_layout(
    legend_title_text='Conference',
    width=1100, height=500,
    margin=dict(l=50,r=200,t=80,b=50)
)

# 2. FEATURE IMPORTANCE FOR AWARD PREDICTION (LOGISTIC REGRESSION)
# Prepare features
X_award = df[['PaperType','Downloads_Xplore','Year','Conference']]
X_award = pd.get_dummies(X_award, drop_first=True)
y_award = df['Award']

# Impute & split
X_award_imp = imputer.fit_transform(X_award)
X_tr, X_te, y_tr, y_te = train_test_split(
    X_award_imp, y_award, test_size=0.2, random_state=42
)

# Train Logistic Regression
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_tr, y_tr)

# Coefficients
coefs = pd.Series(logreg.coef_[0], index=X_award.columns).sort_values()

fig2 = go.Figure(go.Bar(
    x=coefs.values, y=coefs.index,
    orientation='h',
    marker_color='steelblue'
))
fig2.update_layout(
    title='Logistic Regression Feature Importance',
    xaxis_title='Coefficient Value',
    yaxis_title='Feature',
    width=800, height=600,
    margin=dict(l=150, r=50, t=80, b=50)
)

# 3. REGRESSION ANALYSIS FOR DOWNLOAD PREDICTION
X_dl = df[['AminerCitationCount','GraphicsReplicabilityStamp','Year']]
y_dl = df['Downloads_Xplore']

X_dl_imp = imputer.fit_transform(X_dl)
Xtr, Xte, ytr, yte = train_test_split(
    X_dl_imp, y_dl, test_size=0.2, random_state=42
)

lr = LinearRegression()
lr.fit(Xtr, ytr)
y_pred = lr.predict(Xte)

mse = mean_squared_error(yte, y_pred)
r2 = r2_score(yte, y_pred)

# Scatter + 45° line
lims = [min(yte.min(), y_pred.min()), max(yte.max(), y_pred.max())]

fig3 = go.Figure()
fig3.add_trace(go.Scatter(
    x=yte, y=y_pred,
    mode='markers',
    marker=dict(color='teal', size=8, line=dict(color='black', width=0.5)),
    name='Predictions'
))
fig3.add_trace(go.Scatter(
    x=lims, y=lims,
    mode='lines',
    line=dict(dash='dash', color='red', width=1.5),
    name='45° Reference'
))
fig3.update_layout(
    title='Download Prediction: True vs. Predicted',
    xaxis_title='True Download Count',
    yaxis_title='Predicted Download Count',
    width=700, height=700,
    margin=dict(l=60, r=60, t=80, b=60),
    annotations=[dict(
        x=0.05, y=0.95,
        xref='paper', yref='paper',
        text=f"MSE: {mse:.2f}<br>R²: {r2:.2f}",
        showarrow=False,
        align='left',
        bordercolor='black',
        borderwidth=1
    )]
)

# 4. TOP 20 AUTHOR KEYWORDS
# Split, flatten, count
df['AuthorKeywords'] = df['AuthorKeywords']\
    .str.lower()\
    .str.split(',\s*', regex=True)
all_kw = [kw for lst in df['AuthorKeywords'] for kw in lst if kw]
kw_counts = pd.Series(all_kw).value_counts().head(20)

fig4 = px.bar(
    x=kw_counts.values, y=kw_counts.index,
    orientation='h',
    title='Top 20 Author Keywords',
    labels={'x':'Frequency','y':'Keyword'},
    color_discrete_sequence=px.colors.sequential.Viridis
)
fig4.update_layout(
    width=1100, height=500,
    margin=dict(l=150, r=50, t=80, b=50)
)

# 5. IMPACT OF AWARDS ON CITATION COUNT (CROSSREF)
fig5 = px.histogram(
    df, x='CitationCount_CrossRef', color='Award',
    title='Citation Distribution by Award Status',
    labels={'CitationCount_CrossRef':'Citation Count (CrossRef)','density':'Density'},
    histnorm='density',
    barmode='overlay',
    opacity=0.7,
    color_discrete_map={0:'lightgray',1:'goldenrod'}
)
fig5.update_layout(
    legend_title_text='Award',
    legend=dict(
        traceorder='reversed',
        itemsizing='constant',
        title_side='top'
    ),
    width=1100, height=500,
    margin=dict(l=60, r=60, t=80, b=60)
)

# COLLECT ALL FIGURES
figs = [fig1, fig2, fig3, fig4, fig5]

# WRITE TO OUTPUT.HTML
# We will embed plotly.js once, then all divs:
html_parts = []
for i, fig in enumerate(figs):
    html_parts.append(
        fig.to_html(full_html=False,
                    include_plotlyjs=(i==0))
    )

html_page = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Interactive Plotly Figures</title>
</head>
<body>
  {plots}
</body>
</html>
""".format(plots="\n".join(html_parts))

with open('output.html', 'w', encoding='utf-8') as f:
    f.write(html_page)

print("All figures have been saved to output.html")

