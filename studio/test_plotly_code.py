# ---- NEW BLOCK ---- # 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score

import plotly.express as px
import plotly.graph_objects as go

# 0. LOAD & PREPROCESS
# --------------------
df = pd.read_csv('dataset.csv')

# Fill numeric NaNs
for col in ['FirstPage','LastPage','AminerCitationCount','Downloads_Xplore']:
    df[col].fillna(df[col].mean(), inplace=True)

# Fill text NaNs
df['Abstract'].fillna('', inplace=True)
df['AuthorKeywords'].fillna('', inplace=True)
df['GraphicsReplicabilityStamp'].fillna('No', inplace=True)

# Binary/categorical transforms
df['Award'] = df['Award'].notna().astype(int)
df['GraphicsReplicabilityStamp'] = (df['GraphicsReplicabilityStamp']=='Yes').astype(int)

# We will need an "AwardLabel" for plotting
df['AwardLabel'] = df['Award'].map({0:'No',1:'Yes'})

imputer = SimpleImputer(strategy='mean')


# 1. Citation Trend Over Time by Conference
# -----------------------------------------
fig1 = px.line(
    df,
    x='Year', y='AminerCitationCount',
    color='Conference',
    title='Citation Trend Over Time by Conference',
    color_discrete_sequence=px.colors.qualitative.Plotly
)
fig1.update_layout(
    xaxis_title='Year',
    yaxis_title='Aminer Citation Count',
    legend_title='Conference',
    margin={'r':200}
)


# 2. Feature Importance for Award Prediction
# ------------------------------------------
# Prepare data
X_award = df[['PaperType','Downloads_Xplore','Year','Conference']]
X_award = pd.get_dummies(X_award, drop_first=True)
y_award = df['Award']

X_award_imp = imputer.fit_transform(X_award)
X_tr, X_te, y_tr, y_te = train_test_split(X_award_imp, y_award, test_size=0.2, random_state=42)

logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_tr, y_tr)

coefs = pd.Series(logreg.coef_[0], index=X_award.columns).sort_values()

fig2 = go.Figure(
    go.Bar(
        x=coefs.values,
        y=coefs.index,
        orientation='h',
        marker_color='steelblue'
    )
)
fig2.update_layout(
    title='Logistic Regression Feature Importance',
    xaxis_title='Coefficient Value',
    yaxis_title='Feature',
    margin={'l':200}
)


# 3. Regression Analysis for Download Prediction
# ----------------------------------------------
X_dl = df[['AminerCitationCount','GraphicsReplicabilityStamp','Year']]
y_dl = df['Downloads_Xplore']

X_dl_imp = imputer.fit_transform(X_dl)
Xtr, Xte, ytr, yte = train_test_split(X_dl_imp, y_dl, test_size=0.2, random_state=42)

lr = LinearRegression()
lr.fit(Xtr, ytr)
y_pred = lr.predict(Xte)

mse = mean_squared_error(yte, y_pred)
r2  = r2_score(yte, y_pred)

lims = [ min(yte.min(), y_pred.min()), max(yte.max(), y_pred.max()) ]

fig3 = go.Figure()
# scatter
fig3.add_trace(
    go.Scatter(
        x=yte,
        y=y_pred,
        mode='markers',
        marker=dict(color='teal', size=8, opacity=0.6, line=dict(color='black', width=1)),
        name='Pred vs True'
    )
)
# 45° line
fig3.add_trace(
    go.Scatter(
        x=lims, y=lims,
        mode='lines',
        line=dict(color='red', dash='dash', width=1.5),
        showlegend=False
    )
)
# annotation
fig3.add_annotation(
    x=0.05, y=0.95,
    xref='paper', yref='paper',
    text=f"MSE: {mse:.2f}<br>R²: {r2:.2f}",
    showarrow=False,
    align='left',
    bgcolor='white',
    bordercolor='black'
)
fig3.update_layout(
    title='Download Prediction: True vs. Predicted',
    xaxis_title='True Download Count',
    yaxis_title='Predicted Download Count',
    margin={'l':80,'r':80,'t':80,'b':80}
)


# 4. Top 20 Author Keywords
# -------------------------
# explode & count
df['AuthorKeywords'] = df['AuthorKeywords'].str.lower().str.split(',\s*')
all_kw = [kw for sub in df['AuthorKeywords'] for kw in sub if kw]
kw_counts = pd.Series(all_kw).value_counts().head(20)

fig4 = go.Figure(
    go.Bar(
        x=kw_counts.values,
        y=kw_counts.index,
        orientation='h',
        marker=dict(color=px.colors.sequential.Viridis[:20])
    )
)
fig4.update_layout(
    title='Top 20 Author Keywords',
    xaxis_title='Frequency',
    yaxis_title='Keyword',
    margin={'l':200}
)


# 5. Citation Distribution by Award Status
# ----------------------------------------
fig5 = px.histogram(
    df,
    x='CitationCount_CrossRef',
    color='AwardLabel',
    histnorm='density',
    barmode='overlay',
    opacity=0.7,
    color_discrete_map={'No':'lightgray','Yes':'goldenrod'},
    title='Citation Distribution by Award Status'
)
fig5.update_traces(marker_line_width=1.5)
fig5.update_layout(
    xaxis_title='Citation Count (CrossRef)',
    yaxis_title='Density',
    legend_title='Award',
    margin={'l':80,'r':80,'t':80,'b':80}
)


# WRITE TO A SINGLE HTML FILE
# ---------------------------
html_parts = []
# First figure includes plotly.js
html_parts.append(fig1.to_html(full_html=False, include_plotlyjs='cdn'))
# Subsequent figures do not re‐include the library
for fig in [fig2, fig3, fig4, fig5]:
    html_parts.append(fig.to_html(full_html=False, include_plotlyjs=False))

html_body = "\n<hr/>\n".join(html_parts)

html_page = f"""
<!DOCTYPE html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>Interactive Plots</title>
  </head>
  <body>
    {html_body}
  </body>
</html>
"""

with open('output.html', 'w', encoding='utf-8') as f:
    f.write(html_page)

print("All figures have been written to output.html")

