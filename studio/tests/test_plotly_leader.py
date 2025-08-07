from helper_functions import initialize_state_from_csv, define_variables, pretty_print_messages, get_datainfo, data_describe
from agents import plotly_enhancer_leader
from classes import State, Configurable
import re

code ='''
# ---- NEW BLOCK ---- # 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score

import plotly.express as px
import plotly.graph_objects as go

# 0. PREPROCESSING & IMPUTATION
# ----------------------------
df = pd.read_csv('dataset.csv')

df['FirstPage'].fillna(df['FirstPage'].mean(), inplace=True)
df['LastPage'].fillna(df['LastPage'].mean(), inplace=True)
df['AminerCitationCount'].fillna(df['AminerCitationCount'].mean(), inplace=True)
df['Downloads_Xplore'].fillna(df['Downloads_Xplore'].mean(), inplace=True)

df['Abstract'].fillna('', inplace=True)
df['AuthorKeywords'].fillna('', inplace=True)
df['GraphicsReplicabilityStamp'].fillna('No', inplace=True)

df['Award'] = df['Award'].notna().astype(int)
df['GraphicsReplicabilityStamp'] = (df['GraphicsReplicabilityStamp'] == 'Yes').astype(int)

imputer = SimpleImputer(strategy='mean')


# 1. Citation Trend Over Time by Conference
# -----------------------------------------
fig1 = px.line(
    df,
    x='Year',
    y='AminerCitationCount',
    color='Conference',
    title='Citation Trend Over Time by Conference',
    labels={'Year': 'Year', 'AminerCitationCount': 'Aminer Citation Count'},
    color_discrete_sequence=px.colors.qualitative.T10
)
fig1.update_layout(
    legend=dict(title='Conference', x=1.02, y=1, xanchor='left'),
    margin=dict(l=50, r=200, t=50, b=50),
    title_font_size=14,
    xaxis_title_font_size=12,
    yaxis_title_font_size=12
)


# 2. Feature Importance for Award Prediction (Logistic Regression)
# ----------------------------------------------------------------
X_award = df[['PaperType', 'Downloads_Xplore', 'Year', 'Conference']]
X_award = pd.get_dummies(X_award, drop_first=True)
y_award = df['Award']

X_award_imp = imputer.fit_transform(X_award)
X_tr, X_te, y_tr, y_te = train_test_split(
    X_award_imp, y_award, test_size=0.2, random_state=42
)

logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_tr, y_tr)

coefs = pd.Series(logreg.coef_[0], index=X_award.columns).sort_values()

fig2 = go.Figure(go.Bar(
    x=coefs.values,
    y=coefs.index,
    orientation='h',
    marker_color='steelblue'
))
fig2.update_layout(
    title='Logistic Regression Feature Importance',
    xaxis_title='Coefficient Value',
    yaxis_title='Feature',
    height=600,
    width=800,
    title_font_size=14,
    xaxis_title_font_size=12,
    yaxis_title_font_size=12,
    margin=dict(l=150, r=50, t=50, b=50)
)


# 3. Regression Analysis for Download Prediction
# ----------------------------------------------
X_dl = df[['AminerCitationCount', 'GraphicsReplicabilityStamp', 'Year']]
y_dl = df['Downloads_Xplore']

X_dl_imp = imputer.fit_transform(X_dl)
Xtr, Xte, ytr, yte = train_test_split(X_dl_imp, y_dl, test_size=0.2, random_state=42)

lr = LinearRegression()
lr.fit(Xtr, ytr)
y_pred = lr.predict(Xte)

mse = mean_squared_error(yte, y_pred)
r2 = r2_score(yte, y_pred)

lims = [
    min(yte.min(), y_pred.min()),
    max(yte.max(), y_pred.max())
]

fig3 = go.Figure()
# scatter
fig3.add_trace(go.Scatter(
    x=yte,
    y=y_pred,
    mode='markers',
    marker=dict(color='rgba(0,128,128,0.6)', size=10),
    name='Data'
))
# 45° line
fig3.add_trace(go.Scatter(
    x=lims,
    y=lims,
    mode='lines',
    line=dict(color='red', dash='dash', width=1.5),
    name='45° line'
))
# annotation
fig3.add_annotation(
    xref='paper', yref='paper',
    x=0.05, y=0.95,
    text=f"MSE: {mse:.2f}<br>R²: {r2:.2f}",
    showarrow=False,
    bgcolor='white',
    bordercolor='black',
    borderwidth=1,
    opacity=0.7,
    align='left'
)
fig3.update_layout(
    title='Download Prediction: True vs. Predicted',
    xaxis_title='True Download Count',
    yaxis_title='Predicted Download Count',
    height=700,
    width=700,
    title_font_size=14,
    xaxis_title_font_size=12,
    yaxis_title_font_size=12,
    margin=dict(l=50, r=50, t=50, b=50)
)


# 4. Top 20 Author Keywords
# -------------------------
# lowercase, split, flatten
df['AuthorKeywords'] = df['AuthorKeywords'].str.lower().str.split(r',\s*')
all_kw = [kw for sub in df['AuthorKeywords'] for kw in sub if kw]
kw_counts = pd.Series(all_kw).value_counts().head(20)

fig4 = px.bar(
    x=kw_counts.values,
    y=kw_counts.index,
    orientation='h',
    title='Top 20 Author Keywords',
    labels={'x': 'Frequency', 'y': 'Keyword'},
    color=kw_counts.values,
    color_continuous_scale='Viridis'
)
fig4.update_layout(
    height=500,
    width=1100,
    title_font_size=14,
    xaxis_title_font_size=12,
    yaxis_title_font_size=12,
    margin=dict(l=150, r=50, t=50, b=50),
    showlegend=False
)


# 5. Impact of Awards on Citation Count (CrossRef)
# ------------------------------------------------
no_award = df[df['Award'] == 0]['CitationCount_CrossRef']
yes_award = df[df['Award'] == 1]['CitationCount_CrossRef']

fig5 = go.Figure()
fig5.add_trace(go.Histogram(
    x=no_award,
    histnorm='density',
    name='No',
    marker_color='lightgray',
    opacity=0.7
))
fig5.add_trace(go.Histogram(
    x=yes_award,
    histnorm='density',
    name='Yes',
    marker_color='goldenrod',
    opacity=0.7
))
fig5.update_layout(
    barmode='overlay',
    title='Citation Distribution by Award Status',
    xaxis_title='Citation Count (CrossRef)',
    yaxis_title='Density',
    legend=dict(title='Award', x=1, y=1, xanchor='right', yanchor='top'),
    height=500,
    width=1100,
    title_font_size=14,
    xaxis_title_font_size=12,
    yaxis_title_font_size=12,
    margin=dict(l=50, r=200, t=50, b=50)
)


# 6. EXPORT ALL FIGURES TO ONE SINGLE HTML
# ----------------------------------------
# We will load Plotly.js once and then embed each figure's div
html_parts = [
    "<html>",
    "<head>",
    "  <title>Interactive Plotly Figures</title>",
    "  <meta charset='utf-8'/>",
    "  <script src='https://cdn.plot.ly/plotly-latest.min.js'></script>",
    "</head>",
    "<body>",
    "  <h1>Interactive Plotly Visualizations</h1>"
]

for fig in (fig1, fig2, fig3, fig4, fig5):
    # embed each figure; do not include the <head> script again
    html_parts.append(fig.to_html(full_html=False, include_plotlyjs=False))

html_parts.append("</body>")
html_parts.append("</html>")

html_str = "\n".join(html_parts)
with open("output.html", "w", encoding="utf-8") as f:
    f.write(html_str)

print("All figures have been written to output.html")


'''

state = initialize_state_from_csv()
data = state['dataset_info']
data_info = get_datainfo("dataset.csv")
data_description = data_describe("dataset.csv")
dic, config = define_variables(thread = 1, loop_limit = 10, data = data, data_info = data_info, name = "fix-plotly", code = code, data_description= data_description)
dic['revise'] = True
plotly_enhancer_lead = plotly_enhancer_leader(state)
for chunk in plotly_enhancer_lead.stream(input = dic, config = config):
    pretty_print_messages(chunk)
 

print(f"This is the chunk \n {chunk} \n")
result = chunk['code_plotly']['messages'][-1]['content']

code = re.findall(r"```python\n(.*?)\n```", result, re.DOTALL)
print(f"This is the code ! \n {code} \n")
with open("extracted_code.py", "a", encoding="utf-8") as f:
    f.write("# ---- NEW BLOCK ---- # \n")
    for block in code:
        f.write(block + "\n\n")

with open("test_plotly_code.py", "w", encoding="utf-8") as f:
    f.write("# ---- NEW BLOCK ---- # \n")
    for block in code:
        f.write(block + "\n\n")


