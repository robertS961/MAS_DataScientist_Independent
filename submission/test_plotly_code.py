# ---- NEW BLOCK ---- # 
# CODE HERE
# Enhanced interactive Plotly visualization suite for dataset.csv
# - Dark-mode Plotly visuals, improved interactivity (range slider, buttons, toggles),
#   clearer colors, robust handling of NaNs and empty groups, larger figure sizes.
# - Adds academic-style narratives (introduction, per-figure paragraphs that flow, conclusion)
# - Outputs a single HTML file: output.html (includes plotly.js via CDN in each figure div).
#
# Requirements: pandas, numpy, scikit-learn, plotly
# Install if needed:
# pip install pandas numpy scikit-learn plotly

import os
import re
import math
import numpy as np
import pandas as pd
from pathlib import Path

import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.linear_model import HuberRegressor, LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.impute import SimpleImputer

# Use a dark template by default for all figures
pio.templates.default = "plotly_dark"

# -------------------------
# Load and preprocess data
# -------------------------
DATA_PATH = "dataset.csv"
if not Path(DATA_PATH).exists():
    raise FileNotFoundError(f"{DATA_PATH} not found. Please place dataset.csv in working directory.")

df = pd.read_csv(DATA_PATH)

# Defensive: ensure expected columns exist
expected_cols = ["Conference", "Year", "Title", "DOI", "Link", "FirstPage", "LastPage",
                 "PaperType", "Abstract", "AuthorNames-Deduped", "AuthorNames",
                 "AuthorAffiliation", "InternalReferences", "AuthorKeywords",
                 "AminerCitationCount", "CitationCount_CrossRef", "PubsCited_CrossRef",
                 "Downloads_Xplore", "Award", "GraphicsReplicabilityStamp"]
for c in expected_cols:
    if c not in df.columns:
        # if missing, create with NaNs to keep pipeline running
        df[c] = np.nan

# Basic cleaning and derived fields
numeric_cols = ["FirstPage", "LastPage", "AminerCitationCount", "CitationCount_CrossRef",
                "PubsCited_CrossRef", "Downloads_Xplore"]
for c in numeric_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")
    if df[c].isna().all():
        df[c] = df[c].fillna(0.0)
    else:
        med = float(np.nanmedian(df[c].values))
        df[c] = df[c].fillna(med)

# Year as int (fall back to median if problematic)
df['Year'] = pd.to_numeric(df['Year'], errors="coerce")
if df['Year'].isna().all():
    df['Year'] = 2009
else:
    df['Year'] = df['Year'].fillna(int(df['Year'].median())).astype(int)

# Page length derived (LastPage - FirstPage + 1) with sensible fallback
def compute_page_len(row):
    try:
        fp = row.get('FirstPage', np.nan)
        lp = row.get('LastPage', np.nan)
        if not (pd.isna(fp) or pd.isna(lp)):
            return max(1.0, float(lp) - float(fp) + 1.0)
    except Exception:
        pass
    return np.nan

df['PageLength'] = df.apply(compute_page_len, axis=1)
if df['PageLength'].isna().all():
    df['PageLength'] = 1.0
else:
    df['PageLength'] = df['PageLength'].fillna(df['PageLength'].median())

# Internal references count (attempt to count separators)
def count_internal_refs(x):
    if pd.isnull(x):
        return 0
    s = str(x)
    parts = re.split(r"[;,|\n]+", s.strip())
    parts = [p for p in parts if p.strip() != ""]
    # filter out obvious noise
    return max(0, len(parts))

df['InternalRefCount'] = df['InternalReferences'].apply(count_internal_refs)

# Presence of code/data links heuristic
def has_code_link(link, title, abstract):
    s = " ".join([str(link or ""), str(title or ""), str(abstract or "")]).lower()
    if any(k in s for k in ("github.com", "gitlab.com", "zenodo.org", "figshare", "osf.io", "github", "gitlab", "zenodo", "figshare", "osf", "code", "dataset", "data:")):
        return 1
    return 0

df['HasCodeLink'] = df.apply(lambda r: has_code_link(r.get('Link', ''), r.get('Title', ''), r.get('Abstract', '')), axis=1)

# Keywords processing (lowercased string)
def normalize_keywords(x):
    if pd.isnull(x):
        return ""
    s = str(x).strip().lower()
    s = re.sub(r"[;|/]+", ",", s)
    return s

df['KeywordsNorm'] = df['AuthorKeywords'].apply(normalize_keywords)

# Check replicability stamp presence (very sparse)
df['HasReplicabilityStamp'] = df['GraphicsReplicabilityStamp'].notnull().astype(int)

# Citation per year proxy to account for age
current_year = int(df['Year'].max())
df['YearsSince'] = (current_year - df['Year']).clip(lower=1).astype(float)
df['AminerCitationsPerYear'] = df['AminerCitationCount'] / df['YearsSince']
df['CrossRefCitationsPerYear'] = df['CitationCount_CrossRef'] / df['YearsSince']

# Create a simple 'high-impact' label: top 10% by AminerCitationsPerYear (robust to outliers)
cit_series = df['AminerCitationsPerYear'].replace([np.inf, -np.inf], np.nan).fillna(0)
threshold = np.nanpercentile(cit_series, 90) if len(cit_series) > 0 else 0
df['HighImpactTop10'] = (cit_series >= threshold).astype(int)

# -------------------------
# Utility color palettes and layout defaults for dark theme
# -------------------------
PALETTE = px.colors.qualitative.Dark24  # high-contrast palette suitable for dark bg
LIGHT = "#e6eef8"
FIG_WIDTH = 1400
FIG_HEIGHT = 800
FONT = dict(family="Helvetica Neue, Arial, sans-serif", color=LIGHT)

# -------------------------
# Plot 1: Conference Trends Over Time (fixed reindex / duplicate Year issue)
# -------------------------
agg = (df.groupby(['Conference', 'Year'], as_index=False)
         .agg(Papers=('Title', 'count'),
              MedianAminerCites=('AminerCitationCount', 'median'),
              MedianDownloads=('Downloads_Xplore', 'median')))

# Prepare options: All + top 10 conferences by total papers
top_confs = df['Conference'].value_counts().nlargest(10).index.tolist()
options_confs = ['All'] + top_confs

fig_trend = go.Figure()

# Full year range
year_min = int(df['Year'].min())
year_max = int(df['Year'].max())
all_years = list(range(year_min, year_max + 1))
all_years_df = pd.DataFrame({'Year': all_years})

for idx, conf in enumerate(options_confs):
    if conf == 'All':
        subset_all = (agg.groupby('Year', as_index=False)
                        .agg(Papers=('Papers', 'sum'),
                             MedianAminerCites=('MedianAminerCites', 'median'),
                             MedianDownloads=('MedianDownloads', 'median')))
        # Merge with all_years to ensure full timeline
        subset = pd.merge(all_years_df, subset_all, on='Year', how='left').sort_values('Year')
    else:
        subset_conf = agg[agg['Conference'] == conf][['Year', 'Papers', 'MedianAminerCites', 'MedianDownloads']].copy()
        subset = pd.merge(all_years_df, subset_conf, on='Year', how='left').sort_values('Year')

    # Fill NaNs for Papers with 0; leave medians NaN so lines don't mislead if no data that year
    subset['Papers'] = subset['Papers'].fillna(0)

    # Bar trace for number of papers
    fig_trend.add_trace(go.Bar(
        x=subset['Year'],
        y=subset['Papers'],
        name=f"Papers ({conf})",
        marker_color=PALETTE[idx % len(PALETTE)],
        hovertemplate='<b>%{x}</b><br>Papers: %{y}<br>Conference: ' + str(conf) + '<extra></extra>',
        visible=(conf == 'All'),
    ))

    # Line for median Aminer citations
    fig_trend.add_trace(go.Scatter(
        x=subset['Year'],
        y=subset['MedianAminerCites'],
        mode="lines+markers",
        name=f"Median Aminer Cites ({conf})",
        yaxis="y2",
        line=dict(width=3, color=PALETTE[(idx+6) % len(PALETTE)]),
        marker=dict(size=8),
        hovertemplate='<b>%{x}</b><br>Median Aminer Cites: %{y}<br>Conference: ' + str(conf) + '<extra></extra>',
        visible=(conf == 'All'),
    ))

    # Line for median downloads (dashed)
    fig_trend.add_trace(go.Scatter(
        x=subset['Year'],
        y=subset['MedianDownloads'],
        mode="lines",
        name=f"Median Downloads ({conf})",
        yaxis="y3",
        line=dict(width=3, dash='dash', color=PALETTE[(idx+12) % len(PALETTE)]),
        hovertemplate='<b>%{x}</b><br>Median Downloads: %{y}<br>Conference: ' + str(conf) + '<extra></extra>',
        visible=(conf == 'All'),
    ))

# Layout: 3 y-axes; add range slider and selectors
fig_trend.update_layout(
    title="Conference Trends Over Time: Papers vs Median Citations & Downloads",
    xaxis=dict(title="Year", tickmode='linear', range=[year_min-0.5, year_max+0.5],
               rangeslider=dict(visible=True), rangeselector=dict(
                   buttons=list([
                       dict(count=5, label="5y", step="year", stepmode="backward"),
                       dict(count=10, label="10y", step="year", stepmode="backward"),
                       dict(step="all", label="All")
                   ])
               )),
    yaxis=dict(title="Number of Papers", showgrid=True),
    yaxis2=dict(title="Median Aminer Citations", overlaying='y', side='right', position=0.95),
    yaxis3=dict(title="Median Downloads (Xplore)", anchor='free', overlaying='y', side='right', position=0.87),
    legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="left", x=0.01),
    height=FIG_HEIGHT,
    width=FIG_WIDTH,
    template="plotly_dark",
    font=FONT,
    hovermode='x unified',
)

# Dropdown to select conference
n_per_conf = 3
buttons = []
for i, conf in enumerate(options_confs):
    vis = [False] * (len(options_confs) * n_per_conf)
    start = i * n_per_conf
    for j in range(n_per_conf):
        vis[start + j] = True
    buttons.append(dict(label=conf, method="update",
                        args=[{"visible": vis},
                              {"title": f"Conference Trends Over Time: {conf}"}]))
fig_trend.update_layout(
    updatemenus=[dict(active=0, buttons=buttons, x=0.0, y=1.18, xanchor='left',
                      bgcolor="#101217", bordercolor="#2a3f5f", font=dict(color=LIGHT))]
)

# -------------------------
# Plot 2: Cross-source Citation Calibration (Aminer vs CrossRef) with toggles
# -------------------------
# Prepare safe maxes and percentiles to avoid blank-looking plots dominated by outliers
x_full = df['CitationCount_CrossRef'].fillna(0)
y_full = df['AminerCitationCount'].fillna(0)
x_95 = float(np.nanpercentile(x_full, 95))
y_95 = float(np.nanpercentile(y_full, 95))
x_max = max(x_full.max(), 1.0)
y_max = max(y_full.max(), 1.0)

# Robust regression
xr = x_full.values.reshape(-1, 1)
yr = y_full.values
huber = HuberRegressor().fit(xr, yr)
slope = float(huber.coef_[0])
intercept = float(huber.intercept_)

# Scatter with size scaled (but cap size)
size_ref = np.interp(df['AminerCitationCount'], (df['AminerCitationCount'].min(), df['AminerCitationCount'].max()), (6, 36))
size_ref = np.nan_to_num(size_ref, nan=6.0)

fig_cite = go.Figure()

fig_cite.add_trace(go.Scattergl(
    x=x_full,
    y=y_full,
    mode='markers',
    marker=dict(size=size_ref, color=df['Year'], colorscale='Viridis', showscale=True, colorbar=dict(title='Year')),
    hovertemplate="<b>%{text}</b><br>CrossRef: %{x}<br>Aminer: %{y}<extra></extra>",
    text=df['Title'].fillna("No title"),
    name="Papers",
))

# identity y=x and regression lines (computed on full axis)
line_x = np.linspace(0, max(x_max, x_95) * 1.05, 200)
fig_cite.add_trace(go.Scatter(x=line_x, y=line_x, mode='lines', line=dict(color='#888888', dash='dash'), name='y = x'))
fig_cite.add_trace(go.Scatter(x=line_x, y=(slope * line_x + intercept), mode='lines', line=dict(color='#ff6666', width=3),
                              name=f'Robust fit: y = {slope:.2f}x + {intercept:.1f}'))

# Layout and buttons to toggle scale / clip outliers
fig_cite.update_layout(
    title="Aminer vs CrossRef Citation Counts (robust fit)",
    xaxis=dict(title="CitationCount_CrossRef (linear)", range=[0, max(x_95 * 1.1, x_max * 0.1)], showgrid=True),
    yaxis=dict(title="AminerCitationCount (linear)", range=[0, max(y_95 * 1.1, y_max * 0.1)], showgrid=True),
    height=FIG_HEIGHT,
    width=FIG_WIDTH,
    template="plotly_dark",
    font=FONT,
    hovermode='closest',
)

# Buttons: full range, clipped-to-95th, log-log
updatemenus = [
    dict(type="buttons", direction="right", showactive=True, x=0.02, y=1.15,
         buttons=[
             dict(label="Clipped (95th pct)",
                  method="relayout",
                  args=[{
                      "xaxis.range": [0, x_95 * 1.05],
                      "yaxis.range": [0, y_95 * 1.05],
                      "xaxis.type": "linear", "yaxis.type": "linear",
                      "xaxis.title.text": "CitationCount_CrossRef (clipped to 95th pct)",
                      "yaxis.title.text": "AminerCitationCount (clipped to 95th pct)",
                  }]),
             dict(label="Full Range",
                  method="relayout",
                  args=[{
                      "xaxis.range": [0, max(x_max, 1.0) * 1.05],
                      "yaxis.range": [0, max(y_max, 1.0) * 1.05],
                      "xaxis.type": "linear", "yaxis.type": "linear",
                      "xaxis.title.text": "CitationCount_CrossRef (full)",
                      "yaxis.title.text": "AminerCitationCount (full)",
                  }]),
             dict(label="Log-Log",
                  method="relayout",
                  args=[{
                      "xaxis.type": "log",
                      "yaxis.type": "log",
                      "xaxis.title.text": "CitationCount_CrossRef (log)",
                      "yaxis.title.text": "AminerCitationCount (log)",
                  }]),
         ],
         bgcolor="#101217", font=dict(color=LIGHT),
    )
]
fig_cite.update_layout(updatemenus=updatemenus)

# Add marginal histogram of residuals as a second figure (kept large and dark)
residuals = y_full - (slope * x_full + intercept)
fig_resid = px.histogram(residuals, nbins=60, title="Residuals: Aminer - (fitted from CrossRef)",
                         labels={'value': 'Residual (Aminer - fitted)'}, height=500, width=700, template="plotly_dark")
fig_resid.update_layout(font=FONT)

# -------------------------
# Plot 3: Replicability Composite Score and relation to Downloads
# -------------------------
comp_df = df.copy()

def robust_minmax(series):
    lo = np.nanpercentile(series, 5)
    hi = np.nanpercentile(series, 95)
    span = hi - lo if (hi - lo) > 0 else 1.0
    return ((series - lo) / span).clip(0, 1)

comp_df['NormInternalRefCount'] = robust_minmax(comp_df['InternalRefCount'])
comp_df['NormDownloads'] = robust_minmax(comp_df['Downloads_Xplore'])

# Component weights (configurable)
w_stamp = 0.5
w_code = 0.35
w_refs = 0.10
w_downloads = 0.05

comp_df['ReplicabilityScore'] = (
    w_stamp * comp_df['HasReplicabilityStamp'] +
    w_code * comp_df['HasCodeLink'] +
    w_refs * comp_df['NormInternalRefCount'] +
    w_downloads * comp_df['NormDownloads']
)
comp_df['ReplicabilityScore'] = comp_df['ReplicabilityScore'].clip(0, 1)

# Histogram with cumulative overlay
fig_rep_hist = go.Figure()
fig_rep_hist.add_trace(go.Histogram(x=comp_df['ReplicabilityScore'], nbinsx=40, name='count', marker_color='#00CC96', opacity=0.9))
fig_rep_hist.add_trace(go.Histogram(x=comp_df['ReplicabilityScore'], nbinsx=40, histnorm='probability density', name='density', marker_color='rgba(255,255,255,0.05)'))
fig_rep_hist.update_layout(title="Composite Replicability Score Distribution",
                           xaxis_title="Replicability Score (0-1)",
                           yaxis_title="Count",
                           height=600, width=1000, template="plotly_dark", font=FONT)

# Boxplot by PaperType (top types)
top_types = comp_df['PaperType'].value_counts().nlargest(8).index.tolist()
box_df = comp_df[comp_df['PaperType'].isin(top_types)].copy()
# ensure ordering by median score
order = box_df.groupby('PaperType')['ReplicabilityScore'].median().sort_values().index.tolist()
fig_rep_box = px.box(box_df, x='PaperType', y='ReplicabilityScore', category_orders={'PaperType': order},
                     color='PaperType', title="Replicability Score by PaperType (top types)",
                     height=600, width=FIG_WIDTH, template="plotly_dark")
fig_rep_box.update_layout(showlegend=False, font=FONT, xaxis_tickangle=25)

# Scatter: Replicability score vs Downloads, with citation-sized markers and jitter for visibility
scatter_df = comp_df.copy()
# add jitter to x for clarity when many same scores
jitter = (np.random.rand(len(scatter_df)) - 0.5) * 0.02
scatter_df['RepScoreJ'] = (scatter_df['ReplicabilityScore'] + jitter).clip(0, 1)
marker_sizes = np.interp(scatter_df['AminerCitationCount'].fillna(0),
                         (scatter_df['AminerCitationCount'].min(), scatter_df['AminerCitationCount'].max()), (6, 36))
# px.scatter expects column names for size/color, so pass arrays via dataframe
scatter_df['_marker_size'] = marker_sizes
fig_rep_scatter = px.scatter(scatter_df, x='RepScoreJ', y='Downloads_Xplore', size='_marker_size',
                             color='HasCodeLink', color_continuous_scale=['#ffaa00', '#00cc96'],
                             hover_data=['Title', 'AuthorNames-Deduped', 'AminerCitationCount'],
                             title="Replicability Score vs Downloads (bubble size = Aminer citations)",
                             height=FIG_HEIGHT, width=FIG_WIDTH, template="plotly_dark")
fig_rep_scatter.update_layout(xaxis_title="Replicability Score (with jitter)", font=FONT)

# -------------------------
# Plot 4: Topic Modeling (TF-IDF + NMF) + Topic Prevalence Over Time
# -------------------------
text_series = (df['Title'].fillna('') + '. ' + df['Abstract'].fillna('')).astype(str)
tfv = TfidfVectorizer(max_df=0.9, min_df=6, max_features=5000, stop_words='english')
X_tfidf = tfv.fit_transform(text_series.values)

n_topics = 12
nmf = NMF(n_components=n_topics, init='nndsvda', random_state=42, max_iter=400)
W = nmf.fit_transform(X_tfidf)
H = nmf.components_
feature_names = tfv.get_feature_names_out()

# top words per topic
top_n_words = 8
topics_topwords = []
for t in range(n_topics):
    top_indices = H[t].argsort()[::-1][:top_n_words]
    top_words = [feature_names[i] for i in top_indices]
    topics_topwords.append((t, top_words))

topic_df = pd.DataFrame(W, columns=[f"Topic_{i}" for i in range(n_topics)])
topic_df['Year'] = df['Year'].values
topic_year = topic_df.groupby('Year').mean().reset_index().sort_values('Year')

# Choose top 6 topics by overall mean
topic_means = topic_df[[f"Topic_{i}" for i in range(n_topics)]].mean().sort_values(ascending=False)
top_topics_idx = [int(s.split('_')[1]) for s in topic_means.head(6).index.tolist()]

# Stacked area chart: normalized by mean weights
fig_topic_trends = go.Figure()
for i, t in enumerate(top_topics_idx):
    fig_topic_trends.add_trace(go.Scatter(
        x=topic_year['Year'],
        y=topic_year[f"Topic_{t}"],
        stackgroup='one',
        name=f"Topic {t}",
        line=dict(width=1.5, color=PALETTE[i % len(PALETTE)]),
        hovertemplate="Year: %{x}<br>Mean weight: %{y:.4f}<extra></extra>"
    ))
fig_topic_trends.update_layout(title="Top Topic Prevalence Over Years (NMF on Title+Abstract)",
                               xaxis_title="Year", yaxis_title="Mean topic weight",
                               height=FIG_HEIGHT, width=FIG_WIDTH, template="plotly_dark", font=FONT)

# Topic top-words HTML snippet for the dashboard
topic_table_html = "<div style='font-family:Arial, sans-serif; color:#eee;'><h3>Top words per topic (NMF)</h3><ul>"
for tid, words in topics_topwords:
    topic_table_html += f"<li><strong>Topic {tid}:</strong> " + ", ".join(words) + "</li>"
topic_table_html += "</ul></div>"

# -------------------------
# Plot 5: Early-warning High-impact detection (simple logistic model, explainability)
# -------------------------
model_df = df.copy()
model_df['TitleLen'] = model_df['Title'].fillna('').apply(lambda s: len(s.split()))
top_conf_list = df['Conference'].value_counts().nlargest(12).index.tolist()
model_df['ConferenceTop'] = model_df['Conference'].apply(lambda x: x if x in top_conf_list else 'Other')

features = ['Downloads_Xplore', 'PageLength', 'PubsCited_CrossRef', 'InternalRefCount', 'TitleLen', 'HasCodeLink', 'ConferenceTop']
X = model_df[features].copy()
y = model_df['HighImpactTop10'].copy()

# One-hot encode ConferenceTop (compat for sklearn newer versions)
enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
conf_ohe = enc.fit_transform(X[['ConferenceTop']])
conf_categories = [str(c).strip().replace(" ", "_").replace("/", "_") for c in enc.categories_[0]]
conf_cols = [f"Conf_{c}" for c in conf_categories]
conf_ohe_df = pd.DataFrame(conf_ohe, columns=conf_cols, index=X.index)
X = pd.concat([X.drop(columns=['ConferenceTop']), conf_ohe_df], axis=1)

# Impute and scale
imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X_imputed.columns, index=X.index)

# Train-test (stratify if possible)
if y.nunique() == 2 and y.value_counts().min() >= 5:
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.30, random_state=42, stratify=y)
else:
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.30, random_state=42)

clf = LogisticRegression(class_weight='balanced', solver='liblinear', max_iter=1000)
clf.fit(X_train, y_train)

y_score = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else clf.decision_function(X_test)
fpr, tpr, _ = roc_curve(y_test.fillna(0), y_score)
roc_auc = auc(fpr, tpr)

fig_roc = go.Figure()
fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', line=dict(color='#EF553B', width=4),
                             name=f'ROC (AUC = {roc_auc:.3f})'))
fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(color='#888888', dash='dash'), name='Random'))
fig_roc.update_layout(title="ROC Curve: Early-warning High-impact Detection (simple model)",
                      xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
                      height=600, width=900, template="plotly_dark", font=FONT)

coef_df = pd.DataFrame({'feature': X_scaled.columns, 'coef': clf.coef_[0]}).sort_values('coef', key=lambda c: np.abs(c), ascending=False).head(20)
fig_feat = px.bar(coef_df.sort_values('coef'), x='coef', y='feature', orientation='h',
                  color='coef', color_continuous_scale=px.colors.diverging.RdBu, title='Top Feature Coefficients (logistic)',
                  height=FIG_HEIGHT, width=800, template="plotly_dark")
fig_feat.update_layout(font=FONT)

# -------------------------
# Assemble HTML with academic-style narratives + figures
# -------------------------
fragments = []

def fig_html_with_text(fig_html_fragment, narrative_html):
    wrapper = f"""
    <div style="margin:30px 50px; font-family: 'Helvetica Neue', Arial, sans-serif; color:#e6eef8;">
      <div style="background:#0b1220; padding:18px; border-left:6px solid #2a3f5f; margin-bottom:12px;">
        {narrative_html}
      </div>
      <div style="margin-top:12px;">
        {fig_html_fragment}
      </div>
    </div>
    """
    return wrapper

# Introduction paragraph (one academic paragraph describing purpose and flow)
introduction_html = """
<div style="margin:30px 50px; font-family: 'Helvetica Neue', Arial, sans-serif; color:#e6eef8; background:#07101a; padding:18px; border-left:8px solid #2a3f5f;">
  <h1 style="margin-top:0; color:#ffffff;">Introduction</h1>
  <p>
    This report presents an integrated exploratory analysis of a scholarly publication dataset, focusing on five complementary
    themes: (1) conference-level temporal trends in productivity and attention, (2) cross-source citation consistency between
    Aminer and CrossRef, (3) a pragmatic composite replicability score and its association with downloads and citations, (4)
    topic modeling to reveal topical drift across years, and (5) an early-warning prototype to identify papers likely to
    become high-impact. The goal is to provide clear, interpretable visual diagnostics and lightweight predictive baselines
    that a research team can use to prioritize replication resources, study citation dynamics, and monitor emergent topics.
  </p>
</div>
"""

# Narrative for Plot 1: flows into Plot 2
narr1 = """
<h2 style="color:#ffffff; margin-top:0;">1) Conference Trends Over Time</h2>
<p>
  We begin with a longitudinal view of conference activity: yearly paper counts are presented as bars while median Aminer citations
  and median Downloads (Xplore) appear as overlaying lines. This triad allows an analyst to distinguish growth in volume from
  growth in attention — a conference may be producing more papers but not necessarily more influential ones. Observing such patterns
  establishes temporal context for later analyses: citation calibration and topic drift both depend on where and when papers were
  published, so understanding the baseline dynamics is crucial before drawing causal or predictive conclusions.
</p>
"""

# Prepare fig html fragment for fig_trend
frag_trend = pio.to_html(fig_trend, full_html=False, include_plotlyjs='cdn')
fragments.append(fig_html_with_text(frag_trend, narr1))

# Narrative for Plot 2: flows from Plot1 into replicability
narr2 = """
<h2 style="color:#ffffff; margin-top:0;">2) Cross-source Citation Consistency: Aminer vs CrossRef</h2>
<p>
  Citations are a central currency in scholarly analytics yet different sources can disagree. The scatter plot compares Aminer and
  CrossRef counts; a robust Huber regression summarizes systematic deviations and a residual histogram documents dispersion.
  This diagnostic is important because downstream indices (replicability score, early-warning ranks) hinge on citation measurements.
  Reconciling sources or modeling their discrepancies increases confidence in impact estimates and can inform uncertainty-aware ranking.
</p>
"""

# Combine main scatter and residual histogram
frag_cite_main = pio.to_html(fig_cite, full_html=False, include_plotlyjs='cdn')
frag_resid = pio.to_html(fig_resid, full_html=False, include_plotlyjs='cdn')
combined_cite_html = f"""
<div style="display:flex; gap:20px; align-items:flex-start;">
  <div style="flex:1; min-width:700px;">{frag_cite_main}</div>
  <div style="width:420px; min-width:320px;">{frag_resid}</div>
</div>
"""
fragments.append(fig_html_with_text(combined_cite_html, narr2))

# Narrative for Plot 3: composite replicability, flows from citation reconciliation
narr3 = """
<h2 style="color:#ffffff; margin-top:0;">3) Composite Replicability Score & Downloads</h2>
<p>
  Because explicit graphics replicability stamps are rare in the dataset, we construct an interpretable composite replicability score
  that combines explicit stamps (when present), presence of code/data links, internal reference density, and normalized downloads.
  The histogram shows the overall distribution, the boxplot highlights variation by paper type, and the scatter inspects the relationship
  with downloads (bubble size = Aminer citations). These visualizations are intended to guide prioritization of replication efforts:
  papers with high composite scores and high attention are natural candidates for reproducibility audits.
</p>
"""

frag_rep_hist = pio.to_html(fig_rep_hist, full_html=False, include_plotlyjs='cdn')
frag_rep_box = pio.to_html(fig_rep_box, full_html=False, include_plotlyjs='cdn')
frag_rep_scatter = pio.to_html(fig_rep_scatter, full_html=False, include_plotlyjs='cdn')
combined_rep_html = f"""
<div style="display:flex; flex-direction:column; gap:18px;">
  <div style="display:flex; gap:18px;">
    <div style="flex:0 0 420px;">{frag_rep_hist}</div>
    <div style="flex:1;">{frag_rep_box}</div>
  </div>
  <div>{frag_rep_scatter}</div>
</div>
"""
fragments.append(fig_html_with_text(combined_rep_html, narr3))

# Narrative for Plot 4: topic modeling, flows from replicability to thematic structure
narr4 = """
<h2 style="color:#ffffff; margin-top:0;">4) Topic Modeling & Topic Drift</h2>
<p>
  We extract latent topics from Title+Abstract using TF-IDF followed by NMF and chart the prevalence of dominant topics over time.
  The stacked area visualization and the top-words list make it straightforward to detect emergent subfields and to align thematic
  shifts with changes in conference-level metrics or replicability signals. Topic drift is a practical lens for understanding whether
  increases in impact or replicability are localized to particular research themes.
</p>
"""

frag_topic_trends = pio.to_html(fig_topic_trends, full_html=False, include_plotlyjs='cdn')
topic_panel = f"<div style='width:420px; padding:10px; border-left:2px solid #1b2a44; background:#07101a;'>{topic_table_html}</div>"
combined_topic_html = f"""
<div style="display:flex; gap:24px; align-items:flex-start;">
  <div style="flex:1; min-width:700px;">{frag_topic_trends}</div>
  {topic_panel}
</div>
"""
fragments.append(fig_html_with_text(combined_topic_html, narr4))

# Narrative for Plot 5: early-warning, flows from topics and replicability
narr5 = """
<h2 style="color:#ffffff; margin-top:0;">5) Early-warning High-impact Detection (Simple Model)</h2>
<p>
  Finally, we demonstrate a compact early-warning classifier to identify papers that later become high-impact (top 10% by citations/year).
  The model uses early signals—initial downloads, reference counts, page length, title length, code link presence, and venue—and emphasizes
  interpretability via logistic regression coefficients and ROC curves. This prototype illustrates how attention, connectivity, and content
  features can be combined to guide monitoring and curation; however, it should be extended with strict temporal validation and richer text encodings
  before operational deployment.
</p>
"""

frag_roc = pio.to_html(fig_roc, full_html=False, include_plotlyjs='cdn')
frag_feat = pio.to_html(fig_feat, full_html=False, include_plotlyjs='cdn')
combined_early_html = f"""
<div style="display:flex; gap:20px; align-items:flex-start;">
  <div style="flex:0 0 720px;">{frag_roc}</div>
  <div style="flex:1 1 auto;">{frag_feat}</div>
</div>
"""
fragments.append(fig_html_with_text(combined_early_html, narr5))

# Conclusion paragraph (one academic paragraph summarizing and next steps)
conclusion_html = """
<div style="margin:30px 50px; font-family: 'Helvetica Neue', Arial, sans-serif; color:#e6eef8; background:#07101a; padding:18px; border-left:8px solid #2a3f5f;">
  <h2 style="margin-top:0; color:#ffffff;">Conclusion</h2>
  <p>
    In summary, the visual diagnostics presented here form a coherent exploratory pipeline for scholarly analytics: temporal conference trends
    provide the macro context, citation calibration reconciles measurement sources, a composite replicability score offers an interpretable
    prioritization signal, topic modeling reveals thematic drift, and a compact early-warning model demonstrates how early signals can
    forecast later impact. Key next steps include validating the replicability composite against ground-truth stamps where available,
    incorporating contextual text embeddings and network features into predictive models, and developing uncertainty-aware citation
    ensembles for robust ranking. Together, these directions can help research teams allocate verification resources effectively and
    improve the credibility of bibliometric assessments.
  </p>
</div>
"""

# Methodology footer remains
methodology_html = f"""
<div style="margin:30px 50px; font-family: 'Helvetica Neue', Arial, sans-serif; color:#e6eef8;">
  <hr style="border:none; border-top:1px solid #1b2a44;"/>
  <h3 style="color:#ffffff;">Methodology & Notes</h3>
  <ul>
    <li>Numeric missing values were imputed with medians. Categorical missing values handled explicitly.</li>
    <li>Replicability score is an interpretable composite (weights chosen for demonstration). Where ground-truth stamps exist,
        supervise a model for better calibration.</li>
    <li>Topic modeling used TF-IDF + NMF for speed and interpretability; using contextual embeddings (e.g., SBERT) will
        increase semantic quality at higher compute cost.</li>
    <li>Early-warning model is a demonstrator and must be revalidated temporally before operational use.</li>
  </ul>
</div>
"""

# Compose final HTML (intro -> all fragments -> conclusion -> methodology)
html_header = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Scholarly Dataset Interactive Analysis</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body { margin: 0; padding: 0; background: #07101a; color: #e6eef8; }
    h1 { font-family: 'Helvetica Neue', Arial, sans-serif; color: #ffffff; margin-left:50px; }
    .topbar { padding: 18px 50px; background: linear-gradient(90deg,#06111a,#081522); border-bottom:1px solid #0f2740; }
    a { color: #9fd3ff; }
  </style>
</head>
<body>
  <div class="topbar">
    <h1>Scholarly Dataset Interactive Analysis</h1>
    <div style="color:#9fb1c9; margin-top:6px; margin-left:2px;">
      Interactive Plotly figures exploring conference trends, citation calibration, replicability scoring, topic drift, and an early-warning prototype.
    </div>
  </div>
"""

# Join fragments
html_body = introduction_html + "\n".join(fragments) + conclusion_html + methodology_html
html_footer = """
</body>
</html>
"""

final_html = html_header + html_body + html_footer

OUT_FILE = "output.html"
with open(OUT_FILE, 'w', encoding='utf-8') as f:
    f.write(final_html)

print(f"Done. Generated {OUT_FILE}. Open this file in a browser to view interactive visualizations.")

