# Enhanced Plotly Visualization Suite for dataset.csv with Educational Narratives
# - Improved dark-mode visuals, clearer legends, larger plots, and robust interactivity
# - Buttons styled with selectable (light-green) state; no disappearing descriptions
# - Prevents overlapping legends by reserving right margin
# - Handles NaNs, missing values, and sklearn compatibility
# - Writes a standalone output.html (each plot includes_plotlyjs='cdn')
# - Added educational introduction, per-figure narratives, and conclusion in the HTML report
#
# Run: python generate_plots.py
# Output: output.html (open in a browser / server)
#
# Author: Plotly expert assistant
# Date: 2025-08-20

import os
import json
import math
import uuid
import re
import numpy as np
import pandas as pd
from collections import Counter, defaultdict

# Visualization libraries
import plotly.graph_objs as go
import plotly.offline as pyo

# Light ML tools for a prediction example
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

# ---------------------------------------------------------------------
# 1) Load and preprocess dataset
# ---------------------------------------------------------------------
DATA_FILE = "dataset.csv"
if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"{DATA_FILE} not found in current directory.")

df = pd.read_csv(DATA_FILE)

# Basic cleaning and imputations as requested: numeric NaNs -> median, text -> ""
numeric_cols = ["FirstPage", "LastPage", "AminerCitationCount", "CitationCount_CrossRef", "PubsCited_CrossRef", "Downloads_Xplore"]
for col in numeric_cols:
    if col in df.columns:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

# Text columns: fill NaNs with empty string
text_cols = ["Abstract", "Title", "AuthorKeywords", "AuthorNames", "AuthorNames-Deduped", "AuthorAffiliation", "InternalReferences"]
for col in text_cols:
    if col in df.columns:
        df[col] = df[col].fillna("")

# Award and GraphicsReplicabilityStamp may be sparse; fill with empty strings for detection
if "Award" in df.columns:
    df["Award"] = df["Award"].fillna("")
if "GraphicsReplicabilityStamp" in df.columns:
    df["GraphicsReplicabilityStamp"] = df["GraphicsReplicabilityStamp"].fillna("")

# Convert Year to int when possible
if "Year" in df.columns:
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").fillna(df["Year"].median()).astype(int)

# Helper: normalize and split author keywords into list of cleaned keywords.
re_split_keywords = re.compile(r"[;,\|\/]+")
def split_keywords(s):
    if not isinstance(s, str) or s.strip() == "":
        return []
    parts = [p.strip().lower() for p in re_split_keywords.split(s) if p.strip()]
    return parts

if "AuthorKeywords" in df.columns:
    df["KeywordsList"] = df["AuthorKeywords"].apply(lambda s: split_keywords(s))
else:
    df["KeywordsList"] = [[] for _ in range(len(df))]

# Explode all keywords into a long form for counting
keywords_flat = []
for idx, row in df.iterrows():
    kws = row.get("KeywordsList", [])
    if kws:
        for k in kws:
            keywords_flat.append(k)
# compute top keywords overall
kw_counter = Counter([k for k in keywords_flat if k])
top_keywords = [kw for kw, _ in kw_counter.most_common(12)]
top_keywords = top_keywords[:12]

# For robustness, also extract frequent token phrases from Titles (in case of missing keywords)
title_texts = df.get("Title", pd.Series([""]*len(df))).fillna("").str.lower()
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
tok_counter = Counter()
for t in title_texts:
    tokens = re.findall(r"\b[a-zA-Z0-9\-]+\b", t)
    for tk in tokens:
        if tk in ENGLISH_STOP_WORDS or len(tk) < 3 or tk.isdigit():
            continue
        tok_counter[tk] += 1
top_title_tokens = [tk for tk, _ in tok_counter.most_common(8)]

# Merge keywords + title tokens to form final "topic tokens" list
topic_tokens = list(dict.fromkeys(top_keywords + top_title_tokens))[:12]
if len(topic_tokens) == 0:
    topic_tokens = ["method", "network", "learning", "model", "data", "analysis"]

# Build keyword/year/conference counts for plotting
years_sorted = sorted(df["Year"].unique()) if "Year" in df.columns else [2020]
conferences = df["Conference"].value_counts().index.tolist() if "Conference" in df.columns else []
top_conferences = conferences[:6]

# Build aggregated counts
counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
for idx, row in df.iterrows():
    year = row.get("Year", None)
    conf = row.get("Conference", "Unknown")
    kws = row.get("KeywordsList", [])
    if not kws:
        title = str(row.get("Title", "")).lower()
        tks = re.findall(r"\b[a-zA-Z0-9\-]+\b", title)
        kws = [tk for tk in tks if tk in topic_tokens]
    for kw in kws:
        if kw in topic_tokens:
            counts["ALL"][kw][year] += 1
            counts[conf][kw][year] += 1

# Fill missing year entries with zeros to avoid blank lines
for conf in list(counts.keys()) + top_conferences + ["ALL"]:
    for kw in topic_tokens:
        for y in years_sorted:
            _ = counts[conf][kw][y]

# ---------------------------------------------------------------------
# 2) Create Plotly figures (improved layout, legends, ranges)
# ---------------------------------------------------------------------

# Shared styling
PLOT_HEIGHT = 700
PLOT_WIDTH = 1250
DARK_BG = "#0b0c0f"
VIBRANT_BTN = "#00D1FF"      # primary button color
VIBRANT_BTN_TEXT = "#061018" # dark text on buttons
HOVER_BG = "#dfffd6"         # hover label background (light green)
HOVER_FONT = "#041214"
SELECT_BG = "#C6F6D5"        # light green selection background
SELECT_TEXT = "#032a12"

# COMMON_LAYOUT_KW does NOT include 'yaxis' to avoid duplicate keyword issues
COMMON_LAYOUT_KW = dict(
    template="plotly_dark",
    paper_bgcolor=DARK_BG,
    plot_bgcolor=DARK_BG,
    margin=dict(l=70, r=320, t=100, b=120),  # reserve right space for vertical legend
    xaxis=dict(title="Year", tickmode="linear", dtick=1),
    hoverlabel=dict(bgcolor=HOVER_BG, font=dict(color=HOVER_FONT)),
    legend=dict(orientation="v", x=1.02, y=1.0, bordercolor="#222222", borderwidth=1),
    height=PLOT_HEIGHT,
    width=PLOT_WIDTH,
    font=dict(size=12, color="#FFFFFF")
)

# 2.1 Keyword Trends by Year
traces_kw = []
conf_list_for_traces = ["ALL"] + (top_conferences if len(top_conferences) > 0 else ["Unknown"])
for conf in conf_list_for_traces:
    for kw in topic_tokens:
        y_vals = [counts[conf][kw].get(y, 0) for y in years_sorted]
        visible = True if conf == "ALL" else False
        trace = go.Scatter(
            x=years_sorted,
            y=y_vals,
            mode="lines+markers",
            name=f"{kw}  |  {conf}",
            visible=visible,
            hovertemplate="<b>%{text}</b><br>Year %{x}<br>Count %{y}<extra></extra>",
            text=[f"{kw} ({conf})"] * len(years_sorted),
            marker=dict(size=8),
            line=dict(width=3),
            opacity=0.9
        )
        traces_kw.append(trace)

layout_kw = go.Layout(
    **COMMON_LAYOUT_KW,
    title=dict(text="Keyword / Topic Trends Over Time (select conference)", font=dict(size=24, color="#FFFFFF")),
    yaxis=dict(title="Number of papers mentioning token", rangemode="tozero")
)
fig_kw = go.Figure(data=traces_kw, layout=layout_kw)

# 2.2 Reproducibility vs Citations scatter
repro_keywords = ["code", "github", "open source", "dataset", "data available", "reproducible", "open-source", "available at"]
def reproducibility_proxy(row):
    txt = (" ".join([str(row.get("Title", "")), str(row.get("Abstract", "")), str(row.get("AuthorKeywords", ""))])).lower()
    if str(row.get("GraphicsReplicabilityStamp", "")).strip() != "":
        return 1
    for rk in repro_keywords:
        if rk in txt:
            return 1
    return 0

df["ReproProxy"] = df.apply(reproducibility_proxy, axis=1)

citations = df["CitationCount_CrossRef"].astype(float) if "CitationCount_CrossRef" in df.columns else pd.Series([0.0]*len(df))
downloads = df["Downloads_Xplore"].astype(float) if "Downloads_Xplore" in df.columns else pd.Series([0.0]*len(df))
repro = df["ReproProxy"].astype(int)

if downloads.max() - downloads.min() > 0:
    size = ((downloads - downloads.min()) / (downloads.max() - downloads.min()) * 35) + 8
else:
    size = np.ones(len(downloads)) * 10

trace_repro = go.Scatter(
    x=citations,
    y=downloads,
    mode="markers",
    marker=dict(
        size=size,
        color=repro,
        colorscale=[[0, "#FF6B6B"], [1, "#6BFFB8"]],
        showscale=True,
        colorbar=dict(title="Repro (proxy)", orientation="h", x=0.5, xanchor="center", y=-0.18),
        line=dict(width=0.6, color="#101010"),
        opacity=0.85
    ),
    text=df.get("Title", ""),
    hovertemplate="<b>%{text}</b><br>Citations: %{x}<br>Downloads: %{y}<extra></extra>",
    name="Papers"
)

x = np.log1p(citations.fillna(0).astype(float))
y = np.log1p(downloads.fillna(0).astype(float))
mask = np.isfinite(x) & np.isfinite(y)
if mask.sum() >= 3:
    coeffs = np.polyfit(x[mask], y[mask], 1)
    x_line = np.linspace(max(0.0, citations.min()), citations.max() if citations.max() > 0 else 1.0, 200)
    x_line_clipped = np.clip(x_line, 0, None)
    x_line_log = np.log1p(x_line_clipped)
    y_line_log = np.polyval(coeffs, x_line_log)
    y_line = np.expm1(y_line_log)
    trace_line = go.Scatter(x=x_line, y=y_line, mode="lines", name="Trend (log-log)", line=dict(color="#FFD166", width=3))
else:
    trace_line = go.Scatter()

layout_repro = go.Layout(
    template="plotly_dark",
    paper_bgcolor=DARK_BG,
    plot_bgcolor=DARK_BG,
    title=dict(text="Reproducibility Signals vs Citations & Downloads", font=dict(size=24, color="#FFFFFF")),
    margin=dict(l=70, r=300, t=100, b=120),
    xaxis=dict(title="CitationCount_CrossRef", rangemode="tozero"),
    yaxis=dict(title="Downloads_Xplore", rangemode="tozero"),
    hoverlabel=dict(bgcolor=HOVER_BG, font=dict(color=HOVER_FONT)),
    legend=dict(orientation="v", x=1.02, y=1),
    height=PLOT_HEIGHT,
    width=PLOT_WIDTH,
    font=dict(size=12)
)
fig_repro = go.Figure(data=[trace_repro, trace_line], layout=layout_repro)

# 2.3 Downloads prediction (RandomForest baseline)
texts = (df.get("Title", pd.Series([""]*len(df))).fillna("") + " " + df.get("Abstract", pd.Series([""]*len(df))).fillna("")).values
tf = TfidfVectorizer(max_features=2000, stop_words="english")
X_tfidf = tf.fit_transform(texts)
svd = TruncatedSVD(n_components=min(20, X_tfidf.shape[1]-1 if X_tfidf.shape[1]>1 else 1), random_state=42)
X_text_red = svd.fit_transform(X_tfidf)

num_cols = ["CitationCount_CrossRef", "AminerCitationCount", "PubsCited_CrossRef", "Year"]
for c in num_cols:
    if c not in df.columns:
        df[c] = 0.0
num_feats = df[["CitationCount_CrossRef", "AminerCitationCount", "PubsCited_CrossRef", "Year"]].astype(float).values

conf_topk = df["Conference"].value_counts().index[:8].tolist() if "Conference" in df.columns else []
df["ConferenceTop"] = df.get("Conference", pd.Series(["Other"]*len(df))).apply(lambda x: x if x in conf_topk else "Other")

try:
    ohe = OneHotEncoder(sparse=False, handle_unknown="ignore")
except TypeError:
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
conf_ohe = ohe.fit_transform(df[["ConferenceTop"]])

X = np.hstack([X_text_red, num_feats, conf_ohe])
y_downloads = df["Downloads_Xplore"].astype(float).values

unique_years = sorted(df["Year"].unique())
if len(unique_years) >= 2:
    year_threshold = unique_years[max(0, int(len(unique_years) * 0.8) - 1)]
    train_mask = df["Year"] <= year_threshold
    test_mask = ~train_mask
    if test_mask.sum() == 0:
        train_mask = np.arange(len(df)) < int(len(df) * 0.8)
        test_mask = ~train_mask
else:
    train_mask = np.arange(len(df)) < int(len(df) * 0.8)
    test_mask = ~train_mask

X_train = X[np.array(train_mask).astype(bool)]
X_test = X[np.array(test_mask).astype(bool)]
y_train = y_downloads[np.array(train_mask).astype(bool)]
y_test = y_downloads[np.array(test_mask).astype(bool)]

imp = SimpleImputer(strategy="median")
X_train = imp.fit_transform(X_train)
X_test = imp.transform(X_test)

model_rf = RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42)
model_rf.fit(X_train, y_train)
y_pred = model_rf.predict(X_test)
residuals = y_test - y_pred

trace_actual_pred = go.Scatter(
    x=y_test,
    y=y_pred,
    mode="markers",
    marker=dict(size=9, color="#9B5DE5", line=dict(width=0.6, color="#101010"), opacity=0.9),
    hovertemplate="<b>Actual:</b> %{x}<br><b>Pred:</b> %{y}<extra></extra>",
    name="Actual vs Predicted"
)
maxv = max(np.nanmax(y_test) if len(y_test)>0 else 1, np.nanmax(y_pred) if len(y_pred)>0 else 1)
trace_diag = go.Scatter(x=[0, maxv], y=[0, maxv], mode="lines", line=dict(color="#00F5D4", dash="dash"), name="Ideal (y=x)")

layout_pred = go.Layout(
    template="plotly_dark",
    paper_bgcolor=DARK_BG,
    plot_bgcolor=DARK_BG,
    title=dict(text="Predicting Downloads_Xplore (RandomForest baseline)", font=dict(size=24, color="#FFFFFF")),
    margin=dict(l=70, r=300, t=100, b=120),
    xaxis=dict(title="Actual Downloads", rangemode="tozero"),
    yaxis=dict(title="Predicted Downloads", rangemode="tozero"),
    hoverlabel=dict(bgcolor=HOVER_BG, font=dict(color=HOVER_FONT)),
    legend=dict(orientation="v", x=1.02, y=1),
    height=PLOT_HEIGHT,
    width=PLOT_WIDTH
)
fig_pred = go.Figure(data=[trace_actual_pred, trace_diag], layout=layout_pred)

trace_resid = go.Histogram(x=residuals, nbinsx=60, marker=dict(color="#EF476F"), name="Residuals")
layout_resid = go.Layout(
    template="plotly_dark",
    paper_bgcolor=DARK_BG,
    plot_bgcolor=DARK_BG,
    title=dict(text="Residuals Distribution (Actual - Predicted Downloads)", font=dict(size=24, color="#FFFFFF")),
    margin=dict(l=70, r=300, t=100, b=120),
    xaxis=dict(title="Residual"),
    hoverlabel=dict(bgcolor=HOVER_BG, font=dict(color=HOVER_FONT)),
    height=PLOT_HEIGHT,
    width=PLOT_WIDTH
)
fig_resid = go.Figure(data=[trace_resid], layout=layout_resid)

# 2.4 Anomaly detection
df["TitleNorm"] = df.get("Title", pd.Series([""]*len(df))).str.strip().str.lower()
title_dups = df["TitleNorm"].duplicated(keep=False)
doi_dups = df["DOI"].duplicated(keep=False) if "DOI" in df.columns else pd.Series([False]*len(df))

raw_df = pd.read_csv(DATA_FILE)
page_anomaly = pd.Series(False, index=raw_df.index)
if "FirstPage" in raw_df.columns and "LastPage" in raw_df.columns:
    mask_first = raw_df["FirstPage"].notnull()
    mask_last = raw_df["LastPage"].notnull()
    mask_both = mask_first & mask_last
    if mask_both.any():
        page_anomaly.loc[mask_both] = raw_df.loc[mask_both, "LastPage"] < raw_df.loc[mask_both, "FirstPage"]

anomaly_types = []
anomaly_idxs = []
for i in df.index:
    reasons = []
    if title_dups.iloc[i]:
        reasons.append("Duplicate Title")
    if doi_dups.iloc[i]:
        reasons.append("Duplicate DOI")
    if i < len(page_anomaly) and page_anomaly.iloc[i]:
        reasons.append("Page Range Anomaly")
    if reasons:
        anomaly_types.append("; ".join(reasons))
        anomaly_idxs.append(i)

anomaly_df = pd.DataFrame({
    "Index": anomaly_idxs,
    "Title": df.loc[anomaly_idxs, "Title"].values,
    "DOI": df.loc[anomaly_idxs, "DOI"].values if "DOI" in df.columns else [""]*len(anomaly_idxs),
    "Conference": df.loc[anomaly_idxs, "Conference"].values if "Conference" in df.columns else [""]*len(anomaly_idxs),
    "Year": df.loc[anomaly_idxs, "Year"].values if "Year" in df.columns else [""]*len(anomaly_idxs),
    "Issues": anomaly_types
})

anomaly_count = Counter()
for t in anomaly_types:
    for sub in t.split(";"):
        anomaly_count[sub.strip()] += 1
anomaly_items_sorted = sorted(anomaly_count.items(), key=lambda x: -x[1])
anomaly_labels = [a for a, _ in anomaly_items_sorted]
anomaly_values = [v for _, v in anomaly_items_sorted]

trace_anom_bar = go.Bar(x=anomaly_labels, y=anomaly_values,
                       marker=dict(color=["#FF6B6B", "#FFD166", "#06D6A0"], line=dict(color="#FFFFFF", width=0.5)),
                       text=anomaly_values, textposition="auto")
layout_anom = go.Layout(
    template="plotly_dark",
    paper_bgcolor=DARK_BG,
    plot_bgcolor=DARK_BG,
    title=dict(text="Detected Metadata Anomalies (counts)", font=dict(size=20, color="#FFFFFF")),
    margin=dict(l=70, r=300, t=100, b=120),
    xaxis=dict(title="Anomaly Type"),
    yaxis=dict(title="Count"),
    hoverlabel=dict(bgcolor=HOVER_BG, font=dict(color=HOVER_FONT)),
    height=520,
    width=PLOT_WIDTH
)
fig_anom = go.Figure(data=[trace_anom_bar], layout=layout_anom)

# Table of anomalies (top 200)
table_rows = anomaly_df.head(200)
table_header = dict(values=list(table_rows.columns),
                    fill_color="#111111",
                    align="left",
                    font=dict(color="#FFFFFF", size=12))
table_cells = dict(values=[table_rows[col].astype(str).tolist() for col in table_rows.columns],
                   fill_color="#111111",
                   align="left",
                   font=dict(color="#F0F0F0", size=11))
trace_table = go.Table(header=table_header, cells=table_cells)
layout_table = go.Layout(template="plotly_dark", paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
                        margin=dict(l=20, r=20, t=40, b=20), height=600, width=PLOT_WIDTH)
fig_table = go.Figure(data=[trace_table], layout=layout_table)

# ---------------------------------------------------------------------
# 3) Export figures to HTML divs (patching generated div ids for stable JS hooks)
# ---------------------------------------------------------------------
def gen_div_id(prefix):
    return f"{prefix}-{uuid.uuid4().hex[:8]}"

def set_custom_div_id(div_html, new_id):
    new_div_html = re.sub(r'<div id="[^"]+"', f'<div id="{new_id}"', div_html, count=1)
    return new_div_html

div_kw = gen_div_id("fig_kw")
div_repro = gen_div_id("fig_repro")
div_pred = gen_div_id("fig_pred")
div_resid = gen_div_id("fig_resid")
div_anom = gen_div_id("fig_anom")
div_table = gen_div_id("fig_table")

div_kw_html_raw = pyo.plot(fig_kw, include_plotlyjs='cdn', output_type='div', auto_open=False)
div_kw_html = set_custom_div_id(div_kw_html_raw, div_kw)

div_repro_html_raw = pyo.plot(fig_repro, include_plotlyjs='cdn', output_type='div', auto_open=False)
div_repro_html = set_custom_div_id(div_repro_html_raw, div_repro)

div_pred_html_raw = pyo.plot(fig_pred, include_plotlyjs='cdn', output_type='div', auto_open=False)
div_pred_html = set_custom_div_id(div_pred_html_raw, div_pred)

div_resid_html_raw = pyo.plot(fig_resid, include_plotlyjs='cdn', output_type='div', auto_open=False)
div_resid_html = set_custom_div_id(div_resid_html_raw, div_resid)

div_anom_html_raw = pyo.plot(fig_anom, include_plotlyjs='cdn', output_type='div', auto_open=False)
div_anom_html = set_custom_div_id(div_anom_html_raw, div_anom)

div_table_html_raw = pyo.plot(fig_table, include_plotlyjs='cdn', output_type='div', auto_open=False)
div_table_html = set_custom_div_id(div_table_html_raw, div_table)

# Build conference buttons HTML (selectable)
conf_buttons_html = []
for conf in top_conferences:
    safe_conf = json.dumps(conf)
    conf_buttons_html.append(f'<button class="btn conf-btn" onclick="selectConf(this, {safe_conf})">{conf}</button>')
conf_buttons_html = "\n".join(conf_buttons_html)
if len(top_conferences) == 0:
    conf_buttons_html = '<button class="btn conf-btn" onclick="selectConf(this, \'ALL\')">ALL</button>'

# ---------------------------------------------------------------------
# 4) Compose standalone HTML with CSS/JS for polished interactivity and narratives
# ---------------------------------------------------------------------
html_template = f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Interactive Analysis — dataset.csv</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body {{
      background-color: {DARK_BG};
      color: #FFFFFF;
      font-family: "Helvetica Neue", Arial, sans-serif;
      padding: 24px 32px;
    }}
    h1.main-title {{
      font-size: 34px;
      margin: 6px 0 18px 0;
      color: #FFFFFF;
    }}
    .intro, .conclusion {{
      font-size: 17px;
      color: #E6F0E6;
      max-width: 1100px;
      line-height: 1.5;
      margin-bottom: 18px;
    }}
    .section {{
      margin-bottom: 72px;
      padding: 12px 0;
    }}
    .subtitle {{
      color: #DDDDDD;
      font-size: 15px;
      margin-bottom: 12px;
      line-height: 1.4;
      max-width: 1100px;
    }}
    .narrative {{
      font-size: 15px;
      color: #E6E6E6;
      line-height: 1.5;
      max-width: 1100px;
      margin-bottom: 10px;
    }}
    .main-takeaway {{
      font-size: 15px;
      color: #FFFFFF;
      background-color: rgba(0,0,0,0.18);
      padding: 8px 10px;
      border-radius: 6px;
      display: inline-block;
      margin-top: 8px;
    }}
    /* Buttons */
    .btn {{
      display:inline-block;
      padding:10px 16px;
      margin: 6px 10px 12px 0;
      font-size:14px;
      border-radius:8px;
      border: none;
      cursor: pointer;
      transition: transform 0.08s ease, box-shadow 0.08s ease;
      box-shadow: 0 4px 14px rgba(0,0,0,0.55);
      color: {VIBRANT_BTN_TEXT};
      background-color: {VIBRANT_BTN};
    }}
    .btn:hover {{
      transform: translateY(-3px);
      filter: brightness(0.95);
    }}
    .btn:active {{ transform: translateY(0); }}
    .btn-alt {{ background-color: #FF6B6B; color: #FFFFFF; }}
    /* Selected state (light green) */
    .btn-selected {{
      background-color: {SELECT_BG} !important;
      color: {SELECT_TEXT} !important;
      box-shadow: 0 6px 18px rgba(0,0,0,0.6);
    }}
    .control-row {{ display:flex; align-items:center; gap:20px; flex-wrap:wrap; margin-bottom:12px; }}
    .control-desc {{ color:#AAAAAA; font-size:13px; min-width:320px; max-width:800px; }}
    .plot-container {{ overflow-x:auto; padding:12px 0 22px 0; }}
    .legend-note {{ color:#CCCCCC; font-size:12px; margin-top:8px; }}
    .note {{ color:#999999; font-size:12px; margin-top:8px; }}
    .table-wrap {{ overflow-x:auto; margin-top:8px; }}
  </style>
</head>
<body>
  <h1 class="main-title">Research Dataset Interactive Visualizations — Educational Report</h1>

  <!-- Introduction -->
  <div class="intro">
    <p>
      This report provides an interactive, educational exploration of the research dataset (dataset.csv). It is designed to guide readers step-by-step through
      multiple analyses: topic/keyword trends over time, relationships between reproducibility indicators and impact metrics,
      a baseline predictive model for paper downloads, and a metadata quality audit. The visuals are interactive (hover to inspect, toggle lines, and choose views),
      and each section includes plain-language explanations so someone without domain expertise can interpret the graphs and take action.
    </p>
    <p>
      The flow of the report follows a logical path: first we examine the evolution and prominence of research topics; then we study how replicability signals relate to early
      impact (downloads and citations); next we present a simple machine learning baseline that predicts downloads and inspect its errors; finally we show detected metadata anomalies.
      Each section explains the algorithm used, why it was selected, what the plot shows (axes, keys, and interpretation), and what strong or average findings would look like.
    </p>
    <p class="main-takeaway"><strong>Main introduction takeaway:</strong> This report is built to be interpretable — it combines clear interactive visuals with easy-to-read narratives so you can quickly understand trends, relationships, and data quality issues.</p>
  </div>

  <!-- Section 1: Keyword Trends -->
  <div class="section" id="section-keywords">
    <div class="subtitle">
      1) Keyword / Topic Trends Over Time
    </div>

    <div class="narrative">
      <p>
        Algorithm & purpose: This visualization aggregates paper keywords and frequently occurring title tokens and plots their yearly counts. We selected a simple count-based time series approach
        because it provides an intuitive and robust signal of topic prevalence across years and conferences without requiring heavy modelling.
        The graph helps answer: which topics are rising or falling, and how specialized conferences are in certain topics.
      </p>
      <p>
        What the plot shows: each visible line corresponds to a token (keyword or title token) and the x-axis shows Year while the y-axis shows the count of papers mentioning that token.
        The per-conference buttons on the left allow switching between the “ALL” consolidated view and specific conferences. Hover on any point to see the token, year and raw count.
      </p>
      <p>
        Interpretation and significance: a steadily increasing curve indicates a growing area of interest; a sharp recent spike suggests an emerging or hot topic.
        An almost flat low-count curve suggests the token is niche or uncommon. A conference that shows a different trend from the global "ALL" curve is specialized.
      </p>
      <p class="main-takeaway"><strong>Key takeaway:</strong> Look for tokens with consistent upward slopes or recent spikes — these indicate emerging research directions that may merit attention or further investigation.</p>
    </div>

    <div class="control-row">
      <div>
        <button class="btn conf-btn btn-selected" onclick="selectConf(this, 'ALL')">Show ALL</button>
        {conf_buttons_html}
      </div>
      <div class="control-desc">Use the buttons to highlight token trends for specific conferences. Selecting a conference will display only that conference's token timelines.</div>
    </div>

    <div class="plot-container">
      {div_kw_html}
    </div>

    <div class="legend-note">
      Practical interpretation note: An "average" token often has a modest, smooth curve with small year-to-year variance. Significant findings are tokens that depart markedly from this average — large upward slopes or sudden jumps.
    </div>
  </div>

  <!-- Section 2: Reproducibility vs Citations -->
  <div class="section" id="section-repro">
    <div class="subtitle">
      2) Reproducibility Signals vs Citations & Downloads
    </div>

    <div class="narrative">
      <p>
        Algorithm & purpose: We construct a lightweight reproducibility proxy (binary) from metadata and text cues: presence of a graphical replicability stamp, or textual mentions such as
        "code", "github", or "data available" in the title/abstract/keywords. We then plot Downloads (y-axis) against CitationCount_CrossRef (x-axis). Marker color denotes the reproducibility proxy
        and marker size is proportional to downloads. This simple scatter plot was chosen because it clearly shows cross-sectional relationships and is easy to interpret.
      </p>
      <p>
        What the plot shows: each point is a paper; x = citations, y = downloads. Green-ish points indicate the reproducibility proxy is present; red-ish points indicate it is absent.
        The trendline (toggleable) is a log-log linear fit that summarizes the general relationship. Use the "Toggle Log Scale" button to better see heavy-tailed distributions.
      </p>
      <p>
        Interpretation: If reproducible papers (green) cluster toward higher downloads and/or higher citations, that suggests reproducibility is associated with early visibility or impact.
        In contrast, a random scatter of green and red without separation suggests little observable relationship in this dataset. On heavy-tailed metrics (few papers have very high citations/downloads),
        use the log scale for clearer patterns.
      </p>
      <p class="main-takeaway"><strong>Key takeaway:</strong> A visible cluster of reproducible (green) papers at higher downloads/citations would indicate reproducibility correlates with early impact — an actionable insight for promoting reproducible practices.</p>
    </div>

    <div class="control-row">
      <div>
        <button id="btn-trend" class="btn" onclick="toggleTrendline(this)">Toggle Trendline</button>
        <button id="btn-log" class="btn btn-alt" onclick="toggleLogScale(this)">Toggle Log Scale</button>
      </div>
      <div class="control-desc">Use the Trendline to see the overall fitted relationship and Log Scale to inspect heavy-tailed behavior.</div>
    </div>

    <div class="plot-container">
      {div_repro_html}
    </div>

    <div class="note">
      What is impactful here: a strong upward trendline and separation between colors are meaningful; average behavior is a diffuse cloud centered near low-to-moderate citation and download counts.
    </div>
  </div>

  <!-- Section 3: Downloads Prediction -->
  <div class="section" id="section-pred">
    <div class="subtitle">
      3) Predicting Downloads — Baseline Model & Error Analysis
    </div>

    <div class="narrative">
      <p>
        Algorithm & purpose: This section demonstrates a baseline RandomForest regressor trained to predict Downloads_Xplore. Inputs include TF-IDF features from Title+Abstract (reduced via SVD),
        numeric metadata (citation counts, references, year) and a one-hot encoding of the top conferences. RandomForest was selected as a robust ensemble baseline that handles mixed feature types.
      </p>
      <p>
        What the visuals show: the "Actual vs Predicted" scatter plots observed downloads (x) against predicted downloads (y). The diagonal y=x is the ideal — points near it are well predicted.
        The Residuals view (toggle) shows the distribution of prediction errors (Actual - Predicted). Both figures help diagnose bias and heteroskedasticity (wider errors for large counts).
      </p>
      <p>
        Interpretation: A high-performing model places most points near the diagonal with small symmetric residuals centered near zero. If residuals are skewed or heavy-tailed, the model may under/overpredict systematically.
        Use this baseline to iterate: improving text representations (better embeddings), adding author-level features, or modeling counts with negative-binomial approaches can improve performance.
      </p>
      <p class="main-takeaway"><strong>Key takeaway:</strong> Points tightly clustered around the diagonal and a narrow, centered residual distribution indicate good predictive performance — otherwise, refine features or model family.</p>
    </div>

    <div class="control-row">
      <div>
        <button id="btn-view-scatter" class="btn btn-selected" onclick="showPredView('scatter', this)">Actual vs Predicted</button>
        <button id="btn-view-resid" class="btn" onclick="showPredView('residual', this)">Residuals</button>
      </div>
      <div class="control-desc">Switch views to inspect overall fit (scatter) or error distribution (residuals). Both are important for assessing model reliability.</div>
    </div>

    <div class="plot-container" id="pred-container">
      {div_pred_html}
      <div id="resid-div" style="display:none;">
        {div_resid_html}
      </div>
    </div>

    <div class="note">
      What to look for: systematic departures from the diagonal (e.g., consistent underprediction at high download counts) indicate model mismatch. A typical baseline shows broader spread for large-outlier papers.
    </div>
  </div>

  <!-- Section 4: Anomalies & Data QA -->
  <div class="section" id="section-anom">
    <div class="subtitle">
      4) Metadata Anomalies & Dataset Integrity
    </div>

    <div class="narrative">
      <p>
        Algorithm & purpose: This quality audit uses rule-based checks (e.g., LastPage < FirstPage), simple duplicate detection on normalized titles and DOIs, and aggregates counts of flagged issues.
        Ensuring metadata integrity is crucial before building models or making bibliometric conclusions because anomalies can distort results.
      </p>
      <p>
        What the visuals show: a summary bar chart shows counts of anomalies by type (duplicate titles, duplicate DOIs, page-range anomalies). The table lists flagged rows for manual inspection and correction.
      </p>
      <p>
        Interpretation: A small number of anomalies is expected; a large number suggests systemic extraction or parsing errors that should be prioritized. Correcting high-impact anomalies (duplicate DOIs, broken page ranges)
        will reduce duplicated counts and erroneous links between records.
      </p>
      <p class="main-takeaway"><strong>Key takeaway:</strong> Prioritize fixing duplicate DOIs and inverted page ranges — these issues lead to duplicated records and incorrect page metadata that bias downstream analytics.</p>
    </div>

    <div class="control-row">
      <div>
        <button class="btn btn-alt" onclick="scrollToTable()">Go to Flagged Rows</button>
      </div>
      <div class="control-desc">Click the button to scroll to the table of flagged rows for manual review and correction.</div>
    </div>

    <div class="plot-container">
      {div_anom_html}
    </div>
    <div class="table-wrap" id="anomaly-table">
      {div_table_html}
    </div>
    <div class="note">
      Practical implications: Fixing these anomalies improves the reliability of topic counts, prediction targets, and any aggregation used in reports or dashboards.
    </div>
  </div>

  <!-- Conclusion -->
  <div class="conclusion">
    <p>
      Conclusion and next steps: This educational report combined interpretable visualizations and a basic predictive pipeline to highlight topic trends, reproducibility signals, predictive baseline performance, and dataset issues.
      The immediate next steps for research scientists are: (1) perform more advanced topic modeling (dynamic topic models or BERTopic) to get interpretable topic labels; (2) refine the reproducibility proxy with weak supervision or manual labels;
      (3) iterate on predictive models with richer author/conference/time features and uncertainty-aware models; and (4) implement a data cleaning pipeline that resolves the flagged anomalies.
    </p>
    <p class="main-takeaway"><strong>Final takeaway:</strong> Use this interactive report as a starting point—visual inspection, simple models, and data QA together form a practical workflow to turn noisy bibliographic data into actionable insights.</p>
  </div>

  <!-- Custom JS for interactivity (no disappearing descriptions, selected buttons become light green) -->
  <script>
    function getPlotlyGraphDiv(outerDivId) {{
      const outer = document.getElementById(outerDivId);
      if (!outer) {{
        const candidates = document.querySelectorAll("div");
        for (let c of candidates) {{
          if (c.id && c.id.indexOf(outerDivId.split('-')[0]) === 0) return c.querySelector(".plotly-graph-div") || c;
        }}
        return null;
      }}
      return outer.querySelector(".plotly-graph-div") || outer;
    }}

    const kwDiv = getPlotlyGraphDiv("{div_kw}");
    const reproDiv = getPlotlyGraphDiv("{div_repro}");
    const predDiv = getPlotlyGraphDiv("{div_pred}");
    const residDiv = getPlotlyGraphDiv("{div_resid}");
    const anomDiv = getPlotlyGraphDiv("{div_anom}");
    const tableDiv = getPlotlyGraphDiv("{div_table}");

    const topicTokens = {json.dumps(topic_tokens)};
    const topConfs = ["ALL", {', '.join(json.dumps(c) for c in top_conferences)}];

    function setButtonSelected(btn) {{
      const parent = btn.parentNode;
      const buttons = parent.querySelectorAll(".conf-btn");
      buttons.forEach(b => b.classList.remove("btn-selected"));
      btn.classList.add("btn-selected");
    }}

    function selectConf(btnElem, confName) {{
      try {{
        if (typeof btnElem === 'string' || btnElem === undefined) {{
          const all = document.querySelectorAll(".conf-btn");
          all.forEach(b => {{
            if (b.innerText.trim() === confName || (confName === 'ALL' && b.innerText.trim().toUpperCase() === 'SHOW ALL')) {{
              b.classList.add("btn-selected");
            }} else {{
              b.classList.remove("btn-selected");
            }}
          }});
        }} else {{
          setButtonSelected(btnElem);
        }}
      }} catch(e) {{
        console.warn("selectConf button styling:", e);
      }}

      const conf = String(confName);
      const gd = kwDiv;
      if (!gd || !gd.data) return;
      const traces = gd.data;
      const blockSize = topicTokens.length;
      let confIndex = topConfs.indexOf(conf);
      if (confIndex === -1) confIndex = 0;
      const total = traces.length;
      let visibility = new Array(total).fill(false);
      const start = confIndex * blockSize;
      for (let i = 0; i < blockSize; i++) {{
        const idx = start + i;
        if (idx < total) visibility[idx] = true;
      }}
      Plotly.restyle(gd, {{visible: visibility}});
    }}

    function toggleTrendline(btn) {{
      const gd = reproDiv;
      if (!gd || !gd.data) return;
      const trendIdx = 1;
      if (!gd.data[trendIdx]) return;
      const cur = gd.data[trendIdx].visible;
      const newVis = (cur === true || cur === undefined) ? "legendonly" : true;
      Plotly.restyle(gd, {{visible: newVis}}, [trendIdx]);
      if (btn.classList.contains("btn-selected")) btn.classList.remove("btn-selected"); else btn.classList.add("btn-selected");
    }}

    function toggleLogScale(btn) {{
      const gd = reproDiv;
      if (!gd) return;
      const currentX = (gd.layout && gd.layout.xaxis && gd.layout.xaxis.type) ? gd.layout.xaxis.type : "linear";
      const newType = currentX === "linear" ? "log" : "linear";
      Plotly.relayout(gd, {{'xaxis.type': newType, 'yaxis.type': newType}});
      if (btn.classList.contains("btn-selected")) btn.classList.remove("btn-selected"); else btn.classList.add("btn-selected");
    }}

    function showPredView(view, btn) {{
      const scatterOuter = document.getElementById("{div_pred}");
      const residOuter = document.getElementById("{div_resid}");
      const btnScatter = document.getElementById("btn-view-scatter");
      const btnResid = document.getElementById("btn-view-resid");
      if (view === "scatter") {{
        if (scatterOuter) scatterOuter.style.display = "";
        if (residOuter) residOuter.style.display = "none";
        btnScatter.classList.add("btn-selected");
        btnResid.classList.remove("btn-selected");
      }} else {{
        if (scatterOuter) scatterOuter.style.display = "none";
        if (residOuter) residOuter.style.display = "";
        btnResid.classList.add("btn-selected");
        btnScatter.classList.remove("btn-selected");
      }}
    }}

    function scrollToTable() {{
      const el = document.getElementById("anomaly-table");
      if (el) {{
        el.scrollIntoView({{behavior: "smooth", block: "center"}});
        el.style.border = "1px solid #333333";
        setTimeout(()=> el.style.border = "none", 2400);
      }}
    }}

    try {{
      selectConf(document.querySelector(".conf-btn"), "ALL");
      showPredView('scatter');
    }} catch(e) {{
      console.warn("Initialization error:", e);
    }}
  </script>
</body>
</html>
"""

# Write to output.html
out_file = "output.html"
with open(out_file, "w", encoding="utf-8") as f:
    f.write(html_template)

print(f"Generated {out_file}. Open this file in a browser to interact with the plots.")
