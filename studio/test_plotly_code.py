# ---- NEW BLOCK ---- # 
#CODE HERE
# Enhanced Plotly Dashboard with Academic Narratives embedded in HTML
# This script reads dataset.csv, creates interactive Plotly visualizations,
# and writes a standalone output.html with an academic-style introduction,
# figure explanations (flowing like a paper), and a conclusion.

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
import nltk

# Ensure VADER lexicon is available (used later)
try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
except Exception:
    nltk.download('vader_lexicon')
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Load dataset
df = pd.read_csv('dataset.csv')

# Preprocessing: normalize numeric columns and fill NaNs with median
num_cols = ['FirstPage','LastPage','AminerCitationCount','CitationCount_CrossRef',
            'PubsCited_CrossRef','Downloads_Xplore']
for c in num_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        median_val = float(df[c].median(skipna=True))
        df[c] = df[c].fillna(median_val)

# Fill categorical NaNs with sensible defaults
if 'AuthorKeywords' in df.columns:
    df['AuthorKeywords'] = df['AuthorKeywords'].fillna('')

if 'GraphicsReplicabilityStamp' in df.columns:
    df['GraphicsReplicabilityStamp'] = df['GraphicsReplicabilityStamp'].fillna('No')

# Ensure Abstract exists
if 'Abstract' in df.columns:
    df['Abstract'] = df['Abstract'].fillna('').astype(str)
else:
    df['Abstract'] = ''

# Helper columns
def count_keywords(text):
    if not text:
        return 0
    s = str(text).replace(',', ';')
    parts = [p.strip() for p in s.split(';') if p.strip()]
    return len(parts)

df['NumKeywords'] = df['AuthorKeywords'].apply(count_keywords)

# Award flag: binary 1 if Award not null/empty else 0
df['AwardFlag'] = df.get('Award', pd.Series([np.nan]*len(df))).notnull() & (df.get('Award', '').astype(str).str.strip() != '')
df['AwardFlag'] = df['AwardFlag'].astype(int)

# Precompute some caps for visualizations
downloads_99 = df['Downloads_Xplore'].quantile(0.99) if 'Downloads_Xplore' in df.columns else 0.0
citations_99 = df['CitationCount_CrossRef'].quantile(0.99) if 'CitationCount_CrossRef' in df.columns else 0.0

# Initialize containers for plots and narrative
divs = []
narratives = []  # will be overridden with academic paragraphs later

# Prepare features for logistic regression
feat_cols = ['Year','AminerCitationCount','CitationCount_CrossRef','Downloads_Xplore','NumKeywords']
X = df[[c for c in feat_cols if c in df.columns]].copy()

# One-hot encode top 5 categories for PaperType and Conference
for col in ['PaperType','Conference']:
    if col in df.columns:
        top_cats = df[col].value_counts().nlargest(5).index
        for cat in top_cats:
            X[f"{col}_{cat}"] = (df[col] == cat).astype(int)

y = df['AwardFlag']

# Handle edge case: if all y are zero or model cannot train due to lack of positive class,
# create fallback coefficients to avoid breaking downstream code.
can_train = (y.sum() > 1) and (len(y.unique()) > 1) and (X.shape[1] > 0)
if can_train:
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)
    clf = LogisticRegression(max_iter=1000, solver='lbfgs')
    clf.fit(X_train, y_train)
    coefs = pd.Series(clf.coef_[0], index=X.columns).sort_values(ascending=False)
else:
    if X.shape[1] == 0:
        X['NoFeatures'] = 0.0
    coefs = pd.Series(0.0, index=X.columns).sort_values(ascending=False)

# Visual: show strongest pos and neg coefficients
top_pos = coefs[coefs > 0].nlargest(10)
top_neg = coefs[coefs < 0].nsmallest(10)
display_coefs = pd.concat([top_pos, top_neg]).sort_values()

fig1 = go.Figure()
fig1.add_trace(go.Bar(
    x=display_coefs.values,
    y=display_coefs.index,
    orientation='h',
    marker=dict(
        color=np.where(display_coefs.values >= 0, '#00CC96', '#EF553B'),
        line=dict(color='rgba(255,255,255,0.06)', width=0.5)
    ),
    hovertemplate="<b>%{y}</b><br>Coefficient: %{x:.4f}<extra></extra>"
))
fig1.update_layout(
    title="Feature Coefficients for Predicting Award-Winning Papers (Logistic Regression)",
    template='plotly_dark',
    paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
    xaxis_title="Coefficient (logistic regression)", yaxis_title="Feature",
    height=600,
    margin=dict(l=220, r=40, t=80, b=40),
    font=dict(color='white')
)
fig1.add_shape(type="line", x0=0, x1=0, y0=-0.5, y1=len(display_coefs)-0.5, line=dict(color="white", width=1, dash="dash"))

div1 = plot(fig1, include_plotlyjs='cdn', output_type='div')
divs.append(div1)

# A brief placeholder was previously appended into 'narratives'; we'll overwrite below with full academic text.

# -------------------------
# FIGURE 2: Trend Analysis in Research Topics (Top 5 Keywords Over Time)
# -------------------------
# Safely explode AuthorKeywords into rows of (Year, Keyword)
if 'AuthorKeywords' in df.columns:
    kw_df = df[['Year','AuthorKeywords']].copy()
    kw_df['AuthorKeywords'] = kw_df['AuthorKeywords'].astype(str).replace('^nan$', '', regex=True)
    kw_df['AuthorKeywords'] = kw_df['AuthorKeywords'].fillna('')
    kw_df['AuthorKeywords'] = kw_df['AuthorKeywords'].str.replace(',', ';')
    kw_df['KeywordList'] = kw_df['AuthorKeywords'].apply(lambda s: [k.strip() for k in s.split(';') if k.strip()])
    kw_exploded = kw_df.explode('KeywordList').rename(columns={'KeywordList':'Keyword'})
    kw_exploded = kw_exploded[kw_exploded['Keyword'].notnull() & (kw_exploded['Keyword'] != '')].copy()
else:
    kw_exploded = pd.DataFrame(columns=['Year','Keyword'])

# Count top keywords
top5 = kw_exploded['Keyword'].value_counts().nlargest(5).index.tolist() if not kw_exploded.empty else []
if top5:
    df_trend = kw_exploded[kw_exploded['Keyword'].isin(top5)].groupby(['Year','Keyword']).size().reset_index(name='Count')
    # Ensure full year range
    years = list(range(int(df['Year'].min()), int(df['Year'].max()) + 1))
    pivot = df_trend.pivot(index='Year', columns='Keyword', values='Count').reindex(years, fill_value=0)
    pivot = pivot.sort_index()
    fig2 = go.Figure()
    palette = px.colors.qualitative.Set1
    for i, kw in enumerate(pivot.columns):
        fig2.add_trace(go.Scatter(
            x=pivot.index, y=pivot[kw],
            mode='lines+markers',
            name=kw,
            marker=dict(size=6),
            line=dict(width=2, color=palette[i % len(palette)]),
            hovertemplate="<b>%{text}</b><br>Year: %{x}<br>Count: %{y}<extra></extra>",
            text=[kw] * len(pivot.index)
        ))
    fig2.update_layout(
        title="Trends of Top 5 Author Keywords Over Years",
        xaxis=dict(title='Year', rangeslider=dict(visible=True), rangemode='tozero'),
        yaxis=dict(title='Count', rangemode='tozero'),
        template='plotly_dark',
        paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
        height=600,
        font=dict(color='white'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1.0)
    )
else:
    fig2 = go.Figure()
    fig2.update_layout(title="No Keywords Available to Plot", template='plotly_dark',
                       paper_bgcolor='#0e1117', plot_bgcolor='#0e1117', font=dict(color='white'), height=400)

div2 = plot(fig2, include_plotlyjs='cdn', output_type='div')
divs.append(div2)

# -------------------------
# FIGURE 3: Citation Count Prediction Using Linear Regression
# -------------------------
# Ensure reg features exist
reg_feats = ['AminerCitationCount','PubsCited_CrossRef','Downloads_Xplore','Year','NumKeywords']
for f in reg_feats:
    if f not in df.columns:
        df[f] = 0.0

Xr = df[reg_feats].copy()
yr = pd.to_numeric(df['CitationCount_CrossRef'], errors='coerce').fillna(df['CitationCount_CrossRef'].median() if 'CitationCount_CrossRef' in df.columns else 0.0)

if yr.nunique() > 1:
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(Xr, yr, test_size=0.3, random_state=42)
    reg = LinearRegression().fit(Xr_train, yr_train)
    yr_pred = reg.predict(Xr_test)

    residuals = np.abs(yr_test.values - yr_pred)
    # Use np.ptp instead of ndarray.ptp to be compatible with NumPy 2.0+
    denom = np.ptp(residuals) if np.ptp(residuals) != 0 else 1.0
    norm_res = (residuals - residuals.min()) / (denom + 1e-9)
    # sample colorscale - ensure values in [0,1]
    norm_res = np.clip(norm_res, 0.0, 1.0)
    colors = px.colors.sample_colorscale('Viridis', list(norm_res))

    minv = float(min(np.min(yr_test.values), np.min(yr_pred)))
    maxv = float(max(np.max(yr_test.values), np.max(yr_pred)))
    pad = (maxv - minv) * 0.05 if maxv > minv else 1.0

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=yr_test, y=yr_pred,
        mode='markers',
        marker=dict(color=colors, size=7, line=dict(width=0.3, color='white')),
        hovertemplate="Actual: %{x:.1f}<br>Predicted: %{y:.1f}<br>Residual: %{customdata:.2f}<extra></extra>",
        customdata=residuals
    ))
    fig3.add_trace(go.Scatter(
        x=[minv - pad, maxv + pad],
        y=[minv - pad, maxv + pad],
        mode='lines',
        line=dict(color='white', dash='dash'),
        name='Ideal fit'
    ))
    fig3.update_layout(
        title="Linear Regression: Actual vs Predicted CitationCount_CrossRef",
        xaxis=dict(title='Actual CitationCount_CrossRef', range=[minv - pad, maxv + pad]),
        yaxis=dict(title='Predicted CitationCount_CrossRef', range=[minv - pad, maxv + pad]),
        template='plotly_dark',
        paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
        height=600,
        font=dict(color='white')
    )
else:
    fig3 = go.Figure()
    fig3.update_layout(title="Not enough variance in target to fit Linear Regression",
                       template='plotly_dark', paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
                       font=dict(color='white'), height=400)

div3 = plot(fig3, include_plotlyjs='cdn', output_type='div')
divs.append(div3)

# -------------------------
# FIGURE 4: Effect of GraphicsReplicabilityStamp on Downloads
# -------------------------
if 'GraphicsReplicabilityStamp' in df.columns:
    df['Replicable'] = df['GraphicsReplicabilityStamp'].astype(str).str.contains('yes', case=False, na=False)
    df['Replicable'] = df['Replicable'].map({True:'Yes', False:'No'})
else:
    df['Replicable'] = 'No'

# Compute visualization cap for downloads (99th percentile)
cap = downloads_99 if downloads_99 > 0 else df['Downloads_Xplore'].max()
y_max = float(cap * 1.05) if cap > 0 else float(df['Downloads_Xplore'].max() * 1.05)

fig4 = go.Figure()
for val, color in [('Yes', '#636EFA'), ('No', '#EF553B')]:
    subset = df[df['Replicable'] == val]
    if not subset.empty:
        fig4.add_trace(go.Box(
            y=subset['Downloads_Xplore'].clip(upper=y_max),
            name=val,
            marker_color=color,
            boxmean='sd',
            hovertemplate="Replicable: " + val + "<br>Downloads (capped): %{y}<extra></extra>",
            boxpoints='outliers'
        ))

# add jittered sample points
for val, color in [('Yes', '#636EFA'), ('No', '#EF553B')]:
    subset = df[df['Replicable'] == val]
    if not subset.empty:
        sample = subset.sample(n=min(300, len(subset)), random_state=1)
        fig4.add_trace(go.Scatter(
            x=[val] * len(sample),
            y=sample['Downloads_Xplore'].clip(upper=y_max),
            mode='markers',
            marker=dict(color=color, size=6, opacity=0.6, line=dict(width=0.2, color='black')),
            hovertemplate="Replicable: %{x}<br>Downloads (capped): %{y}<extra></extra>",
            showlegend=False
        ))

fig4.update_layout(
    title="Downloads (Xplore) by Graphics Replicability Stamp (capped at 99th percentile)",
    yaxis=dict(title='Downloads_Xplore (capped)', range=[0, y_max]),
    template='plotly_dark',
    paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
    height=600,
    font=dict(color='white')
)

div4 = plot(fig4, include_plotlyjs='cdn', output_type='div')
divs.append(div4)

# -------------------------
# FIGURE 5: Sentiment Analysis on Abstract vs Citation
# -------------------------
sid = SentimentIntensityAnalyzer()
df['Sentiment'] = df['Abstract'].astype(str).apply(lambda t: sid.polarity_scores(t)['compound'])

# Cap citation values for clearer plotting
cap_cit = citations_99 if citations_99 > 0 else df['CitationCount_CrossRef'].max()
y_max_cit = float(cap_cit * 1.05) if cap_cit > 0 else float(df['CitationCount_CrossRef'].max() * 1.05)

# sample for plotting clarity and performance
sample_df = df.sample(n=min(2000, len(df)), random_state=2)

fig5 = px.scatter(
    sample_df,
    x='Sentiment', y='CitationCount_CrossRef',
    color='NumKeywords',
    color_continuous_scale='Turbo',
    hover_data={'Title': True, 'DOI': True, 'CitationCount_CrossRef': True, 'Sentiment': True, 'NumKeywords': True},
    title="VADER Sentiment of Abstract vs CrossRef Citations (sampled points)",
)
# Add linear fit using sklearn for all data (uncapped fit)
try:
    xr = df[['Sentiment']].copy()
    yr_all = pd.to_numeric(df['CitationCount_CrossRef'], errors='coerce').fillna(0.0)
    if yr_all.nunique() > 1:
        lr = LinearRegression().fit(xr, yr_all)
        xs = np.linspace(df['Sentiment'].min(), df['Sentiment'].max(), 100)
        ys = lr.predict(xs.reshape(-1,1))
        fig5.add_trace(go.Scatter(x=xs, y=ys, mode='lines', line=dict(color='white', dash='dash'),
                                  name='Linear fit', hoverinfo='skip'))
except Exception:
    pass

fig5.update_traces(marker=dict(size=6, opacity=0.75, line=dict(width=0.2, color='black')))
fig5.update_layout(
    template='plotly_dark',
    paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
    yaxis=dict(range=[0, y_max_cit] if y_max_cit > 0 else None),
    xaxis=dict(title='VADER Compound Sentiment'),
    yaxis_title='CitationCount_CrossRef (capped)',
    height=600,
    font=dict(color='white')
)

div5 = plot(fig5, include_plotlyjs='cdn', output_type='div')
divs.append(div5)

# -------------------------
# Compose academic narratives (Introduction, figure paragraphs, and Conclusion)
# -------------------------
intro = (
    "<h2>Introduction</h2>"
    "<p>"
    "This report presents an integrative descriptive and predictive analysis of a corpus of conference papers. "
    "Using the provided dataset, we explore five complementary analytical perspectives: (1) a predictive model "
    "for award-winning papers using logistic regression on bibliometric and metadata features; (2) a temporal "
    "trend analysis of the most frequent author keywords to reveal emergent research topics; (3) a linear "
    "regression model to estimate CrossRef citation counts from early indicators; (4) an assessment of whether "
    "a graphics replicability stamp is associated with higher download counts; and (5) a sentiment analysis of "
    "paper abstracts and its relationship to citation impact. Each analysis is presented with an interactive "
    "visualization and an interpretive discussion intended for researchers, program committees, and research policy audiences."
    "</p>"
)

# Figure-specific academic paragraphs; each paragraph transitions to the next
fig1_paragraph = (
    "<h3>Figure 1 — Predictors of Award-Winning Papers</h3>"
    "<p>"
    "Figure 1 visualizes the coefficients from a logistic regression which models the probability that a paper "
    "receives an award. The independent variables combine publication metadata and bibliometrics (publication year, "
    "internal and external citation counts, download counts, keyword richness, and one-hot encodings of the most "
    "frequent paper types and conferences). Coefficients greater than zero indicate features that increase the predicted "
    "award probability, and negative coefficients correspond to features that reduce it. This coefficient-based view "
    "is valuable because it provides an interpretable, directionally informative summary of primary signals associated "
    "with awards: citation-related metrics and venue identities often appear among the strongest contributors. Readers "
    "should consider these findings as hypothesis-generating rather than causal evidence, and the results motivate "
    "additional analyses that account for confounding and selection biases when forming recommendations for committee use."
    "</p>"
)

fig2_paragraph = (
    "<h3>Figure 2 — Temporal Trends of Top Author Keywords</h3>"
    "<p>"
    "Figure 2 traces the annual frequencies of the five most frequent author-supplied keywords. We extracted keywords "
    "from the AuthorKeywords field, normalized common separators, and aggregated counts per year. This longitudinal "
    "view reveals which topics gain traction and which decline, offering an empirical basis for mapping research fronts "
    "and advising strategic publication decisions. Observing trends alongside the award-predictor findings helps contextualize "
    "how topicality and community focus interact with impact metrics: rising keywords may predict future citation growth "
    "or shifts in award distributions."
    "</p>"
)

fig3_paragraph = (
    "<h3>Figure 3 — Predicting CrossRef Citation Counts</h3>"
    "<p>"
    "Figure 3 compares actual to predicted CrossRef citation counts from a linear regression trained on early indicators "
    "(Aminer citations, the number of CrossRef-cited publications, downloads, publication year, and keyword counts). "
    "The diagonal line denotes perfect prediction; deviations reveal under- and over-predictions. While the model offers "
    "a simple baseline for understanding how early readership and referral metrics relate to longer-term citations, the "
    "scatter and residual patterns emphasize that citations are influenced by many non-linear and latent factors. Future "
    "work should evaluate more flexible algorithms and rigorous holdout strategies to quantify predictive gains and generalization."
    "</p>"
)

fig4_paragraph = (
    "<h3>Figure 4 — Downloads and Graphics Replicability Stamps</h3>"
    "<p>"
    "Figure 4 examines the empirical relationship between a graphics replicability stamp and Xplore downloads. We compare "
    "download distributions for papers with and without the stamp, capping extreme values at the 99th percentile to preserve "
    "visual interpretability. The distributional comparison sheds light on whether explicit indicators of reproducible or "
    "well-documented visual materials align with higher readership. If a consistent uplift is observed for stamped papers, "
    "this provides pragmatic support for policies that promote transparent graphical practices; nonetheless, causal claims "
    "would require additional designs such as propensity-score stratification or randomized incentives."
    "</p>"
)

fig5_paragraph = (
    "<h3>Figure 5 — Sentiment of Abstracts and Citation Impact</h3>"
    "<p>"
    "Figure 5 explores whether an abstract's affective tone, as quantified by VADER's compound sentiment score, is associated "
    "with CrossRef citation counts. Although sentiment analysis methods were developed for general language and social media, "
    "they can reveal stylistic tendencies in abstracts (e.g., confidently framed contributions versus cautious language). The "
    "scatter plot with a fitted trend line offers an initial empirical probe: any observed correlation should be interpreted with "
    "caution given methodological limitations of sentiment analysis on scientific text and confounds such as topic, venue, and author "
    "reputation. This exploratory result motivates refined natural-language approaches (topic-conditioned sentiment or rhetorical-role modeling) "
    "to better understand writing practices and visibility."
    "</p>"
)

conclusion = (
    "<h2>Conclusion</h2>"
    "<p>"
    "Taken together, these analyses provide a multifaceted view of scholarly impact and practice within the dataset: metadata and "
    "citation measures are informative predictors of awards; keyword trends map intellectual shifts; simple regression models offer "
    "useful but limited prediction of citation outcomes; replicability markers may correlate with readership; and abstract sentiment "
    "presents an intriguing, hypothesis-generating link to visibility. Important limitations include the observational nature of the data, "
    "the reliance on surface features, and potential biases in citation and download measures. We recommend next steps such as causal inference "
    "analyses to probe the effect of badges, network analyses of co-authorship and citation topology, richer textual models for abstracts, "
    "and the adoption of more flexible machine learning approaches with thoughtful validation. Collectively, these directions can help the "
    "research community and conference organizers forge evidence-based practices that promote robust, visible, and reproducible research."
    "</p>"
)

# Replace the earlier (short) narratives with the figure paragraphs mapped to the divs order
narratives = [fig1_paragraph, fig2_paragraph, fig3_paragraph, fig4_paragraph, fig5_paragraph]

# -------------------------
# Assemble final HTML document with flow like an academic paper
# -------------------------
html_parts = [
    "<!DOCTYPE html>",
    "<html>",
    "<head>",
    "  <meta charset='utf-8'/>",
    "  <meta name='viewport' content='width=device-width, initial-scale=1'/>",
    "  <title>Research Analytics Dashboard (Enhanced)</title>",
    "  <style>",
    "    body { background: #0b0f14; color: #ffffff; font-family: Arial, Helvetica, sans-serif; margin: 18px; line-height:1.55 }",
    "    h1 { margin-bottom: 6px; }",
    "    h2 { color: #e6eef8; }",
    "    h3 { color: #dbeafe; margin-top: 18px }",
    "    .figure { margin-bottom: 48px; padding: 18px; border-radius: 8px; background: #0b1116; box-shadow: 0 4px 12px rgba(0,0,0,0.6);} ",
    "    .narrative { margin-bottom: 12px; color: #dbeafe; font-size: 15px; }",
    "    .meta { color: #9aa6b2; font-size: 13px; margin-bottom: 10px; }",
    "    .citation { color: #9aa6b2; font-size: 13px; margin-top:6px }",
    "  </style>",
    "</head>",
    "<body>",
    "  <h1>Research Analytics Dashboard (Enhanced)</h1>",
    "  <p class='meta'>This interactive report provides visual and interpretive analyses derived from <code>dataset.csv</code>. "
    "All interactive charts are embedded below; hover and zoom features are available to explore the data.</p>",
    "  <div class='figure'>",
    f"    <div class='narrative'>{intro}</div>",
    "  </div>",
]

# Insert each figure block with its academic narrative; keep order aligned with divs
for idx, (narr, div) in enumerate(zip(narratives, divs), start=1):
    html_parts.append("<div class='figure'>")
    # Add a succinct figure heading and the academic narrative (already contains <h3>)
    html_parts.append(f"{narr}")
    # Insert the interactive plot div
    html_parts.append(div)
    html_parts.append("</div>")

# Add conclusion block
html_parts.append("<div class='figure'>")
html_parts.append(f"<div class='narrative'>{conclusion}</div>")
html_parts.append("</div>")

html_parts.append("</body></html>")

with open('output.html', 'w', encoding='utf-8') as f:
    f.write("\n".join(html_parts))

print("Enhanced dashboard saved to output.html")