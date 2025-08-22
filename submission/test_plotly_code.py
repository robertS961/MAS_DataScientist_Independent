# ---- NEW BLOCK ---- # 
# -*- coding: utf-8 -*-
"""
Enhanced interactive Plotly visualization script for dataset.csv with educational report narratives.
- Dark theme, large figures, spaced annotations, non-overlapping controls
- Adds an introduction, Per-figure educational narratives, and a conclusion
- Produces output.html with each figure embedded using Plotly CDN (include_plotlyjs='cdn')

Requirements:
    pip install pandas numpy scikit-learn plotly scipy
Run:
    python this_script.py
"""
import os
import math
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.impute import SimpleImputer
from scipy import sparse
import plotly.graph_objs as go
import plotly.io as pio
from plotly.subplots import make_subplots

# Global figure sizing and theme
FIG_WIDTH = 1300
FIG_HEIGHT = 720
DARK_BG = "#0e1117"
PLOT_BG = DARK_BG
PAPER_BG = DARK_BG
FONT_COLOR = "#E6EEF8"
HOVER_BG = "#222222"
LEGEND_Y = -0.22  # place legend below plots
BUTTON_BG = "#223338"
BUTTON_FONT = {'color': '#FFFFFF', 'size': 12}
BUTTON_ACTIVE_BG = "#A8E6A1"  # light green for active hint (menu background only)

# -------------------------
# 1. Load dataset & preprocess
# -------------------------
df = pd.read_csv("dataset.csv")

# Basic cleaning and derived features
df['Title'] = df['Title'].fillna("")
df['Abstract'] = df['Abstract'].fillna("")
df['AuthorNames-Deduped'] = df['AuthorNames-Deduped'].fillna(df['AuthorNames'].fillna(''))
df['AuthorNames'] = df['AuthorNames'].fillna('')
df['AuthorAffiliation'] = df['AuthorAffiliation'].fillna('unknown')
df['AuthorKeywords'] = df['AuthorKeywords'].fillna('')

# Numeric imputation (median)
num_cols = ['FirstPage', 'LastPage', 'AminerCitationCount', 'CitationCount_CrossRef',
            'PubsCited_CrossRef', 'Downloads_Xplore']
imp = SimpleImputer(strategy='median')
df[num_cols] = imp.fit_transform(df[num_cols])

# Derived features
def compute_page_count(row):
    try:
        fp = row.get('FirstPage', np.nan)
        lp = row.get('LastPage', np.nan)
        if not (pd.isna(fp) or pd.isna(lp)):
            return max(1, int(lp - fp + 1))
    except Exception:
        pass
    return 1

df['page_count'] = df.apply(compute_page_count, axis=1)

def count_authors(s):
    if not isinstance(s, str) or s.strip()=='':
        return 1
    for sep in [';', ',', ' and ', ' & ', '|', '\n']:
        if sep in s:
            parts = [x.strip() for x in s.split(sep) if x.strip()!='']
            return max(1, len(parts))
    return 1

df['num_authors'] = df['AuthorNames-Deduped'].apply(count_authors)

df['title_length'] = df['Title'].astype(str).apply(lambda x: len(x.split()))
df['abstract_length'] = df['Abstract'].astype(str).apply(lambda x: len(x.split()))
df['year_since'] = df['Year'].max() - df['Year'] + 1
df['citations_per_year'] = (df['CitationCount_CrossRef'] + 1) / df['year_since']

# Award flag and replicability flag
df['Award_flag'] = df['Award'].notnull().astype(int)
df['Replicability_flag'] = df['GraphicsReplicabilityStamp'].notnull().astype(int)

# Ensure types consistent
df['Year'] = df['Year'].astype(int)

# -------------------------
# 2. Topic modeling (TF-IDF -> SVD -> KMeans)
# -------------------------
df['text_for_topic'] = (df['Title'] + ' . ' + df['Abstract'] + ' . ' + df['AuthorKeywords'])

tfidf = TfidfVectorizer(max_features=4000, ngram_range=(1,2), stop_words='english')
X_tfidf = tfidf.fit_transform(df['text_for_topic'])

# SVD reduction (keep components <= min(n_samples-1, 100))
n_comp = min(100, X_tfidf.shape[0]-1)
if n_comp <= 0:
    n_comp = 1
svd = TruncatedSVD(n_components=n_comp, random_state=42)
X_reduced = svd.fit_transform(X_tfidf)

n_topics = 8
n_topics = min(n_topics, max(1, df.shape[0]//10))  # avoid too many clusters for tiny datasets
kmeans = KMeans(n_clusters=n_topics, random_state=42, n_init=20)
topic_labels = kmeans.fit_predict(X_reduced)
df['topic'] = topic_labels

terms = np.array(tfidf.get_feature_names_out())
top_terms = {}
for t in range(n_topics):
    idx = np.where(df['topic'] == t)[0]
    if len(idx) == 0:
        top_terms[t] = ["(no docs)"]
        continue
    mean_tfidf = np.asarray(X_tfidf[idx].mean(axis=0)).ravel()
    top_feats = terms[np.argsort(mean_tfidf)[-8:]][::-1]
    top_terms[t] = [t for t in top_feats]

topic_year = df.groupby(['Year', 'topic']).size().reset_index(name='count')
years_sorted = sorted(df['Year'].unique())
topic_prevalence = topic_year.pivot(index='Year', columns='topic', values='count').reindex(years_sorted).fillna(0)
topic_prevalence_frac = topic_prevalence.div(topic_prevalence.sum(axis=1).replace(0,1), axis=0)

# Colors for topics - distinct palette
topic_colors = ['#FF6B6B', '#FFD93D', '#6BCB77', '#4D96FF', '#9B5DE5', '#FF6EC7', '#00C2A8', '#FF8C42']
import itertools
topic_colors = list(itertools.islice(itertools.cycle(topic_colors), n_topics))

# Build stacked area chart (relative) with interactive toggle to absolute
fig_topics = go.Figure()
for t in range(n_topics):
    y_rel = topic_prevalence_frac[t] if t in topic_prevalence_frac.columns else np.zeros(len(topic_prevalence_frac.index))
    y_abs = topic_prevalence[t] if t in topic_prevalence.columns else np.zeros(len(topic_prevalence.index))
    fig_topics.add_trace(go.Scatter(
        x=topic_prevalence_frac.index,
        y=y_rel,
        mode='lines',
        stackgroup='one',
        name=f"Topic {t}",
        hoverinfo='name+x+y',
        fillcolor=topic_colors[t],
        line=dict(width=1),
        customdata=[t]*len(topic_prevalence_frac.index),
    ))

# layout and controls
fig_topics.update_layout(
    title={'text': 'Topic Prevalence Over Years — KMeans on TF-IDF (relative proportions)', 'x':0.5},
    plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG, font={'color': FONT_COLOR},
    hovermode='x unified',
    width=FIG_WIDTH, height=FIG_HEIGHT,
    margin={'t': 140, 'b': 100, 'l': 90, 'r': 120},
    legend={'orientation': 'h', 'y': LEGEND_Y}
)

# Add topic keywords annotation with spacing above controls
topic_kw_html = "<br>".join([f"<b>Topic {t}:</b> " + ", ".join(top_terms[t][:6]) for t in range(n_topics)])
fig_topics.add_annotation(
    dict(text=topic_kw_html,
         x=0.5, xanchor='center',
         y=1.20, yanchor='bottom',
         xref='paper', yref='paper',
         showarrow=False, bgcolor=PAPER_BG, bordercolor='#2b2f38', font={'color': '#CCCCCC', 'size': 12})
)

# Place controls below the keywords to avoid overlap
topic_buttons = [
    dict(label='Relative (fraction)',
         method='update',
         args=[{'y': [topic_prevalence_frac[t].values for t in range(n_topics)]},
               {'title': {'text': 'Topic Prevalence Over Years — Relative fraction (stacked area)'}}]),
    dict(label='Absolute (counts)',
         method='update',
         args=[{'y': [topic_prevalence[t].values for t in range(n_topics)]},
               {'title': {'text': 'Topic Prevalence Over Years — Absolute counts (stacked area)'}}])
]
fig_topics.update_layout(
    updatemenus=[dict(
        buttons=topic_buttons,
        direction='right',
        pad={'r': 6, 't': 6},
        showactive=True,
        x=0.5, xanchor='center',
        y=1.12, yanchor='bottom',
        bgcolor=BUTTON_BG,
        bordercolor="#3b4a4f",
        font=BUTTON_FONT
    )]
)

# -------------------------
# 3. Stylometric analysis (violins) for top authors
# -------------------------
def sentence_count(text):
    if not isinstance(text, str) or text.strip()=='':
        return 1
    s = [s for s in text.replace('\n', ' ').split('.') if s.strip()!='']
    if len(s) == 0:
        return 1
    return max(1, len(s))

df['abstract_sentence_count'] = df['Abstract'].apply(sentence_count)
df['avg_words_per_sentence'] = df['abstract_length'] / df['abstract_sentence_count'].replace(0,1)
df['avg_word_length'] = df['Abstract'].apply(lambda x: np.mean([len(w) for w in x.split()]) if len(x.split())>0 else 0)

author_count_series = df['AuthorNames-Deduped'].value_counts().head(8)
top_authors = [a for a in author_count_series.index if isinstance(a, str) and a.strip()!='']
# ensure at least 2 authors to show violin comparisons
if len(top_authors) < 2:
    top_authors = df['AuthorNames-Deduped'].dropna().unique()[:2].tolist()

stylometric_df_list = []
for author in top_authors:
    sel = df[df['AuthorNames-Deduped'].astype(str).str.contains(author, regex=False, na=False)]
    if sel.shape[0] == 0:
        continue
    stylometric_df_list.append((author, sel[['avg_words_per_sentence', 'avg_word_length', 'abstract_length', 'title_length']].copy()))

metrics = ['avg_words_per_sentence', 'avg_word_length', 'abstract_length', 'title_length']
metric_labels = {
    'avg_words_per_sentence': 'Average words per sentence',
    'avg_word_length': 'Average word length (chars)',
    'abstract_length': 'Abstract length (words)',
    'title_length': 'Title length (words)'
}
styl_colors = ['#FF6B6B', '#FFD93D', '#6BCB77', '#4D96FF', '#9B5DE5', '#FF6EC7', '#00C2A8', '#FF8C42']

if len(stylometric_df_list) == 0:
    fig_styl = go.Figure()
    fig_styl.add_annotation({'text': "Not enough author data for stylometric comparison", 'xref':'paper','yref':'paper','showarrow':False,'font':{'color':FONT_COLOR}})
    fig_styl.update_layout(plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG, width=FIG_WIDTH, height=600)
else:
    initial_metric = metrics[0]
    fig_styl = go.Figure()
    for i, (author, subdf) in enumerate(stylometric_df_list):
        # handle small subdf: duplicate single value to let Violin render
        yvals = subdf[initial_metric].values
        if len(yvals) == 1:
            # add a tiny jitter to show a narrow violin
            yvals = np.concatenate([yvals, yvals + 1e-6])
        fig_styl.add_trace(go.Violin(
            x=[author]*len(yvals),
            y=yvals,
            name=author,
            line_color=styl_colors[i % len(styl_colors)],
            fillcolor=styl_colors[i % len(styl_colors)],
            box_visible=True, meanline_visible=True, opacity=0.85, showlegend=False
        ))

    fig_styl.update_layout(
        title={'text': 'Stylometric comparison across prolific authors — metric: ' + metric_labels[initial_metric], 'x':0.5},
        plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG, font={'color': FONT_COLOR},
        width=FIG_WIDTH, height=FIG_HEIGHT, margin={'t': 140, 'b': 140, 'l': 100, 'r': 100},
        hovermode='closest', legend={'orientation':'h', 'y': LEGEND_Y}
    )

    # Prepare restyle args for all traces - create list-of-lists y for each trace
    restyle_args = []
    for metric in metrics:
        ys_for_traces = []
        for (author, subdf) in stylometric_df_list:
            yv = subdf[metric].values
            if len(yv) == 1:
                yv = np.concatenate([yv, yv + 1e-6])
            ys_for_traces.append(yv.tolist())
        restyle_args.append(ys_for_traces)

    buttons_custom = []
    for i, metric in enumerate(metrics):
        buttons_custom.append(
            dict(label=metric_labels[metric],
                 method='update',
                 args=[{'y': restyle_args[i]},
                       {'title': {'text': 'Stylometric comparison across prolific authors — metric: ' + metric_labels[metric]}}])
        )

    fig_styl.update_layout(
        updatemenus=[dict(
            buttons=buttons_custom,
            direction='right', pad={'r':6,'t':6},
            showactive=True, x=0.5, xanchor='center', y=1.10, yanchor='bottom',
            bgcolor=BUTTON_BG, bordercolor="#3b4a4f", font=BUTTON_FONT
        )]
    )

    fig_styl.add_annotation(dict(
        text="Interpretation: violin widths show density; box indicates IQR; meanline shows central tendency.",
        x=0.5, xanchor='center', y=1.18, xref='paper', yref='paper', showarrow=False, font={'color':'#CCCCCC', 'size':12}, bgcolor=PAPER_BG
    ))

# -------------------------
# 4. Replicability stamp vs Citations/Downloads (boxes + toggles & log view)
# -------------------------
df['Replicability_cat'] = df['Replicability_flag'].map({1: 'Stamped', 0: 'Not Stamped'})

# Prepare data needed and safe transforms for log visualization
cit_raw = df['CitationCount_CrossRef'].fillna(0).astype(float).values
dl_raw = df['Downloads_Xplore'].fillna(0).astype(float).values
# small positive for log transform to avoid -inf
eps = 1e-6
cit_log = np.log1p(cit_raw)
dl_log = np.log1p(dl_raw)

# Create subplots: left = citations, right = downloads
fig_rep = make_subplots(rows=1, cols=2, subplot_titles=("Citations (CrossRef)", "Downloads (Xplore)"))

# Boxplots (raw)
fig_rep.add_trace(go.Box(x=df['Replicability_cat'], y=cit_raw, marker_color='#FF6B6B', name='Citations (raw)', boxmean='sd', showlegend=False), row=1, col=1)
fig_rep.add_trace(go.Box(x=df['Replicability_cat'], y=dl_raw, marker_color='#6BCB77', name='Downloads (raw)', boxmean='sd', showlegend=False), row=1, col=2)

# Boxplots (log1p) - hidden initially
fig_rep.add_trace(go.Box(x=df['Replicability_cat'], y=cit_log, marker_color='#FFD93D', name='Citations (log1p)', boxmean='sd', visible=False, showlegend=False), row=1, col=1)
fig_rep.add_trace(go.Box(x=df['Replicability_cat'], y=dl_log, marker_color='#9B5DE5', name='Downloads (log1p)', boxmean='sd', visible=False, showlegend=False), row=1, col=2)

# Jittered points (raw) - hidden initially
def jitter(arr, jitter_amt=0.1):
    return arr + np.random.uniform(-jitter_amt, jitter_amt, size=len(arr))

cat_map = {'Stamped': 0, 'Not Stamped': 1}
x_positions = df['Replicability_cat'].map(cat_map).values
fig_rep.add_trace(go.Scatter(x=jitter(x_positions), y=cit_raw, mode='markers', marker=dict(color='#FFFFFF', size=6, opacity=0.6), name='Citations points (raw)', visible=False, showlegend=False, hovertemplate='Replicability: %{x}<br>Citations: %{y}'), row=1, col=1)
fig_rep.add_trace(go.Scatter(x=jitter(x_positions), y=dl_raw, mode='markers', marker=dict(color='#FFFFFF', size=6, opacity=0.6), name='Downloads points (raw)', visible=False, showlegend=False, hovertemplate='Replicability: %{x}<br>Downloads: %{y}'), row=1, col=2)

fig_rep.update_layout(
    title={'text': 'Replicability stamp vs Citations and Downloads', 'x':0.5},
    plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG, font={'color': FONT_COLOR},
    width=FIG_WIDTH, height=FIG_HEIGHT, margin={'t':140, 'b':120, 'l':100, 'r':120},
    hovermode='closest', legend={'orientation':'h', 'y': LEGEND_Y}
)

# Updatemenu for view selection; position below the topic keywords and above title to avoid overlap
rep_buttons = [
    dict(label='Boxplots (raw)',
         method='update',
         args=[{'visible': [True, True, False, False, False, False]},  # traces 0..5
               {'yaxis': {'title':'Citations (raw)'}, 'yaxis2': {'title':'Downloads (raw)'}, 'title': {'text':'Replicability — Boxplots (raw)'}}]),
    dict(label='Boxplots (log1p)',
         method='update',
         args=[{'visible': [False, False, True, True, False, False]},
               {'yaxis': {'title':'log1p(Citations)'}, 'yaxis2': {'title':'log1p(Downloads)'}, 'title': {'text':'Replicability — Boxplots (log1p)'}}]),
    dict(label='Show points',
         method='update',
         args=[{'visible': [False, False, False, False, True, True]},
               {'title': {'text':'Replicability — Jittered points (raw)'}}]),
]
fig_rep.update_layout(
    updatemenus=[dict(buttons=rep_buttons, direction='right', pad={'r':6,'t':6}, showactive=True,
                     x=0.5, xanchor='center', y=1.06, yanchor='bottom',
                     bgcolor=BUTTON_BG, bordercolor="#3b4a4f", font=BUTTON_FONT)]
)

fig_rep.add_annotation(dict(text="Toggle between raw boxplots, log1p boxplots (to reduce outlier effects), or view individual points.",
                            x=0.5, xanchor='center', y=1.14, xref='paper', yref='paper', showarrow=False, font={'color':'#CCCCCC', 'size':12}, bgcolor=PAPER_BG))

# -------------------------
# 5. Award prediction: logistic regression + ROC/PR + top numeric coefficients
# -------------------------
# Prepare features: numeric + TF-IDF(title)
award_df = df.copy()
num_features = ['page_count', 'num_authors', 'title_length', 'abstract_length',
                'Downloads_Xplore', 'CitationCount_CrossRef', 'AminerCitationCount', 'citations_per_year']
X_num = award_df[num_features].fillna(0).values
scaler = StandardScaler()
X_num_scaled = scaler.fit_transform(X_num)

vect_title = TfidfVectorizer(max_features=800, ngram_range=(1,2), stop_words='english')
X_title = vect_title.fit_transform(award_df['Title'].fillna(''))

if sparse.issparse(X_title):
    X = sparse.hstack([sparse.csr_matrix(X_num_scaled), X_title]).tocsr()
else:
    X = np.hstack([X_num_scaled, X_title.toarray()])

y = award_df['Award_flag'].values

if y.sum() == 0 or len(np.unique(y)) == 1:
    fig_award = go.Figure()
    fig_award.add_annotation({'text': "Not enough labeled award examples to train a classifier. Award_flag has zero or one unique label.",
                              'xref':'paper','yref':'paper','showarrow':False,'font':{'size':14,'color':FONT_COLOR}})
    fig_award.update_layout(title={'text':'Award prediction — insufficient positive labels', 'x':0.5},
                            plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG, font={'color':FONT_COLOR},
                            width=FIG_WIDTH, height=560)
else:
    clf = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42, solver='liblinear')
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # cross-validated probabilities
    y_proba = cross_val_predict(clf, X, y, cv=cv, method='predict_proba')[:, 1]
    clf.fit(X, y)
    fpr, tpr, _ = roc_curve(y, y_proba)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y, y_proba)
    pr_auc = auc(recall, precision)

    coef = clf.coef_.ravel()
    coef_num = coef[:len(num_features)]
    fi_df = pd.DataFrame({'feature': num_features, 'coef': coef_num})
    fi_df['abscoef'] = fi_df['coef'].abs()
    fi_df = fi_df.sort_values('abscoef', ascending=False).head(12)

    fig_award = make_subplots(rows=1, cols=3, subplot_titles=("ROC Curve", "Precision-Recall Curve", "Top numeric coefficients"), column_widths=[0.33,0.33,0.34])

    # ROC traces
    fig_award.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', line=dict(color='#FF6B6B', width=3), name=f'ROC (AUC={roc_auc:.3f})', hoverinfo='x+y+name'), row=1, col=1)
    fig_award.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', line=dict(color='#444444', dash='dash'), showlegend=False), row=1, col=1)

    # PR trace
    fig_award.add_trace(go.Scatter(x=recall, y=precision, mode='lines', line=dict(color='#6BCB77', width=3), name=f'PR (AUC={pr_auc:.3f})'), row=1, col=2)

    # Feature importance bar
    fig_award.add_trace(go.Bar(x=fi_df['coef'], y=fi_df['feature'], orientation='h', marker_color='#4D96FF', showlegend=False), row=1, col=3)

    fig_award.update_layout(
        title={'text': 'Award prediction — cross-validated ROC & Precision-Recall; numeric coefficients', 'x':0.5},
        plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG, font={'color':FONT_COLOR},
        width=FIG_WIDTH, height=FIG_HEIGHT, margin={'t':140, 'b':120}
    )

    fig_award.update_xaxes(title_text="False positive rate", row=1, col=1)
    fig_award.update_yaxes(title_text="True positive rate", row=1, col=1)
    fig_award.update_xaxes(title_text="Recall", row=1, col=2)
    fig_award.update_yaxes(title_text="Precision", row=1, col=2)
    fig_award.update_xaxes(title_text="Coefficient value", row=1, col=3)

    # Updatemenu to toggle ROC/PR visibility; compute number of traces and their indices
    # Trace layout: [ROC_line (0), ROC_diag (1), PR_line (2), Bar (3)]
    fig_award.update_layout(
        updatemenus=[dict(
            buttons=[
                dict(label='Show ROC & features',
                     method='update',
                     args=[{'visible': [True, True, False, True]},
                           {'title': {'text':'Award prediction — ROC & Top features'}}]),
                dict(label='Show PR & features',
                     method='update',
                     args=[{'visible': [False, False, True, True]},
                           {'title': {'text':'Award prediction — Precision-Recall & Top features'}}])
            ],
            direction='right', pad={'r':6,'t':6}, showactive=True,
            x=0.5, xanchor='center', y=1.06, yanchor='bottom',
            bgcolor=BUTTON_BG, bordercolor="#3b4a4f", font=BUTTON_FONT
        )]
    )

    fig_award.add_annotation(dict(text="ROC and PR curves from stratified CV. Bars show numeric feature direction and magnitude.",
                                  x=0.5, xanchor='center', y=1.14, xref='paper', yref='paper', showarrow=False, font={'color':'#CCCCCC', 'size':12}, bgcolor=PAPER_BG))

# -------------------------
# 6. Duplicate Title Similarity heatmap (TF-IDF on titles)
# -------------------------
N = 200
title_counts = df['Title'].astype(str).value_counts()
if len(title_counts) == 0:
    fig_sim = go.Figure()
    fig_sim.add_annotation({'text': 'No titles available for similarity heatmap.', 'xref':'paper','yref':'paper','showarrow':False,'font':{'color':FONT_COLOR}})
    fig_sim.update_layout(plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG, width=FIG_WIDTH, height=600)
    sample_titles = []
else:
    sample_titles = title_counts.head(N).index.tolist()
    # Ensure a minimum of 2 titles
    if len(sample_titles) < 2:
        unique_titles = df['Title'].dropna().unique().tolist()
        sample_titles = unique_titles[:min(len(unique_titles), N)]

    titles_df = pd.DataFrame({'Title': sample_titles})
    vect_t = TfidfVectorizer(max_features=2000, stop_words='english')
    T_title = vect_t.fit_transform(titles_df['Title'])
    sim = cosine_similarity(T_title)
    np.fill_diagonal(sim, 0.0)

    # Truncate labels for axis ticks to avoid overlap, but keep mapping for hover via annotation/table
    def short_label(s, n=30):
        s = s.strip()
        return (s[:n-3] + '...') if len(s) > n else s
    ticklabels = [short_label(t, 28) for t in sample_titles]

    # Build heatmap
    hovertext = []
    for i in range(len(sample_titles)):
        row_ht = []
        for j in range(len(sample_titles)):
            row_ht.append(f"<b>i</b>: {i}<br><b>j</b>: {j}<br><b>sim</b>: {sim[i,j]:.3f}<br><b>Title i</b>: {sample_titles[i]}<br><b>Title j</b>: {sample_titles[j]}")
        hovertext.append(row_ht)

    fig_sim = go.Figure(data=go.Heatmap(
        z=sim,
        x=list(range(len(sample_titles))),
        y=list(range(len(sample_titles))),
        text=hovertext,
        hoverinfo='text',
        colorscale='Turbo',
        colorbar=dict(title="Cosine sim", lenmode='fraction', len=0.6, y=0.5),
    ))

    fig_sim.update_layout(
        title={'text': f'Title similarity heatmap (cosine) — top {len(sample_titles)} titles', 'x':0.5},
        plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG, font={'color': FONT_COLOR},
        width=FIG_WIDTH, height=FIG_HEIGHT+80, margin={'t':160,'b':120,'l':120,'r':120}
    )

    # set tick labels on axes and rotate to avoid overlap
    fig_sim.update_xaxes(tickmode='array', tickvals=list(range(len(ticklabels))), ticktext=ticklabels, tickangle=45)
    fig_sim.update_yaxes(tickmode='array', tickvals=list(range(len(ticklabels))), ticktext=ticklabels, autorange='reversed')

    # Build top pairs table HTML
    pairs = []
    n_items = sim.shape[0]
    for i in range(n_items):
        for j in range(i+1, n_items):
            pairs.append((i, j, sim[i, j], sample_titles[i], sample_titles[j]))
    pairs_sorted = sorted(pairs, key=lambda x: x[2], reverse=True)
    top_pairs = pairs_sorted[:60]

    pairs_html_rows = ""
    for (i, j, score, t1, t2) in top_pairs:
        t1_esc = str(t1).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        t2_esc = str(t2).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        pairs_html_rows += f"<tr style='border-bottom:1px solid #2b2f38;'><td style='padding:6px; color:#ddd;'> {i} </td><td style='padding:6px; color:#ddd;'> {j} </td><td style='padding:6px; color:#ddd;'> {score:.3f} </td><td style='padding:6px; color:#ddd;'> {t1_esc} </td><td style='padding:6px; color:#ddd;'> {t2_esc} </td></tr>\n"

    pairs_table_html = f"""
    <div style="max-height:320px; overflow:auto; border:1px solid #2b2f38; padding:10px; background:{PAPER_BG};">
    <table style="width:100%; color:#ffffff; border-collapse: collapse;">
    <thead><tr style="border-bottom:1px solid #444;"><th style='text-align:left; padding:6px;'>i</th><th style='text-align:left; padding:6px;'>j</th><th style='text-align:left; padding:6px;'>similarity</th><th style='text-align:left; padding:6px;'>Title i</th><th style='text-align:left; padding:6px;'>Title j</th></tr></thead>
    <tbody>{pairs_html_rows}</tbody></table></div>
    """

# -------------------------
# 7. Assemble HTML with all figures embedded (include_plotlyjs='cdn' for each)
#    — Added educational narratives: introduction, per-figure explanations, and conclusion.
# -------------------------
html_parts = []

page_style = f"""
<style>
body {{ background-color: {PAPER_BG}; color: {FONT_COLOR}; font-family: Arial, sans-serif; margin: 24px; }}
.section {{ padding: 28px 0; border-bottom: 1px solid #1f2430; margin-bottom: 20px; }}
.section h1 {{ color: #ffffff; font-size: 28px; margin-bottom: 8px; }}
.section h2 {{ color: #cfe5ff; font-size: 20px; margin-top: 10px; }}
.caption {{ color: #bfcbdc; margin-bottom: 12px; font-size: 15px; }}
.plot-container {{ padding: 10px 0 40px 0; }}
.table-scroll {{ max-height: 320px; overflow:auto; border: 1px solid #2b2f38; padding:10px; background:{PAPER_BG}; }}
a {{ color: #4D96FF; }}
.upnote {{ color: #BEE3DB; font-size: 13px; margin-top:8px; }}
.button-note {{ color: #9fb2d0; font-size: 13px; margin-top:6px; }}
.report-paragraph {{ font-size: 16px; line-height:1.5; color: #d8e6ff; }}
.report-bold {{ font-size: 16px; line-height:1.5; color: #ffffff; font-weight: bold; }}
</style>
"""

# Introduction section: purpose and flow
intro_html = f"""
<div style="padding: 12px 0 12px 0;">
  <h1>Exploratory Report: dataset.csv — topics, stylometrics, replicability, awards, and duplicates</h1>
  <p class="report-paragraph">
    This interactive report demonstrates several practical data science analyses applied to a scientific publications dataset.
    You will see a sequence of visualizations that together explore: how topics evolve over time, textual stylistic differences across prolific authors,
    the relationship between reported replicability stamps and impact (citations & downloads), a simple award prediction model evaluation, and a duplicate-title detection heatmap.
    Each figure is accompanied by an accessible, educational explanation of the algorithm used, why it was chosen, how to read the plot, and what an important finding would look like.
  </p>
  <p class="report-paragraph">
    The goal is to provide an approachable, reproducible dashboard that both explains the underlying methods and surfaces actionable insights.
    Use the controls above each plot to switch views (for example, absolute vs relative topic prevalence or raw vs log-scale comparisons) and hover to reveal details.
  </p>
</div>
"""
html_parts.append(page_style + intro_html)

def append_figure_section(title, subtitle, fig, narrative_html):
    fig_html = pio.to_html(fig, include_plotlyjs='cdn', full_html=False)
    section = f"""
    <div class="section">
      <h2>{title}</h2>
      <div class="caption">{subtitle}</div>
      <div class="plot-container">{fig_html}</div>
      <div>{narrative_html}</div>
    </div>
    """
    html_parts.append(section)

# Narrative for Topic Prevalence
topic_narrative = f"""
  <p class="report-paragraph">
    This figure uses TF-IDF to convert each paper's title, abstract, and keywords into a vector representation and then applies truncated SVD
    for compactness before clustering with KMeans to form topics. KMeans is used because it is fast, deterministic (with fixed random seed),
    and yields easily interpretable clusters for exploration. This pipeline (TF-IDF → SVD → KMeans) is a common lightweight approach to topic discovery when you need
    quick, scalable groupings without heavy language model compute.
  </p>
  <p class="report-paragraph">
    The stacked area plot shows the <em>relative prevalence</em> of each discovered topic per year (you can toggle to absolute counts).
    - X axis: Year. Y axis: fraction of papers in each topic (stacked to 1). Each color is a topic; the linked keywords above the plot summarize what terms drive that topic.
    - Interpretation: a rising colored band means that topic is becoming more common over time. A shrinking band means declining interest.
    - Impactful findings: sustained growth of a topic's relative share across several years suggests an emerging research area. Sudden spikes may indicate one-time workshops or trend effects.
  </p>
  <p class="report-paragraph report-bold">
    Main takeaway: <strong>If a topic shows consistent upward trend across years, it signals an emerging or rapidly growing research direction; declining or flat topics indicate stable or waning interest.</strong>
  </p>
  <p class="report-paragraph">
    From here, researchers could drill down by filtering papers within a topic to inspect representative abstracts or use more advanced dynamic topic models (e.g., BERTopic) to capture topic drift more precisely.
  </p>
"""

append_figure_section(
    "Dynamic Topic Prevalence Over Time",
    "KMeans topics (n=8) derived from TF-IDF on Title+Abstract+Keywords. Toggle between relative prevalence and absolute counts. Topic keywords listed above the control.",
    fig_topics,
    topic_narrative
)

# Narrative for Stylometric comparison
styl_narrative = f"""
  <p class="report-paragraph">
    This comparison extracts simple stylometric features from paper abstracts (average words per sentence, average word length, abstract length, title length).
    Violins visualize the distribution of each metric for the most prolific authors. We select these features because they are intuitive, computationally cheap,
    and effective at capturing broad writing-style differences. Violin plots are selected to display full distributional shape rather than just summary statistics.
  </p>
  <p class="report-paragraph">
    How to read: each violin represents the distribution of the chosen metric for an author: wide regions indicate many papers with that value, narrow regions fewer.
    The inner box shows the interquartile range and the mean line indicates the average. This helps detect authors with consistently shorter abstracts, longer sentences, or unusual word-length patterns.
    Significant findings would include an author exhibiting consistently shorter abstracts or a markedly different average words-per-sentence compared to peers, which might reflect a distinct writing style or editorial constraints.
  </p>
  <p class="report-paragraph report-bold">
    Main takeaway: <strong>Consistent differences in stylometric features across authors can reveal distinct writing conventions or editorial norms; large deviations may warrant further investigation (e.g., author-specific practices or potential copy-paste patterns).</strong>
  </p>
  <p class="report-paragraph">
    Next steps might include training a stylometric classifier to attribute anonymous text to likely authors or to highlight anomalous submissions for manual review.
  </p>
"""

append_figure_section(
    "Stylometric Comparison Across Prolific Authors",
    "Violin plots comparing text-level features (avg words per sentence, avg word length, abstract length, title length). Choose metric via the buttons above the chart.",
    fig_styl,
    styl_narrative
)

# Narrative for Replicability vs Impact
rep_narrative = f"""
  <p class="report-paragraph">
    This section compares papers with and without a reported graphics replicability stamp across two impact measures: CrossRef citations and IEEE Xplore downloads.
    Because raw citation and download counts are often heavy-tailed, the visualization offers both raw and log1p views and a jittered point view to inspect individual observations.
    We use boxplots for distributional comparisons (median, IQR, whiskers) and jittered points to reveal outliers.
  </p>
  <p class="report-paragraph">
    How to read: each panel shows the distribution for stamped vs not-stamped papers.
    - If the 'Stamped' group has a visibly higher median box and/or a generally higher distribution in both citations and downloads, that suggests an association between reported replicability practices and impact.
    - However, causality is not established here — confounding factors (conference, year, topic) can influence both replicability reporting and impact.
    Important signals: consistently higher medians and shifted distributions for stamped papers across both metrics would be noteworthy.
  </p>
  <p class="report-paragraph report-bold">
    Main takeaway: <strong>If stamped papers systematically show higher citations and downloads, this suggests replicable research may correlate with higher scholarly impact — but follow-up causal analysis is needed to confirm causation.</strong>
  </p>
  <p class="report-paragraph">
    Next steps could include causal inference (propensity score matching or difference-in-differences) controlling for confounders like conference and year to estimate the replicability stamp's effect on impact.
  </p>
"""

append_figure_section(
    "Graphics Replicability Stamp vs Citations & Downloads",
    "Compare papers with a replicability stamp vs those without. Toggle between raw boxplots, log1p boxplots (reduces outlier impact), or jittered points.",
    fig_rep,
    rep_narrative
)

# Narrative for Award prediction
award_narrative = f"""
  <p class="report-paragraph">
    We build a simple, interpretable baseline model to predict whether a paper received an award using numeric metadata (e.g., page count, author count, citations, downloads) and TF-IDF features from the title.
    Logistic regression with balanced class weights is chosen because it provides well-calibrated probabilities and coefficients that are easy to interpret.
    Given the rarity of awards, we emphasize cross-validated evaluation (ROC and Precision-Recall) to account for class imbalance.
  </p>
  <p class="report-paragraph">
    How to read: the left panel shows ROC curve (trade-off between true positive and false positive rates) and its AUC.
    The middle panel is the Precision-Recall curve which is more informative under class imbalance; a higher area indicates better precision at high recall values.
    The right panel shows the most influential numeric features (by coefficient magnitude) and the sign indicates whether a higher value increases or decreases award probability.
    Impactful findings would include a model with strong PR-AUC (much higher than random) and interpretable features consistent with domain knowledge (for example, higher early downloads predicting awards).
  </p>
  <p class="report-paragraph report-bold">
    Main takeaway: <strong>Interpretable models can surface which observable features are associated with awards; however, due to label sparsity and temporal changes, this is a hypothesis-generating step rather than conclusive prediction.</strong>
  </p>
  <p class="report-paragraph">
    Recommended next steps: add richer text embeddings, perform temporal holdout validation, and use SHAP to explain predictions at the paper level.
  </p>
"""

append_figure_section(
    "Award Prediction — Model Evaluation",
    "Logistic regression trained with numeric features + TF-IDF(title). Displays cross-validated ROC and Precision-Recall curves; numeric coefficients for interpretability.",
    fig_award,
    award_narrative
)

# Narrative for Similarity heatmap
if 'fig_sim' in locals() and isinstance(fig_sim, go.Figure):
    sim_narrative = f"""
      <p class="report-paragraph">
        The title similarity heatmap computes TF-IDF vectors for paper titles and then the cosine similarity between them.
        High similarity values (hotter colors) indicate near-duplicate or highly related titles. This simple approach is fast and useful for identifying potential duplicate submissions,
        obvious retitles of the same work, or clustering alike contributions.
      </p>
      <p class="report-paragraph">
        How to read: each heatmap cell corresponds to the similarity between Title i (row) and Title j (column). Hover to see the full titles and the similarity score.
        The accompanying table lists the top similar pairs sorted by similarity for manual review.
        Significant findings include many high-similarity pairs (for example, >0.9) indicating potential duplicates or multiple submissions of the same content.
      </p>
      <p class="report-paragraph report-bold">
        Main takeaway: <strong>Hot, dense blocks or many high-similarity pairs suggest duplicated or closely related titles — these pairs merit manual inspection for integrity checks or dataset cleaning.</strong>
      </p>
      <p class="report-paragraph">
        Next steps: apply full-text similarity checks for flagged pairs, inspect DOIs and links, and consider deduplication or consolidation for downstream analyses.
      </p>
    """
    append_figure_section(
        "Title Similarity Heatmap — Duplicate / Near-duplicate Detection",
        f"Cosine similarity on TF-IDF of top {len(sample_titles)} titles. Hotter colors indicate high similarity. Hover a cell to see the full titles. Table below shows top similar pairs.",
        fig_sim,
        sim_narrative
    )

# Conclusion
conclusion_html = f"""
<div style="padding-top:12px;">
  <h2>Conclusion & Next Steps</h2>
  <p class="report-paragraph">
    This report demonstrated a compact set of analyses that combine simple, interpretable algorithms with interactive visualizations to surface meaningful patterns in a publication dataset.
    We used TF-IDF + SVD + KMeans for quick topic discovery, stylometric summaries and violin plots to compare authors, distributional comparisons for replicability and impact,
    a baseline interpretable classification model for award prediction, and TF-IDF similarity for duplicate detection.
  </p>
  <p class="report-paragraph">
    Each visualization is intended to be actionable: trending topics point to emerging research areas; stylometric outliers or duplicate titles can trigger data-quality or integrity workflows;
    replicability correlations suggest avenues for causal analysis; and the award model provides candidate features for deeper modeling.
  </p>
  <p class="report-paragraph report-bold">
    Final takeaway: <strong>Combining lightweight, explainable algorithms with careful visual exploration provides powerful first-pass insights. The plots here should be followed by targeted, rigorous analyses (temporal validation, causal inference, and human-in-the-loop review) to confirm findings.</strong>
  </p>
  <p class="report-paragraph">
    If you would like, I can provide runnable starter notebooks for any of the follow-up analyses mentioned (dynamic topic models, SHAP explanations, causal inference pipelines, or full-text duplicate checks).
  </p>
</div>
"""
html_parts.append(conclusion_html)

# Footer metadata
footer_html = f"""
<div style="padding-top:18px; color:#9fb2d0;">
  <div>Generated by script. Source file: <code>dataset.csv</code></div>
  <div style="margin-top:6px;">Open <code>output.html</code> in a browser. Each plot embeds Plotly JS from CDN for compatibility.</div>
</div>
"""
html_parts.append(footer_html)

# Combine and write to output.html
full_html = "<html><head><meta charset='utf-8'><title>Dataset Visualizations — Educational Report</title></head><body>" + "\n".join(html_parts) + "</body></html>"
with open("output.html", "w", encoding="utf-8") as f:
    f.write(full_html)

print("Wrote output.html — open it in a browser to view the interactive visualizations and educational report.")

