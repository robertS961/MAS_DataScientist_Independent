# ---- NEW BLOCK ---- # 
#CODE HERE
# Enhanced interactive Plotly visualizations for dataset.csv
# - Dark theme, improved interactivity, clearer axes and ranges
# - Robust handling of NaNs, duplicates, and older/newer sklearn API
# - Exports a single output.html with narrative descriptions before each figure
# Run this script in the same folder as dataset.csv

import pandas as pd
import numpy as np
import math

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio

# ML imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_curve, auc, r2_score

# Set default template to dark theme
pio.templates.default = "plotly_dark"

# ---------------------------
# Load & Preprocessing
# ---------------------------
df = pd.read_csv("dataset.csv")

# Basic defensive checks
if df.shape[0] == 0:
    raise RuntimeError("dataset.csv appears empty.")

# Fill numeric NaNs with median (robust to outliers)
num_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
for col in num_cols:
    median_val = df[col].median()
    df[col] = df[col].fillna(median_val)

# Fill text/categorical NaNs
text_defaults = {
    "Award": "NoAward",
    "GraphicsReplicabilityStamp": "NoStamp",
    "InternalReferences": "",
    "AuthorKeywords": "",
    "AuthorAffiliation": "",
}
for col, default in text_defaults.items():
    if col in df.columns:
        df[col] = df[col].fillna(default)
    else:
        df[col] = default

# Ensure AuthorNames-Deduped exists
if "AuthorNames-Deduped" not in df.columns:
    df["AuthorNames-Deduped"] = df.get("AuthorNames", "").fillna("")

# Ensure Year is int
df["Year"] = df["Year"].astype(int)

# Remove exact duplicate rows (defensive)
df = df.drop_duplicates()

# Feature engineering
def count_internal_refs(s):
    if not isinstance(s, str) or s.strip() == "":
        return 0
    # handle different separators
    parts = [p.strip() for p in s.replace(";", ",").split(",") if p.strip()]
    return len(parts)

def count_keywords(s):
    if not isinstance(s, str) or s.strip() == "":
        return 0
    parts = [p.strip() for p in s.replace(";", ",").split(",") if p.strip()]
    return len(parts)

df["internal_refs_count"] = df["InternalReferences"].apply(count_internal_refs)
df["keyword_count"] = df["AuthorKeywords"].apply(count_keywords)

# Short helper for robust OneHotEncoder creation (compat for sklearn versions)
def safe_ohe(**kwargs):
    try:
        # sklearn >=1.2 uses sparse_output
        return OneHotEncoder(**{**kwargs, "sparse_output": False})
    except TypeError:
        # older sklearn fallback
        params = kwargs.copy()
        params.pop("sparse_output", None)
        return OneHotEncoder(**{**params, "sparse": False})

# ---------------------------
# 1) Enhanced Trend Analysis on Citation Counts
# ---------------------------
# Narrative (used later in HTML): We visualize how citations evolve over time for the top conferences.
# This helps identify venues that historically contributed more to paper impact.

# Choose top conferences by count (limit to top 5 for clarity)
top_confs = df["Conference"].value_counts().nlargest(5).index.tolist()
df_trend_base = df[df["Conference"].isin(top_confs)].copy()

# Aggregate mean by Year & Conference and handle potential duplicates by grouping
df_trend = (
    df_trend_base.groupby(["Year", "Conference"], as_index=False)[
        ["AminerCitationCount", "CitationCount_CrossRef"]
    ]
    .mean()
    .melt(id_vars=["Year", "Conference"], value_vars=["AminerCitationCount", "CitationCount_CrossRef"],
          var_name="Metric", value_name="Count")
)

# If any negative or NaN counts appear, coerce
df_trend["Count"] = pd.to_numeric(df_trend["Count"], errors="coerce").fillna(0)
min_year = int(df_trend["Year"].min())
max_year = int(df_trend["Year"].max())
ymax = max(1, df_trend["Count"].max())
y_range = [0, math.ceil(ymax * 1.12)]

# Build a multi-trace figure so we can provide a dropdown to toggle metric
metrics = df_trend["Metric"].unique().tolist()
fig1 = go.Figure()
color_seq = px.colors.qualitative.Alphabet  # distinct colors

# Add traces: each (conference, metric) pair
trace_idx = 0
for metric in metrics:
    subset_metric = df_trend[df_trend["Metric"] == metric]
    for i, conf in enumerate(top_confs):
        s = subset_metric[subset_metric["Conference"] == conf].sort_values("Year")
        # Add as a trace (initially only the first metric visible)
        visible = True if metric == metrics[0] else False
        fig1.add_trace(
            go.Scatter(
                x=s["Year"],
                y=s["Count"],
                name=f"{conf} — {metric}",
                mode="lines+markers",
                marker=dict(size=6),
                line=dict(width=2, color=color_seq[i % len(color_seq)]),
                visible=visible,
                hovertemplate="<b>%{text}</b><br>Year: %{x}<br>Count: %{y:.1f}",
                text=[f"{conf} — {metric}"] * s.shape[0],
            )
        )
        trace_idx += 1

# Dropdown buttons to toggle metric
buttons = []
n_confs = len(top_confs)
for m_i, metric in enumerate(metrics):
    visible_list = []
    for m_j in range(len(metrics)):
        # For all traces: visible if this trace belongs to chosen metric index
        visible_list.extend([m_j == m_i] * n_confs)
    buttons.append(
        dict(
            label=metric.replace("_", " "),
            method="update",
            args=[
                {"visible": visible_list},
                {
                    "title": f"Citation Trends by Conference ({metric})",
                    "yaxis": {"range": y_range},
                    "xaxis": {"range": [min_year - 0.5, max_year + 0.5], "dtick": 1},
                },
            ],
        )
    )

# Add a button to show both metrics (makes all traces visible)
buttons.append(
    dict(
        label="Both Metrics",
        method="update",
        args=[
            {"visible": [True] * (len(metrics) * n_confs)},
            {
                "title": "Citation Trends by Conference (Both Metrics)",
                "yaxis": {"range": y_range},
                "xaxis": {"range": [min_year - 0.5, max_year + 0.5], "dtick": 1},
            },
        ],
    )
)

fig1.update_layout(
    title=f"Citation Trends by Conference ({metrics[0]})",
    updatemenus=[
        dict(active=0, buttons=buttons, x=0.0, y=1.12, xanchor="left", bgcolor="#222", bordercolor="#444")
    ],
    xaxis=dict(title="Year", range=[min_year - 0.5, max_year + 0.5], dtick=1),
    yaxis=dict(title="Average Citation Count", range=y_range),
    legend=dict(title="Conference — Metric", orientation="h", yanchor="bottom", y=-0.25),
    hovermode="closest",
)

# ---------------------------
# 2) Machine Learning: Predict Awards (Classification)
# ---------------------------
# Narrative: We build a logistic regression model to estimate the probability that a paper wins an award.
# Features: PaperType, Conference, keyword_count, internal_refs_count, Downloads_Xplore, Year.
# We present ROC curve and top features (coefficients) so users understand predictive drivers.

df_ml = df.copy()
df_ml["AwardBinary"] = (df_ml["Award"] != "NoAward").astype(int)

# Select features
X = df_ml[["PaperType", "Conference", "keyword_count", "internal_refs_count", "Downloads_Xplore", "Year"]].copy()
y = df_ml["AwardBinary"].copy()

# If there are no positive examples, skip ML gracefully
if y.sum() < 2:
    raise RuntimeError("Not enough award examples to train a classifier.")

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

cat_feats = ["PaperType", "Conference"]
num_feats = ["keyword_count", "internal_refs_count", "Downloads_Xplore", "Year"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", safe_ohe(drop="if_binary", handle_unknown="ignore"), cat_feats),
        ("num", StandardScaler(), num_feats),
    ],
    remainder="drop",
)

clf_pipeline = Pipeline(
    [
        ("prep", preprocessor),
        ("model", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)),
    ]
)

clf_pipeline.fit(X_train, y_train)
y_proba = clf_pipeline.predict_proba(X_test)[:, 1]

fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

# ROC figure
fig2 = go.Figure()
fig2.add_trace(
    go.Scatter(
        x=fpr,
        y=tpr,
        mode="lines",
        name=f"ROC (AUC={roc_auc:.3f})",
        line=dict(width=3, color="#ff7f0e"),
        hovertemplate="FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>",
    )
)
fig2.add_trace(
    go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        name="Random",
        line=dict(width=2, color="#888", dash="dash"),
        hoverinfo="skip",
    )
)
fig2.update_layout(
    title="ROC Curve for Award Prediction (Logistic Regression)",
    xaxis=dict(title="False Positive Rate", range=[0, 1]),
    yaxis=dict(title="True Positive Rate", range=[0, 1]),
    legend=dict(orientation="h", y=-0.2),
    width=700,
    height=500,
)

# Feature importance from logistic regression coefficients
# Extract OHE feature names robustly
ohe = clf_pipeline.named_steps["prep"].named_transformers_["cat"]
try:
    ohe_names = ohe.get_feature_names_out(cat_feats).tolist()
except Exception:
    # older sklearn fallback
    ohe_names = []
    for idx, feat in enumerate(cat_feats):
        cats = ohe.categories_[idx]
        ohe_names += [f"{feat}_{str(cat)}" for cat in cats]

feature_names = ohe_names + num_feats
coefs = clf_pipeline.named_steps["model"].coef_[0]
feat_imp = pd.DataFrame({"feature": feature_names, "coef": coefs})
feat_imp["abs_coef"] = feat_imp["coef"].abs()
feat_imp = feat_imp.sort_values("abs_coef", ascending=False).head(15)

fig2_imp = px.bar(
    feat_imp[::-1],
    x="coef",
    y="feature",
    orientation="h",
    color="coef",
    color_continuous_scale=px.colors.diverging.RdYlBu,
    title="Top Features (by absolute coefficient) for Award Prediction",
    labels={"coef": "Coefficient", "feature": "Feature"},
)
fig2_imp.update_layout(width=800, height=420)

# ---------------------------
# 3) Regression Analysis: Predict Downloads
# ---------------------------
# Narrative: Linear regression to predict Downloads_Xplore. We show Predicted vs Actual + Residuals to judge fit.

df_reg = df.copy()
df_reg["stamp_flag"] = (df_reg["GraphicsReplicabilityStamp"] != "NoStamp").astype(int)

Xr = df_reg[["AminerCitationCount", "internal_refs_count", "stamp_flag", "Year", "Conference"]].copy()
yr = df_reg["Downloads_Xplore"].copy()

# Use a small subset if very large, but here dataset ~4k rows (ok)
Xr_train, Xr_test, yr_train, yr_test = train_test_split(Xr, yr, test_size=0.3, random_state=42)

preproc_reg = ColumnTransformer(
    transformers=[
        ("cat", safe_ohe(drop="if_binary", handle_unknown="ignore"), ["Conference"]),
        ("num", StandardScaler(), ["AminerCitationCount", "internal_refs_count", "stamp_flag", "Year"]),
    ],
    remainder="drop",
)

reg_pipeline = Pipeline([("prep", preproc_reg), ("model", LinearRegression())])
reg_pipeline.fit(Xr_train, yr_train)
yr_pred = reg_pipeline.predict(Xr_test)

r2 = r2_score(yr_test, yr_pred)

# Pred vs Actual + Residuals subplot
fig3 = make_subplots(rows=1, cols=2, subplot_titles=(f"Predicted vs Actual (R2={r2:.3f})", "Residuals vs Predicted"))

# Scatter Predicted vs Actual
fig3.add_trace(
    go.Scatter(
        x=yr_test,
        y=yr_pred,
        mode="markers",
        marker=dict(color="#17becf", size=6),
        hovertemplate="Actual: %{x:.1f}<br>Predicted: %{y:.1f}<extra></extra>",
        name="Pred vs Actual",
    ),
    row=1,
    col=1,
)
# Identity line
min_val = float(min(yr_test.min(), yr_pred.min()))
max_val = float(max(yr_test.max(), yr_pred.max()))
fig3.add_trace(
    go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode="lines",
        line=dict(color="#888", dash="dash"),
        showlegend=False,
    ),
    row=1,
    col=1,
)

# Residuals
residuals = yr_test - yr_pred
fig3.add_trace(
    go.Scatter(
        x=yr_pred,
        y=residuals,
        mode="markers",
        marker=dict(color="#9467bd", size=6),
        hovertemplate="Predicted: %{x:.1f}<br>Residual: %{y:.1f}<extra></extra>",
        name="Residuals",
    ),
    row=1,
    col=2,
)

fig3.update_xaxes(title_text="Actual Downloads", row=1, col=1)
fig3.update_yaxes(title_text="Predicted Downloads", row=1, col=1)
fig3.update_xaxes(title_text="Predicted Downloads", row=1, col=2)
fig3.update_yaxes(title_text="Residuals", row=1, col=2)
fig3.update_layout(title_text="Download Prediction Analysis (Linear Regression)", width=1000, height=450)

# ---------------------------
# 4) Keyword Trend Analysis
# ---------------------------
# Narrative: Top keywords by frequency and how they change over time (top 10 keywords).
kw_df = df[["Year", "AuthorKeywords"]].copy()
kw_df["AuthorKeywords"] = kw_df["AuthorKeywords"].astype(str).str.replace(";", ",", regex=False)
kw_df = kw_df.assign(keyword=kw_df["AuthorKeywords"].str.split(",")).explode("keyword")
kw_df["keyword"] = kw_df["keyword"].astype(str).str.strip().str.lower()
kw_df = kw_df[kw_df["keyword"] != ""]
kw_counts = kw_df.groupby("keyword").size().reset_index(name="total_counts")
top_keywords = kw_counts.nlargest(10, "total_counts")["keyword"].tolist()

kw_trend = kw_df[kw_df["keyword"].isin(top_keywords)].groupby(["Year", "keyword"]).size().reset_index(name="count")
kw_trend = kw_trend.sort_values(["keyword", "Year"])

# Ensure year axis includes range and slider
years_sorted = sorted(kw_trend["Year"].unique())
ymax_kw = max(1, kw_trend["count"].max())

fig4 = px.line(
    kw_trend,
    x="Year",
    y="count",
    color="keyword",
    markers=True,
    title="Top 10 Keywords Trends Over Time",
    color_discrete_sequence=px.colors.qualitative.Plotly,
)
fig4.update_layout(
    xaxis=dict(range=[min(years_sorted) - 0.5, max(years_sorted) + 0.5], dtick=1, rangeslider=dict(visible=True)),
    yaxis=dict(range=[0, math.ceil(ymax_kw * 1.12)], title="Keyword Count"),
    legend=dict(title="Keyword"),
    width=1000,
    height=520,
)

# ---------------------------
# 5) Award Impact (Boxplots): Citations & Downloads by Award Status
# ---------------------------
df_aw = df.copy()
df_aw["AwardFlag"] = np.where(df_aw["Award"] != "NoAward", "Awarded", "Not Awarded")

# Prepare axis ranges to avoid blank-looking plots
cit_max = max(1, df_aw["CitationCount_CrossRef"].max())
dl_max = max(1, df_aw["Downloads_Xplore"].max())

fig5 = make_subplots(rows=1, cols=2, subplot_titles=("CitationCount_CrossRef by Award", "Downloads_Xplore by Award"))
fig5.add_trace(
    go.Box(
        x=df_aw["AwardFlag"],
        y=df_aw["CitationCount_CrossRef"],
        marker_color="#FFD700",
        name="Citations",
        boxmean=True,
        hovertemplate="Award: %{x}<br>Citations: %{y}<extra></extra>",
    ),
    row=1,
    col=1,
)
fig5.add_trace(
    go.Box(
        x=df_aw["AwardFlag"],
        y=df_aw["Downloads_Xplore"],
        marker_color="#2CA02C",
        name="Downloads",
        boxmean=True,
        hovertemplate="Award: %{x}<br>Downloads: %{y}<extra></extra>",
    ),
    row=1,
    col=2,
)

fig5.update_yaxes(range=[0, math.ceil(cit_max * 1.12)], row=1, col=1, title_text="CitationCount_CrossRef")
fig5.update_yaxes(range=[0, math.ceil(dl_max * 1.12)], row=1, col=2, title_text="Downloads_Xplore")
fig5.update_layout(title_text="Award Impact on Citations and Downloads", width=1000, height=480)

# ---------------------------
# Assemble HTML with narratives and include_plotlyjs='cdn' for each figure
# ---------------------------

# Helper to render figure to div (include_plotlyjs='cdn' as requested)
def fig_to_div(fig):
    return pio.to_html(fig, include_plotlyjs="cdn", full_html=False)

# Narrative descriptions to be included above each figure (academic, flowing)
narratives = {
    "introduction": """
    <h2>Introduction</h2>
    <p>
    This report presents a structured, multi-method analysis of a scholarly dataset encompassing conference papers,
    citation metrics, download counts, keywords, and award information. Our objective is to provide an empirical
    and interpretable assessment of temporal citation dynamics, the predictive determinants of awards, the drivers of
    downloads, and evolving topical emphases as expressed through author keywords. The visualizations combine aggregate
    descriptive statistics with simple supervised models to enable both exploratory insight and interpretable inference.
    Subsequent sections present results, interpretive commentary, and suggestions for future analytical directions.
    </p>
    """,
    "fig1": """
    <h3>1. Citation Trends by Conference</h3>
    <p>
    The first analysis examines longitudinal patterns in citation exposure across the most prolific conferences in the
    dataset. By plotting two citation metrics (Aminer-derived and CrossRef-derived) we can compare repository-specific
    coverage and cross-source consistency. This view highlights periods of rising or falling scholarly impact for venues,
    making it possible to identify cohorts of years where particular conferences produced especially influential work.
    The interactive dropdown allows selection of either metric or both simultaneously; this helps researchers discern
    whether a signal is supported across citation sources or driven by idiosyncrasies of a single indexing system.
    Understanding these venue-level trends is important for longitudinal bibliometrics and for stakeholders allocating
    attention or resources toward impactful conferences.
    </p>
    <p>
    This analysis sets the stage for modeling questions: if certain venues or years show higher average citation counts,
    we can ask which paper-level features (e.g., keywords, references, downloads) best explain award outcomes and attention.
    The next section develops a predictive model that investigates those relationships in a supervised learning framework.
    </p>
    """,
    "fig2": """
    <h3>2. Award Prediction — Model Discrimination (ROC)</h3>
    <p>
    Building on descriptive trends, we trained a logistic regression model to estimate the probability that a paper receives
    an award. The ROC curve summarizes the model's discriminative ability across decision thresholds: the area under the curve (AUC)
    quantifies how well the model distinguishes awarded from non-awarded papers. Logistic regression is used here primarily
    for its transparency—coefficients provide direction and magnitude of association—and class balancing mitigates prevalence bias.
    A strong ROC suggests features in the dataset contain signal about award likelihood; a weak ROC indicates awards are influenced
    by factors not captured here (e.g., novelty, reviewer dynamics, or external reputational effects).
    </p>
    <p>
    The subsequent visualization of feature coefficients supports interpretability by identifying which covariates
    contribute most strongly to award probability and in which direction. Together, these plots guide hypothesis generation
    about the interplay between paper attributes, venue, and scholarly recognition.
    </p>
    """,
    "fig2_imp": """
    <h3>2a. Award Prediction — Feature Importance (Coefficients)</h3>
    <p>
    This bar chart displays the top model coefficients (by absolute magnitude) from the logistic regression. Each coefficient
    indicates the expected change in the log-odds of receiving an award associated with a unit change in the predictor,
    holding other variables constant. Positive coefficients identify features that increase award likelihood, while negative
    coefficients indicate features associated with lower likelihood. Because the model includes one-hot encodings for categorical
    variables (e.g., conference and paper type), individual venue or format effects may appear.
    Interpreting these coefficients requires contextual knowledge: large effects may reflect genuine merit signals or selection
    artifacts driven by the peer-review process. These results are useful for scholars and organizers studying recognition dynamics.
    </p>
    <p>
    Having examined award prediction, we next turn to download behavior to evaluate whether the drivers of recognition overlap
    with drivers of readership and accessibility.
    </p>
    """,
    "fig3": """
    <h3>3. Download Prediction and Residual Analysis</h3>
    <p>
    This section fits a linear regression model to predict downloads using citation counts, internal references, a replicability
    stamp indicator, year, and conference. The left panel compares predicted to actual downloads: alignment along the identity
    line indicates good fit, whereas systematic departures reveal bias. The right panel presents residuals (actual minus predicted)
    versus predicted values to detect heteroscedasticity, nonlinearities, or groups of papers systematically under- or over-predicted.
    Understanding these patterns informs whether simple linear relationships suffice or whether more flexible modeling is warranted.
    </p>
    <p>
    The download analysis complements the award analysis: if the same features predict both downloads and awards, it suggests
    overlapping pathways to attention and recognition. Conversely, divergence points to distinct mechanisms for readership and adjudication.
    The next section explores thematic dynamics through keyword evolution, offering content-based context for these quantitative patterns.
    </p>
    """,
    "fig4": """
    <h3>4. Keyword Trend Analysis — Top Topics Over Time</h3>
    <p>
    Topic evolution is a central interest in scientometrics. Here we plot the yearly frequency trajectories for the ten most
    frequent author-assigned keywords. These curves reveal the emergence, persistence, or decline of topical foci, which helps
    interpret the temporal patterns observed in citations and downloads. A rising keyword indicates an emergent or rapidly growing
    research niche; a declining curve may reflect topic maturation or shifting priorities.
    The interactive range slider enables close inspection of sub-periods and facilitates comparisons across keyword life-cycles.
    </p>
    <p>
    By juxtaposing keyword trends with citation and award dynamics, researchers can investigate whether emergent topics receive
    disproportionate recognition or readership, thereby linking content to impact.
    The final empirical section summarizes award-related differences in attention metrics.
    </p>
    """,
    "fig5": """
    <h3>5. Award Impact on Citations and Downloads</h3>
    <p>
    This pair of boxplots contrasts awarded versus non-awarded papers with respect to CrossRef citations and downloads. Boxplots
    succinctly summarize central tendency and dispersion, which allows assessment of whether awarded papers systematically enjoy higher
    attention. Such comparisons are essential when evaluating the downstream effects of awards on visibility and scholarly dissemination.
    Care must be taken when interpreting causality: awards might both reflect prior attention and subsequently catalyze additional attention.
    </p>
    <p>
    Together with the regression and classification analyses above, these descriptive comparisons provide a multi-faceted view of
    how recognition, readership, and citations interrelate. The concluding section synthesizes these insights and outlines future work.
    </p>
    """,
    "conclusion": """
    <h2>Conclusion</h2>
    <p>
    This report combines descriptive and inferential approaches to examine scholarly impact within a multi-conference dataset.
    The temporal citation analysis identified venue-specific impact dynamics; the award prediction model highlighted which features
    correlate with recognition; regression diagnostics for downloads indicated the extent to which simple predictors account for readership;
    and keyword trends situated these outcomes within evolving topical landscapes. Collectively, these analyses emphasize that impact
    is multi-dimensional and that no single metric fully captures scholarly influence.
    Future work should incorporate richer textual representations (embeddings of abstracts), longitudinal survival analyses of attention,
    and causal inference strategies to better establish whether awards drive attention or are merely correlated with it. The dashboard
    and accompanying figures provide a reproducible starting point for such investigations.
    </p>
    """
}

# Create HTML content: narrative + figure divs
html_blocks = []
html_blocks.append("<!doctype html>\n<html>\n<head>\n<meta charset='utf-8'>\n<title>Dataset Analysis Visualizations - Dark Theme</title>\n")
# Basic CSS for dark page and readability
html_blocks.append(
    "<style>\n"
    "body { background: #0f1115; color: #f3f4f6; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; }\n"
    "h1 { color: #ffffff }\n"
    ".section { padding: 10px 0 30px 0; border-bottom: 1px solid #212428; }\n"
    ".figure-container { background: #0b0c0f; padding: 12px; border-radius: 6px; box-shadow: 0 2px 8px rgba(0,0,0,0.6); }\n"
    "p { color: #c7ccd1; max-width: 1100px; line-height: 1.5; }\n"
    "h2,h3 { color: #ffffff; margin-top: 8px; }\n"
    "footer { color:#777; margin-top: 20px; }\n"
    "</style>\n"
)
html_blocks.append("</head>\n<body>\n")
html_blocks.append("<h1>Dataset Analysis Visualizations (Dark Mode)</h1>\n")

# Introduction
html_blocks.append("<div class='section'><div class='figure-container'>")
html_blocks.append(narratives["introduction"])
html_blocks.append("</div></div>\n")

# Add each section: narrative + figure
sections = [
    ("fig1", fig1),
    ("fig2", fig2),
    ("fig2_imp", fig2_imp),
    ("fig3", fig3),
    ("fig4", fig4),
    ("fig5", fig5),
]

for key, fig in sections:
    narrative = narratives.get(key, "")
    html_blocks.append(f"<div class='section'><div class='figure-container'>{narrative}")
    # Convert fig to html div (include_plotlyjs='cdn' included here for compatibility)
    div = fig_to_div(fig)
    # pio.to_html returns a full <div>... including <script> tag referencing plotly.js via CDN when requested.
    # We will embed it directly.
    html_blocks.append(div)
    html_blocks.append("</div></div>\n")

# Conclusion
html_blocks.append("<div class='section'><div class='figure-container'>")
html_blocks.append(narratives["conclusion"])
html_blocks.append("</div></div>\n")

# Footer
html_blocks.append("<footer><p>Generated by Plotly & scikit-learn. Figures are interactive: zoom, pan, hover. The analysis is reproducible and ready for extensions.</p></footer>\n")
html_blocks.append("</body>\n</html>")

html_page = "\n".join(html_blocks)

# Write to output file
output_file = "output.html"
with open(output_file, "w", encoding="utf-8") as f:
    f.write(html_page)

print(f"Exported interactive dashboard to {output_file}")

