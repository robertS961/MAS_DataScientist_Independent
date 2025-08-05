from helper_functions import initialize_state_from_csv, define_variables, pretty_print_messages, get_last_ai_message, generate_pdf_report, get_datainfo
from agents import ploty_agent, plotly_leader
from classes import State, Configurable
import re

code = """"
# ---- NEW BLOCK ---- # 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv('dataset.csv')

# 0. PREPROCESSING & IMPUTATION
# ----------------------------
# Fill NaNs in numeric columns
df['FirstPage'].fillna(df['FirstPage'].mean(), inplace=True)
df['LastPage'].fillna(df['LastPage'].mean(), inplace=True)
df['AminerCitationCount'].fillna(df['AminerCitationCount'].mean(), inplace=True)
df['Downloads_Xplore'].fillna(df['Downloads_Xplore'].mean(), inplace=True)
# Fill NaNs in text columns
df['Abstract'].fillna('', inplace=True)
df['AuthorKeywords'].fillna('', inplace=True)
# GraphicsReplicabilityStamp: assume 'Yes'/'No'
df['GraphicsReplicabilityStamp'].fillna('No', inplace=True)

# Convert categorical/binary columns
df['Award'] = df['Award'].notna().astype(int)
df['GraphicsReplicabilityStamp'] = (df['GraphicsReplicabilityStamp'] == 'Yes').astype(int)

# Reusable imputer for feature matrices
imputer = SimpleImputer(strategy='mean')


# 1. Trend of Aminer Citation Counts by Conference
# ------------------------------------------------
plt.figure(figsize=(11, 5))
sns.lineplot(
    data=df, x='Year', y='AminerCitationCount',
    hue='Conference', palette='tab10', ci=None
)
plt.title('Citation Trend Over Time by Conference', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Aminer Citation Count', fontsize=12)
plt.legend(title='Conference', bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.show()


# 2. Feature Importance for Award Prediction (Logistic Regression)
# ----------------------------------------------------------------
# Prepare features
X_award = df[['PaperType', 'Downloads_Xplore', 'Year', 'Conference']]
X_award = pd.get_dummies(X_award, drop_first=True)
y_award = df['Award']

# Impute and split
X_award_imp = imputer.fit_transform(X_award)
X_tr, X_te, y_tr, y_te = train_test_split(X_award_imp, y_award, test_size=0.2, random_state=42)

# Train
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_tr, y_tr)

# Extract & sort coefficients
coefs = pd.Series(logreg.coef_[0], index=X_award.columns)
coefs = coefs.sort_values()

plt.figure(figsize=(8, 6))
coefs.plot(kind='barh', color='steelblue')
plt.title('Logistic Regression Feature Importance', fontsize=14)
plt.xlabel('Coefficient Value', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.tight_layout()
plt.show()


# 3. Regression Analysis for Download Prediction
# ----------------------------------------------
X_dl = df[['AminerCitationCount', 'GraphicsReplicabilityStamp', 'Year']]
y_dl = df['Downloads_Xplore']

# Impute, split, train
X_dl_imp = imputer.fit_transform(X_dl)
Xtr, Xte, ytr, yte = train_test_split(X_dl_imp, y_dl, test_size=0.2, random_state=42)

lr = LinearRegression()
lr.fit(Xtr, ytr)
y_pred = lr.predict(Xte)

mse = mean_squared_error(yte, y_pred)
r2 = r2_score(yte, y_pred)

plt.figure(figsize=(7, 7))
plt.scatter(yte, y_pred, color='teal', alpha=0.6, edgecolor='k', s=50)
# 45° reference line
lims = [min(yte.min(), y_pred.min()), max(yte.max(), y_pred.max())]
plt.plot(lims, lims, 'r--', linewidth=1.5)
plt.title('Download Prediction: True vs. Predicted', fontsize=14)
plt.xlabel('True Download Count', fontsize=12)
plt.ylabel('Predicted Download Count', fontsize=12)
# Annotate MSE and R²
textstr = f'MSE: {mse:.2f}\nR\u00b2: {r2:.2f}'
plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes,
               fontsize=11, verticalalignment='top',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
plt.tight_layout()
plt.show()


# 4. Top 20 Author Keywords
# -------------------------
# Split, flatten and count
df['AuthorKeywords'] = df['AuthorKeywords'].str.lower().str.split(',\s*')
all_kw = [kw for sub in df['AuthorKeywords'] for kw in sub if kw]
kw_counts = pd.Series(all_kw).value_counts().head(20)

plt.figure(figsize=(11, 5))
sns.barplot(x=kw_counts.values, y=kw_counts.index, palette='viridis')
plt.title('Top 20 Author Keywords', fontsize=14)
plt.xlabel('Frequency', fontsize=12)
plt.ylabel('Keyword', fontsize=12)
plt.tight_layout()
plt.show()


# 5. Impact of Awards on Citation Count (CrossRef)
# ------------------------------------------------
plt.figure(figsize=(11, 5))
sns.histplot(
    data=df, x='CitationCount_CrossRef', hue='Award',
    element='step', stat='density', common_norm=False,
    palette={0: 'lightgray', 1: 'goldenrod'}, alpha=0.7
)
plt.title('Citation Distribution by Award Status', fontsize=14)
plt.xlabel('Citation Count (CrossRef)', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend(title='Award', labels=['No', 'Yes'], loc='upper right')
plt.tight_layout()
plt.show()

"""
previous = ''' Apologies for the oversight. Here are the refined ideas with suggested enhancements:

1. **Enhanced Trend Analysis on Citation Counts (Data Science)**
   - **Objective**: Utilize `Year`, `AminerCitationCount`, and `CitationCount_CrossRef` while incorporating `Conference` and `PaperType`.
   - **Approach**: Conduct time-series analysis to identify citation trends and examine whether conferences or paper types exhibit distinct patterns.

2. **Network Analysis of Authors and Affiliations (Data Science)**
   - **Objective**: Use `AuthorNames-Deduped`, `AuthorAffiliation`, merged with `InternalReferences`.
   - **Approach**: Build networks depicting author connections through shared affiliations and citation links within the dataset, identifying influential researchers.

3. **Machine Learning to Predict Awards (Data Science)**
   - **Objective**: Implement models using `PaperType`, `AuthorKeywords`, `InternalReferences`, `Downloads_Xplore`, and additionally, `Year` and `Conference`.
   - **Approach**: Develop classification algorithms to predict the likelihood of winning an award, considering temporal and conference-based patterns.

4. **Regression Analysis for Download Prediction (Statistical Learning)**
   - **Objective**: Analyze `Downloads_Xplore` with predictors like `AminerCitationCount`, `GraphicsReplicabilityStamp`, `InternalReferences`, and add `Year` and `Conference`.
   - **Approach**: Perform a multi-variable regression analysis to understand the factors impacting download numbers.

5. **Factor Analysis of Research Topics (Statistical Learning)**
   - **Objective**: Use `AuthorKeywords` and `Abstract` to delve into research themes.
   - **Approach**: Conduct factor analysis pairing with abstract analysis for a comprehensive exploration of research topics.

6. **Survival Analysis on Paper Popularity (Statistical Learning)**
   - **Objective**: Use `Year`, `Downloads_Xplore`, `CitationCount_CrossRef`, `PaperType`, and `Conference`.
   - **Approach**: Leverage survival analysis to study paper popularity lifecycles and assess how particular paper types or conferences impact popularity over time.

These ideas incorporate suggestions for using additional columns to derive more in-depth insights and improvement opportunities.
 
Here are the consolidated ideas from the agents, along with additional suggestions:

**Data Science Ideas:**

1. **Collaborative Network Analysis:**
   - Construct a network graph using `AuthorNames-Deduped` and `AuthorAffiliation` to identify collaboration patterns. Analyze centrality measures to find influential authors and explore network clustering to uncover collaboration communities across different `Conference` events, and visualize evolution over `Year`.

2. **Conference Impact Assessment:**
   - Evaluate the impact of different `Conferences` on citation metrics (`AminerCitationCount`, `CitationCount_CrossRef`) and their relationship with `PaperType` and `Year` to understand the historical significance of conferences.

3. **Keyword Trend Analysis:**
   - Use NLP to analyze the `AuthorKeywords` over `Year` for frequency and trend analysis to uncover emerging topics and shift in research interests. Visualization can illustrate these keyword evolutions.

4. **Citation Prediction Model:**
   - Develop machine learning models (e.g., regression, XGBoost) to predict `CitationCount_CrossRef` using features like `Year`, `PaperType`, `AuthorAffiliation`, `Downloads_Xplore`, and others extracted from `Abstract`.

**Statistical Learning Ideas:**

5. **Factor Analysis on Author Affiliations:**
   - Apply factor analysis on `AuthorAffiliation` to identify hidden factors influencing research output. Correlate these with `AminerCitationCount` and `Downloads_Xplore` to study institutional impact.

6. **Meta-analysis of Graphics Replicability:**
   - Perform meta-analysis on the `GraphicsReplicabilityStamp` to examine correlation with `CitationCount_CrossRef`, `Downloads_Xplore`, and `Award`, emphasizing the role of replicability.

**Additional Ideas:**

7. **Abstract Sentiment Analysis:**
   - Perform sentiment analysis on `Abstract` texts to explore if sentiment correlates with `CitationCount_CrossRef`. Positive sentiment might influence citation attractiveness.

8. **Author Influence Over Time:**
   - Analyze changes in `AminerCitationCount` over `Year` for top authors to identify how their influence grows or wanes over time.

9. **Research Content Clustering:**
   - Cluster papers based on abstracts and `AuthorKeywords` to identify thematic groups and compare them across different `Conferences`.

10. **Impact of Awards on Citations:**
    - Evaluate if papers with `Award` exhibit higher `CitationCount_CrossRef` and `Downloads_Xplore`, and explore other patterns linked with award-winning papers.

This comprehensive approach will harness both data science and statistical methodologies for valuable insights from the dataset.
 
'''

state = initialize_state_from_csv()
data = state['dataset_info']
data_info = get_datainfo("dataset.csv")
dic, config = define_variables(thread = 1, loop_limit = 10, data = data, data_info = data_info, name = "plotly", input = previous)
dic['revise'] = True
plotly_lead = plotly_leader(state)
for chunk in plotly_lead.stream(input = dic, config = config):
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


