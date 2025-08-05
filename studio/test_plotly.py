from helper_functions import initialize_state_from_csv, define_variables, pretty_print_messages, get_last_ai_message, generate_pdf_report, get_datainfo
from agents import ploty_agent
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

state = initialize_state_from_csv()
data = state['dataset_info']
data_info = get_datainfo("dataset.csv")
dic, config = define_variables(thread = 1, loop_limit = 10, data = data, data_info = data_info, name = "plotly", code = code)
dic['revise'] = True
plotly_ag = ploty_agent(State)
for chunk in plotly_ag.stream(input = dic, config = config):
    pretty_print_messages(chunk)
 

print(f"This is the chunk \n {chunk} \n")
result = chunk['agent']['messages'][-1].content

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


