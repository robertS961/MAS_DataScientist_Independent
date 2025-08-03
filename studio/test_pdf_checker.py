
from helper_functions import initialize_state_from_csv, define_variables, pretty_print_messages, get_last_ai_message, generate_pdf_report, get_datainfo
from agents import machinelearning_agent, create_code, pdf_checker_agent
from classes import State, Configurable
import re
'''
import re
from helper_functions import initialize_state_from_csv, define_variables, pretty_print_messages, get_last_ai_message, generate_pdf_report, get_datainfo
from agents import machinelearning_agent, create_code
from classes import State, Configurable



state = initialize_state_from_csv()
data = state['dataset_info']
data_info = get_datainfo("dataset.csv")
dic, config = define_variables(thread = 1, loop_limit = 10, data = data, data_info = data_info, name = "ml")

agent = machinelearning_agent(State)

result = agent.invoke(dic, config)
return_message = get_last_ai_message(result['messages'])
print(return_message)

dic, config = define_variables(thread = 1, loop_limit = 12, data = data, data_info = data_info, name = "code", input = return_message)

graph = create_code(State)
print("Graph for Visualization Created! \n")

for chunk in graph.stream(input = dic, config = config):
    pretty_print_messages(chunk)


print(f"This is the chunk \n {chunk} \n")
result = chunk['code_agent']['messages'][-1]['content']

code = re.findall(r"```python\n(.*?)\n```", result, re.DOTALL)
print(f"This is the code ! \n {code} \n")
with open("extracted_code.py", "a", encoding="utf-8") as f:
    f.write("# ---- NEW BLOCK ---- # \n")
    for block in code:
        f.write(block + "\n\n")

generate_pdf_report(result, 'output.pdf')

'''

code = '''# ---- NEW BLOCK ---- # 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression

# Load the dataset
df = pd.read_csv('dataset.csv')

# Fill NaN values with modern methods
df['AminerCitationCount'].fillna(df['AminerCitationCount'].median(), inplace=True)
df['CitationCount_CrossRef'].fillna(df['CitationCount_CrossRef'].median(), inplace=True)
df['Downloads_Xplore'].fillna(df['Downloads_Xplore'].median(), inplace=True)
df['AuthorKeywords'].fillna('', inplace=True)
df['Award'] = df['Award'].fillna('No Award')
df['GraphicsReplicabilityStamp'] = df['GraphicsReplicabilityStamp'].fillna('No Stamp')
df['InternalReferences'] = df['InternalReferences'].fillna('')

# Convert categorical variables
df['Conference'] = df['Conference'].astype('category')
df['PaperType'] = df['PaperType'].astype('category')
df['Award'] = df['Award'].astype('category')

# Enhanced Trend Analysis on Citation Counts
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x='Year', y='AminerCitationCount', hue='Conference', ci=None)
sns.lineplot(data=df, x='Year', y='CitationCount_CrossRef', hue='PaperType', ci=None, linestyle='--')
plt.title('Citation Trends Over the Years by Conference and Paper Type')
plt.xlabel('Year')
plt.ylabel('Citation Count')
plt.legend(title='Legend', fontsize='small')
plt.show()

# Machine Learning to Predict Awards
X = df[['PaperType', 'Downloads_Xplore', 'Year', 'Conference', 'AuthorKeywords']].copy()
X['Conference'] = LabelEncoder().fit_transform(X['Conference'])
X['PaperType'] = LabelEncoder().fit_transform(X['PaperType'])
X['AuthorKeywords'] = X['AuthorKeywords'].apply(lambda x: len(x.split(',')))

y = df['Award']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)
y_pred = rf_classifier.predict(X_test)

# Display tree feature importance
importances = rf_classifier.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 5))
plt.title("Feature Importance from RandomForest")
sns.barplot(x=[X.columns[i] for i in indices], y=importances[indices])
plt.show()

# Convert InternalReferences from strings of DOIs to counts of references
df['InternalReferencesCount'] = df['InternalReferences'].apply(lambda x: len(x.split(';')) if x else 0)

# Regression Analysis for Download Prediction
predictors = ['AminerCitationCount', 'GraphicsReplicabilityStamp', 'InternalReferencesCount', 'Year', 'Conference']
X = df[predictors].copy()
X['Conference'] = LabelEncoder().fit_transform(X['Conference'])
X['GraphicsReplicabilityStamp'] = X['GraphicsReplicabilityStamp'].apply(lambda x: 1 if x == 'Has Stamp' else 0)

y = df['Downloads_Xplore']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

# Visualization of the actual vs predicted download values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.title('Actual vs Predicted Downloads')
plt.xlabel('Actual Downloads')
plt.ylabel('Predicted Downloads')
plt.show()

# Keyword Trend Analysis
all_keywords = df['AuthorKeywords'].str.cat(sep=', ')
wordcloud = WordCloud(width=800, height=400, max_words=150, background_color='white').generate(all_keywords)

plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.title('Keyword Frequency WordCloud')
plt.axis('off')
plt.show()

# Impact of Awards on Citations
award_citations = df[df['Award'] != 'No Award']['CitationCount_CrossRef']
no_award_citations = df[df['Award'] == 'No Award']['CitationCount_CrossRef']

plt.figure(figsize=(12, 6))
sns.boxplot(data=[award_citations, no_award_citations], palette='pastel')
plt.xticks([0, 1], ['Awarded Papers', 'Non Awarded Papers'])
plt.title('Impact of Awards on Citation Count')
plt.ylabel('CitationCount_CrossRef')
plt.show()

'''
state = initialize_state_from_csv()
data = state['dataset_info']
data_info = get_datainfo("dataset.csv")
dic, config = define_variables(thread = 1, loop_limit = 10, data = data, data_info = data_info, name = "fix-vis", code = code)

agent = pdf_checker_agent(State)
for chunk in agent.stream(input = dic, config = config):
    pretty_print_messages(chunk)


print(f"This is the chunk \n {chunk} \n")
result = chunk['agent']['messages'][-1].content

code = re.findall(r"```python\n(.*?)\n```", result, re.DOTALL)
print(f"This is the code ! \n {code} \n")
with open("extracted_code.py", "a", encoding="utf-8") as f:
    f.write("# ---- NEW BLOCK ---- # \n")
    for block in code:
        f.write(block + "\n\n")

generate_pdf_report(result, 'output.pdf')
