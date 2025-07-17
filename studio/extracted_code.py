import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('dataset.csv')

# Set the style of seaborn
sns.set(style="whitegrid")

# Plot 1: Number of Papers per Year
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='Year', order=sorted(df['Year'].unique()))
plt.title('Number of Papers per Year')
plt.xticks(rotation=45)
plt.xlabel('Year')
plt.ylabel('Number of Papers')
plt.tight_layout()
plt.show()

# Plot 2: Top 10 Conferences by Number of Papers
top_conferences = df['Conference'].value_counts().nlargest(10)
plt.figure(figsize=(12, 6))
sns.barplot(x=top_conferences.values, y=top_conferences.index, palette='viridis')
plt.title('Top 10 Conferences by Number of Papers')
plt.xlabel('Number of Papers')
plt.ylabel('Conference')
plt.tight_layout()
plt.show()

# Plot 3: Distribution of Citation Counts
plt.figure(figsize=(12, 6))
sns.histplot(df['CitationCount_CrossRef'].dropna(), bins=30, kde=True)
plt.title('Distribution of Citation Counts')
plt.xlabel('Citation Count')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# Plot 4: Downloads vs Citation Count
plt.figure(figsize=(12, 6))
sns.scatterplot(data=df, x='Downloads_Xplore', y='CitationCount_CrossRef', alpha=0.6)
plt.title('Downloads vs Citation Count')
plt.xlabel('Downloads')
plt.ylabel('Citation Count')
plt.tight_layout()
plt.show()

# Plot 5: Paper Types Distribution
plt.figure(figsize=(12, 6))
sns.countplot(data=df, y='PaperType', order=df['PaperType'].value_counts().index, palette='coolwarm')
plt.title('Distribution of Paper Types')
plt.xlabel('Count')
plt.ylabel('Paper Type')
plt.tight_layout()
plt.show()

# --- New Code Block --- 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('dataset.csv')

# Set the style for seaborn
sns.set(style="whitegrid")

# Plot 1: Number of Papers per Year
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='Year', palette='viridis')
plt.title('Number of Papers per Year')
plt.xticks(rotation=45)
plt.xlabel('Year')
plt.ylabel('Number of Papers')
plt.tight_layout()
plt.show()

# Plot 2: Top 10 Conferences by Number of Papers
top_conferences = df['Conference'].value_counts().nlargest(10)
plt.figure(figsize=(12, 6))
sns.barplot(x=top_conferences.values, y=top_conferences.index, palette='magma')
plt.title('Top 10 Conferences by Number of Papers')
plt.xlabel('Number of Papers')
plt.ylabel('Conference')
plt.tight_layout()
plt.show()

# Plot 3: Distribution of Citation Counts
plt.figure(figsize=(12, 6))
sns.histplot(df['CitationCount_CrossRef'].dropna(), bins=30, kde=True, color='skyblue')
plt.title('Distribution of Citation Counts')
plt.xlabel('Citation Count')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# Plot 4: Average Citation Count by Paper Type
avg_citations_by_type = df.groupby('PaperType')['CitationCount_CrossRef'].mean().sort_values(ascending=False)
plt.figure(figsize=(12, 6))
sns.barplot(x=avg_citations_by_type.values, y=avg_citations_by_type.index, palette='coolwarm')
plt.title('Average Citation Count by Paper Type')
plt.xlabel('Average Citation Count')
plt.ylabel('Paper Type')
plt.tight_layout()
plt.show()

# Plot 5: Downloads vs Citation Count
plt.figure(figsize=(12, 6))
sns.scatterplot(data=df, x='Downloads_Xplore', y='CitationCount_CrossRef', hue='PaperType', palette='deep', alpha=0.7)
plt.title('Downloads vs Citation Count')
plt.xlabel('Downloads')
plt.ylabel('Citation Count')
plt.legend(title='Paper Type')
plt.tight_layout()
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('dataset.csv')

# Convert 'Year' to datetime for better handling
df['Year'] = pd.to_datetime(df['Year'], format='%Y')

# Plot 1: Number of Papers Published Per Year
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Year', palette='viridis')
plt.title('Number of Papers Published Per Year')
plt.xlabel('Year')
plt.ylabel('Number of Papers')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot 2: Top 10 Conferences by Number of Papers
top_conferences = df['Conference'].value_counts().nlargest(10)
plt.figure(figsize=(10, 6))
sns.barplot(x=top_conferences.values, y=top_conferences.index, palette='magma')
plt.title('Top 10 Conferences by Number of Papers')
plt.xlabel('Number of Papers')
plt.ylabel('Conference')
plt.tight_layout()
plt.show()

# Plot 3: Distribution of Citation Counts
plt.figure(figsize=(10, 6))
sns.histplot(df['CitationCount_CrossRef'].dropna(), bins=30, kde=True, color='skyblue')
plt.title('Distribution of Citation Counts')
plt.xlabel('Citation Count')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# Plot 4: Average Citation Count by Paper Type
avg_citation_by_type = df.groupby('PaperType')['CitationCount_CrossRef'].mean().sort_values()
plt.figure(figsize=(10, 6))
sns.barplot(x=avg_citation_by_type.values, y=avg_citation_by_type.index, palette='coolwarm')
plt.title('Average Citation Count by Paper Type')
plt.xlabel('Average Citation Count')
plt.ylabel('Paper Type')
plt.tight_layout()
plt.show()

# Plot 5: Downloads vs Citation Count
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Downloads_Xplore', y='CitationCount_CrossRef', hue='PaperType', palette='Set2', alpha=0.7)
plt.title('Downloads vs Citation Count')
plt.xlabel('Downloads')
plt.ylabel('Citation Count')
plt.legend(title='Paper Type')
plt.tight_layout()
plt.show()
#--- New Code Block --- 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('dataset.csv')

# Set the style of seaborn
sns.set(style="whitegrid")

# Plot 1: Number of Papers per Year
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='Year', palette='viridis')
plt.title('Number of Papers per Year')
plt.xticks(rotation=45)
plt.xlabel('Year')
plt.ylabel('Number of Papers')
plt.tight_layout()
plt.show()

# Plot 2: Top 10 Conferences by Number of Papers
top_conferences = df['Conference'].value_counts().nlargest(10)
plt.figure(figsize=(12, 6))
sns.barplot(x=top_conferences.values, y=top_conferences.index, palette='magma')
plt.title('Top 10 Conferences by Number of Papers')
plt.xlabel('Number of Papers')
plt.ylabel('Conference')
plt.tight_layout()
plt.show()

# Plot 3: Distribution of Citation Counts
plt.figure(figsize=(12, 6))
sns.histplot(df['AminerCitationCount'].dropna(), bins=30, kde=True, color='skyblue')
plt.title('Distribution of Aminer Citation Counts')
plt.xlabel('Aminer Citation Count')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# Plot 4: Downloads vs Citation Count
plt.figure(figsize=(12, 6))
sns.scatterplot(data=df, x='Downloads_Xplore', y='AminerCitationCount', hue='Year', palette='coolwarm', alpha=0.7)
plt.title('Downloads vs Aminer Citation Count')
plt.xlabel('Downloads (Xplore)')
plt.ylabel('Aminer Citation Count')
plt.legend(title='Year', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Plot 5: Paper Types Distribution
plt.figure(figsize=(12, 6))
sns.countplot(data=df, y='PaperType', order=df['PaperType'].value_counts().index, palette='cubehelix')
plt.title('Distribution of Paper Types')
plt.xlabel('Number of Papers')
plt.ylabel('Paper Type')
plt.tight_layout()
plt.show()



#--- New Code Block --- 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('dataset.csv')

# Set the style for seaborn
sns.set(style="whitegrid")

# Plot 1: Number of Papers per Year
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='Year', order=sorted(df['Year'].unique()))
plt.title('Number of Papers per Year')
plt.xticks(rotation=45)
plt.xlabel('Year')
plt.ylabel('Number of Papers')
plt.tight_layout()
plt.show()

# Plot 2: Top 10 Conferences by Number of Papers
plt.figure(figsize=(12, 6))
top_conferences = df['Conference'].value_counts().nlargest(10)
sns.barplot(x=top_conferences.values, y=top_conferences.index, palette='viridis')
plt.title('Top 10 Conferences by Number of Papers')
plt.xlabel('Number of Papers')
plt.ylabel('Conference')
plt.tight_layout()
plt.show()

# Plot 3: Distribution of Citation Counts
plt.figure(figsize=(12, 6))
sns.histplot(df['AminerCitationCount'].dropna(), bins=30, kde=True, color='skyblue')
plt.title('Distribution of Aminer Citation Counts')
plt.xlabel('Aminer Citation Count')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# Plot 4: Downloads vs Citation Count
plt.figure(figsize=(12, 6))
sns.scatterplot(data=df, x='Downloads_Xplore', y='AminerCitationCount', hue='PaperType', alpha=0.7)
plt.title('Downloads vs Aminer Citation Count')
plt.xlabel('Downloads (Xplore)')
plt.ylabel('Aminer Citation Count')
plt.legend(title='Paper Type')
plt.tight_layout()
plt.show()

# Plot 5: Heatmap of Internal References
plt.figure(figsize=(12, 6))
internal_references = df['InternalReferences'].str.split(';').apply(len)
sns.histplot(internal_references, bins=30, kde=True, color='lightcoral')
plt.title('Distribution of Internal References')
plt.xlabel('Number of Internal References')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()



#--- New Code Block --- 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from wordcloud import WordCloud

# Load the dataset
df = pd.read_csv('dataset.csv')

# 1. Temporal Keyword Analysis
def temporal_keyword_analysis(df):
    df['Year'] = pd.to_datetime(df['Year'], format='%Y')
    keywords_by_year = df.groupby(df['Year'].dt.year)['AuthorKeywords'].apply(lambda x: ' '.join(x.dropna()))
    vectorizer = CountVectorizer(stop_words='english')
    keywords_matrix = vectorizer.fit_transform(keywords_by_year)
    keywords_df = pd.DataFrame(keywords_matrix.toarray(), index=keywords_by_year.index, columns=vectorizer.get_feature_names_out())
    top_keywords = keywords_df.sum().sort_values(ascending=False).head(10)
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=keywords_df[top_keywords.index])
    plt.title('Temporal Keyword Analysis')
    plt.xlabel('Year')
    plt.ylabel('Keyword Frequency')
    plt.legend(title='Keywords')
    plt.show()

# 2. Co-authorship Networks
def co_authorship_network(df):
    G = nx.Graph()
    for authors in df['AuthorNames-Deduped'].dropna():
        author_list = authors.split(';')
        for i in range(len(author_list)):
            for j in range(i + 1, len(author_list)):
                G.add_edge(author_list[i], author_list[j])
    
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G, k=0.1)
    nx.draw(G, pos, node_size=20, with_labels=False)
    plt.title('Co-authorship Network')
    plt.show()

# 3. Paper Impact Prediction (Visualization not applicable directly)

# 4. Impact of Awards on Citation
def impact_of_awards_on_citation(df):
    df['Award'] = df['Award'].fillna('No Award')
    sns.boxplot(x='Award', y='AminerCitationCount', data=df)
    plt.title('Impact of Awards on Citation Count')
    plt.xlabel('Award')
    plt.ylabel('Aminer Citation Count')
    plt.show()

# 5. Internal Citations Dynamics
def internal_citations_dynamics(df):
    G = nx.DiGraph()
    for index, row in df.iterrows():
        if pd.notna(row['InternalReferences']):
            references = row['InternalReferences'].split(';')
            for ref in references:
                G.add_edge(row['Title'], ref)
    
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G, k=0.1)
    nx.draw(G, pos, node_size=20, with_labels=False)
    plt.title('Internal Citations Dynamics')
    plt.show()

# 6. Effect of Page Length
def effect_of_page_length(df):
    df['PageLength'] = df['LastPage'] - df['FirstPage']
    sns.scatterplot(x='PageLength', y='AminerCitationCount', data=df)
    plt.title('Effect of Page Length on Citation Count')
    plt.xlabel('Page Length')
    plt.ylabel('Aminer Citation Count')
    plt.show()

# 7. Textual Analysis for Innovation Detection
def textual_analysis_for_innovation(df):
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['Abstract'].dropna())
    lda = LatentDirichletAllocation(n_components=5, random_state=42)
    lda.fit(X)
    
    for index, topic in enumerate(lda.components_):
        print(f'Topic #{index}:')
        print([vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-10:]])
    
    wordcloud = WordCloud(width=800, height=400, max_words=100).generate(' '.join(df['Abstract'].dropna()))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud of Abstracts')
    plt.show()

# 8. Interpretation of Conference Impact
def interpretation_of_conference_impact(df):
    conference_impact = df.groupby('Conference')['AminerCitationCount'].mean().sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=conference_impact.index, y=conference_impact.values)
    plt.xticks(rotation=90)
    plt.title('Conference Impact Based on Citation Count')
    plt.xlabel('Conference')
    plt.ylabel('Average Aminer Citation Count')
    plt.show()

# Run the functions
temporal_keyword_analysis(df)
co_authorship_network(df)
impact_of_awards_on_citation(df)
internal_citations_dynamics(df)
effect_of_page_length(df)
textual_analysis_for_innovation(df)
interpretation_of_conference_impact(df)



#--- New Code Block --- 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Load the dataset
df = pd.read_csv('dataset.csv')

# 1. Citation Analysis and Patterns Detection
def plot_citation_analysis(df):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='AminerCitationCount', y='CitationCount_CrossRef', size='PubsCited_CrossRef', hue='PubsCited_CrossRef', sizes=(20, 200), palette='viridis')
    plt.title('Citation Analysis')
    plt.xlabel('Aminer Citation Count')
    plt.ylabel('Citation Count CrossRef')
    plt.legend(title='Pubs Cited CrossRef')
    plt.show()

plot_citation_analysis(df)

# 2. Temporal Trend Analysis
def plot_temporal_trend(df):
    plt.figure(figsize=(12, 6))
    sns.countplot(data=df, x='Year', hue='Conference', palette='tab10')
    plt.title('Publications Trend Over Years by Conference')
    plt.xticks(rotation=45)
    plt.xlabel('Year')
    plt.ylabel('Number of Publications')
    plt.legend(title='Conference')
    plt.show()

plot_temporal_trend(df)

# 3. Authors Collaboration Network
def plot_author_collaboration(df):
    G = nx.Graph()
    for _, row in df.iterrows():
        authors = row['AuthorNames-Deduped'].split(';')
        for i in range(len(authors)):
            for j in range(i + 1, len(authors)):
                G.add_edge(authors[i], authors[j])
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G, k=0.1)
    nx.draw(G, pos, node_size=20, node_color='blue', edge_color='gray', with_labels=False)
    plt.title('Authors Collaboration Network')
    plt.show()

plot_author_collaboration(df)

# 4. Keyword Evolution and Topic Modeling
def plot_keyword_evolution(df):
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    dtm = vectorizer.fit_transform(df['AuthorKeywords'].fillna(''))
    lda = LatentDirichletAllocation(n_components=5, random_state=42)
    lda.fit(dtm)
    for index, topic in enumerate(lda.components_):
        print(f'Top 10 words for Topic #{index}')
        print([vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-10:]])
        print('\n')

plot_keyword_evolution(df)

# 5. Content-based Recommendation System
def recommend_papers(df, paper_index):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['Abstract'].fillna(''))
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    sim_scores = list(enumerate(cosine_sim[paper_index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    paper_indices = [i[0] for i in sim_scores]
    return df['Title'].iloc[paper_indices]

print(recommend_papers(df, 0))

# 6. Influence of Awards and Replicability on Impact
def plot_award_influence(df):
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='Award', y='Downloads_Xplore', hue='GraphicsReplicabilityStamp', palette='coolwarm')
    plt.title('Influence of Awards and Replicability on Downloads')
    plt.xlabel('Award')
    plt.ylabel('Downloads Xplore')
    plt.legend(title='Graphics Replicability Stamp')
    plt.show()

plot_award_influence(df)



#--- New Code Block --- 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from wordcloud import WordCloud
from collections import Counter
import numpy as np

# Load the dataset
df = pd.read_csv('dataset.csv')

# 1. Temporal Trends in Citation Growth
plt.figure(figsize=(10, 6))
df.groupby('Year')['AminerCitationCount'].sum().plot()
plt.title('Temporal Trends in Citation Growth')
plt.xlabel('Year')
plt.ylabel('Total Citations')
plt.grid(True)
plt.show()

# 2. Paper Type Analysis
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='PaperType', y='Downloads_Xplore')
plt.title('Downloads by Paper Type')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='PaperType', y='AminerCitationCount')
plt.title('Citations by Paper Type')
plt.xticks(rotation=45)
plt.show()

# 3. Collaboration Networks and Impact
author_collab = df['AuthorNames-Deduped'].str.split(';').apply(lambda x: list(set(x)))
G = nx.Graph()

for authors in author_collab:
    for i in range(len(authors)):
        for j in range(i + 1, len(authors)):
            G.add_edge(authors[i], authors[j])

plt.figure(figsize=(12, 12))
pos = nx.spring_layout(G, k=0.15)
nx.draw(G, pos, node_size=20, node_color='blue', edge_color='gray', with_labels=False)
plt.title('Collaboration Network')
plt.show()

# 4. Award Influence on Citation
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='Award', y='AminerCitationCount')
plt.title('Citation Count by Award Status')
plt.show()

# 5. Reference Patterns and Success
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='InternalReferences', y='AminerCitationCount')
plt.title('Internal References vs Citation Count')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='PubsCited_CrossRef', y='AminerCitationCount')
plt.title('Pubs Cited vs Citation Count')
plt.show()

# 6. Textual Analysis of Abstracts
abstracts = ' '.join(df['Abstract'].dropna())
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(abstracts)

plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Abstracts')
plt.show()

# 7. Download Patterns Across Conferences
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='Conference', y='Downloads_Xplore')
plt.title('Downloads by Conference')
plt.xticks(rotation=45)
plt.show()



#--- New Code Block --- 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('dataset.csv')

# Set the style for seaborn
sns.set(style="whitegrid")

# 1. Bar plot of the number of papers per conference
plt.figure(figsize=(12, 6))
conference_counts = df['Conference'].value_counts()
sns.barplot(x=conference_counts.index, y=conference_counts.values, palette='viridis')
plt.title('Number of Papers per Conference')
plt.xlabel('Conference')
plt.ylabel('Number of Papers')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 2. Line plot of the number of papers published each year
plt.figure(figsize=(12, 6))
yearly_counts = df['Year'].value_counts().sort_index()
sns.lineplot(x=yearly_counts.index, y=yearly_counts.values, marker='o')
plt.title('Number of Papers Published Each Year')
plt.xlabel('Year')
plt.ylabel('Number of Papers')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 3. Scatter plot of Citation Count vs Downloads
plt.figure(figsize=(12, 6))
sns.scatterplot(data=df, x='CitationCount_CrossRef', y='Downloads_Xplore', hue='Conference', palette='deep')
plt.title('Citation Count vs Downloads')
plt.xlabel('Citation Count (CrossRef)')
plt.ylabel('Downloads (Xplore)')
plt.legend(title='Conference', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# 4. Heatmap of the correlation matrix
plt.figure(figsize=(12, 6))
correlation_matrix = df[['AminerCitationCount', 'CitationCount_CrossRef', 'Downloads_Xplore']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()



#--- New Code Block --- 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
import numpy as np

# Load the dataset
df = pd.read_csv('dataset.csv')

# 1. Influence of Conference on Citation Count
plt.figure(figsize=(12, 6))
sns.boxplot(x='Conference', y='AminerCitationCount', data=df)
plt.xticks(rotation=90)
plt.title('Influence of Conference on Aminer Citation Count')
plt.tight_layout()
plt.show()

# 2. Predicting Paper Awards with Machine Learning
features = ['Year', 'PaperType', 'AminerCitationCount', 'Downloads_Xplore']
df['Award'] = df['Award'].fillna(0)  # Assuming NaN means no award
X = df[features]
y = df['Award']

# Encode categorical variables
le = LabelEncoder()
X['PaperType'] = le.fit_transform(X['PaperType'])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest Classifier
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

print("Classification Report for Award Prediction:")
print(classification_report(y_test, y_pred))

# 3. Author Collaboration Network Analysis
G = nx.Graph()
for _, row in df.iterrows():
    authors = row['AuthorNames-Deduped'].split(';')
    for i in range(len(authors)):
        for j in range(i + 1, len(authors)):
            G.add_edge(authors[i], authors[j])

plt.figure(figsize=(12, 12))
pos = nx.spring_layout(G, k=0.15)
nx.draw(G, pos, node_size=20, with_labels=False)
plt.title('Author Collaboration Network')
plt.show()

# 4. Keyword Trends over Time
df['AuthorKeywords'] = df['AuthorKeywords'].fillna('')
keywords = df['AuthorKeywords'].str.get_dummies(sep=';')
keywords['Year'] = df['Year']
keywords_grouped = keywords.groupby('Year').sum()

plt.figure(figsize=(12, 6))
for keyword in keywords.columns[:-1]:
    plt.plot(keywords_grouped.index, keywords_grouped[keyword], label=keyword)
plt.title('Keyword Trends Over Time')
plt.xlabel('Year')
plt.ylabel('Frequency')
plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
plt.tight_layout()
plt.show()

# 5. Assessing Paper Quality through Graphics and Replicability
df['GraphicsReplicabilityStamp'] = df['GraphicsReplicabilityStamp'].fillna(0)
X = df[['GraphicsReplicabilityStamp']]
y = df['AminerCitationCount']

# Linear Regression
lr = LinearRegression()
lr.fit(X, y)
y_pred = lr.predict(X)

plt.figure(figsize=(8, 6))
plt.scatter(X, y, color='blue', label='Actual')
plt.plot(X, y_pred, color='red', linewidth=2, label='Predicted')
plt.title('Impact of Graphics Replicability on Citation Count')
plt.xlabel('Graphics Replicability Stamp')
plt.ylabel('Aminer Citation Count')
plt.legend()
plt.show()



#--- New Code Block --- 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from lifelines import KaplanMeierFitter
from sklearn.decomposition import FactorAnalysis
from statsmodels.graphics.tsaplots import plot_acf
import numpy as np

# Load the dataset
df = pd.read_csv('dataset.csv')

# 1. Causal Inference on Citations
# Visualize the effect of Conference type and Award status on CitationCount_CrossRef
plt.figure(figsize=(12, 6))
sns.boxplot(x='Conference', y='CitationCount_CrossRef', hue='Award', data=df)
plt.title('Effect of Conference Type and Award Status on Citation Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 2. Survival Analysis
# Kaplan-Meier survival curve for paper relevance based on CitationCount_CrossRef
kmf = KaplanMeierFitter()
kmf.fit(durations=df['CitationCount_CrossRef'], event_observed=df['Downloads_Xplore'])
plt.figure(figsize=(10, 6))
kmf.plot_survival_function()
plt.title('Survival Analysis of Paper Relevance')
plt.xlabel('Citation Count')
plt.ylabel('Survival Probability')
plt.show()

# 3. Latent Variable Modeling
# Factor Analysis to identify latent constructs
fa = FactorAnalysis(n_components=2)
latent_factors = fa.fit_transform(df[['InternalReferences', 'Abstract', 'AuthorKeywords']].fillna(0))
df['Factor1'], df['Factor2'] = latent_factors[:, 0], latent_factors[:, 1]

plt.figure(figsize=(10, 6))
sns.scatterplot(x='Factor1', y='Factor2', hue='CitationCount_CrossRef', data=df, palette='viridis')
plt.title('Latent Variable Modeling')
plt.show()

# 4. Network Analysis of Citation Data
# Create a citation network
G = nx.from_pandas_edgelist(df, source='DOI', target='PubsCited_CrossRef')
plt.figure(figsize=(12, 12))
pos = nx.spring_layout(G, k=0.1)
nx.draw(G, pos, node_size=20, node_color='blue', edge_color='gray', with_labels=False)
plt.title('Citation Network Analysis')
plt.show()

# 5. Time Series Analysis of Trends
# Time series analysis on yearly trends in AminerCitationCount
df['Year'] = pd.to_datetime(df['Year'], format='%Y')
yearly_trends = df.groupby(df['Year'].dt.year)['AminerCitationCount'].sum()

plt.figure(figsize=(12, 6))
yearly_trends.plot()
plt.title('Yearly Trends in Aminer Citation Count')
plt.xlabel('Year')
plt.ylabel('Aminer Citation Count')
plt.show()

# Autocorrelation plot
plot_acf(yearly_trends)
plt.title('Autocorrelation of Yearly Trends in Aminer Citation Count')
plt.show()



#--- New Code Block --- 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('dataset.csv')

# Set the style for seaborn
sns.set(style="whitegrid")

# 1. Bar plot of the number of papers per conference
plt.figure(figsize=(12, 6))
conference_counts = df['Conference'].value_counts()
sns.barplot(x=conference_counts.index, y=conference_counts.values, palette='viridis')
plt.title('Number of Papers per Conference')
plt.xlabel('Conference')
plt.ylabel('Number of Papers')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 2. Line plot of the average citation count per year
plt.figure(figsize=(12, 6))
avg_citations_per_year = df.groupby('Year')['CitationCount_CrossRef'].mean()
sns.lineplot(x=avg_citations_per_year.index, y=avg_citations_per_year.values, marker='o')
plt.title('Average Citation Count per Year')
plt.xlabel('Year')
plt.ylabel('Average Citation Count')
plt.grid(True)
plt.tight_layout()
plt.show()

# 3. Scatter plot of Downloads vs Citation Count
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Downloads_Xplore', y='CitationCount_CrossRef', hue='PaperType', palette='deep')
plt.title('Downloads vs Citation Count')
plt.xlabel('Downloads (Xplore)')
plt.ylabel('Citation Count (CrossRef)')
plt.legend(title='Paper Type')
plt.tight_layout()
plt.show()

# 4. Heatmap of the correlation matrix
plt.figure(figsize=(10, 8))
correlation_matrix = df[['AminerCitationCount', 'CitationCount_CrossRef', 'Downloads_Xplore']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()

# 5. Box plot of citation counts by paper type
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='PaperType', y='CitationCount_CrossRef', palette='pastel')
plt.title('Citation Counts by Paper Type')
plt.xlabel('Paper Type')
plt.ylabel('Citation Count (CrossRef)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

