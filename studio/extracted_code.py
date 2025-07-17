import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Load the dataset
df = pd.read_csv('dataset.csv')

# 1. Network Analysis of Author Collaborations
def plot_author_collaboration_network(df):
    G = nx.Graph()
    for authors in df['AuthorNames-Deduped'].dropna():
        author_list = authors.split(';')
        for i in range(len(author_list)):
            for j in range(i + 1, len(author_list)):
                G.add_edge(author_list[i], author_list[j])
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G, k=0.15)
    nx.draw(G, pos, node_size=20, with_labels=False)
    plt.title('Author Collaboration Network')
    plt.show()

plot_author_collaboration_network(df)

# 2. Trend Analysis of Research Topics
def plot_research_trends(df):
    df['Year'] = pd.to_datetime(df['Year'], format='%Y')
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['AuthorKeywords'].fillna(''))
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X.toarray())
    df['PCA1'] = X_pca[:, 0]
    df['PCA2'] = X_pca[:, 1]
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Year', palette='viridis', alpha=0.7)
    plt.title('Research Topic Trends Over Time')
    plt.show()

plot_research_trends(df)

# 3. Citation Prediction Models
def citation_prediction_model(df):
    features = ['AminerCitationCount', 'PubsCited_CrossRef']
    df = df.dropna(subset=features + ['CitationCount_CrossRef'])
    X = df[features]
    y = df['CitationCount_CrossRef']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')

citation_prediction_model(df)

# 4. Comparative Analysis of Citation Sources
def plot_citation_comparison(df):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='AminerCitationCount', y='CitationCount_CrossRef', alpha=0.7)
    plt.title('Comparative Analysis of Citation Sources')
    plt.xlabel('Aminer Citation Count')
    plt.ylabel('CrossRef Citation Count')
    plt.show()

plot_citation_comparison(df)

# 5. Impact of Conference Presentation on Downloads
def plot_conference_impact_on_downloads(df):
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='Conference', y='Downloads_Xplore')
    plt.xticks(rotation=90)
    plt.title('Impact of Conference Presentation on Downloads')
    plt.show()

plot_conference_impact_on_downloads(df)

# 6. Assessment of Graphical Content on Replicability
def plot_graphics_replicability_impact(df):
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='GraphicsReplicabilityStamp', y='Downloads_Xplore')
    plt.title('Impact of Graphical Content on Downloads')
    plt.show()

plot_graphics_replicability_impact(df)

# 7. Exploration of Award Impact on Author Collaborations
def plot_award_impact_on_collaborations(df):
    award_authors = df[df['Award'].notna()]['AuthorNames-Deduped'].dropna()
    award_collaborations = []
    for authors in award_authors:
        award_collaborations.extend(authors.split(';'))
    award_collaborations = pd.Series(award_collaborations).value_counts()
    plt.figure(figsize=(10, 6))
    sns.barplot(x=award_collaborations.index[:10], y=award_collaborations.values[:10])
    plt.xticks(rotation=90)
    plt.title('Top Collaborators Among Awarded Authors')
    plt.show()

plot_award_impact_on_collaborations(df)

# ---- NEW BLOCK ---- # 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('dataset.csv')

# Set the style for seaborn
sns.set(style="whitegrid")

# 1. Distribution of Paper Types
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='PaperType', order=df['PaperType'].value_counts().index, palette='viridis')
plt.title('Distribution of Paper Types')
plt.xlabel('Paper Type')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 2. Number of Papers per Year
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Year', order=sorted(df['Year'].unique()), palette='coolwarm')
plt.title('Number of Papers per Year')
plt.xlabel('Year')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 3. Top 10 Authors by Number of Papers
top_authors = df['AuthorNames-Deduped'].value_counts().head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x=top_authors.values, y=top_authors.index, palette='magma')
plt.title('Top 10 Authors by Number of Papers')
plt.xlabel('Number of Papers')
plt.ylabel('Author')
plt.tight_layout()
plt.show()

# 4. Average Citation Count by Conference
avg_citations = df.groupby('Conference')['CitationCount_CrossRef'].mean().sort_values(ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x=avg_citations.values, y=avg_citations.index, palette='plasma')
plt.title('Average Citation Count by Conference')
plt.xlabel('Average Citation Count')
plt.ylabel('Conference')
plt.tight_layout()
plt.show()

# 5. Downloads vs Citation Count
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Downloads_Xplore', y='CitationCount_CrossRef', hue='PaperType', palette='deep')
plt.title('Downloads vs Citation Count')
plt.xlabel('Downloads (Xplore)')
plt.ylabel('Citation Count (CrossRef)')
plt.legend(title='Paper Type')
plt.tight_layout()
plt.show()

# ---- NEW BLOCK ---- # 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
dataset = pd.read_csv('dataset.csv')

# Identify missing data in each column
missing_data = dataset.isnull().sum()

# Visualize the missing data
plt.figure(figsize=(12, 8))
sns.barplot(x=missing_data.index, y=missing_data.values)
plt.xticks(rotation=90)
plt.title('Missing Data in Each Column')
plt.ylabel('Number of Missing Values')
plt.xlabel('Columns')
plt.show()

