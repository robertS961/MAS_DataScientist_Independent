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
'''
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
'''
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
'''
# ---- NEW BLOCK ---- # 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('dataset.csv')

# Data-Driven Conference Profiling: Citations and Downloads over Years
plt.figure(figsize=(14, 6))
sns.lineplot(data=df, x='Year', y='AminerCitationCount', hue='Conference', marker='o')
plt.title('Conference Citations Over Years')
plt.xlabel('Year')
plt.ylabel('Aminer Citation Count')
plt.legend(title='Conference', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.show()

# Citation Network Analysis: Co-Citation Heatmap
citation_matrix = df[['InternalReferences', 'AminerCitationCount']].dropna().corr()
plt.figure(figsize=(8, 6))
sns.heatmap(citation_matrix, annot=True, cmap='Blues')
plt.title('Citation Network Heatmap')
plt.show()

# Keyword Evolution Study: Keyword Frequency Over Time
df_exploded = df.assign(AuthorKeywords=df['AuthorKeywords'].str.split(',')).explode('AuthorKeywords')
keyword_by_year = df_exploded.groupby(['Year', 'AuthorKeywords']).size().unstack(fill_value=0)
plt.figure(figsize=(12, 8))
sns.heatmap(keyword_by_year, cmap='YlGnBu')
plt.title('Keyword Evolution Over Time')
plt.xlabel('Keywords')
plt.ylabel('Year')
plt.show()

# Replicability Analysis: Citations vs Graphics Replicability
plt.figure(figsize=(10, 5))
sns.boxplot(data=df, x='GraphicsReplicabilityStamp', y='AminerCitationCount')
plt.title('Citations by Graphics Replicability Stamp')
plt.xlabel('Graphics Replicability Stamp')
plt.ylabel('Aminer Citation Count')
plt.grid(True)
plt.show()

# Page Count Influence: Citations vs Page Count
df['PageCount'] = df['LastPage'] - df['FirstPage']
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='PageCount', y='AminerCitationCount', hue='Conference', style='Award')
plt.title('Page Count Influence on Citations')
plt.xlabel('Page Count')
plt.ylabel('Aminer Citation Count')
plt.legend(title='Conference | Award')
plt.grid(True)
plt.show()

'''# ---- NEW BLOCK ---- # 
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns

# Load the dataset
data = pd.read_csv('dataset.csv')

# Ensure 'AuthorNames-Deduped' is treated as string
data['AuthorNames-Deduped'] = data['AuthorNames-Deduped'].astype(str)
data['AuthorKeywords'] = data['AuthorKeywords'].astype(str)

# Author Collaboration Network Analysis
def plot_author_collaboration_network(data):
    G = nx.Graph()
    for index, row in data.iterrows():
        authors = row['AuthorNames-Deduped'].split(', ')
        G.add_nodes_from(authors)
        for i in range(len(authors)):
            for j in range(i + 1, len(authors)):
                G.add_edge(authors[i], authors[j])

    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G, k=0.3)
    nx.draw_networkx_nodes(G, pos, node_size=50, node_color='skyblue')
    nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color='gray')
    plt.title('Author Collaboration Network')
    plt.show()

plot_author_collaboration_network(data)

# Research Trend Analysis using Author Keywords
def plot_research_trends(data):
    keywords = data['AuthorKeywords'].str.cat(sep=', ').lower().split(', ')
    keyword_df = pd.DataFrame(keywords, columns=['Keyword'])
    keyword_count = keyword_df['Keyword'].value_counts().head(20)

    plt.figure(figsize=(12, 8))
    sns.barplot(x=keyword_count.values, y=keyword_count.index, palette='viridis')
    plt.title('Top 20 Research Keywords')
    plt.xlabel('Frequency')
    plt.ylabel('Keyword')
    plt.tight_layout()
    plt.show()

plot_research_trends(data)

# Temporal Analysis of Conference Impact
def plot_temporal_conference_impact(data):
    citation_count_by_year = data.groupby('Year')['CitationCount_CrossRef'].sum().reset_index()

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=citation_count_by_year, x='Year', y='CitationCount_CrossRef', marker='o', color='c')
    plt.title('Yearly Citation Count')
    plt.xlabel('Year')
    plt.ylabel('Total Citation Count')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_temporal_conference_impact(data)

# ---- NEW BLOCK ---- # 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud

# Load the dataset
data = pd.read_csv('dataset.csv')

# Sample data inspect:
# data.head()

# Idea 1: Author Influence Analysis - Visualizing Author Influence
def plot_author_influence():
    author_influence = data.groupby('AuthorNames-Deduped')['AminerCitationCount'].sum().reset_index()
    top_authors = author_influence.sort_values(by='AminerCitationCount', ascending=False).head(10)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='AminerCitationCount', y='AuthorNames-Deduped', data=top_authors, palette="viridis")
    plt.title('Top 10 Influential Authors by Aminer Citation Count')
    plt.xlabel('Total Aminer Citation Count')
    plt.ylabel('Author Names')
    plt.show()

plot_author_influence()

# Idea 2: Topic Modeling on Abstracts - Visualizing topics using Word Clouds
def plot_lda_wordclouds():
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    X = vectorizer.fit_transform(data['Abstract'].fillna(''))
    lda = LatentDirichletAllocation(n_components=5, random_state=42)
    lda.fit(X)
    
    for i, topic in enumerate(lda.components_):
        plt.figure(figsize=(8, 8))
        plt.imshow(WordCloud(background_color='white').fit_words({vectorizer.get_feature_names_out()[j]: topic[j] for j in topic.argsort()[:-15 - 1:-1]}))
        plt.axis('off')
        plt.title(f'Topic {i+1}')
        plt.show()

plot_lda_wordclouds()

# Idea 5: Trend Analysis of Research Topics - Visualizing Trends Over Years
def plot_research_trends():
    keywords_trend = data['AuthorKeywords'].str.get_dummies(sep=';').sum()
    keywords_trend = keywords_trend.sort_values(ascending=False).head(10)
    years = data['Year'].unique()
    
    trends_over_years = {key: [] for key in keywords_trend.index}
    for year in sorted(years):
        yearly_data = data[data['Year'] == year]
        keywords_yearly = yearly_data['AuthorKeywords'].str.get_dummies(sep=';').sum()
        for key in trends_over_years:
            trends_over_years[key].append(keywords_yearly.get(key, 0))
    
    plt.figure(figsize=(14, 8))
    for key in trends_over_years:
        plt.plot(sorted(years), trends_over_years[key], label=key)
    
    plt.title('Trend Analysis of Top Research Topics Over Years')
    plt.xlabel('Year')
    plt.ylabel('Keyword Appearance Count')
    plt.legend(title='Keywords')
    plt.show()

plot_research_trends()

# ---- NEW BLOCK ---- # 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Load dataset
df = pd.read_csv('dataset.csv')

# Preprocess: Ensure NaN handling
df.fillna({'Abstract': '', 'AuthorNames-Deduped': ''}, inplace=True)

# 1. Predictive Citation Analysis
plt.figure(figsize=(12, 7))
sns.scatterplot(x='AminerCitationCount', y='CitationCount_CrossRef', size='PubsCited_CrossRef', hue='AuthorKeywords', data=df, palette="viridis", alpha=0.6)
plt.title('Predictive Citation Analysis')
plt.xlabel('Aminer Citation Count')
plt.ylabel('CrossRef Citation Count')
plt.legend(title='Author Keywords')
plt.show()

# 2. Author Collaboration Networks
G = nx.Graph()
for index, row in df.iterrows():
    authors = row['AuthorNames-Deduped'].split(';')
    for author in authors:
        G.add_node(author.strip())
    for i, author1 in enumerate(authors):
        for author2 in authors[i+1:]:
            G.add_edge(author1.strip(), author2.strip())

plt.figure(figsize=(14, 14))
pos = nx.spring_layout(G, k=0.1)
nx.draw(G, pos, node_size=10, node_color="blue", edge_color="gray", with_labels=False, alpha=0.7)
plt.title('Author Collaboration Network')
plt.show()

# 3. Text Analysis for Research Trends
vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
X = vectorizer.fit_transform(df['Abstract'])

lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda.fit(X)

# Display the topic keywords
terms = vectorizer.get_feature_names_out()
for index, topic in enumerate(lda.components_):
    print(f'Topic {index}:')
    print([terms[i] for i in topic.argsort()[-10:]])

# 4. Impact of Graphics on Paper Success
plt.figure(figsize=(12, 7))
sns.boxplot(x='GraphicsReplicabilityStamp', y='Downloads_Xplore', data=df, palette='coolwarm')
plt.title('Impact of Graphics Replicability on Downloads')
plt.xlabel('Graphics Replicability Stamp')
plt.ylabel('Downloads in Xplore')
plt.show()

# 5. Award-winning Research Characteristics
plt.figure(figsize=(12, 7))
sns.countplot(x='PaperType', hue='Award', data=df, palette='Set2')
plt.title('Award-winning Research Characteristics by PaperType')
plt.xlabel('Paper Type')
plt.ylabel('Count')
plt.legend(title='Award Won')
plt.show()

