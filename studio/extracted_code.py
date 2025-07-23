'''
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
'''
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
'''
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

# ---- NEW BLOCK ---- # 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from wordcloud import WordCloud

# Load the dataset
df = pd.read_csv('dataset.csv')

# Fix potential issues with split by ensuring the column is strings
df['AuthorNames-Deduped'] = df['AuthorNames-Deduped'].astype(str)

# Idea 1: Conference Research Trends Over Time
plt.figure(figsize=(14, 6))
sns.countplot(data=df, x='Year', hue='Conference')
plt.title('Research Paper Publication Trends by Conference Over Time')
plt.xlabel('Year')
plt.ylabel('Number of Papers')
plt.xticks(rotation=45)
plt.legend(title='Conference', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Idea 2: Author Collaboration Network Graph
author_collab = nx.Graph()
for i, row in df.iterrows():
    authors = row['AuthorNames-Deduped'].split(',')
    for author in authors:
        author_collab.add_node(author.strip())
    for j in range(len(authors)):
        for k in range(j + 1, len(authors)):
            author_collab.add_edge(authors[j].strip(), authors[k].strip())

plt.figure(figsize=(12, 12))
pos = nx.spring_layout(author_collab, k=0.1)
nx.draw_networkx_edges(author_collab, pos, alpha=0.1)
nx.draw_networkx_nodes(author_collab, pos, node_size=20)
plt.title('Author Collaboration Network')
plt.axis('off')
plt.show()

# Idea 3: Word Cloud of Author Keywords
all_keywords = ' '.join(df['AuthorKeywords'].dropna().astype(str))
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_keywords)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.title('Word Cloud of Author Keywords')
plt.axis('off')
plt.show()

# Idea 4: Distribution of Paper Types
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='PaperType')
plt.title('Distribution of Paper Types')
plt.xlabel('Paper Type')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Idea 5: Award-winning Papers Based on Downloads
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Award', y='Downloads_Xplore')
plt.title('Distribution of Downloads for Award and Non-award Winning Papers')
plt.xlabel('Award')
plt.ylabel('Number of Downloads')
plt.tight_layout()
plt.show()

# Ensure there are no plotting errors
plt.close('all')



# ---- NEW BLOCK ---- # 
# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from wordcloud import WordCloud
import seaborn as sns

# Load the dataset
df = pd.read_csv('dataset.csv')

# Idea 1: Network Analysis of Authors and Affiliations
def plot_author_affiliation_network(df):
    G = nx.Graph()
    for _, row in df.iterrows():
        if pd.notnull(row['AuthorNames']) and pd.notnull(row['AuthorAffiliation']):
            authors = str(row['AuthorNames']).split(';')
            affiliations = str(row['AuthorAffiliation']).split(';')
            G.add_edges_from([(a, aff) for a in authors for aff in affiliations])
    plt.figure(figsize=(12, 12))
    nx.draw(G, with_labels=True, node_size=20, font_size=8)
    plt.title('Author and Affiliation Network')
    plt.show()

# Idea 2: Topic Modeling in Abstracts
def plot_topic_modeling(df, n_topics=5):
    abstracts = df['Abstract'].dropna().astype(str)
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf = tfidf_vectorizer.fit_transform(abstracts)
    nmf = NMF(n_components=n_topics, random_state=1)
    nmf.fit(tfidf)
    feature_names = tfidf_vectorizer.get_feature_names_out()
    fig, axes = plt.subplots(1, n_topics, figsize=(15, 5), sharex=True)
    for topic_idx, topic in enumerate(nmf.components_):
        wordcloud = WordCloud(background_color='white').generate_from_frequencies(
            dict(zip(feature_names, topic)))
        axes[topic_idx].imshow(wordcloud, interpolation='bilinear')
        axes[topic_idx].axis('off')
        axes[topic_idx].set_title(f'Topic {topic_idx}')
    plt.show()

# Idea 3: Predicting Citation Impact
def plot_citation_impact(df):
    sns.boxplot(data=df, x='Year', y='AminerCitationCount')
    plt.title('Citation Count by Year')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

# Idea 4: Trend Analysis of Research Keywords
def plot_keyword_trends(df):
    keywords = df['AuthorKeywords'].dropna().str.split(';').explode()
    trends = keywords.value_counts().head(10)
    trends.plot(kind='bar', figsize=(10, 6))
    plt.title('Top 10 Research Keywords')
    plt.xlabel('Keywords')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

# Idea 5: Awards and Recognition Modeling
def plot_award_downloads(df):
    sns.scatterplot(data=df, x='Downloads_Xplore', y='Award', hue='Award', palette='viridis')
    plt.title('Downloads vs Award Status')
    plt.xlabel('Downloads')
    plt.ylabel('Award (Yes/No)')
    plt.tight_layout()
    plt.show()

# Execute the visualizations
#plot_author_affiliation_network(df)
plot_topic_modeling(df, n_topics=5)
plot_citation_impact(df)
plot_keyword_trends(df)
plot_award_downloads(df)

# ---- NEW BLOCK ---- # 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('dataset.csv')

# Visualization 1: Citation and Download Analysis
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='AminerCitationCount', y='Downloads_Xplore', hue='PaperType')
plt.title('Aminer Citation Count vs Downloads')
plt.xlabel('Aminer Citation Count')
plt.ylabel('Downloads in Xplore')
plt.legend(title='Paper Type')
plt.show()

# Visualization 2: Conference Impact Evaluation
award_count = df.groupby('Conference')['Award'].sum().reset_index()
plt.figure(figsize=(12, 6))
sns.barplot(data=award_count, x='Conference', y='Award')
plt.xticks(rotation=45)
plt.title('Number of Awards per Conference')
plt.xlabel('Conference')
plt.ylabel('Number of Awards')
plt.show()

# Visualization 3: Author and Affiliation Network Analysis (heatmap of collaborations)
affiliation_network = df['AuthorAffiliation'].value_counts().head(10)
plt.figure(figsize=(12, 6))
sns.barplot(x=affiliation_network.index, y=affiliation_network.values)
plt.xticks(rotation=45)
plt.title('Top 10 Affiliation by Number of Papers')
plt.xlabel('Affiliation')
plt.ylabel('Number of Papers')
plt.show()

# Visualization 4: Abstract Text Mining for Trends (Word frequency)
from wordcloud import WordCloud

all_abstracts = ' '.join(df['Abstract'].dropna())
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_abstracts)

plt.figure(figsize=(15, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Wordcloud of Abstracts')
plt.show()

# Visualization 5: Historical Trends in Author Keywords
keywords_year = df.explode('AuthorKeywords').groupby(['Year', 'AuthorKeywords']).size().reset_index(name='counts')
top_keywords = keywords_year.groupby('AuthorKeywords')['counts'].sum().nlargest(10).index
filtered_keywords = keywords_year[keywords_year['AuthorKeywords'].isin(top_keywords)]

plt.figure(figsize=(12, 8))
sns.lineplot(data=filtered_keywords, x='Year', y='counts', hue='AuthorKeywords', marker="o")
plt.title('Trends of Top 10 Author Keywords Over Time')
plt.xlabel('Year')
plt.ylabel('Keyword Appearance Count')
plt.legend(title='Author Keywords')
plt.show()

# ---- NEW BLOCK ---- # 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from wordcloud import WordCloud
import numpy as np
from collections import Counter
import re

# Load dataset 
data = pd.read_csv('dataset.csv')

# Plot 1: Citation Prediction (Year vs. AminerCitationCount)
plt.figure(figsize=(10,6))
sns.lineplot(data=data, x='Year', y='AminerCitationCount', ci=None)
plt.title('Trend of Aminer Citation Count Over Years')
plt.xlabel('Year')
plt.ylabel('Aminer Citation Count')
plt.show()

# Plot 2: Authorship Network (Dummy Visualization for Cohesion)
# In a practical scenario, network plots might need libraries like NetworkX for detailed graphs.
author_affiliation_counts = data['AuthorAffiliation'].value_counts().head(10)
plt.figure(figsize=(8,6))
sns.barplot(y=author_affiliation_counts.index, x=author_affiliation_counts.values, palette="viridis")
plt.title('Top Author Affiliations by Paper Count')
plt.xlabel('Number of Papers')
plt.ylabel('Affiliation')
plt.show()

# Plot 3: Bayesian Analysis for Citations Prediction (Conference vs. CitationCount_CrossRef)
conference_citation_counts = data.groupby('Conference')['CitationCount_CrossRef'].mean().sort_values(ascending=False).head(10)
plt.figure(figsize=(10,6))
sns.barplot(y=conference_citation_counts.index, x=conference_citation_counts.values, palette="Blues_d")
plt.title('Mean Citation Count CrossRef by Conference')
plt.xlabel('Mean Citation Count (CrossRef)')
plt.ylabel('Conference')
plt.show()

# Plot 4: Word Cloud for Author Keywords
author_keywords = ' '.join(data['AuthorKeywords'].dropna())
word_cloud = WordCloud(width=1000, height=500, max_font_size=80, max_words=100, background_color='white').generate(author_keywords)
plt.figure(figsize=(15, 7.5))
plt.imshow(word_cloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud for Author Keywords')
plt.show()

# Plot 5: Temporal Trends in Research (AuthorKeywords over Years)
# Extract common keywords to analyze
keywords_sets = data['AuthorKeywords'].dropna().apply(lambda x: re.split('; |, |\*|\n', x))
all_keywords = [keyword for sublist in keywords_sets for keyword in sublist]
common_keywords = [item[0] for item in Counter(all_keywords).most_common(10)]

years = sorted(data['Year'].unique())
keyword_trends = pd.DataFrame(0, index=years, columns=common_keywords)

for idx, row in data.iterrows():
    if isinstance(row['AuthorKeywords'], str):
        year = row['Year']
        keywords = re.split('; |, |\*|\n', row['AuthorKeywords'])
        for keyword in set(keywords):
            if keyword in common_keywords:
                keyword_trends.at[year, keyword] += 1

plt.figure(figsize=(12,8))
sns.heatmap(keyword_trends.T, cmap="YlGnBu", cbar_kws={'label': 'Frequency in Year'})
plt.title('Temporal Trends in Research - Keyword Frequency per Year')
plt.xlabel('Year')
plt.ylabel('Keywords')
plt.show()

# Plot 6: Award vs Non-Award Papers comparison
award_summary = data.groupby('Award')['AminerCitationCount'].mean()
plt.figure(figsize=(6,6))
sns.barplot(x=award_summary.index, y=award_summary.values, palette='pastel')
plt.title('Average Aminer Citation Count: Awarded vs Non-Awarded Papers')
plt.xlabel('Awarded')
plt.ylabel('Average Aminer Citation Count')
plt.show()

# Plot 7: Graphics Replicability Stamp vs. Downloads
graphics_replicability = data.groupby('GraphicsReplicabilityStamp')['Downloads_Xplore'].mean()
plt.figure(figsize=(8,6))
sns.barplot(x=graphics_replicability.index, y=graphics_replicability.values, palette='bright')
plt.title('Average Downloads with Graphics Replicability Stamp')
plt.xlabel('Graphics Replicability Stamp')
plt.ylabel('Average Downloads (Xplore)')
plt.show()


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from wordcloud import WordCloud
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from textblob import TextBlob

# Load dataset
df = pd.read_csv('dataset.csv')

# Idea 1: Author Collaboration Networks
def plot_author_collaboration_network(df):
    # Creating a network graph
    G = nx.Graph()
    for authors in df['AuthorNames-Deduped'].fillna(''):
        author_list = authors.split(';')
        for author in author_list:
            G.add_node(author)
        for i in range(len(author_list)):
            for j in range(i+1, len(author_list)):
                G.add_edge(author_list[i], author_list[j])
    
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G, k=0.15, iterations=20)
    nx.draw(G, pos, node_size=20, with_labels=False, edge_color='gray', alpha=0.7)
    plt.title('Author Collaboration Network')
    plt.show()

plot_author_collaboration_network(df)

# Idea 2: Sentiment Analysis of Abstracts
def sentiment_analysis_of_abstracts(df):
    df['Sentiment'] = df['Abstract'].dropna().apply(lambda text: TextBlob(str(text)).sentiment.polarity)
    plt.figure(figsize=(12, 6))
    sns.histplot(df['Sentiment'].dropna(), bins=30, kde=True, color='skyblue')
    plt.title('Sentiment Analysis of Abstracts')
    plt.xlabel('Sentiment Polarity')
    plt.ylabel('Frequency')
    plt.show()

sentiment_analysis_of_abstracts(df)

# Idea 3: Temporal Trends in Research Topics
def plot_keyword_trends(df):
    df['Year'] = pd.to_datetime(df['Year'], format='%Y')
    all_keywords = df['AuthorKeywords'].dropna().str.get_dummies(sep=';').sum().reset_index()
    all_keywords.columns = ['Keyword', 'Count']
    most_frequent_keywords = all_keywords.nlargest(10, 'Count')['Keyword']

    trend_data = pd.DataFrame(index=df['Year'].dropna().unique())
    for keyword in most_frequent_keywords:
        trend_data[keyword] = df[df['AuthorKeywords'].str.contains(keyword, na=False)]['Year'].value_counts()

    trend_data = trend_data.fillna(0).sort_index()
    trend_data.plot(figsize=(14, 8), marker='o')
    plt.title('Trends of Research Topics Over Time')
    plt.xlabel('Year')
    plt.ylabel('Frequency')
    plt.show()

plot_keyword_trends(df)

# Idea 4: Interactive Data Visualizations using Plotly
def plot_interactive_visualization(df):
    fig = px.scatter(df, x='Downloads_Xplore', y='AminerCitationCount',
                     size='CitationCount_CrossRef', color='Year',
                     hover_name='Title', log_x=True, size_max=60,
                     title="Interactive Visualization: Downloads vs Citations")
    fig.show()

plot_interactive_visualization(df)

# Idea 5: Clustering Research Fields
def cluster_research_fields(df):
    keywords = df['AuthorKeywords'].dropna().str.get_dummies(sep=';')
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(keywords)
    
    kmeans = KMeans(n_clusters=5)
    kmeans.fit(reduced_data)
    df['Cluster'] = kmeans.labels_
    
    plt.figure(figsize=(10, 6))
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=df['Cluster'], cmap='viridis', alpha=0.6)
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.title('Clustering of Research Fields')
    plt.colorbar(label='Cluster')
    plt.show()

cluster_research_fields(df)

# New Block

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from wordcloud import WordCloud
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from textblob import TextBlob

# Load dataset
df = pd.read_csv('dataset.csv')

# Idea 1: Author Collaboration Networks
def plot_author_collaboration_network(df):
    # Creating a network graph
    G = nx.Graph()
    for authors in df['AuthorNames-Deduped'].fillna(''):
        author_list = authors.split(';')
        for author in author_list:
            G.add_node(author)
        for i in range(len(author_list)):
            for j in range(i+1, len(author_list)):
                G.add_edge(author_list[i], author_list[j])
    
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G, k=0.15, iterations=20)
    nx.draw(G, pos, node_size=20, with_labels=False, edge_color='gray', alpha=0.7)
    plt.title('Author Collaboration Network')
    plt.show()

plot_author_collaboration_network(df)

# Idea 2: Sentiment Analysis of Abstracts
def sentiment_analysis_of_abstracts(df):
    df['Sentiment'] = df['Abstract'].dropna().apply(lambda text: TextBlob(str(text)).sentiment.polarity)
    plt.figure(figsize=(12, 6))
    sns.histplot(df['Sentiment'].dropna(), bins=30, kde=True, color='skyblue')
    plt.title('Sentiment Analysis of Abstracts')
    plt.xlabel('Sentiment Polarity')
    plt.ylabel('Frequency')
    plt.show()

sentiment_analysis_of_abstracts(df)

# Idea 3: Temporal Trends in Research Topics
def plot_keyword_trends(df):
    df['Year'] = pd.to_datetime(df['Year'], format='%Y')
    all_keywords = df['AuthorKeywords'].dropna().str.get_dummies(sep=';').sum().reset_index()
    all_keywords.columns = ['Keyword', 'Count']
    most_frequent_keywords = all_keywords.nlargest(10, 'Count')['Keyword']

    trend_data = pd.DataFrame(index=df['Year'].dropna().unique())
    for keyword in most_frequent_keywords:
        trend_data[keyword] = df[df['AuthorKeywords'].str.contains(keyword, na=False)]['Year'].value_counts()

    trend_data = trend_data.fillna(0).sort_index()
    trend_data.plot(figsize=(14, 8), marker='o')
    plt.title('Trends of Research Topics Over Time')
    plt.xlabel('Year')
    plt.ylabel('Frequency')
    plt.show()

plot_keyword_trends(df)

# Idea 4: Interactive Data Visualizations using Plotly
def plot_interactive_visualization(df):
    fig = px.scatter(df, x='Downloads_Xplore', y='AminerCitationCount',
                     size='CitationCount_CrossRef', color='Year',
                     hover_name='Title', log_x=True, size_max=60,
                     title="Interactive Visualization: Downloads vs Citations")
    fig.show()

plot_interactive_visualization(df)

# Idea 5: Clustering Research Fields
def cluster_research_fields(df):
    keywords = df['AuthorKeywords'].dropna().str.get_dummies(sep=';')
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(keywords)
    
    kmeans = KMeans(n_clusters=5)
    kmeans.fit(reduced_data)
    df['Cluster'] = kmeans.labels_
    
    plt.figure(figsize=(10, 6))
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=df['Cluster'], cmap='viridis', alpha=0.6)
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.title('Clustering of Research Fields')
    plt.colorbar(label='Cluster')
    plt.show()

cluster_research_fields(df)# ---- NEW BLOCK ---- # 
# ---- NEW BLOCK ---- # 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px

# Load the dataset
df = pd.read_csv('dataset.csv')

# 1. Exploratory Data Analysis on Publication Trends

plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='PaperType', order=df['PaperType'].value_counts().index)
plt.title('Publication Trends by Paper Type')
plt.xlabel('Paper Type')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 2. Citation Network Analysis

yearly_citations = df.groupby('Year')['CitationCount_CrossRef'].sum()

plt.figure(figsize=(10, 6))
plt.plot(yearly_citations.index, yearly_citations.values, marker='o')
plt.title('Trend of Citations Over the Years')
plt.xlabel('Year')
plt.ylabel('Total Citations')
plt.grid()
plt.tight_layout()
plt.show()

# 3. Text Mining and Topic Modeling (using PCA on keywords for illustrative purposes)

author_keywords = df['AuthorKeywords'].dropna().str.split(';')
keyword_matrix = author_keywords.str.join(sep=' ').str.get_dummies(sep=' ')
pca = PCA(n_components=2)
pca_result = pca.fit_transform(keyword_matrix)

plt.figure(figsize=(10, 6))
plt.scatter(pca_result[:, 0], pca_result[:, 1])
plt.title('PCA of Author Keywords')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.grid()
plt.tight_layout()
plt.show()

# 4. Predictive Analysis for Award-Winning Papers

award_papers = df[df['Award'] == True]
non_award_papers = df[df['Award'] == False]

plt.figure(figsize=(10, 6))
sns.kdeplot(award_papers['Downloads_Xplore'], label='Awarded Papers', shade=True)
sns.kdeplot(non_award_papers['Downloads_Xplore'], label='Non-Awarded Papers', shade=True)
plt.title('Downloads Distribution for Award vs Non-Award Papers')
plt.xlabel('Downloads_Xplore')
plt.ylabel('Density')
plt.legend()
plt.tight_layout()
plt.show()

# 5. Bibliometric Analysis Using Clustering (KMeans on PCA of author keywords)

kmeans = KMeans(n_clusters=3, random_state=0).fit(pca_result)

plt.figure(figsize=(10, 6))
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=kmeans.labels_)
plt.title('KMeans Clustering on PCA of Author Keywords')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.grid()
plt.tight_layout()
plt.show()

# 6. Network Dynamics of Author Collaborations Over the Years

author_collaborations = df['AuthorNames-Deduped'].dropna().apply(lambda x: x.split(';'))
author_collabs_flat = [item for sublist in author_collaborations for item in sublist]

author_df = pd.DataFrame(author_collabs_flat, columns=['Author'])
author_year_count = author_df.value_counts().groupby(df['Year']).count()

plt.figure(figsize=(10, 6))
plt.plot(author_year_count.index, author_year_count.values, marker='o')
plt.title('Author Collaborations Over the Years')
plt.xlabel('Year')
plt.ylabel('Number of Collaborations')
plt.grid()
plt.tight_layout()
plt.show()

# ---- NEW BLOCK ---- # 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from wordcloud import WordCloud
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Load the dataset
data = pd.read_csv('dataset.csv')

# Handle missing values in the target column
data['CitationCount_CrossRef'] = data['CitationCount_CrossRef'].fillna(data['CitationCount_CrossRef'].mean())

# --- Predictive Analytics for Conference Impact Assessment ---
features = ['Award', 'PaperType']
target = 'CitationCount_CrossRef'

# Encode categorical variables
data['Award'] = data['Award'].astype('category').cat.codes
data['PaperType'] = data['PaperType'].astype('category').cat.codes

X = pd.get_dummies(data[features], drop_first=True)
y = data[target]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fit the Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel('Actual Citation Counts')
plt.ylabel('Predicted Citation Counts')
plt.title(f'Predicted vs Actual Citation Counts\nMSE: {mse:.2f}, R^2: {r2:.2f}')
plt.show()

# --- Analysis of Research Trends via Bibliometrics ---
years = data['Year'].value_counts().sort_index()
plt.figure(figsize=(12, 6))
sns.lineplot(data=years)
plt.title('Research Publication Trends Over Years')
plt.xlabel('Year')
plt.ylabel('Number of Publications')
plt.show()

# --- Textual Analysis of Abstracts and Titles ---
abstracts = ' '.join(data['Abstract'].dropna().tolist())
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(abstracts)
plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Abstract Themes')
plt.show()

# --- Graphics and Replicability Assessment ---
plt.figure(figsize=(10, 6))
sns.violinplot(x='GraphicsReplicabilityStamp', y='CitationCount_CrossRef', data=data)
plt.title('Impact of Graphics Replicability Stamp on Citations')
plt.show()

# --- Network Analysis of Author Collaboration ---
G = nx.Graph()
author_connections = data['AuthorNames-Deduped'].dropna().str.split('; ')

for authors in author_connections:
    for i in range(len(authors)):
        for j in range(i + 1, len(authors)):
            G.add_edge(authors[i].strip(), authors[j].strip())

plt.figure(figsize=(10, 10))
pos = nx.spring_layout(G, k=0.3)
nx.draw(G, pos, with_labels=False, node_size=20, font_size=8, width=0.5)
plt.title("Author Collaboration Network")
plt.show()

# --- Enhanced Tabular Data Analysis with Multi-representation DeepInsight ---
plt.title('Placeholder for Multi-representation DeepInsight Analysis')
plt.text(0.5, 0.5, 'DeepInsight Analysis Here', horizontalalignment='center', verticalalignment='center')
plt.axis('off')
plt.show()
'''

# New Code Block

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv('dataset.csv')

# Ensure that empty or non-string AuthorNames-Deduped field do not cause a split error
df['AuthorNames-Deduped'] = df['AuthorNames-Deduped'].fillna('')

# 1. Temporal Citation Trend Analysis
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='Year', y='AminerCitationCount', label='Aminer')
sns.lineplot(data=df, x='Year', y='CitationCount_CrossRef', label='CrossRef')
plt.title('Temporal Citation Trend Analysis')
plt.ylabel('Citation Count')
plt.xlabel('Year')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 2. Author Collaboration Network Analysis
G = nx.Graph()
for idx, row in df.iterrows():
    authors = row['AuthorNames-Deduped'].split(';')
    for a in authors:
        G.add_node(a)
    for i in range(len(authors)):
        for j in range(i + 1, len(authors)):
            G.add_edge(authors[i], authors[j])
plt.figure(figsize=(12, 12))
nx.draw(G, with_labels=True, node_color='lightblue', edge_color='grey', node_size=25, font_size=8)
plt.title('Author Collaboration Network')
plt.tight_layout()
plt.show()

# 3. Predictive Model for Award-Winning Papers
df_encoded = df.copy()
df_encoded['Award'] = df_encoded['Award'].fillna(0)
X = df_encoded[['Downloads_Xplore', 'PubsCited_CrossRef']].fillna(0)
y = df_encoded['Award'].apply(lambda x: 1 if x else 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# 4. Graphics Influence Study
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='GraphicsReplicabilityStamp', y='Downloads_Xplore', palette='Set2')
plt.title('Graphics Influence on Downloads')
plt.tight_layout()
plt.show()

# 5. Keyword Evolution Mapping
# Use safe fillna value to prevent split issues
df['AuthorKeywords'] = df['AuthorKeywords'].fillna('')

pivot_table = df.pivot_table(index='AuthorKeywords', columns='Year', aggfunc='size').fillna(0)
plt.figure(figsize=(12, 8))
sns.heatmap(pivot_table, cmap="YlGnBu", linewidths=.5)
plt.title('Keyword Evolution Mapping')
plt.tight_layout()
plt.show()

# 6. Citation and Internal Reference Correlation
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='InternalReferences', y='CitationCount_CrossRef', hue='Year', palette='viridis')
plt.title('Correlation between Internal References and Citation Count')
plt.tight_layout()
plt.show()

# 7. Author Diversity Impact on Research Quality
def count_unique_affiliations(affiliations):
    try:
        return len(set(affiliations.split(';')))
    except AttributeError:
        return 0

plt.figure(figsize=(10, 6))
author_diversity = df['AuthorAffiliation'].apply(count_unique_affiliations)
plt.scatter(author_diversity, df['AminerCitationCount'], alpha=0.5)
plt.title('Author Diversity vs. Citation Count')
plt.xlabel('Author Diversity (Unique Affiliations)')
plt.ylabel('Aminer Citation Count')
plt.tight_layout()
plt.show()
 # ---- NEW BLOCK ---- # 
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('dataset.csv')

# Extract relevant columns
authors_affiliations = data[['AuthorNames-Deduped', 'AuthorAffiliation']]

# Create a collaboration network
G = nx.Graph()

for _, row in authors_affiliations.iterrows():
    authors = row['AuthorNames-Deduped'].split(';')
    affiliations = row['AuthorAffiliation'].split(';')
    for i, author in enumerate(authors):
        author_affiliation = f"{author} ({affiliations[i]})"
        for coauthor in authors:
            if author != coauthor:
                coauthor_affiliation = f"{coauthor} ({affiliations[i]})"
                G.add_edge(author_affiliation, coauthor_affiliation)

# Plot the network
plt.figure(figsize=(12, 8))
nx.draw(G, with_labels=True, node_size=50, font_size=8)
plt.title('Collaboration Network')
plt.show()

import seaborn as sns

# Aggregate data by year and conference
conference_trends = data.groupby(['Year', 'Conference']).size().reset_index(name='PaperCount')

# Plot the trend over time
plt.figure(figsize=(14, 6))
sns.lineplot(data=conference_trends, x='Year', y='PaperCount', hue='Conference', marker='o')
plt.title('Conference Trends Over Time')
plt.xlabel('Year')
plt.ylabel('Number of Papers')
plt.legend(title='Conference')
plt.show()

# Analyze impact of awards on citation counts
award_data = data[['Award', 'AminerCitationCount', 'CitationCount_CrossRef']].dropna()

# Boxplot for citations based on awards
plt.figure(figsize=(10, 6))
sns.boxplot(x='Award', y='AminerCitationCount', data=award_data)
plt.title('Impact of Awards on Aminer Citation Count')
plt.xlabel('Award Received')
plt.ylabel('Aminer Citation Count')
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Prepare predictive model dataset
features = data[['Year', 'PaperType', 'GraphicsReplicabilityStamp']]
features = pd.get_dummies(features, drop_first=True)
target = data['Downloads_Xplore'].fillna(0)

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Train a Random Forest model
model = RandomForestRegressor()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')

from wordcloud import WordCloud

# Extract keywords by year
keywords_by_year = data[['Year', 'AuthorKeywords']].dropna()

# Plot a word cloud for each year
unique_years = keywords_by_year['Year'].unique()

plt.figure(figsize=(20, 15))
for i, year in enumerate(unique_years, 1):
    year_keywords = ' '.join(keywords_by_year[keywords_by_year['Year'] == year]['AuthorKeywords'].tolist())
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(year_keywords)
    
    plt.subplot(len(unique_years)//2, 2, i)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Keywords in {year}')
plt.tight_layout()
plt.show()

# ---- NEW BLOCK ---- # 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import networkx as nx
import numpy as np

# Load the dataset
df = pd.read_csv('dataset.csv')

# Idea 1: Citation Network Analysis
plt.figure(figsize=(10, 6))
citation_matrix = np.zeros((len(df), len(df)))
# Assume 'InternalReferences' column contains IDs that refer to other papers in the same dataset
for i, refs in enumerate(df['InternalReferences'].fillna('')):
    for ref in refs.split(','):
        if ref and int(ref) < len(df):
            citation_matrix[i, int(ref)] = 1
G = nx.from_numpy_matrix(citation_matrix, create_using=nx.DiGraph)
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, node_size=20, node_color='blue', with_labels=False, arrowsize=8)
plt.title('Citation Network Analysis')
plt.show()

# Idea 2: Predictive Modeling for Conference Awards
award_features = ['Year', 'Conference', 'PaperType', 'Downloads_Xplore', 'GraphicsReplicabilityStamp']
sns.pairplot(df, vars=award_features, hue='Award')
plt.suptitle('Predictive Modeling for Conference Awards', y=1.02)
plt.show()

# Idea 3: Temporal Analysis of Research Impact
plt.figure(figsize=(10, 6))
df.groupby('Year')['CitationCount_CrossRef'].mean().plot()
plt.title('Temporal Analysis of Research Impact')
plt.xlabel('Year')
plt.ylabel('Average Citation Count')
plt.show()

# Idea 4: Topic Modeling on Abstracts and Keywords
abstracts = df['Abstract'].fillna('')
tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
tfidf_matrix = tfidf.fit_transform(abstracts)
nmf = NMF(n_components=5, random_state=42)
nmf_topics = nmf.fit_transform(tfidf_matrix)
plt.figure(figsize=(10, 6))
for topic_idx, topic in enumerate(nmf.components_):
    plt.bar([f'Topic {i}' for i in range(len(topic))], topic, label=f'Topic {topic_idx}')
plt.title('Topic Modeling on Abstracts and Keywords')
plt.xticks(rotation=45)
plt.ylabel('Weight')
plt.legend()
plt.show()

# Idea 5: Author Collaboration Patterns
authors = df['AuthorNames-Deduped'].apply(lambda x: x.split(','))
collab_edges = [(author, other_author) for row in authors for author in row for other_author in row if author != other_author]
G_authors = nx.Graph()
G_authors.add_edges_from(set(collab_edges))
plt.figure(figsize=(10, 6))
pos = nx.spring_layout(G_authors, seed=42)
nx.draw(G_authors, pos, node_size=20, node_color='green', with_labels=False)
plt.title('Author Collaboration Patterns')
plt.show()

# Idea 6: Download and Citation Correlation Analysis
plt.figure(figsize=(10, 6))
sns.regplot(x='Downloads_Xplore', y='CitationCount_CrossRef', data=df)
plt.title('Download and Citation Correlation Analysis')
plt.xlabel('Downloads in Xplore')
plt.ylabel('Citation Count in CrossRef')
plt.show()

# ---- NEW BLOCK ---- # 
# ---- NEW BLOCK ---- # 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import networkx as nx

# Load the dataset
df = pd.read_csv('dataset.csv')

# Idea 1: Advanced Topic Modeling
def topic_modeling():
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    X = vectorizer.fit_transform(df['Abstract'].fillna(''))
    
    lda = LatentDirichletAllocation(n_components=5, random_state=42)
    lda.fit(X)
    
    feature_names = vectorizer.get_feature_names_out()
    for topic_idx, topic in enumerate(lda.components_):
        print(f"Topic {topic_idx}: ", [feature_names[i] for i in topic.argsort()[:-6:-1]])

topic_modeling()

# Idea 2: Author Network Analysis
def author_network():
    G = nx.Graph()

    for index, row in df.iterrows():
        authors = str(row['AuthorNames']).split(';')
        for i in range(len(authors)):
            for j in range(i + 1, len(authors)):
                G.add_edge(authors[i].strip(), authors[j].strip())

    plt.figure(figsize=(12, 12))
    nx.draw(G, node_size=5, with_labels=False, edge_color="b", alpha=0.2)
    plt.title("Co-authorship Network")
    plt.show()

author_network()

# Idea 3: Citation Impact Prediction
def citation_prediction():
    sns.regplot(x='Downloads_Xplore', y='AminerCitationCount', data=df)
    plt.title('Downloads vs. Citation Count')
    plt.xlabel('Downloads')
    plt.ylabel('Aminer Citation Count')
    plt.show()

citation_prediction()

# Idea 4: Sequential Pattern Analysis
def sequential_pattern():
    citations_year = df.groupby('Year')['InternalReferences'].count()
    plt.plot(citations_year.index, citations_year.values, marker='o')
    plt.title('Internal References Over Years')
    plt.xlabel('Year')
    plt.ylabel('Number of Internal References')
    plt.grid(True)
    plt.show()

sequential_pattern()

# Idea 5: Topic Evolution Mapping
def topic_evolution():
    keywords_by_year = df.groupby('Year')['AuthorKeywords'].apply(lambda x: ';'.join(x.dropna()))
    sns.barplot(x=keywords_by_year.index, y=keywords_by_year.apply(lambda x: len(set(x.split(';')))))
    plt.title('Unique Keywords by Year')
    plt.xlabel('Year')
    plt.ylabel('Unique Keywords Count')
    plt.xticks(rotation=45)
    plt.show()

topic_evolution()

# Idea 6: Replicability and Award Analysis
def replicability_award():
    sns.boxplot(x='Award', y='CitationCount_CrossRef', hue='GraphicsReplicabilityStamp', data=df)
    plt.title('Citation Count by Award and Graphics Replicability Stamp')
    plt.xlabel('Award Status')
    plt.ylabel('CrossRef Citation Count')
    plt.show()

replicability_award()

# ---- NEW BLOCK ---- # 
# ---- NEW BLOCK ---- # 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# Load the dataset
df = pd.read_csv('dataset.csv')

# Preprocessing
df.fillna(0, inplace=True)
le = LabelEncoder()
df['PaperType'] = le.fit_transform(df['PaperType'])
df['Conference'] = le.fit_transform(df['Conference'])

# Safely convert 'Award' to binary, handling potential non-integer values
def safe_convert_to_binary(value):
    try:
        return int(float(value)) if float(value) > 0 else 0
    except ValueError:
        return 0

df['Award'] = df['Award'].apply(safe_convert_to_binary)

# Define features and target
features = ['PaperType', 'Downloads_Xplore', 'Year', 'Conference']
X = df[features]
y = df['Award']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Predictions and evaluation
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# ---- NEW BLOCK ---- # 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingRegressor
from sklearn.metrics import classification_report, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

# Load the dataset
df = pd.read_csv('dataset.csv')

# Handle missing values in the entire dataset
imputer = SimpleImputer(strategy='most_frequent')
df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Convert appropriate columns back to their numeric types if they were object
df['AminerCitationCount'] = pd.to_numeric(df['AminerCitationCount'], errors='coerce')
df['CitationCount_CrossRef'] = pd.to_numeric(df['CitationCount_CrossRef'], errors='coerce')
df['Downloads_Xplore'] = pd.to_numeric(df['Downloads_Xplore'], errors='coerce')
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')

# Enhanced Trend Analysis on Citation Counts
def trend_analysis_plot():
    plt.figure(figsize=(14, 7))
    sns.lineplot(data=df, x='Year', y='AminerCitationCount', hue='Conference', style='PaperType', markers=True, dashes=False)
    plt.title('Trend Analysis: Citation Counts Over Years')
    plt.ylabel('Citation Count')
    plt.grid(True)
    plt.show()

# Machine Learning to Predict Awards
def machine_learning_prediction():
    features = df[['PaperType', 'AuthorKeywords', 'InternalReferences', 'Downloads_Xplore', 'Year', 'Conference']]
    target = df['Award'].apply(lambda x: 1 if x == 'Yes' else 0)
    features = pd.get_dummies(features)

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(classification_report(y_test, predictions))

# Regression Analysis for Download Prediction
def download_prediction_regression():
    features = df[['AminerCitationCount', 'GraphicsReplicabilityStamp', 'InternalReferences', 'Year', 'Conference']]
    features = pd.get_dummies(features)
    target = df['Downloads_Xplore']

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    model = HistGradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(f'Regression R2 Score: {r2_score(y_test, predictions)}')

# Keyword Trend Analysis
def keyword_trend_analysis():
    vectorizer = CountVectorizer()
    keyword_matrix = vectorizer.fit_transform(df['AuthorKeywords'].fillna(''))
    keyword_freq = np.asarray(keyword_matrix.sum(axis=0)).flatten()
    keywords = vectorizer.get_feature_names_out()

    keyword_df = pd.DataFrame(list(zip(keywords, keyword_freq)), columns=['Keyword', 'Frequency'])
    keyword_df = keyword_df.sort_values(by='Frequency', ascending=False).head(20)

    plt.figure(figsize=(14, 7))
    sns.barplot(data=keyword_df, x='Frequency', y='Keyword', palette='viridis')
    plt.title('Top 20 Keywords by Frequency')
    plt.grid(True)
    plt.show()

# Impact of Awards on Citations
def award_impact_analysis():
    plt.figure(figsize=(14, 7))
    sns.boxplot(x='Award', y='CitationCount_CrossRef', data=df)
    plt.title('Impact of Awards on Citations')
    plt.ylabel('Citation Count')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(14, 7))
    sns.boxplot(x='Award', y='Downloads_Xplore', data=df)
    plt.title('Impact of Awards on Downloads')
    plt.ylabel('Downloads')
    plt.grid(True)
    plt.show()

# Execute functions to create plots
trend_analysis_plot()
machine_learning_prediction()
download_prediction_regression()
keyword_trend_analysis()
award_impact_analysis()

# ---- NEW BLOCK ---- # 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from textblob import TextBlob

# Load dataset
data = pd.read_csv('dataset.csv')

# Handle NaN values
data.fillna(method='ffill', inplace=True)  # Forward fill for simplicity; could also use 'bfill' or other strategies

### 1. Enhanced Trend Analysis on Citation Counts
# Group by Year and get mean citation counts
citation_trend = data.groupby('Year')[['AminerCitationCount', 'CitationCount_CrossRef']].mean().reset_index()

# Plot citation trends
plt.figure(figsize=(10, 5))
sns.lineplot(data=citation_trend, x='Year', y='AminerCitationCount', label='AminerCitationCount')
sns.lineplot(data=citation_trend, x='Year', y='CitationCount_CrossRef', label='CitationCount_CrossRef')

plt.title('Citation Trends Over the Years')
plt.xlabel('Year')
plt.ylabel('Average Citation Count')
plt.legend()
plt.show()

### 2. Machine Learning to Predict Awards
# Define features and target variable
features = data[['PaperType', 'Downloads_Xplore', 'Year', 'Conference']]
target = data['Award']

# Drop rows with NaN target values to prevent errors
features, target = features.dropna(), target.dropna()

# Convert categorical features to dummy variables
features = pd.get_dummies(features)

# Split data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict and print report
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))

### 3. Conference Impact Assessment
# Group data by Conference and Year, calculate mean citation counts
conference_impact = data.groupby(['Conference', 'Year'])[['AminerCitationCount', 'CitationCount_CrossRef']].mean().reset_index()

# Plot
plt.figure(figsize=(12, 6))
sns.lineplot(data=conference_impact, x='Year', y='AminerCitationCount', hue='Conference', legend=None, estimator=None)
sns.lineplot(data=conference_impact, x='Year', y='CitationCount_CrossRef', hue='Conference', legend=None, estimator=None)

plt.title('Conference Impact Over the Years')
plt.xlabel('Year')
plt.ylabel('Mean Citation Count')
plt.show()

### 4. Abstract Sentiment Analysis
# Compute sentiment polarity
data['Sentiment'] = data['Abstract'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)

# Analyze correlation between sentiment and CitationCount_CrossRef
correlation = data[['Sentiment', 'CitationCount_CrossRef']].corr().iloc[0, 1]
print(f"Correlation between sentiment and CitationCount_CrossRef: {correlation}")

### 5. Impact of Awards on Citations
# Compare citation counts for awarded vs non-awarded papers
award_impact = data.groupby('Award')[['CitationCount_CrossRef', 'Downloads_Xplore']].mean().reset_index()

# Plot
plt.figure(figsize=(7, 5))
award_impact.plot(x='Award', y=['CitationCount_CrossRef', 'Downloads_Xplore'], kind='bar')
plt.title('Impact of Awards on Citations and Downloads')
plt.ylabel('Average Count')
plt.show()

# --- New Block --- # 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import FactorAnalysis
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from lifelines import KaplanMeierFitter
import numpy as np
from wordcloud import WordCloud
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import warnings

# Ignore warnings
warnings.filterwarnings("ignore")

# Load the dataset
data = pd.read_csv('dataset.csv')

# Replace problematic strings in numeric fields before converting or analysis
data['Year'] = pd.to_numeric(data['Year'], errors='coerce')
data['AminerCitationCount'] = pd.to_numeric(data['AminerCitationCount'], errors='coerce')
data['CitationCount_CrossRef'] = pd.to_numeric(data['CitationCount_CrossRef'], errors='coerce')
data['Downloads_Xplore'] = pd.to_numeric(data['Downloads_Xplore'], errors='coerce')
data['GraphicsReplicabilityStamp'] = pd.to_numeric(data['GraphicsReplicabilityStamp'], errors='coerce')

# Handle NaN values after conversion
data.fillna({
    'Downloads_Xplore': 0, 
    'CitationCount_CrossRef': 0, 
    'AminerCitationCount': 0, 
    'Year': data['Year'].mean(), 
    'GraphicsReplicabilityStamp': 0,
    'Award': 0,
    'AuthorAffiliation': '', 
    'AuthorKeywords': '', 
    'Abstract': '', 
    'PaperType': 'unknown'
}, inplace=True)

# Ensure the 'Award' column is a binary category
data['Award'] = data['Award'].apply(lambda x: 1 if x else 0)

# 1. Enhanced Trend Analysis on Citation Counts
plt.figure(figsize=(10, 6))
sns.lineplot(data=data, x='Year', y='AminerCitationCount', hue='Conference', style='PaperType')
plt.title('Trend Analysis on Citation Counts by Year, Conference, and PaperType')
plt.show()
'''
# 2. Network Analysis of Authors and Affiliations
plt.figure(figsize=(12, 8))
G = nx.Graph()
author_aff_pairs = set()
for index, row in data.iterrows():
    authors = row['AuthorNames-Deduped'].split(';') if pd.notna(row['AuthorNames-Deduped']) else []
    affiliations = row['AuthorAffiliation'].split(';') if pd.notna(row['AuthorAffiliation']) else []
    for author in authors:
        for affiliation in affiliations:
            author_aff_pairs.add((author.strip(), affiliation.strip()))
G.add_edges_from(author_aff_pairs)
pos = nx.spring_layout(G)
nx.draw(G, pos, node_size=20, font_size=8, with_labels=False)
plt.title('Network Analysis of Authors and Affiliations')
plt.show()
'''

# 3. Machine Learning to Predict Awards
features = data[['PaperType', 'AuthorKeywords', 'InternalReferences', 'Downloads_Xplore', 'Year', 'Conference']]
target = data['Award']
features = pd.get_dummies(features)

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)
model = tree.DecisionTreeClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# 4. Regression Analysis for Download Prediction
reg_features = data[['AminerCitationCount', 'GraphicsReplicabilityStamp', 'InternalReferences', 'Year', 'Conference']]
reg_features = pd.get_dummies(reg_features)
reg = LinearRegression().fit(reg_features, data['Downloads_Xplore'])
print(f'Regression coefficients: {reg.coef_}')

# 5. Factor Analysis of Research Topics
fa = FactorAnalysis(n_components=2, random_state=0)
abstract_keywords = data['Abstract'] + ' ' + data['AuthorKeywords']
wordcloud = WordCloud().generate(' '.join(abstract_keywords))

plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Factor Analysis: WordCloud of Research Topics')
plt.show()

# 6. Survival Analysis on Paper Popularity
kmf = KaplanMeierFitter()
T = data['Year']
E = (data['Downloads_Xplore'] + data['CitationCount_CrossRef']) > 0
kmf.fit(T, event_observed=E)

plt.figure(figsize=(10, 6))
kmf.plot_survival_function()
plt.title('Survival Analysis on Paper Popularity')
plt.xlabel('Year')
plt.ylabel('Survival Probability')
plt.show()

# 7. Collaborative Network Analysis
plt.figure(figsize=(12, 8))
author_collab = data['AuthorNames-Deduped'].apply(lambda x: x.split(';') if pd.notna(x) else [])
collab_pairs = [(a.strip(), b.strip()) for sublist in author_collab for a in sublist for b in sublist if a != b]
G = nx.Graph()
G.add_edges_from(collab_pairs)
pos = nx.spring_layout(G)
nx.draw(G, pos, node_size=20, font_size=8, with_labels=False)
plt.title('Collaborative Network Analysis')
plt.show()

# 8. Conference Impact Assessment
plt.figure(figsize=(12, 8))
sns.barplot(data=data, x='Conference', y='CitationCount_CrossRef', hue='PaperType')
plt.title('Conference Impact Assessment on Citations')
plt.xticks(rotation=45)
plt.show()

# 9. Keyword Trend Analysis
author_keywords = data[['AuthorKeywords', 'Year']]
author_keywords_melted = author_keywords['AuthorKeywords'].str.split(';', expand=True).stack().reset_index(drop=True)
author_keywords_series = pd.Series(author_keywords_melted).value_counts()

wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(author_keywords_series.to_dict())
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Keyword Trend Analysis: Word Cloud')
plt.show()

# 10. Citation Prediction Model
features = data[['Year', 'PaperType', 'AuthorAffiliation', 'Downloads_Xplore']]
features = pd.get_dummies(features)
target = data['CitationCount_CrossRef']

X_train, X_test, y_train, y_test = train_test_split(features.fillna(0), target, test_size=0.3, random_state=42)
regressor = LinearRegression()
regressor.fit(X_train, y_train)
print(f'Regression coefficients for Citation Prediction: {regressor.coef_}')

# 11. Factor Analysis on Author Affiliations
affiliations = data['AuthorAffiliation'].str.split(';', expand=True).stack().reset_index(drop=True)
affiliations_series = affiliations.value_counts()

fa = FactorAnalysis(n_components=2, random_state=0)
scaled_affiliations = StandardScaler().fit_transform(affiliations_series.values.reshape(-1, 1))

fa.fit(scaled_affiliations)
print('Factor Analysis on Author Affiliations:', fa.components_)

# 12. Meta-analysis of Graphics Replicability
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='GraphicsReplicabilityStamp', y='CitationCount_CrossRef')
plt.title('Meta-analysis of Graphics Replicability with Citation Count')
plt.show()

# 13. Abstract Sentiment Analysis
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()
abstract_sentiments = data['Abstract'].apply(lambda x: sia.polarity_scores(x)['compound'])

plt.figure(figsize=(10, 6))
sns.histplot(abstract_sentiments, bins=20)
plt.title('Sentiment Analysis of Abstracts')
plt.xlabel('Sentiment Score')
plt.ylabel('Frequency')
plt.show()

# 14. Author Influence Over Time
author_citations_over_time = data.groupby('AuthorNames-Deduped')['AminerCitationCount'].sum().sort_values(ascending=False).head(10)
author_citations_over_time.plot(kind='bar', figsize=(12, 6))
plt.title('Top Authors Influence Over Time')
plt.xlabel('Authors')
plt.ylabel('Aminer Citation Count')
plt.show()

# 15. Research Content Clustering
abstracts_scaled = StandardScaler().fit_transform(data['Abstract'].apply(lambda x: len(x) if x else 0).values.reshape(-1, 1))
kmeans = KMeans(n_clusters=5, random_state=0).fit(abstracts_scaled)
data['Cluster'] = kmeans.labels_

plt.figure(figsize=(10, 6))
sns.countplot(data=data, x='Cluster', hue='Conference')
plt.title('Research Content Clustering')
plt.show()

# 16. Impact of Awards on Citations
plt.figure(figsize=(10, 6))
sns.boxplot(data=data, x='Award', y='CitationCount_CrossRef')
plt.title('Impact of Awards on Citation Count')
plt.show()

# ---- NEW BLOCK ---- # 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import statsmodels.api as sm
from sklearn.feature_extraction.text import CountVectorizer

# Load the dataset
df = pd.read_csv('dataset.csv')

# Ensure 'Award' is binary for classification
df['Award'] = df['Award'].fillna(0)  # Fill NaNs with 0s
df['Award'] = df['Award'].apply(lambda x: 1 if x else 0)  # Convert to binary (0 or 1)

# 1. Enhanced Trend Analysis on Citation Counts
def trend_analysis(df):
    trend_data = df.groupby(['Year', 'Conference', 'PaperType']).agg({
        'AminerCitationCount': 'mean',
        'CitationCount_CrossRef': 'mean'
    }).reset_index()
    sns.lineplot(data=trend_data, x='Year', y='AminerCitationCount', hue='Conference', style='PaperType')
    plt.title('Trend Analysis on Citation Counts')
    plt.xlabel('Year')
    plt.ylabel('Average Citation Count')
    plt.show()

# 2. Machine Learning to Predict Awards
def award_prediction(df):
    features = ['PaperType', 'AuthorKeywords', 'InternalReferences', 'Downloads_Xplore', 'Year', 'Conference']
    X = pd.get_dummies(df[features])
    y = df['Award'].astype(int)  # Ensures 'Award' is int type for classification
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(f'Accuracy: {accuracy_score(y_test, predictions)}')
    print(confusion_matrix(y_test, predictions))

# 3. Regression Analysis for Download Prediction
def download_prediction(df):
    df['Downloads_Xplore'] = df['Downloads_Xplore'].fillna(df['Downloads_Xplore'].mean())  # Fill missing values
    features = ['AminerCitationCount', 'GraphicsReplicabilityStamp', 'InternalReferences', 'Year', 'Conference']
    
    # Convert all features to dummy variables to ensure numeric input
    X = pd.get_dummies(df[features])
    y = df['Downloads_Xplore']
    
    # Ensure valid numeric types for regression
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0).astype(float)
    y = pd.to_numeric(y, errors='coerce').fillna(0).astype(float)

    # Adding constant for intercept
    X = sm.add_constant(X)
    
    model = sm.OLS(y, X).fit()
    print(model.summary())

# 4. Keyword Trend Analysis
def keyword_trend_analysis(df):
    df['AuthorKeywords'] = df['AuthorKeywords'].fillna('')  # Handle missing values
    cv = CountVectorizer()
    keyword_counts = cv.fit_transform(df['AuthorKeywords'])
    keyword_trends = pd.DataFrame(keyword_counts.toarray(), columns=cv.get_feature_names_out())
    keyword_trends['Year'] = df['Year']
    yearly_keyword_counts = keyword_trends.groupby('Year').sum()
    yearly_keyword_counts.T.plot(kind='line', stacked=True, figsize=(14,7))
    plt.title('Keyword Trend Analysis Over Years')
    plt.xlabel('Year')
    plt.ylabel('Keyword Frequency')
    plt.show()

# 5. Impact of Awards on Citations
def impact_of_awards_on_citations(df):
    sns.boxplot(x='Award', y='CitationCount_CrossRef', data=df)
    plt.title('Impact of Awards on Citation Count')
    plt.xlabel('Award')
    plt.ylabel('Citation Count (CrossRef)')
    plt.show()
    print(df.groupby('Award').agg({'CitationCount_CrossRef': 'mean'}).reset_index())

# Running the functions
trend_analysis(df)
award_prediction(df)
download_prediction(df)
keyword_trend_analysis(df)
impact_of_awards_on_citations(df)

# ---- NEW BLOCK ---- # 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import numpy as np

# Load dataset
df = pd.read_csv('dataset.csv')

# 1. Time Series Analysis on Paper Popularity
def time_series_analysis():
    plt.figure(figsize=(10, 5))
    df.groupby('Year')['Downloads_Xplore'].sum().plot(color='skyblue')
    plt.title('Trend of Paper Downloads Over Years')
    plt.xlabel('Year')
    plt.ylabel('Total Downloads')
    plt.grid(True)
    plt.show()

# 2. Correlation Analysis
def correlation_analysis():
    plt.figure(figsize=(8, 6))
    correlation_matrix = df[['AminerCitationCount', 'CitationCount_CrossRef', 'PubsCited_CrossRef']].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation among Citation Metrics')
    plt.show()

# 3. Citation Impact Analysis
def citation_impact_analysis():
    X = df[['Downloads_Xplore']].values.reshape(-1, 1)
    y = df['CitationCount_CrossRef'].values
    # Use imputer to handle NaNs
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(X)
    y = np.nan_to_num(y)  # Set NaN values in y to zero

    model = LinearRegression()
    model.fit(X, y)

    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='lightcoral')
    plt.plot(X, model.predict(X), color='blue')
    plt.title('Downloads vs Citation Count Regression')
    plt.xlabel('Downloads_Xplore')
    plt.ylabel('CitationCount_CrossRef')
    plt.grid(True)
    plt.show()
    
    print(f'Regression Coefficient: {model.coef_[0]}')
    print(f'Regression Intercept: {model.intercept_}')

# 4. Trend Analysis on Citation Counts
def citation_trend_analysis():
    plt.figure(figsize=(12, 8))
    citation_trend = df.groupby(['Year', 'Conference'])['CitationCount_CrossRef'].mean().unstack()
    citation_trend.plot(marker='o')
    plt.title('Citation Counts Trend Over Years for Each Conference')
    plt.xlabel('Year')
    plt.ylabel('Average Citation Count')
    plt.grid(True)
    plt.legend(title='Conference', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()

# 5. Predictive Modeling for Award-Winning Papers
def predictive_modeling_for_awards():
    # Preprocess feature data by imputing missing values
    features = df[['Downloads_Xplore', 'CitationCount_CrossRef', 'AminerCitationCount', 'PubsCited_CrossRef']].fillna(0)
    target = df['Award'].apply(lambda x: 1 if pd.notna(x) else 0)
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, predictions)
    print(f'Accuracy of Predictive Model: {accuracy:.2f}')

# Run the functions to see the output
time_series_analysis()
correlation_analysis()
citation_impact_analysis()
citation_trend_analysis()
predictive_modeling_for_awards()

