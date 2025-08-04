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

# ---- NEW BLOCK ---- # 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Load the dataset
data = pd.read_csv('dataset.csv')

# Fill or drop NaN values for numeric computations
imputer = SimpleImputer(strategy='constant', fill_value=0)
data['Downloads_Xplore'] = imputer.fit_transform(data[['Downloads_Xplore']])
data['AminerCitationCount'] = data['AminerCitationCount'].fillna(0)
data['PubsCited_CrossRef'] = imputer.fit_transform(data[['PubsCited_CrossRef']])

# Ensure categorical variables are strings
data['Conference'] = data['Conference'].fillna('Unknown').astype(str)
data['PaperType'] = data['PaperType'].fillna('Unknown').astype(str)
data['GraphicsReplicabilityStamp'] = data['GraphicsReplicabilityStamp'].fillna('Unknown').astype(str)
data['Award'] = data['Award'].fillna(0).apply(lambda x: 1 if x else 0)  # Binary classification

# Visualization 1: Linear Regression for Citation Count Prediction
X_predict = data[['Conference', 'Year', 'PaperType', 'AuthorKeywords', 'Downloads_Xplore']].copy()
le = LabelEncoder()
X_predict['Conference'] = le.fit_transform(X_predict['Conference'])
X_predict['PaperType'] = le.fit_transform(X_predict['PaperType'])
tfidf = TfidfVectorizer(stop_words='english', max_features=100)
X_predict_keywords = tfidf.fit_transform(X_predict['AuthorKeywords'].fillna('')).toarray()
X_predict_tfidf = pd.DataFrame(X_predict_keywords, columns=tfidf.get_feature_names_out())
X_predict.drop('AuthorKeywords', axis=1, inplace=True)
X_predict = pd.concat([X_predict, X_predict_tfidf], axis=1)
y_predict = data['AminerCitationCount']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_predict, y_predict, test_size=0.2, random_state=42)

# Train linear regression model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Plot actual vs predicted citation counts
y_pred = lr.predict(X_test)
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel("Actual Citation Counts")
plt.ylabel("Predicted Citation Counts")
plt.title("Actual vs Predicted Citation Counts")
plt.grid(True)
plt.show()

# Visualization 2: Random Forest Classification for Award Prediction
X_award = data[['GraphicsReplicabilityStamp', 'AuthorKeywords', 'Downloads_Xplore', 'PubsCited_CrossRef']].copy()
X_award['GraphicsReplicabilityStamp'] = le.fit_transform(X_award['GraphicsReplicabilityStamp'])
X_award_keywords = tfidf.fit_transform(X_award['AuthorKeywords'].fillna('')).toarray()
X_award_tfidf = pd.DataFrame(X_award_keywords, columns=tfidf.get_feature_names_out())
X_award = pd.concat([X_award.drop('AuthorKeywords', axis=1), X_award_tfidf], axis=1)
y_award = data['Award']  # Binary labels

X_train_award, X_test_award, y_train_award, y_test_award = train_test_split(X_award, y_award, test_size=0.2, random_state=42)
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train_award, y_train_award)

# Plot Feature Importance for Random Forest
importance = rf_clf.feature_importances_
indices = importance.argsort()[-10:][::-1]
features = X_train_award.columns
plt.figure(figsize=(10, 7))
plt.title('Random Forest Feature Importance for Award Prediction')
plt.barh(range(len(indices)), importance[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()

# Visualization 3: LASSO Regression for Keyword Relevance
lasso = Lasso(alpha=0.1)
lasso.fit(X_train.drop(columns=['Downloads_Xplore']), y_train)

# Plot non-zero coefficients
coef = pd.Series(lasso.coef_, index=X_train.drop(columns=['Downloads_Xplore']).columns)
coef = coef.sort_values()
important_coef = coef[coef != 0]
plt.figure(figsize=(10, 8))
important_coef.plot(kind="barh")
plt.title("LASSO Model - Keyword Relevance in Citation Prediction")
plt.xlabel('Coefficient Value')
plt.show()

# Visualization 4: Semantic Analysis using TF-IDF and Linear Regression
X_semantic = data['Abstract'].fillna('')
X_semantic_tfidf = tfidf.fit_transform(X_semantic).toarray()
y_semantic = data['AminerCitationCount']

X_train_sem, X_test_sem, y_train_sem, y_test_sem = train_test_split(X_semantic_tfidf, y_semantic, test_size=0.2, random_state=42)
lr_sem = LinearRegression()
lr_sem.fit(X_train_sem, y_train_sem)

# Plot predicted vs actual values for semantic analysis
y_pred_sem = lr_sem.predict(X_test_sem)
plt.figure(figsize=(10, 6))
plt.scatter(y_test_sem, y_pred_sem, alpha=0.7)
plt.xlabel("Actual Citation Counts from Semantic Analysis")
plt.ylabel("Predicted Citation Counts from Semantic Analysis")
plt.title("Actual vs Predicted Citation Counts from Semantic Analysis Using TF-IDF")
plt.grid(True)
plt.show()

# ---- NEW BLOCK ---- # 
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from textblob import TextBlob

# Load dataset
df = pd.read_csv('dataset.csv')

# Handle missing values by filling them with appropriate replacements
df.fillna({
    'Abstract': '',
    'AuthorKeywords': '',
    'Downloads_Xplore': 0,
    'CitationCount_CrossRef': 0,
    'PubsCited_CrossRef': 0
}, inplace=True)

# 1. Topic Modeling on Abstracts using LDA from sklearn
def topic_modeling_lda(df, num_topics=5):
    # Prepare data
    documents = df['Abstract'].tolist()
    vectorizer = CountVectorizer(stop_words='english', max_features=5000)
    X = vectorizer.fit_transform(documents)
    
    # Build LDA model
    lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda_model.fit(X)
    
    # Display topics
    for i, topic in enumerate(lda_model.components_):
        topic_terms = [vectorizer.get_feature_names_out()[index] for index in topic.argsort()[-10:]]
        print(f"Topic {i}: {' '.join(topic_terms)}")

# 2. Classification of PaperType
def classify_paper_type(df):
    # Prepare data
    features = df[['Year', 'Downloads_Xplore']]
    labels = df['PaperType']
    
    # Use TF-IDF for AuthorKeywords
    keywords_vectorizer = TfidfVectorizer(max_features=100)
    keywords = keywords_vectorizer.fit_transform(df['AuthorKeywords'])
    
    # Concatenate features
    features = pd.concat([features, pd.DataFrame(keywords.toarray())], axis=1)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)
    
    # Train RandomForest
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    
    # Print classification report
    print(classification_report(y_test, y_pred))

# 3. Sentiment Analysis
def sentiment_analysis(df):
    # Calculate sentiment using TextBlob
    df['Sentiment'] = df['Abstract'].apply(lambda x: TextBlob(x).sentiment.polarity)
    
    # Determine sentiment labels
    conditions = [
        (df['Sentiment'] > 0),
        (df['Sentiment'] == 0),
        (df['Sentiment'] < 0)
    ]
    sentiments = ['positive', 'neutral', 'negative']
    df['SentimentLabel'] = pd.cut(df['Sentiment'], bins=3, labels=sentiments)
    print("Sentiment Analysis Labels:\n", df['SentimentLabel'].value_counts())

# 4. Principal Component Analysis (PCA)
def apply_pca(df):
    # Prepare data
    features = df[['CitationCount_CrossRef', 'PubsCited_CrossRef', 'Downloads_Xplore']]
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Apply PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(features_scaled)
    df_pca = pd.DataFrame(pca_result, columns=['PCA1', 'PCA2'])
    
    # Plot results
    sns.scatterplot(x='PCA1', y='PCA2', data=df_pca)
    plt.title('PCA of Numerical Features')
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    plt.show()

# 5. Trends in Paper Topics
def analyze_trends(df):
    # Calculate trends over years
    df['Year'] = df['Year'].astype(int)
    unique_keywords_per_year = df.groupby('Year')['AuthorKeywords'].apply(lambda x: x.str.split(';').explode().nunique())
    
    # Plot trends
    plt.figure(figsize=(10,5))
    unique_keywords_per_year.plot(kind='line', marker='o')
    plt.title('Trends in Paper Topics Over Years')
    plt.xlabel('Year')
    plt.ylabel('Unique Topics')
    plt.grid(True)
    plt.show()

# Example function calls
# topic_modeling_lda(df)
# classify_paper_type(df)
# sentiment_analysis(df)
# apply_pca(df)
# analyze_trends(df)

# ---- NEW BLOCK ---- # 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report, mean_squared_error

# Load dataset
df = pd.read_csv('dataset.csv')

# Handle NaN values by filling them with the median of the column or dropping if necessary
df.fillna(df.median(numeric_only=True), inplace=True)
df.dropna(subset=['AuthorKeywords', 'InternalReferences'], inplace=True)

# Encode categorical variables
label_encoders = {}
categorical_columns = ['Conference', 'PaperType', 'AuthorAffiliation', 'Award', 'GraphicsReplicabilityStamp']
for column in categorical_columns:
    le = LabelEncoder()
    df[column] = df[column].astype(str)
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Idea 1: Logistic Regression for Predicting Awards
X_award = df[['Conference', 'Year', 'PaperType', 'AminerCitationCount', 
              'CitationCount_CrossRef', 'Downloads_Xplore', 'GraphicsReplicabilityStamp']]
y_award = df['Award']
X_train, X_test, y_train, y_test = train_test_split(X_award, y_award, test_size=0.3, random_state=42)
model_award = LogisticRegression()
model_award.fit(X_train, y_train)
y_pred = model_award.predict(X_test)
print("Logistic Regression for Predicting Awards")
print(classification_report(y_test, y_pred))

# Idea 2: SVM for Graphics Replicability Stamp Prediction
X_stamp = df[['Year', 'PaperType', 'AminerCitationCount', 'CitationCount_CrossRef', 'Downloads_Xplore']]
y_stamp = df['GraphicsReplicabilityStamp']
X_train, X_test, y_train, y_test = train_test_split(X_stamp, y_stamp, test_size=0.3, random_state=42)
model_stamp = SVC()
model_stamp.fit(X_train, y_train)
y_pred = model_stamp.predict(X_test)
print("SVM for Graphics Replicability Stamp Prediction")
print(classification_report(y_test, y_pred))

# Idea 3: Random Forest for Citation Count Prediction
X_citation = df[['Conference', 'Year', 'PaperType', 'AuthorAffiliation', 
                 'AminerCitationCount', 'Downloads_Xplore']]
y_citation = df['CitationCount_CrossRef']
X_train, X_test, y_train, y_test = train_test_split(X_citation, y_citation, test_size=0.3, random_state=42)
model_citation = RandomForestRegressor()
model_citation.fit(X_train, y_train)
y_pred = model_citation.predict(X_test)
print("Random Forest for Citation Count Prediction")
print("MSE:", mean_squared_error(y_test, y_pred))

# Idea 4: K-Nearest Neighbors for Similar Paper Suggestion
# Here we demonstrate KNN with regression for illustration purposes
X_knn = df[['AminerCitationCount', 'PubsCited_CrossRef']]
y_knn = df['PubsCited_CrossRef']
X_train, X_test, y_train, y_test = train_test_split(X_knn, y_knn, test_size=0.3, random_state=42)
model_knn = KNeighborsRegressor(n_neighbors=3)
model_knn.fit(X_train, y_train)
y_pred = model_knn.predict(X_test)
print("KNN for Similar Paper Suggestion")
print("MSE:", mean_squared_error(y_test, y_pred))

# Idea 5: Linear Regression for Download Prediction
X_downloads = df[['Conference', 'Year', 'PaperType', 'AminerCitationCount', 'CitationCount_CrossRef']]
y_downloads = df['Downloads_Xplore']
X_train, X_test, y_train, y_test = train_test_split(X_downloads, y_downloads, test_size=0.3, random_state=42)
model_downloads = LinearRegression()
model_downloads.fit(X_train, y_train)
y_pred = model_downloads.predict(X_test)
print("Linear Regression for Download Prediction")
print("MSE:", mean_squared_error(y_test, y_pred))

# ---- NEW BLOCK ---- # 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score

# Load the dataset
df = pd.read_csv('dataset.csv')

# Fill NaN values with appropriate method
df.fillna({'Abstract': '', 'AuthorKeywords': '', 'AminerCitationCount': 0,
           'CitationCount_CrossRef': 0, 'Downloads_Xplore': 0}, inplace=True)

# Convert Award to binary
df['Award'] = df['Award'].notnull().astype(int)

# Check and convert necessary columns to appropriate types.
df['Year'] = df['Year'].astype(int)

# 1. Topic Modeling with LDA (Latent Dirichlet Allocation)
vectorizer = TfidfVectorizer(stop_words='english')
X_lda = vectorizer.fit_transform(df['Abstract'] + ' ' + df['Title'] + ' ' + df['AuthorKeywords'])

lda = LatentDirichletAllocation(n_components=5, random_state=0)
lda.fit(X_lda)

# Display the top words in each topic
def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("\nTopic %d:" % (topic_idx))
        print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))

no_top_words = 10
display_topics(lda, vectorizer.get_feature_names_out(), no_top_words)

# 2. Citation Prediction using Regression
X = df[['Year', 'AminerCitationCount', 'CitationCount_CrossRef']]
y = df['CitationCount_CrossRef']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

# Plot the predicted vs actual values
plt.scatter(y_test, y_pred, alpha=0.3)
plt.xlabel("Actual Citation Count")
plt.ylabel("Predicted Citation Count")
plt.title("Actual vs Predicted Citation Count using Regression")
plt.show()

# 3. Classification of Award-Winning Papers using Logistic Regression
X = df[['AminerCitationCount', 'CitationCount_CrossRef', 'Downloads_Xplore']]
y = df['Award']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)

# Plot the confusion matrix
conf_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
sns.heatmap(conf_matrix, annot=True, fmt='d')
plt.title("Confusion Matrix for Award-Winning Paper Classification")
plt.show()

# 4. Exploring Correlation and Impact Analysis through Exploratory Data Analysis
corr_matrix = df[['AminerCitationCount', 'CitationCount_CrossRef', 'Downloads_Xplore']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix of Numerical Features")
plt.show()

# 5. Dimensionality Reduction for Visualization using PCA
X_pca = df[['AminerCitationCount', 'CitationCount_CrossRef', 'Downloads_Xplore']].fillna(0)

pca = PCA(n_components=2)
principal_components = pca.fit_transform(X_pca)

plt.figure(figsize=(10, 7))
plt.scatter(principal_components[:, 0], principal_components[:, 1], alpha=0.3)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Numerical Features')
plt.show()

# ---- NEW BLOCK ---- # 
from sklearn.model_selection import train_test_split
import pandas as pd

# Make sure to load the full dataset
df = pd.read_csv('dataset.csv')

# Assure there are no NaN entries
df.dropna(inplace=True)

# Check the number of samples
num_samples = df.shape[0]

# Only proceed with splitting if the number of rows is greater than 1
if num_samples > 1:
    # Select features and target variable
    X = df[['Year', 'Conference', 'PaperType', 'AuthorKeywords', 'Downloads_Xplore']].copy()
    y = df['AminerCitationCount']

    # Encoding categorical variables
    X_encoded = pd.get_dummies(X, drop_first=True)

    # Adjust train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=min(0.2, num_samples-1), random_state=42)

    print(f"Train set size: {X_train.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples")
else:
    print("Insufficient number of samples for train-test split!")

# ---- NEW BLOCK ---- # 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.express as px

# Load the dataset
df = pd.read_csv("dataset.csv")

# Drop rows with NaN values
df.dropna(subset=['Abstract', 'AuthorKeywords', 'AminerCitationCount', 'Downloads_Xplore', 'InternalReferences'], inplace=True)

# Preparing data
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
X = tfidf_vectorizer.fit_transform(df['Abstract'])
y = df['PaperType']

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the model
model_nb = MultinomialNB()
model_nb.fit(X_train, y_train)

# Predict and visualize
y_pred = model_nb.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix for Paper Type Classification')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

print(classification_report(y_test, y_pred))

# Variables
features = df[['Year', 'PubsCited_CrossRef']]
target = df['AminerCitationCount']

# Split data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Train model
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)

# Visualization of predictions
y_pred = model_lr.predict(X_test)
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Citation Count')
plt.ylabel('Predicted Citation Count')
plt.title('Regression: Actual vs Predicted Citation Count')
plt.show()

# Numeric transformation for clustering
tfidf_vectors = tfidf_vectorizer.fit_transform(df['Abstract']).toarray()
scaler = StandardScaler()
scaled_data = scaler.fit_transform(tfidf_vectors)

# K-Means Clustering
kmeans = KMeans(n_clusters=5, random_state=42)
cluster_labels = kmeans.fit_predict(scaled_data)

# Visualizing the clusters using PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(scaled_data)

plt.scatter(principal_components[:, 0], principal_components[:, 1], c=cluster_labels, cmap='viridis', alpha=0.6)
plt.title('Clustering of Research Papers')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.show()

# Applying t-SNE
tsne = TSNE(n_components=2, random_state=42)
tsne_result = tsne.fit_transform(scaled_data)

plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=cluster_labels, cmap='viridis', alpha=0.6)
plt.title('t-SNE Dimensionality Reduction')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.show()

df_ts = df.groupby('Year').agg({'AminerCitationCount': 'sum', 'Downloads_Xplore': 'sum'}).reset_index()

# Plotting
plt.figure(figsize=(14, 7))
sns.lineplot(x='Year', y='AminerCitationCount', data=df_ts, label='Aminer Citation Count')
sns.lineplot(x='Year', y='Downloads_Xplore', data=df_ts, label='Downloads Xplore')
plt.title('Trends Over Time for Citations and Downloads')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# ---- NEW BLOCK ---- # 
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.impute import SimpleImputer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('dataset.csv')

# Fill NaN values
imputer = SimpleImputer(strategy='most_frequent')
df_filled = pd.DataFrame(imputer.fit_transform(df))
df_filled.columns = df.columns

# Convert categorical columns if needed
df_filled['PaperType'] = df_filled['PaperType'].astype('category')
df_filled['Award'] = df_filled['Award'].astype('category')

# 1. Text Classification on PaperType using TF-IDF and Logistic Regression
def text_classification_on_paper_type():
    tfidf = TfidfVectorizer(stop_words='english')
    X = tfidf.fit_transform(df_filled['Abstract'].fillna('') + ' ' + df_filled['Title'].fillna(''))
    y = df_filled['PaperType'].cat.codes

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    # Plotting WordCloud for Text Visualization
    plt.figure(figsize=(10, 6))
    text = ' '.join(df_filled['Abstract'].fillna(''))
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title('Word Cloud of Abstracts')
    plt.axis('off')
    plt.show()

# 2. Citation Prediction using Linear Regression
def citation_prediction():
    features = ['Year', 'AminerCitationCount', 'CitationCount_CrossRef']
    df_filled['AuthorKeywords'] = df_filled['AuthorKeywords'].fillna('')
    df_filled['keywords_count'] = df_filled['AuthorKeywords'].apply(lambda x: len(x.split(',')))

    X = df_filled[features + ['keywords_count']]
    y = df_filled['CitationCount_CrossRef']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    
    # Plotting Predicted vs Actual Citations
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred)
    plt.xlabel('Actual Citations')
    plt.ylabel('Predicted Citations')
    plt.title('Actual vs Predicted Citations (MSE: {:.2f})'.format(mse))
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')
    plt.show()

# 3. Conference Impact Analysis using Clustering
def conference_impact_analysis():
    from sklearn.cluster import KMeans

    features = df_filled[['AminerCitationCount', 'CitationCount_CrossRef', 'Downloads_Xplore']]
    features = features.fillna(0)  # Replace NaNs for clustering

    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(features)
    
    # Plotting Clusters
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=features['AminerCitationCount'], y=features['CitationCount_CrossRef'], hue=clusters, palette='viridis')
    plt.title('Conference Impact Clustering')
    plt.show()

# 4. Recommendation System for Papers using TF-IDF
def recommendation_system_for_papers():
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df_filled['Abstract'].fillna(''))
    
    # Generate Word Cloud for a quick visualization
    plt.figure(figsize=(10, 6))
    text = ' '.join(df_filled['Title'].fillna(''))
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title('Word Cloud of Titles')
    plt.axis('off')
    plt.show()

# 5. Award Prediction Model using Random Forest
def award_prediction_model():
    from sklearn.ensemble import RandomForestClassifier

    features = df_filled[['Year', 'AminerCitationCount', 'CitationCount_CrossRef', 'Downloads_Xplore']].fillna(0)
    y = df_filled['Award'].cat.codes  # Converting categories to codes

    X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    # Plotting Actual vs Predicted Awards
    plt.figure(figsize=(10, 6))
    sns.histplot(y_pred, kde=False, label='Predicted', color='blue', stat='density', bins=9)
    sns.histplot(y_test, kde=False, label='Actual', color='orange', stat='density', bins=9)
    plt.title('Predicted vs Actual Awards')
    plt.legend()
    plt.show()

# Run all functions
text_classification_on_paper_type()
citation_prediction()
conference_impact_analysis()
recommendation_system_for_papers()
award_prediction_model()

# ---- NEW BLOCK ---- # 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the dataset
df = pd.read_csv('dataset.csv')

# Fill NaN values for numerical columns with median
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[numerical_cols] = df[numerical_cols].apply(lambda col: col.fillna(col.median()), axis=0)

# Fill NaN values for categorical columns with mode
categorical_cols = df.select_dtypes(include=['object']).columns
df[categorical_cols] = df[categorical_cols].apply(lambda col: col.fillna(col.mode()[0]), axis=0)

# Function 1: Trend Analysis in Conference Topics
def trend_analysis_conference_topics():
    trend_data = df.groupby(['Year', 'Conference']).size().unstack().fillna(0)
    trend_data.plot(kind='line', figsize=(14, 8))
    plt.title('Trend Analysis of Conference Topics Over the Years')
    plt.xlabel('Year')
    plt.ylabel('Number of Papers')
    plt.legend(title='Conference', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

# Function 2: Impact of Awards on Citations
def impact_awards_on_citations():
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Award', y='AminerCitationCount', data=df)
    plt.title('Impact of Awards on Aminer Citation Count')
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Award', y='CitationCount_CrossRef', data=df)
    plt.title('Impact of Awards on CrossRef Citation Count')
    plt.show()

# Function 3: Regression Analysis on Download Counts
def regression_analysis_download_counts():
    features = df[['Year', 'PubsCited_CrossRef', 'GraphicsReplicabilityStamp']]
    features = pd.get_dummies(features, drop_first=True)
    target = df['Downloads_Xplore']

    # Scaling and fitting the model
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    model = LinearRegression()
    model.fit(features_scaled, target)
    coefficients = pd.DataFrame(model.coef_, features.columns, columns=['Coefficient'])

    coefficients.plot(kind='barh', figsize=(8, 6))
    plt.title('Feature Coefficients for Predicting Downloads')
    plt.xlabel('Coefficient Value')
    plt.ylabel('Feature')
    plt.show()

# Function 4: Text Mining for Topic Modeling
def text_mining_topic_modeling():
    abstracts = df['Abstract'].fillna('')
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(abstracts)

    lda = LatentDirichletAllocation(n_components=5, random_state=0)
    lda.fit(tfidf_matrix)

    feature_names = tfidf_vectorizer.get_feature_names_out()
    for index, topic in enumerate(lda.components_):
        print(f'Top 10 words for topic #{index + 1}:')
        print([feature_names[i] for i in topic.argsort()[-10:]])
        print('\n')

# Function 5: Keyword Frequency and Topic Clustering
def keyword_frequency_topic_clustering():
    all_keywords = df['AuthorKeywords'].dropna().str.split(';').explode()
    keyword_counts = all_keywords.value_counts()

    # Plot top keywords by frequency
    top_keywords = keyword_counts.head(10)
    top_keywords.plot(kind='bar', figsize=(10, 6))
    plt.title('Top 10 Keywords by Frequency')
    plt.xlabel('Keywords')
    plt.ylabel('Frequency')
    plt.show()

    # Clustering
    tfidf_vectorizer = TfidfVectorizer()
    keyword_embeddings = tfidf_vectorizer.fit_transform(all_keywords.dropna().unique()).toarray()
    keyword_clusters = KMeans(n_clusters=5, random_state=0)
    keyword_clusters.fit(keyword_embeddings)

    plt.scatter(keyword_embeddings[:, 0], keyword_embeddings[:, 1], c=keyword_clusters.labels_, cmap='viridis')
    plt.title('Keyword Topic Clustering')
    plt.xlabel('Embedding Dimension 1')
    plt.ylabel('Embedding Dimension 2')
    plt.show()

# Execute the defined functions
trend_analysis_conference_topics()
impact_awards_on_citations()
regression_analysis_download_counts()
text_mining_topic_modeling()
keyword_frequency_topic_clustering()

# ---- NEW BLOCK ---- # 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import numpy as np

# Load the dataset
data = pd.read_csv('dataset.csv')

# Data Cleaning: Convert to correct dtypes
numeric_columns = [
    'Year', 'FirstPage', 'LastPage',
    'AminerCitationCount', 'CitationCount_CrossRef',
    'PubsCited_CrossRef', 'Downloads_Xplore'
]

# Force numeric columns to be numeric and fill NAs
data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors='coerce').fillna(0)

# Convert categorical to suitable format
data['Award'] = data['Award'].notnull().astype('int')

# Visual Helper Function
def show_feature_importance(model, X, title):
    sns.barplot(x=model.feature_importances_, y=X.columns)
    plt.title(title)
    plt.show()

# 1. Temporal Analysis of Citations using Holt-Winters method (without seasonality)
def temporal_analysis(data):
    df_citations = data.groupby('Year').agg({'CitationCount_CrossRef': 'sum'}).reset_index()
    model = ExponentialSmoothing(df_citations['CitationCount_CrossRef'], trend='add')
    model_fit = model.fit()
    
    forecast = model_fit.forecast(steps=5)
    
    plt.plot(df_citations['Year'], df_citations['CitationCount_CrossRef'], label='Observed', marker='o')
    plt.plot(range(df_citations['Year'].max() + 1, df_citations['Year'].max() + 6), forecast, label='Forecast', marker='o')
    plt.title('Temporal Analysis of Citations using Exponential Smoothing')
    plt.xlabel('Year')
    plt.ylabel('Citations')
    plt.legend()
    plt.show()

temporal_analysis(data)

# 2. Predictive Modeling for Award-Winning Papers
def award_prediction(data):
    features = ['Year', 'PaperType', 'CitationCount_CrossRef', 'AuthorAffiliation', 'Downloads_Xplore']
    X = pd.get_dummies(data[features], drop_first=True)
    y = data['Award']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"Award Prediction Accuracy: {acc}")
    show_feature_importance(model, X, "Feature Importance for Award Prediction")

award_prediction(data)

# 3. Latent Semantic Analysis on Abstracts
def latent_semantic_analysis(data):
    vectorizer = TfidfVectorizer(max_features=500)
    X_abstract = vectorizer.fit_transform(data['Abstract'].fillna(''))
    
    svd = TruncatedSVD(n_components=5)
    topics = svd.fit_transform(X_abstract)
    
    sns.heatmap(topics, cmap='viridis', cbar_kws={'label': 'Topic Weights'})
    plt.title("Latent Semantic Analysis on Abstracts")
    plt.xlabel('Topics')
    plt.show()

latent_semantic_analysis(data)

# 4. Classification for Paper Type
def paper_type_classification(data):
    features = ['Conference', 'Year', 'AminerCitationCount', 'CitationCount_CrossRef', 'Downloads_Xplore']
    X = pd.get_dummies(data[features], drop_first=True)
    y = data['PaperType']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"Paper Type Classification Accuracy: {acc}")
    show_feature_importance(model, X, "Feature Importance in Paper Type Classification")

paper_type_classification(data)

# 5. RandomForest for Download Prediction
def boosting_for_download_prediction(data):
    features = ['Year', 'Conference', 'CitationCount_CrossRef', 'Award', 'GraphicsReplicabilityStamp']
    X = pd.get_dummies(data[features], drop_first=True)
    y = data['Downloads_Xplore']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mse = np.mean((y_pred - y_test) ** 2)
    
    print(f"Mean Squared Error for Download Prediction: {mse}")
    show_feature_importance(model, X, "Feature Importance in Download Prediction")

boosting_for_download_prediction(data)

# ---- NEW BLOCK ---- # 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
data = pd.read_csv('dataset.csv')

# Fill NaN values
data.fillna(method='ffill', inplace=True)

### Encode Categorical Columns ###
# Encoding categorical features for models
# An example of encoding 'Conference' as it might be used numerically:
data['Conference_encoded'] = LabelEncoder().fit_transform(data['Conference'])

### Idea 1: Text Classification ###
# Visualize the distribution of PaperType
plt.figure(figsize=(10, 6))
sns.countplot(x='PaperType', data=data, order=data['PaperType'].value_counts().index)
plt.title('Distribution of Paper Types')
plt.xticks(rotation=45)
plt.show()

# Simple text classification with Naive Bayes
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X_text = vectorizer.fit_transform(data['Abstract'].astype(str) + " " + data['Title'].astype(str))
y_type = data['PaperType']
X_train, X_test, y_train, y_test = train_test_split(X_text, y_type, test_size=0.3, random_state=42)

clf = MultinomialNB()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Classification Report for PaperType Prediction:")
print(classification_report(y_test, y_pred))

### Idea 2: Citation Count Prediction ###
# Plot correlations
plt.figure(figsize=(10, 6))
sns.heatmap(data[['Year', 'PubsCited_CrossRef', 'Downloads_Xplore', 'AminerCitationCount', 'Conference_encoded']].corr(), annot=True)
plt.title('Correlation Heatmap')
plt.show()

# Random Forest Regression for AminerCitationCount
features = ['Year', 'PubsCited_CrossRef', 'Downloads_Xplore', 'Conference_encoded']
X_features = data[features].fillna(0)
y_citation = data['AminerCitationCount'].fillna(0)
X_train, X_test, y_train, y_test = train_test_split(X_features, y_citation, test_size=0.3, random_state=42)

regressor = RandomForestRegressor(n_estimators=100, random_state=42)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.title('Citation Count Prediction')
plt.xlabel('Actual Citation Counts')
plt.ylabel('Predicted Citation Counts')
plt.show()

### Idea 3: Recommendation System for Papers ###
# Content-based filtering using TF-IDF for Title and Keywords
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
content_matrix = vectorizer.fit_transform(data['Title'].astype(str) + " " + data['AuthorKeywords'].astype(str))
cosine_sim = cosine_similarity(content_matrix, content_matrix)

def get_recommendations(title, cosine_sim=cosine_sim):
    indices = data[data['Title'].str.contains(title, case=False, na=False)].index
    if indices.size > 0:
        idx = indices[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:6]
        paper_indices = [i[0] for i in sim_scores]
        return data['Title'].iloc[paper_indices]
    else:
        return "No recommendations found. Please ensure the title exists."

# Example Recommendation
example_title = data['Title'].iloc[0]  # Use the first title for demonstration
print(f"Recommendations for the paper titled: '{example_title}'")
print(get_recommendations(example_title))

### Idea 4: Predictive Modelling for Awards ###
# Logistic Regression predicting Award based on attributes (skipping due to low Award count but plotting)
plt.figure(figsize=(10, 6))
sns.heatmap(data[['Year', 'Downloads_Xplore', 'Conference_encoded']].corr(), annot=True)
plt.title('Correlation Heatmap for Award Prediction Attributes')
plt.show()

### Idea 5: Keyword Trend Analysis ###
# Example: Count the usage of a keyword over the years

def plot_keyword_trend(keyword):
    keyword_data = data[data['AuthorKeywords'].str.contains(keyword, na=False, case=False)]
    keyword_trend = keyword_data.groupby('Year').size()

    plt.figure(figsize=(10, 6))
    plt.plot(keyword_trend.index, keyword_trend.values, marker='o')
    plt.title(f'Trend of "{keyword}" over Years')
    plt.xlabel('Year')
    plt.ylabel('Number of Papers')
    plt.grid(True)
    plt.show()

# Plot trend for an example keyword
plot_keyword_trend('machine learning')

# ---- NEW BLOCK ---- # 
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

# ---- NEW BLOCK ---- # 
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

# ---------------------------
# Enhanced Trend Analysis
# ---------------------------
plt.figure(figsize=(14, 6))

# Subplot 1: AminerCitationCount by Conference
plt.subplot(1, 2, 1)
sns.lineplot(
    data=df,
    x='Year', y='AminerCitationCount',
    hue='Conference',
    marker='o',
    palette='tab10',
    ci=None
)
plt.title('Aminer Citation Trends by Conference', fontsize=14)
plt.xlabel('Publication Year', fontsize=12)
plt.ylabel('Aminer Citation Count', fontsize=12)
plt.legend(title='Conference', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
plt.grid(alpha=0.3)

# Subplot 2: CrossRef Citations by Paper Type
plt.subplot(1, 2, 2)
sns.lineplot(
    data=df,
    x='Year', y='CitationCount_CrossRef',
    hue='PaperType',
    style='PaperType',
    markers=True,
    dashes=True,
    palette='Set2',
    ci=None
)
plt.title('CrossRef Citation Trends by Paper Type', fontsize=14)
plt.xlabel('Publication Year', fontsize=12)
plt.ylabel('CrossRef Citation Count', fontsize=12)
plt.legend(title='Paper Type', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# ---------------------------
# Feature Importance Plot
# ---------------------------
# Machine Learning to Predict Awards
X = df[['PaperType', 'Downloads_Xplore', 'Year', 'Conference', 'AuthorKeywords']].copy()
X['Conference'] = LabelEncoder().fit_transform(X['Conference'])
X['PaperType']   = LabelEncoder().fit_transform(X['PaperType'])
X['AuthorKeywords'] = X['AuthorKeywords'].apply(lambda x: len(x.split(',')))
y = df['Award']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

importances = rf_classifier.feature_importances_
indices = np.argsort(importances)[::-1]
feat_names = [X.columns[i] for i in indices]

plt.figure(figsize=(8, 5))
sns.barplot(x=importances[indices], y=feat_names, palette='viridis')
plt.title('Random Forest Feature Importances', fontsize=14)
plt.xlabel('Importance', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.xlim(0, max(importances)*1.1)
plt.tight_layout()
plt.show()

# ---------------------------
# Regression: Actual vs Predicted Downloads
# ---------------------------
# Prepare data
df['InternalReferencesCount'] = df['InternalReferences'].apply(lambda x: len(x.split(';')) if x else 0)
predictors = ['AminerCitationCount', 'GraphicsReplicabilityStamp', 'InternalReferencesCount', 'Year', 'Conference']
Xr = df[predictors].copy()
Xr['Conference'] = LabelEncoder().fit_transform(Xr['Conference'])
Xr['GraphicsReplicabilityStamp'] = Xr['GraphicsReplicabilityStamp'].apply(lambda x: 1 if x == 'Has Stamp' else 0)
yr = df['Downloads_Xplore']

Xr_train, Xr_test, yr_train, yr_test = train_test_split(Xr, yr, test_size=0.3, random_state=42)
lr = LinearRegression()
lr.fit(Xr_train, yr_train)
yr_pred = lr.predict(Xr_test)

# Compute R^2 for annotation
from sklearn.metrics import r2_score
r2 = r2_score(yr_test, yr_pred)

plt.figure(figsize=(8, 6))
plt.scatter(yr_test, yr_pred, alpha=0.6, edgecolor='k', color='teal')
lims = [min(yr_test.min(), yr_pred.min()), max(yr_test.max(), yr_pred.max())]
plt.plot(lims, lims, 'r--', linewidth=1)
plt.title('Actual vs. Predicted Downloads', fontsize=14)
plt.xlabel('Actual Downloads_Xplore', fontsize=12)
plt.ylabel('Predicted Downloads_Xplore', fontsize=12)
plt.text(
    0.05, 0.95,
    f'$R^2$ = {r2:.2f}',
    transform=plt.gca().transAxes,
    fontsize=12, verticalalignment='top',
    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ---------------------------
# Keyword Trend Analysis (WordCloud)
# ---------------------------
all_keywords = df['AuthorKeywords'].str.cat(sep=', ')
wordcloud = WordCloud(
    width=800, height=400,
    max_words=150, background_color='white',
    colormap='plasma'
).generate(all_keywords)

plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('WordCloud of Author Keywords', fontsize=16)
plt.tight_layout()
plt.show()

# ---------------------------
# Impact of Awards on Citations (Boxplot)
# ---------------------------
award_cit = df[df['Award'] != 'No Award']['CitationCount_CrossRef']
no_award_cit = df[df['Award'] == 'No Award']['CitationCount_CrossRef']

plt.figure(figsize=(8, 6))
sns.boxplot(
    data=[award_cit, no_award_cit],
    palette=['#66c2a5', '#fc8d62']
)
plt.xticks([0, 1], ['Awarded Papers', 'Non-Awarded Papers'], fontsize=12)
plt.title('Effect of Awards on CrossRef Citation Counts', fontsize=14)
plt.ylabel('CitationCount_CrossRef', fontsize=12)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# ---- NEW BLOCK ---- # 
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

# --- Enhanced Trend Analysis on Citation Counts ---
plt.figure()
fig, ax = plt.subplots(figsize=(12, 6))
# Solid lines for AminerCitationCount by Conference
sns.lineplot(
    data=df,
    x='Year',
    y='AminerCitationCount',
    hue='Conference',
    marker='o',
    ci=None,
    ax=ax
)
# Dashed lines with different style for CrossRef citations by PaperType
sns.lineplot(
    data=df,
    x='Year',
    y='CitationCount_CrossRef',
    style='PaperType',
    markers=True,
    dashes=True,
    ci=None,
    legend='brief',
    ax=ax
)
ax.set_title('Citation Trends Over Years\nSolid=Conference | Dashed=Paper Type', fontsize=14)
ax.set_xlabel('Publication Year', fontsize=12)
ax.set_ylabel('Citation Count', fontsize=12)
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(title='Group', fontsize='small', title_fontsize='medium')
plt.tight_layout()
plt.show()


# --- Machine Learning to Predict Awards ---
X = df[['PaperType', 'Downloads_Xplore', 'Year', 'Conference', 'AuthorKeywords']].copy()
X['Conference'] = LabelEncoder().fit_transform(X['Conference'])
X['PaperType'] = LabelEncoder().fit_transform(X['PaperType'])
X['AuthorKeywords'] = X['AuthorKeywords'].apply(lambda x: len(x.split(',')))
y = df['Award']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)
y_pred_award = rf_classifier.predict(X_test)

# Display tree feature importance
importances = rf_classifier.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(10, 6))
sns.barplot(
    x=importances[indices],
    y=[X.columns[i] for i in indices],
    palette='Blues_d'
)
plt.title('Random Forest Feature Importance', fontsize=14)
plt.xlabel('Importance Score', fontsize=12)
plt.ylabel('Features', fontsize=12)
# Annotate each bar
for idx, val in enumerate(importances[indices]):
    plt.text(val + 0.005, idx, f"{val:.2f}", va='center')
plt.tight_layout()
plt.show()


# Convert InternalReferences from strings of DOIs to counts of references
df['InternalReferencesCount'] = df['InternalReferences'].apply(
    lambda x: len(x.split(';')) if x else 0
)

# Regression Analysis for Download Prediction
predictors = [
    'AminerCitationCount',
    'GraphicsReplicabilityStamp',
    'InternalReferencesCount',
    'Year',
    'Conference'
]
X_reg = df[predictors].copy()
X_reg['Conference'] = LabelEncoder().fit_transform(X_reg['Conference'])
X_reg['GraphicsReplicabilityStamp'] = X_reg['GraphicsReplicabilityStamp'].apply(
    lambda x: 1 if x == 'Has Stamp' else 0
)
y_reg = df['Downloads_Xplore']

Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    X_reg, y_reg, test_size=0.3, random_state=42
)

lr = LinearRegression()
lr.fit(Xr_train, yr_train)
y_pred_downloads = lr.predict(Xr_test)

# --- Visualization of Actual vs Predicted Downloads ---
plt.figure(figsize=(10, 6))
plt.scatter(
    yr_test, y_pred_downloads,
    alpha=0.6, color='teal', edgecolor='k'
)
# 45-degree reference line
min_val = min(min(yr_test), min(y_pred_downloads))
max_val = max(max(yr_test), max(y_pred_downloads))
plt.plot([min_val, max_val], [min_val, max_val], '--r', linewidth=1)
plt.title('Actual vs Predicted Downloads', fontsize=14)
plt.xlabel('Actual Download Count', fontsize=12)
plt.ylabel('Predicted Download Count', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


# --- Keyword Trend Analysis (Word Cloud) ---
all_keywords = df['AuthorKeywords'].str.cat(sep=', ')
wordcloud = WordCloud(
    width=800,
    height=400,
    max_words=150,
    background_color='white'
).generate(all_keywords)

plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.title('Keyword Frequency Word Cloud', fontsize=14)
plt.axis('off')
plt.tight_layout()
plt.show()


# --- Impact of Awards on Citations ---
df['AwardedStatus'] = np.where(df['Award'] != 'No Award',
                               'Awarded Paper',
                               'Non-Awarded Paper')

plt.figure(figsize=(10, 6))
sns.boxplot(
    x='AwardedStatus',
    y='CitationCount_CrossRef',
    data=df,
    palette='pastel'
)
plt.title('Impact of Awards on CrossRef Citation Counts', fontsize=14)
plt.xlabel('')
plt.ylabel('CrossRef Citation Count', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# ---- NEW BLOCK ---- # 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer

# Global styling
sns.set_style("whitegrid")
sns.set_context("talk")

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

# ---- IMPROVED TREND ANALYSIS ----
plt.figure(figsize=(12, 6), dpi=100)
sns.lineplot(
    data=df,
    x='Year',
    y='AminerCitationCount',
    hue='Conference',
    marker='o',
    palette='tab10',
    ci=None,
    linewidth=2
)
sns.lineplot(
    data=df,
    x='Year',
    y='CitationCount_CrossRef',
    hue='PaperType',
    marker='s',
    palette='tab20',
    ci=None,
    linestyle='--',
    linewidth=2
)
plt.title('Citation Trends Over the Years by Conference and Paper Type', fontsize=16)
plt.xlabel('Publication Year', fontsize=14)
plt.ylabel('Citation Count', fontsize=14)
plt.legend(
    title='Group',
    bbox_to_anchor=(1.02, 1),
    loc='upper left',
    borderaxespad=0
)
plt.tight_layout()
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

# ---- IMPROVED FEATURE IMPORTANCE ----
importances = rf_classifier.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6), dpi=100)
sns.barplot(
    x=importances[indices],
    y=[X.columns[i] for i in indices],
    palette='viridis'
)
plt.title('Feature Importance from RandomForest Classifier', fontsize=16)
plt.xlabel('Importance Score', fontsize=14)
plt.ylabel('Feature', fontsize=14)
for i, v in enumerate(importances[indices]):
    plt.text(v + 0.005, i, f"{v:.2f}", va='center', fontsize=12)
plt.tight_layout()
plt.show()

# Convert InternalReferences to counts
df['InternalReferencesCount'] = df['InternalReferences'].apply(
    lambda x: len(x.split(';')) if x else 0
)

# Regression Analysis for Download Prediction
predictors = [
    'AminerCitationCount',
    'GraphicsReplicabilityStamp',
    'InternalReferencesCount',
    'Year',
    'Conference'
]
X = df[predictors].copy()
X['Conference'] = LabelEncoder().fit_transform(X['Conference'])
X['GraphicsReplicabilityStamp'] = X['GraphicsReplicabilityStamp'].map(
    {'Has Stamp': 1, 'No Stamp': 0}
)

# Handle missing values in the predictors
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

y = df['Downloads_Xplore']
y = imputer.fit_transform(y.values.reshape(-1, 1)).flatten()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

# ---- IMPROVED ACTUAL VS PREDICTED SCATTER ----
plt.figure(figsize=(10, 6), dpi=100)
plt.scatter(
    y_test,
    y_pred,
    alpha=0.6,
    edgecolor='k',
    color='teal'
)
max_val = max(max(y_test), max(y_pred))
plt.plot([0, max_val], [0, max_val], 'r--', lw=2)
r2 = lr.score(X_test, y_test)
plt.title('Actual vs. Predicted Downloads', fontsize=16)
plt.xlabel('Actual Downloads', fontsize=14)
plt.ylabel('Predicted Downloads', fontsize=14)
plt.text(0.05 * max_val, 0.9 * max_val, f'R\u00b2 = {r2:.2f}', fontsize=12, color='darkred')
plt.tight_layout()
plt.show()

# Keyword Trend Analysis
all_keywords = df['AuthorKeywords'].str.cat(sep=', ')
wordcloud = WordCloud(
    width=800,
    height=400,
    max_words=150,
    background_color='white'
).generate(all_keywords)

# ---- IMPROVED WORDCLOUD ----
plt.figure(figsize=(12, 6), dpi=100)
plt.imshow(
    wordcloud.recolor(colormap='inferno', random_state=42),
    interpolation='bilinear'
)
plt.title('Keyword Frequency Word Cloud', fontsize=16)
plt.axis('off')
plt.tight_layout()
plt.show()

# Impact of Awards on Citations
plt.figure(figsize=(8, 6), dpi=100)
sns.boxplot(
    x='Award',
    y='CitationCount_CrossRef',
    data=df,
    palette='pastel'
)
plt.title('Impact of Awards on CrossRef Citation Counts', fontsize=16)
plt.xlabel('Award Category', fontsize=14)
plt.ylabel('Citation Count (CrossRef)', fontsize=14)
plt.xticks(rotation=15, fontsize=12)
plt.tight_layout()
plt.show()

# ---- NEW BLOCK ---- # 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Set a consistent style
sns.set_style('whitegrid')

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

# ---- ENHANCED TREND ANALYSIS ON CITATION COUNTS ----
plt.figure(figsize=(14, 6))
# Solid lines for conferences
sns.lineplot(
    data=df,
    x='Year', y='AminerCitationCount',
    hue='Conference',
    palette='tab10',
    linewidth=2.2,
    legend='full'
)
# Dashed lines for paper types
sns.lineplot(
    data=df,
    x='Year', y='CitationCount_CrossRef',
    hue='PaperType',
    palette='Set2',
    linestyle='--',
    linewidth=2.2,
    legend='full'
)
plt.title('Citation Trends Over Time\nSolid = Conference | Dashed = Paper Type', fontsize=16, weight='bold')
plt.xlabel('Year', fontsize=13)
plt.ylabel('Citation Count', fontsize=13)
plt.legend(title='Line Legend', fontsize=10, title_fontsize=11, bbox_to_anchor=(1.02, 1), loc='upper left')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ---- MACHINE LEARNING TO PREDICT AWARDS ----
X = df[['PaperType', 'Downloads_Xplore', 'Year', 'Conference', 'AuthorKeywords']].copy()
X['Conference'] = LabelEncoder().fit_transform(X['Conference'])
X['PaperType'] = LabelEncoder().fit_transform(X['PaperType'])
X['AuthorKeywords'] = X['AuthorKeywords'].apply(lambda x: len(x.split(',')))
y = df['Award']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)
y_pred = rf_classifier.predict(X_test)

# Display tree feature importance
importances = rf_classifier.feature_importances_
indices = np.argsort(importances)[::-1]
feat_names = [X.columns[i] for i in indices]

plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices], y=feat_names, palette='viridis')
plt.title("Random Forest Feature Importances", fontsize=15, weight='bold')
plt.xlabel("Importance Score", fontsize=12)
plt.ylabel("Feature", fontsize=12)

# Annotate bar values
for i, v in enumerate(importances[indices]):
    plt.text(v + 0.002, i, f"{v:.2f}", va='center', fontsize=10)

plt.tight_layout()
plt.show()


# Convert InternalReferences from strings of DOIs to counts of references
df['InternalReferencesCount'] = df['InternalReferences'].apply(
    lambda x: len(x.split(';')) if x else 0
)

# ---- REGRESSION ANALYSIS FOR DOWNLOAD PREDICTION ----
predictors = [
    'AminerCitationCount',
    'GraphicsReplicabilityStamp',
    'InternalReferencesCount',
    'Year',
    'Conference'
]
X2 = df[predictors].copy()
X2['Conference'] = LabelEncoder().fit_transform(X2['Conference'])
X2['GraphicsReplicabilityStamp'] = X2['GraphicsReplicabilityStamp'].apply(
    lambda x: 1 if x == 'Has Stamp' else 0
)
y2 = df['Downloads_Xplore']

X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2, y2, test_size=0.3, random_state=42
)

lr = LinearRegression()
lr.fit(X2_train, y2_train)
y2_pred = lr.predict(X2_test)

r2 = metrics.r2_score(y2_test, y2_pred)
rmse = np.sqrt(metrics.mean_squared_error(y2_test, y2_pred))

plt.figure(figsize=(10, 6))
plt.scatter(
    y2_test, y2_pred,
    alpha=0.5, color='steelblue', edgecolor='k', linewidth=0.3
)
# 45-degree perfect prediction line
lims = [
    np.min([plt.gca().get_xlim(), plt.gca().get_ylim()]),  # min of both axes
    np.max([plt.gca().get_xlim(), plt.gca().get_ylim()]),  # max of both axes
]
plt.plot(lims, lims, '--', color='gray', linewidth=1)
plt.xlim(lims)
plt.ylim(lims)

plt.title('Actual vs. Predicted Downloads', fontsize=15, weight='bold')
plt.xlabel('Actual Downloads', fontsize=13)
plt.ylabel('Predicted Downloads', fontsize=13)
plt.text(
    0.05, 0.95,
    f"$R^2$ = {r2:.2f}\nRMSE = {rmse:.0f}",
    transform=plt.gca().transAxes,
    fontsize=11,
    verticalalignment='top',
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7)
)
plt.tight_layout()
plt.show()


# ---- KEYWORD TREND ANALYSIS (WORDCLOUD) ----
all_keywords = df['AuthorKeywords'].str.cat(sep=', ')
wordcloud = WordCloud(
    width=800,
    height=400,
    max_words=150,
    background_color='white',
    colormap='tab20'
).generate(all_keywords)

plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.title('Word Cloud of Author Keywords', fontsize=16, weight='bold')
plt.axis('off')
plt.tight_layout()
plt.show()


# ---- IMPACT OF AWARDS ON CITATIONS ----
award_cites = df[df['Award'] != 'No Award']['CitationCount_CrossRef']
no_award_cites = df[df['Award'] == 'No Award']['CitationCount_CrossRef']

plt.figure(figsize=(10, 6))
sns.boxplot(
    data=[award_cites, no_award_cites],
    palette=['#8dd3c7', '#fb8072'],
    notch=True
)
plt.xticks([0, 1], ['Awarded Papers', 'Non-Awarded Papers'], fontsize=12)
plt.ylabel('Citation Count (CrossRef)', fontsize=13)
plt.title('Distribution of CrossRef Citations\nby Award Status', fontsize=16, weight='bold')
plt.tight_layout()
plt.show()

# ---- NEW BLOCK ---- # 
# ---- NEW BLOCK ---- # 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
from wordcloud import WordCloud, STOPWORDS
import numpy as np

# Load the dataset
data = pd.read_csv('dataset.csv')

# Fill NaN values with appropriate methods
data.fillna({
    'FirstPage': data['FirstPage'].mean(),
    'LastPage': data['LastPage'].mean(),
    'Abstract': '',
    'AuthorNames-Deduped': 'Unknown',
    'AuthorNames': 'Unknown',
    'AuthorAffiliation': 'Unknown',
    'InternalReferences': '',
    'AuthorKeywords': '',
    'AminerCitationCount': data['AminerCitationCount'].mean(),
    'Downloads_Xplore': data['Downloads_Xplore'].mean(),
    'Award': 'No',
    'GraphicsReplicabilityStamp': 'Not Applicable'
}, inplace=True)

# Convert categorical variables to type category
data['Conference'] = data['Conference'].astype('category')
data['PaperType'] = data['PaperType'].astype('category')
data['Award'] = data['Award'].apply(lambda x: 'Yes' if x != 'No' else 'No').astype('category')

# 1. Enhanced Trend Analysis on Citation Counts
def trend_analysis():
    sns.set(style="darkgrid")
    plt.figure(figsize=(14, 7))
    sns.lineplot(
        data=data,
        x="Year",
        y="AminerCitationCount",
        hue="Conference",
        style="PaperType",
        markers=True,
        dashes=False,
        palette="tab10",
        linewidth=2,
        markeredgecolor="black",
        markeredgewidth=0.5
    )
    plt.title('Trend of Aminer Citation Counts Over Years by Conference & Paper Type', fontsize=16)
    plt.xlabel('Publication Year', fontsize=12)
    plt.ylabel('Average Aminer Citation Count', fontsize=12)
    plt.xticks(rotation=45)
    plt.legend(title='Conference / PaperType', loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()

# 2. Machine Learning to Predict Awards
def predict_awards():
    features = ['PaperType', 'AuthorKeywords', 'InternalReferences', 'Downloads_Xplore', 'Year', 'Conference']
    X = pd.get_dummies(data[features])
    y = data['Award'].map({'No': 0, 'Yes': 1})

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    counts = [accuracy * len(y_test), (1 - accuracy) * len(y_test)]
    labels = ['Correct', 'Incorrect']

    plt.figure(figsize=(8, 5))
    bar = sns.barplot(
        x=labels,
        y=counts,
        palette=['#4caf50', '#f44336']
    )
    plt.title(f'Award Prediction Results (Accuracy: {accuracy:.2%})', fontsize=14)
    plt.ylabel('Number of Test Samples', fontsize=12)
    plt.ylim(0, len(y_test) + 5)
    # Annotate counts on bars
    for idx, val in enumerate(counts):
        plt.text(idx, val + 1, f'{int(val)}', ha='center', fontweight='bold')
    plt.tight_layout()
    plt.show()

# 3. Regression Analysis for Download Prediction
def download_prediction():
    features = ['AminerCitationCount', 'GraphicsReplicabilityStamp', 'InternalReferences', 'Year', 'Conference']
    X = pd.get_dummies(data[features])
    y = data['Downloads_Xplore']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)

    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, predictions, alpha=0.6, color='steelblue', label='Predicted vs Actual')
    # Perfect-prediction line
    min_val, max_val = y_test.min(), y_test.max()
    plt.plot([min_val, max_val], [min_val, max_val], '--', color='red', label='Ideal: y = x')
    plt.title(f'Download Prediction: True vs. Predicted\nMean Squared Error: {mse:.2f}', fontsize=14)
    plt.xlabel('Actual Download Count', fontsize=12)
    plt.ylabel('Predicted Download Count', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.show()

# 4. Keyword Trend Analysis
def keyword_trend_analysis():
    plt.figure(figsize=(12, 6))
    all_keywords = ' '.join(data['AuthorKeywords'].dropna().astype(str))
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        stopwords=STOPWORDS,
        colormap='viridis',
        max_words=100,
        random_state=42
    ).generate(all_keywords)

    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud of Author-Provided Keywords', fontsize=16)
    plt.tight_layout()
    plt.show()

# 5. Impact of Awards on Citations
def impact_of_awards():
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        x='Award',
        y='CitationCount_CrossRef',
        data=data,
        palette='Pastel1'
    )
    plt.title('Impact of Awards on CrossRef Citation Counts', fontsize=16)
    plt.xlabel('Received Award?', fontsize=12)
    plt.ylabel('Citation Count (CrossRef)', fontsize=12)
    plt.tight_layout()
    plt.show()

# Running the analyses
trend_analysis()
predict_awards()
download_prediction()
keyword_trend_analysis()
impact_of_awards()

# ---- NEW BLOCK ---- # 
# ---- NEW BLOCK ---- # 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the dataset
df = pd.read_csv('dataset.csv')

# Fill missing numerical data with median
numerical_columns = ['AminerCitationCount', 'CitationCount_CrossRef', 
                     'PubsCited_CrossRef', 'Downloads_Xplore', 
                     'FirstPage', 'LastPage']
df[numerical_columns] = df[numerical_columns].fillna(df[numerical_columns].median())

# Fill missing categorical data with mode
# Ensure all string/text fields are filled with ''
text_columns = ['AuthorKeywords', 'InternalReferences']
df[text_columns] = df[text_columns].fillna('')

# Fill other categorical data with mode
categorical_columns = ['PaperType', 'Award', 'GraphicsReplicabilityStamp']
for column in categorical_columns:
    df[column] = df[column].fillna(df[column].mode()[0])

# Convert relevant columns to categorical or maintain as string where applicable
df['Conference'] = df['Conference'].astype('category')
df['PaperType'] = df['PaperType'].astype('category')
df['Year'] = pd.to_datetime(df['Year'], format='%Y').dt.year

# Convert 'Award' to binary representing Yes or No
df['Award'] = df['Award'].apply(lambda x: 1 if x == 'Yes' else 0)

# Assume handling non-numeric properly for 'InternalReferences'
# Count the number of references
df['InternalReferencesCount'] = df['InternalReferences'].apply(lambda x: len(str(x).split(';')) if x else 0)

# Features and target for machine learning, ensuring no NaNs in text
features = df[['PaperType', 'AuthorKeywords', 'InternalReferencesCount', 'Downloads_Xplore', 'Year', 'Conference']]
target = df['Award']

# Column Transformer and Pipeline Setup
preprocessor = ColumnTransformer(
    transformers=[
        ('keyword', CountVectorizer(max_features=50, stop_words='english'), 'AuthorKeywords'),
        ('paper_type', OneHotEncoder(), ['PaperType']),
        ('conference', OneHotEncoder(), ['Conference'])
    ],
    remainder='passthrough'  # Keeps InternalReferencesCount, Downloads_Xplore, Year as they are.
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Fit the Model
pipeline.fit(X_train, y_train)

# Predictions and Evaluation
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

# Confusion Matrix Visualization
plt.figure(figsize=(8, 6))
sns.heatmap(
    confusion_matrix(y_test, y_pred),
    annot=True,
    fmt='d',
    cmap='Blues',
    cbar_kws={'label': 'Number of Papers'},
    xticklabels=['No Award', 'Award'],
    yticklabels=['No Award', 'Award']
)
plt.title('Confusion Matrix: Award Prediction', fontsize=14, pad=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.tight_layout()
plt.show()

# ---- NEW BLOCK ---- # 
import pandas as pd
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from wordcloud import WordCloud
import numpy as np

# Use a clean Seaborn style and consistent palette
sns.set_theme(style="whitegrid", palette="Set2")

# Load the dataset
df = pd.read_csv('dataset.csv')

# Fill NaN values with appropriate imputation
num_imputer = SimpleImputer(strategy='mean')
df[['FirstPage', 'LastPage', 'AminerCitationCount', 'CitationCount_CrossRef',
    'PubsCited_CrossRef', 'Downloads_Xplore']] = num_imputer.fit_transform(
    df[['FirstPage', 'LastPage', 'AminerCitationCount', 'CitationCount_CrossRef',
        'PubsCited_CrossRef', 'Downloads_Xplore']])

cat_imputer = SimpleImputer(strategy='most_frequent')
df[['Conference', 'PaperType', 'AuthorNames-Deduped',
    'AuthorKeywords', 'Award']] = cat_imputer.fit_transform(
    df[['Conference', 'PaperType', 'AuthorNames-Deduped',
        'AuthorKeywords', 'Award']])

df['Conference'] = df['Conference'].astype('category')
df['PaperType'] = df['PaperType'].astype('category')
df['Award']      = df['Award'].astype('category')


# 1. Enhanced Trend Analysis on Citation Counts
def trend_analysis(df):
    plt.figure(figsize=(14, 7))
    sns.lineplot(
        data=df.sort_values('Year'),
        x='Year', y='AminerCitationCount',
        hue='Conference',
        marker='o', linewidth=2, markersize=8
    )
    plt.title('Trend of Aminer Citation Counts by Conference', fontsize=16)
    plt.xlabel('Publication Year', fontsize=14)
    plt.ylabel('Aminer Citation Count', fontsize=14)
    plt.xticks(df['Year'].unique(), rotation=45)
    plt.legend(title='Conference', loc='upper left', bbox_to_anchor=(1,1))
    plt.tight_layout()
    plt.show()


# 2. Machine Learning to Predict Awards
def predict_awards(df):
    features = df[['PaperType','Downloads_Xplore','Year','Conference']].copy()
    target   = df['Award'].apply(lambda x: 1 if str(x).lower().startswith('y') else 0)
    features = pd.get_dummies(features, drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    print(f'Model Accuracy on Test Set: {acc:.3f}')

    # -- Feature Importance Plot --
    importances = pd.Series(model.feature_importances_, index=features.columns)
    imp_sorted  = importances.sort_values(ascending=False).head(10)
    plt.figure(figsize=(10,6))
    sns.barplot(x=imp_sorted.values, y=imp_sorted.index, palette="viridis")
    plt.title('Top 10 Feature Importances for Award Prediction', fontsize=14)
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()
    plt.show()


# 3. Keyword Trend Analysis
def keyword_trend_analysis(df):
    all_keywords = ' '.join(df['AuthorKeywords'].dropna())
    wordcloud = WordCloud(
        width=800, height=400,
        background_color='white',
        colormap='tab10'
    ).generate(all_keywords)
    plt.figure(figsize=(15, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title('Word Cloud of Author Keywords', fontsize=18)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


# 4. Citation Prediction Model
def citation_prediction(df):
    features = df[['Year','PaperType','Downloads_Xplore']].copy()
    target   = df['CitationCount_CrossRef']
    features = pd.get_dummies(features, drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )
    reg = LinearRegression()
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)

    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, color='teal', edgecolor='k')
    # plot ideal diagonal
    lims = [
        np.min([plt.gca().get_xlim(), plt.gca().get_ylim()]),  # min of both axes
        np.max([plt.gca().get_xlim(), plt.gca().get_ylim()]),  # max of both axes
    ]
    plt.plot(lims, lims, 'r--', linewidth=2, label='Ideal Prediction')
    plt.xlim(lims)
    plt.ylim(lims)
    plt.title('Actual vs Predicted Citation Counts', fontsize=16)
    plt.xlabel('Actual Citation Count', fontsize=14)
    plt.ylabel('Predicted Citation Count', fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.show()


# 5. Impact of Awards on Citations
def impact_of_awards(df):
    fig, ax = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
    sns.boxplot(
        data=df, x='Award', y='CitationCount_CrossRef',
        palette='Set2', ax=ax[0]
    )
    ax[0].set_title('Awards vs. Citation Count', fontsize=14)
    ax[0].set_xlabel('Award Received', fontsize=12)
    ax[0].set_ylabel('Citation Count (CrossRef)', fontsize=12)

    sns.boxplot(
        data=df, x='Award', y='Downloads_Xplore',
        palette='Set2', ax=ax[1]
    )
    ax[1].set_title('Awards vs. Downloads', fontsize=14)
    ax[1].set_xlabel('Award Received', fontsize=12)
    ax[1].set_ylabel('Downloads (Xplore)', fontsize=12)

    for subax in ax:
        subax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


# Execute all
trend_analysis(df)
predict_awards(df)
keyword_trend_analysis(df)
citation_prediction(df)
impact_of_awards(df)

# ---- NEW BLOCK ---- # 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, r2_score
from sklearn.linear_model import LinearRegression
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# Set a consistent style
sns.set_style("whitegrid")
plt.rcParams.update({'figure.dpi': 100, 'font.size': 12})

# Load Dataset
df = pd.read_csv('dataset.csv')

# Fill NaN values using modern methods
df['FirstPage'].fillna(df['FirstPage'].median(), inplace=True)
df['LastPage'].fillna(df['LastPage'].median(), inplace=True)
df['Abstract'].fillna('', inplace=True)
df['AuthorKeywords'].fillna('', inplace=True)
df['InternalReferences'].fillna('', inplace=True)
df['AminerCitationCount'].fillna(df['AminerCitationCount'].median(), inplace=True)
df['Downloads_Xplore'].fillna(df['Downloads_Xplore'].median(), inplace=True)

# Convert the 'Award' column to binary
df['Award'] = df['Award'].notnull().astype(int)

# 1. Enhanced Trend Analysis on Citation Counts
def trend_analysis(df):
    plt.figure(figsize=(14, 7))
    ax = sns.lineplot(
        data=df, 
        x='Year', 
        y='CitationCount_CrossRef', 
        hue='Conference',
        marker='o'
    )
    ax.set_title('Citation Trends Over Years by Conference', fontsize=16)
    ax.set_xlabel('Publication Year', fontsize=14)
    ax.set_ylabel('CrossRef Citation Count', fontsize=14)
    ax.legend(title='Conference', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

trend_analysis(df)


# 2. Machine Learning to Predict Awards
def award_prediction(df):
    features = ['PaperType', 'AuthorKeywords', 'InternalReferences', 'Downloads_Xplore', 'Year', 'Conference']
    df_dummies = pd.get_dummies(df[features], drop_first=True)
    X = df_dummies
    y = df['Award']
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=0.3, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    print("Classification Report:\n")
    print(classification_report(y_test, predictions, target_names=['No Award','Awarded']))
    
    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(6, 5))
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    ax.set_title('Confusion Matrix for Award Prediction', fontsize=14)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xticklabels(['No Award','Awarded'])
    ax.set_yticklabels(['No Award','Awarded'], rotation=0)
    plt.tight_layout()
    plt.show()

award_prediction(df)


# 3. Regression Analysis for Download Prediction
def download_prediction(df):
    features = ['AminerCitationCount', 'InternalReferences', 'Year', 'Conference']
    df_dummies = pd.get_dummies(df[features], drop_first=True)
    X = df_dummies
    y = df['Downloads_Xplore']
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=0.3, random_state=42)

    reg = LinearRegression()
    reg.fit(X_train, y_train)
    predictions = reg.predict(X_test)
    r2 = r2_score(y_test, predictions)

    plt.figure(figsize=(7, 7))
    plt.scatter(y_test, predictions, alpha=0.6, edgecolor='k')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
             'r--', lw=2, label='Ideal Fit (y = x)')
    plt.xlabel("Actual Downloads", fontsize=14)
    plt.ylabel("Predicted Downloads", fontsize=14)
    plt.title(f"Actual vs Predicted Downloads (R² = {r2:.2f})", fontsize=16)
    plt.legend()
    plt.tight_layout()
    plt.show()

download_prediction(df)


# 4. Keyword Trend Analysis
def keyword_trend_analysis(df):
    vectorizer = TfidfVectorizer(max_features=100)
    X_tfidf = vectorizer.fit_transform(df['AuthorKeywords'])
    svd = TruncatedSVD(n_components=2, random_state=42)
    keywords_2d = svd.fit_transform(X_tfidf)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        keywords_2d[:, 0], keywords_2d[:, 1], 
        c=df['Year'], cmap='viridis', alpha=0.7, s=50
    )
    cbar = plt.colorbar(scatter)
    cbar.set_label('Publication Year', fontsize=12)
    plt.title('2D Projection of Author Keywords Over Time', fontsize=16)
    plt.xlabel('Component 1', fontsize=14)
    plt.ylabel('Component 2', fontsize=14)
    plt.tight_layout()
    plt.show()

keyword_trend_analysis(df)


# 5. Impact of Awards on Citations
def impact_of_awards(df):
    plt.figure(figsize=(8, 6))
    ax = sns.boxplot(
        x='Award', 
        y='CitationCount_CrossRef', 
        data=df, 
        palette=['lightgray', 'skyblue']
    )
    ax.set_title('Comparison of CrossRef Citations by Award Status', fontsize=16)
    ax.set_xlabel('Award Status', fontsize=14)
    ax.set_ylabel('CrossRef Citation Count', fontsize=14)
    ax.set_xticklabels(['No Award', 'Awarded'])
    plt.tight_layout()
    plt.show()

impact_of_awards(df)

# ---- NEW BLOCK ---- # 
# ---- NEW BLOCK ---- # 
# ---- NEW BLOCK ---- # 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import statsmodels.api as sm
from collections import Counter

# Load data
df = pd.read_csv('dataset.csv')

# Fill NaN values
df['AminerCitationCount'].fillna(df['AminerCitationCount'].median(), inplace=True)
df['CitationCount_CrossRef'].fillna(df['CitationCount_CrossRef'].median(), inplace=True)
df['Downloads_Xplore'].fillna(df['Downloads_Xplore'].median(), inplace=True)
df['Award'].fillna('No Award', inplace=True)
df['GraphicsReplicabilityStamp'].fillna('No Stamp', inplace=True)
df['InternalReferences'].fillna('', inplace=True)
df['AuthorKeywords'].fillna('', inplace=True)

# Convert categorical columns as needed
df['PaperType'] = df['PaperType'].astype('category')
df['Conference'] = df['Conference'].astype('category')

# Global style
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 100

# 1. Enhanced Trend Analysis on Citation Counts
def enhanced_trend_analysis():
    trend_data = df.groupby(['Year', 'Conference', 'PaperType'])[['AminerCitationCount',
                                                                   'CitationCount_CrossRef']].sum().reset_index()
    plt.figure(figsize=(14, 6))
    sns.lineplot(data=trend_data,
                 x='Year',
                 y='AminerCitationCount',
                 hue='Conference',
                 style='PaperType',
                 markers=True,
                 dashes=False,
                 linewidth=2,
                 palette='tab10')
    plt.title('Trend of Aminer Citations Over Time by Conference & Paper Type', fontsize=14)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Total Aminer Citation Count', fontsize=12)
    plt.legend(title='Conference / PaperType', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

# 2. Machine Learning to Predict Awards
def predict_awards():
    features = pd.get_dummies(df[['PaperType', 'Downloads_Xplore', 'Year', 'Conference']])
    target = (df['Award'] == 'Award').astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print("Classification Report for Award Prediction:\n")
    print(classification_report(y_test, preds, target_names=['No Award', 'Award']))

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    names = [features.columns[i] for i in indices]

    plt.figure(figsize=(10, 8))
    sns.barplot(x=importances[indices],
                y=names,
                palette='viridis')
    plt.title('Feature Importances in RandomForest Award Model', fontsize=14)
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()
    plt.show()

# 3. Regression Analysis for Download Prediction
def regression_analysis():
    # Convert internal references to a simple count
    df['NumInternalRefs'] = df['InternalReferences'].apply(
        lambda s: len(s.split(';')) if s else 0)

    features = ['AminerCitationCount', 'NumInternalRefs', 'Year', 'Conference']
    X = pd.get_dummies(df[features], drop_first=True)
    X = sm.add_constant(X)
    y = df['Downloads_Xplore']

    model = sm.OLS(y, X).fit()
    print(model.summary())

    # Scatter plot of predicted vs actual with R^2 annotation
    y_pred = model.predict(X)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_pred, y=y, alpha=0.6, color='teal')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=1)
    plt.title('Predicted vs Actual Downloads', fontsize=14)
    plt.xlabel('Predicted Downloads', fontsize=12)
    plt.ylabel('Actual Downloads', fontsize=12)
    plt.annotate(f'$R^2$ = {model.rsquared:.2f}',
                 xy=(0.05, 0.95),
                 xycoords='axes fraction',
                 fontsize=12,
                 backgroundcolor='white')
    plt.tight_layout()
    plt.show()

    # Residuals plot
    plt.figure(figsize=(8, 6))
    sns.residplot(x=y_pred, y=y - y_pred, lowess=True, color='purple', line_kws={'lw':1})
    plt.axhline(0, linestyle='--', color='gray', linewidth=1)
    plt.title('Residuals vs Fitted Values', fontsize=14)
    plt.xlabel('Fitted Values', fontsize=12)
    plt.ylabel('Residuals', fontsize=12)
    plt.tight_layout()
    plt.show()

# 4. Keyword Trend Analysis
def keyword_trend_analysis():
    all_keywords = [kw.strip() for row in df['AuthorKeywords'] for kw in row.split(';') if kw]
    keyword_freq = Counter(all_keywords).most_common(10)
    keywords, counts = zip(*keyword_freq)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(counts),
                y=list(keywords),
                palette='magma')
    plt.title('Top 10 Author Keywords', fontsize=14)
    plt.xlabel('Count of Occurrences', fontsize=12)
    plt.ylabel('Keyword', fontsize=12)
    plt.tight_layout()
    plt.show()

# 5. Impact of Awards on Citations
def impact_of_awards_on_citations():
    award_citations = df.groupby('Award')[['CitationCount_CrossRef',
                                           'Downloads_Xplore']].mean().reset_index()

    plt.figure(figsize=(8, 6))
    sns.barplot(x='Award',
                y='CitationCount_CrossRef',
                data=award_citations,
                palette='coolwarm')
    plt.title('Mean CrossRef Citations by Award Status', fontsize=14)
    plt.xlabel('Award Status', fontsize=12)
    plt.ylabel('Average CrossRef Citation Count', fontsize=12)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.barplot(x='Award',
                y='Downloads_Xplore',
                data=award_citations,
                palette='cubehelix')
    plt.title('Mean Xplore Downloads by Award Status', fontsize=14)
    plt.xlabel('Award Status', fontsize=12)
    plt.ylabel('Average Downloads', fontsize=12)
    plt.tight_layout()
    plt.show()

# Run all visualizations
enhanced_trend_analysis()
predict_awards()
regression_analysis()
keyword_trend_analysis()
impact_of_awards_on_citations()

# ---- NEW BLOCK ---- # 
Below is the **complete** revised script. I have kept all of your original data‐loading, preprocessing, and model‐building steps untouched, but **enhanced every plotting function** for clarity, appropriate sizing, color palettes, axis labels, legends, and—where relevant—added metrics (e.g. R², MSE, confusion matrix).  


# ---- NEW BLOCK ---- # 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report, mean_squared_error, r2_score
from collections import Counter
import numpy as np

# Load the dataset
data = pd.read_csv('dataset.csv')

# Fill NaN values with some meaningful metrics (for simplicity, using forward fill and back fill)
data['AminerCitationCount'].fillna(method='ffill', inplace=True)
data['AminerCitationCount'].fillna(method='bfill', inplace=True)
data['CitationCount_CrossRef'].fillna(data['CitationCount_CrossRef'].mean(), inplace=True)
data['Downloads_Xplore'].fillna(data['Downloads_Xplore'].median(), inplace=True)

# 1. Enhanced Trend Analysis on Citation Counts
trend_data = data.groupby(['Year', 'Conference', 'PaperType']).agg({
    'AminerCitationCount': 'mean',
    'CitationCount_CrossRef': 'mean'
}).reset_index()

plt.figure(figsize=(14, 7), dpi=100)
sns.set_style("whitegrid")
sns.lineplot(
    data=trend_data,
    x='Year', y='AminerCitationCount',
    hue='Conference', style='PaperType',
    palette='tab10', markers=True, dashes=False
)
plt.title('Average Aminer Citation Count by Year, Conference & Paper Type', fontsize=16)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Avg. Aminer Citation Count', fontsize=12)
plt.legend(title='Conference / Paper Type', bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.show()

# 2. Machine Learning to Predict Awards
features = data[['PaperType','AuthorKeywords','InternalReferences','Downloads_Xplore','Year','Conference']].fillna('')
target = data['Award'].apply(lambda x: 1 if x=='Yes' else 0)
features = pd.get_dummies(features)

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))

feature_importances = pd.Series(model.feature_importances_, index=features.columns)
plt.figure(figsize=(10, 6), dpi=100)
feature_importances.nlargest(10).sort_values().plot(
    kind='barh', color='steelblue'
)
plt.title('Top 10 Feature Importances for Award Prediction', fontsize=14)
plt.xlabel('Importance Score', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.tight_layout()
plt.show()

# 3. Regression Analysis for Download Prediction
features = data[['AminerCitationCount','GraphicsReplicabilityStamp','InternalReferences','Year','Conference']].fillna('')
features = pd.get_dummies(features)
downloads = data['Downloads_Xplore']

X_train, X_test, y_train, y_test = train_test_split(features, downloads, test_size=0.3, random_state=42)
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred = linear_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse:.2f}')
print(f'R² Score: {r2:.2f}')

plt.figure(figsize=(10, 6), dpi=100)
plt.scatter(y_test, y_pred, alpha=0.6, color='darkorange', edgecolors='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         'k--', lw=2, label='Ideal Fit')
plt.title('Actual vs Predicted Downloads', fontsize=16)
plt.xlabel('Actual Downloads', fontsize=12)
plt.ylabel('Predicted Downloads', fontsize=12)
plt.text(
    0.05, 0.90,
    f'R² = {r2:.2f}\nMSE = {mse:.2f}',
    transform=plt.gca().transAxes,
    fontsize=12,
    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
)
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

# 4. Keyword Trend Analysis
year_keywords = data[['Year','AuthorKeywords']].dropna()
year_trends = year_keywords.groupby('Year')['AuthorKeywords'].apply(lambda x: ' '.join(x)).reset_index()

trend_list = []
for _, row in year_trends.iterrows():
    counter = Counter(row['AuthorKeywords'].split(';'))
    for kw, count in counter.items():
        trend_list.append({'Year': row['Year'], 'Keyword': kw.strip(), 'Frequency': count})
trend_df = pd.DataFrame(trend_list)

plt.figure(figsize=(14, 7), dpi=100)
top_keywords = trend_df['Keyword'].value_counts().head(10).index
sns.lineplot(
    data=trend_df[trend_df['Keyword'].isin(top_keywords)],
    x='Year', y='Frequency', hue='Keyword',
    palette='tab10', marker='o'
)
plt.title('Top 10 Keyword Frequencies Over Time', fontsize=16)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.legend(title='Keyword', bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.show()

# 5. Impact of Awards on Citations
award_impact = data.groupby('Award').agg({
    'CitationCount_CrossRef': 'mean',
    'Downloads_Xplore': 'mean'
}).reset_index()

fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=100)
sns.barplot(
    data=award_impact, x='Award', y='CitationCount_CrossRef',
    palette=['salmon','seagreen'], ax=axes[0]
)
axes[0].set_title('Avg. CrossRef Citations by Award', fontsize=14)
axes[0].set_xlabel('Award', fontsize=12)
axes[0].set_ylabel('Avg. CitationCount_CrossRef', fontsize=12)

sns.barplot(
    data=award_impact, x='Award', y='Downloads_Xplore',
    palette=['salmon','seagreen'], ax=axes[1]
)
axes[1].set_title('Avg. Downloads by Award', fontsize=14)
axes[1].set_xlabel('Award', fontsize=12)
axes[1].set_ylabel('Avg. Downloads_Xplore', fontsize=12)

plt.tight_layout()
plt.show()

# ---- NEW BLOCK ---- # 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report, confusion_matrix, r2_score
import numpy as np

# Load the dataset
data = pd.read_csv('dataset.csv')

# Preprocessing: Filling NaN values
data.fillna({
    'FirstPage': data['FirstPage'].mean(),
    'LastPage': data['LastPage'].mean(),
    'Abstract': '',
    'AuthorKeywords': '',
    'AminerCitationCount': data['AminerCitationCount'].mean(),
    'Downloads_Xplore': data['Downloads_Xplore'].mean(),
    'Award': 'No',
    'GraphicsReplicabilityStamp': 'No'
}, inplace=True)

# Convert Award and GraphicsReplicabilityStamp to binary
data['Award'] = data['Award'].apply(lambda x: 1 if x == 'Yes' else 0)
data['GraphicsReplicabilityStamp'] = data['GraphicsReplicabilityStamp'].apply(lambda x: 1 if x == 'Yes' else 0)


# 1. Enhanced Trend Analysis on Citation Counts
def citation_trend_analysis(data):
    plt.figure(figsize=(12, 6))
    sns.set_palette("tab10")
    sns.lineplot(data=data, x='Year', y='AminerCitationCount',
                 hue='Conference', marker='o', linewidth=2.0)
    plt.title('Yearly Aminer Citation Trends by Conference', fontsize=14)
    plt.xlabel('Publication Year', fontsize=12)
    plt.ylabel('Aminer Citation Count', fontsize=12)
    plt.legend(title='Conference', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

citation_trend_analysis(data)


# 2. Machine Learning to Predict Awards
def ml_predict_awards(data):
    features = ['PaperType', 'Downloads_Xplore', 'Year', 'Conference']
    X = pd.get_dummies(data[features], drop_first=True)
    y = data['Award']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print('Classification Report:\n', classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', cbar=False,
                xticklabels=['No Award', 'Award'],
                yticklabels=['No Award', 'Award'])
    plt.title('Confusion Matrix: Award Prediction', fontsize=14)
    plt.ylabel('Actual', fontsize=12)
    plt.xlabel('Predicted', fontsize=12)
    plt.tight_layout()
    plt.show()

ml_predict_awards(data)


# 3. Regression Analysis for Download Prediction
def regression_analysis_downloads(data):
    features = ['AminerCitationCount', 'GraphicsReplicabilityStamp', 'Year', 'Conference']
    X = pd.get_dummies(data[features], drop_first=True)
    y = data['Downloads_Xplore']

    model = LinearRegression()
    model.fit(X, y)
    preds = model.predict(X)
    r2 = r2_score(y, preds)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y, y=preds, color='darkorange', alpha=0.6)
    # 45-degree line
    max_val = max(y.max(), preds.max())
    plt.plot([0, max_val], [0, max_val], ls="--", color="grey")
    plt.title('Actual vs. Predicted Downloads on Xplore', fontsize=14)
    plt.xlabel('Actual Downloads', fontsize=12)
    plt.ylabel('Predicted Downloads', fontsize=12)
    plt.text(0.05*max_val, 0.9*max_val, f'R² = {r2:.2f}',
             fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
    plt.tight_layout()
    plt.show()

regression_analysis_downloads(data)


# 4. Keyword Trend Analysis
def keyword_trend_analysis(data):
    # Build a dataframe of top-5 keywords per year
    all_years = sorted(data['Year'].unique())
    top_keywords = {}
    for year in all_years:
        words = ' '.join(data.loc[data['Year'] == year, 'AuthorKeywords']).split()
        freq = pd.Series(words).value_counts().head(5)
        top_keywords[year] = freq

    plt.figure(figsize=(14, 8))
    palette = sns.color_palette('tab20', n_colors=len(all_years))
    for idx, year in enumerate(all_years):
        freq = top_keywords[year]
        sns.barplot(x=freq.values, y=freq.index,
                    color=palette[idx], alpha=0.7, label=str(year))

    plt.title('Top 5 Author Keywords by Year', fontsize=14)
    plt.xlabel('Frequency', fontsize=12)
    plt.ylabel('Keyword', fontsize=12)
    plt.legend(title='Year', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

keyword_trend_analysis(data)


# 5. Impact of Awards on Citations
def impact_of_awards_on_citations(data):
    # Map back to Yes/No labels for clarity
    data['AwardLabel'] = data['Award'].map({0: 'No', 1: 'Yes'})
    plt.figure(figsize=(8, 6))
    sns.set_palette("Set2")
    sns.boxplot(data=data, x='AwardLabel', y='CitationCount_CrossRef')
    plt.title('Citation Counts by Award Status', fontsize=14)
    plt.xlabel('Award Won', fontsize=12)
    plt.ylabel('CrossRef Citation Count', fontsize=12)
    plt.tight_layout()
    plt.show()

impact_of_awards_on_citations(data)

# ---- NEW BLOCK ---- # 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the dataset
df = pd.read_csv('dataset.csv')

# Fill missing values
df.fillna(method='ffill', inplace=True)
df.fillna(method='bfill', inplace=True)

# Group and take median of both citation counts
trend_df = (
    df
    .groupby(['Year', 'PaperType'])[['AminerCitationCount', 'CitationCount_CrossRef']]
    .median()
    .reset_index()
)

# Melt into long form so we can style by source
trend_long = trend_df.melt(
    id_vars=['Year', 'PaperType'],
    value_vars=['AminerCitationCount', 'CitationCount_CrossRef'],
    var_name='CitationSource',
    value_name='MedianCitation'
)

# Rename sources for display
trend_long['CitationSource'] = trend_long['CitationSource'].map({
    'AminerCitationCount':     'Aminer',
    'CitationCount_CrossRef':  'CrossRef'
})

# Set global style and context
sns.set(style='whitegrid', context='talk')

plt.figure(figsize=(16, 9))
sns.lineplot(
    data=trend_long,
    x='Year',
    y='MedianCitation',
    hue='PaperType',
    style='CitationSource',
    markers=True,
    # solid for Aminer, dashed for CrossRef
    dashes={'Aminer':       '',       # solid
            'CrossRef':     (2, 2)},  # dashed
    palette='tab10'
)

plt.title('Median Citation Counts per Year by Paper Type and Source', fontsize=18)
plt.xlabel('Publication Year', fontsize=14)
plt.ylabel('Median Citation Count', fontsize=14)
plt.xticks(rotation=45)

plt.legend(
    title='Paper Type / Source',
    title_fontsize=12,
    fontsize=10,
    loc='upper left',
    bbox_to_anchor=(1.05, 1)
)

plt.tight_layout()
plt.show()

# ---- NEW BLOCK ---- # 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import statsmodels.api as sm
from wordcloud import WordCloud
from sklearn.impute import SimpleImputer

# Set a clean, consistent style
sns.set(style='whitegrid', palette='muted')

# Load dataset
df = pd.read_csv('dataset.csv')

# Fill NaN values using SimpleImputer
imputer_mean = SimpleImputer(strategy='mean')
imputer_mode = SimpleImputer(strategy='most_frequent')

# Fill numerical columns with mean
numerical_cols = ['AminerCitationCount', 'CitationCount_CrossRef', 'Downloads_Xplore']
for col in numerical_cols:
    df[col] = imputer_mean.fit_transform(df[[col]])

# Handle potential non-convertible strings in numerical columns
for col in numerical_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Fill categorical columns with the mode
categorical_cols = ['PaperType', 'AuthorKeywords', 'InternalReferences', 'Conference', 'Award']
df[categorical_cols] = imputer_mode.fit_transform(df[categorical_cols])

# Convert 'Award' column to category
df['Award'] = df['Award'].astype('category')

# Enhanced Trend Analysis on Citation Counts
citation_trends = df.groupby(['Year', 'Conference']).agg({
    'AminerCitationCount': 'mean',
    'CitationCount_CrossRef': 'mean'
}).reset_index()

# Aminer Citation Trends
plt.figure(figsize=(12, 6))
sns.lineplot(
    data=citation_trends,
    x='Year', y='AminerCitationCount',
    hue='Conference',
    marker='o',
    palette='tab10'
)
plt.title('Average Aminer Citation Count Over Time by Conference', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Mean Aminer Citation Count', fontsize=12)
plt.legend(title='Conference', bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.show()

# CrossRef Citation Trends
plt.figure(figsize=(12, 6))
sns.lineplot(
    data=citation_trends,
    x='Year', y='CitationCount_CrossRef',
    hue='Conference',
    marker='s',
    palette='tab10'
)
plt.title('Average CrossRef Citation Count Over Time by Conference', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Mean CrossRef Citation Count', fontsize=12)
plt.legend(title='Conference', bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Machine Learning to Predict Awards
X = df[['Year', 'PaperType', 'AuthorKeywords', 'InternalReferences', 'Downloads_Xplore', 'Conference']]
y = df['Award']
X = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Confusion Matrix
disp = ConfusionMatrixDisplay.from_estimator(
    model, X_test, y_test,
    cmap='Blues',
    colorbar=True
)
plt.title('Confusion Matrix for Award Prediction', fontsize=14)
plt.xlabel('Predicted Award', fontsize=12)
plt.ylabel('True Award', fontsize=12)
plt.tight_layout()
plt.show()

print(classification_report(y_test, y_pred))

# Regression Analysis for Download Prediction
X2 = df[['AminerCitationCount', 'GraphicsReplicabilityStamp', 'Year']]
X2 = X2.apply(pd.to_numeric, errors='coerce').fillna(0)
X2 = sm.add_constant(X2)

y2 = df['Downloads_Xplore']
download_model = sm.OLS(y2, X2).fit()
print(download_model.summary())

# Keyword Trend Analysis via WordClouds
keywords_per_year = df.groupby('Year')['AuthorKeywords'] \
                      .apply(lambda kw: ' '.join(kw)) \
                      .reset_index()

for _, row in keywords_per_year.iterrows():
    fig, ax = plt.subplots(figsize=(10, 5))
    wc = WordCloud(
        width=800, height=400,
        background_color='white',
        colormap='tab20'
    ).generate(row['AuthorKeywords'])
    ax.imshow(wc, interpolation='bilinear')
    ax.set_title(f'Keyword Trends for Year {row["Year"]}', fontsize=14)
    ax.axis('off')
    plt.tight_layout()
    plt.show()

# Impact of Awards on Citations
plt.figure(figsize=(10, 6))
sns.boxplot(
    data=df,
    x='Award', y='CitationCount_CrossRef',
    palette='Set2'
)
plt.title('Impact of Awards on CrossRef Citation Count', fontsize=14)
plt.xlabel('Award Received', fontsize=12)
plt.ylabel('CrossRef Citation Count', fontsize=12)
plt.tight_layout()
plt.show()

# ---- NEW BLOCK ---- # 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, r2_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Load the dataset
df = pd.read_csv('dataset.csv')

# Fill NaN values
df['FirstPage'].fillna(df['FirstPage'].mean(), inplace=True)
df['LastPage'].fillna(df['LastPage'].mean(), inplace=True)
df['Abstract'].fillna('No Abstract', inplace=True)
df['InternalReferences'].fillna('No References', inplace=True)
df['AuthorKeywords'].fillna('No Keywords', inplace=True)
df['AminerCitationCount'].fillna(df['AminerCitationCount'].mean(), inplace=True)
df['Downloads_Xplore'].fillna(df['Downloads_Xplore'].mean(), inplace=True)

# Convert categorical variables
le = LabelEncoder()
df['Award'].fillna('No Award', inplace=True)
df['GraphicsReplicabilityStamp'].fillna('No Stamp', inplace=True)

# 1. Enhanced Trend Analysis on Citation Counts
plt.figure(figsize=(12, 6))
sns.lineplot(
    data=df, 
    x='Year', 
    y='AminerCitationCount', 
    hue='Conference',
    palette='tab10',
    marker='o',
    linewidth=2
)
plt.title('Trend of Aminer Citation Counts by Conference', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Aminer Citation Count', fontsize=12)
plt.legend(title='Conference', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 2. Machine Learning to Predict Awards
df['Award'] = le.fit_transform(df['Award'])
X = df[['PaperType', 'Downloads_Xplore', 'Year', 'Conference']]
y = df['Award']
X = pd.get_dummies(X, columns=['PaperType', 'Conference'])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm, 
    annot=True, 
    fmt='d', 
    cmap='Blues',
    cbar=False,
    xticklabels=le.classes_,
    yticklabels=le.classes_
)
plt.title('Confusion Matrix: Award Prediction', fontsize=14)
plt.xlabel('Predicted Award (Label)', fontsize=12)
plt.ylabel('True Award (Label)', fontsize=12)
plt.tight_layout()
plt.show()

# 3. Regression Analysis for Download Prediction
X = df[['AminerCitationCount', 'Year', 'Conference']]
X = pd.get_dummies(X, columns=['Conference'])
y = df['Downloads_Xplore']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
reg_model = LinearRegression()
reg_model.fit(X_train, y_train)
y_pred_reg = reg_model.predict(X_test)
r2 = r2_score(y_test, y_pred_reg)

plt.figure(figsize=(7, 7))
plt.scatter(y_test, y_pred_reg, alpha=0.6, edgecolor='k')
lims = [min(y_test.min(), y_pred_reg.min()), max(y_test.max(), y_pred_reg.max())]
plt.plot(lims, lims, 'r--', linewidth=1)
plt.title('Download Prediction: Actual vs. Predicted', fontsize=14)
plt.xlabel('Actual Downloads (Xplore)', fontsize=12)
plt.ylabel('Predicted Downloads (Xplore)', fontsize=12)
plt.text(
    0.05, 0.95, 
    f'$R^2 = {r2:.2f}$', 
    transform=plt.gca().transAxes,
    fontsize=12,
    verticalalignment='top'
)
plt.tight_layout()
plt.show()

# 4. Keyword Trend Analysis
keywords_per_year = (
    df['AuthorKeywords']
    .str.get_dummies(sep='; ')
    .groupby(df['Year'])
    .sum()
)
plt.figure(figsize=(14, 8))
sns.heatmap(
    keywords_per_year, 
    cmap='YlGnBu', 
    cbar_kws={'label': 'Frequency'},
    linewidths=0.5
)
plt.title('Keyword Usage Trends Over Years', fontsize=14)
plt.xlabel('Keywords', fontsize=12)
plt.ylabel('Year', fontsize=12)
plt.tight_layout()
plt.show()

# 5. Impact of Awards on Citations
award_citations = (
    df.groupby('Award')['CitationCount_CrossRef']
    .mean()
    .reset_index()
    .sort_values(by='CitationCount_CrossRef', ascending=False)
)
plt.figure(figsize=(8, 5))
sns.barplot(
    data=award_citations, 
    x='Award', 
    y='CitationCount_CrossRef',
    palette='Set2'
)
plt.title('Mean CrossRef Citations: Awarded vs. Non-Awarded Papers', fontsize=14)
plt.xlabel('Award Status', fontsize=12)
plt.ylabel('Average CrossRef Citation Count', fontsize=12)
plt.tight_layout()
plt.show()

# ---- NEW BLOCK ---- # 
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import nltk
from wordcloud import WordCloud

# Set a consistent theme
sns.set_theme(style='whitegrid', context='talk')

# Load the dataset
dataset = pd.read_csv('dataset.csv')

# Handle missing values by filling with modern methods
dataset['FirstPage'].fillna(dataset['FirstPage'].median(), inplace=True)
dataset['LastPage'].fillna(dataset['LastPage'].median(), inplace=True)
dataset['Abstract'].fillna('', inplace=True)
dataset['AuthorNames-Deduped'].fillna('Unknown', inplace=True)
dataset['AuthorAffiliation'].fillna('Unknown', inplace=True)
dataset['InternalReferences'].fillna('[]', inplace=True)
dataset['AuthorKeywords'].fillna('', inplace=True)
dataset['AminerCitationCount'].fillna(dataset['AminerCitationCount'].median(), inplace=True)
dataset['GraphicsReplicabilityStamp'].fillna('No', inplace=True)
dataset['Downloads_Xplore'].fillna(dataset['Downloads_Xplore'].median(), inplace=True)

# Ensure correct data types
dataset['Year'] = dataset['Year'].astype(int)
dataset['Conference'] = dataset['Conference'].astype('category')
dataset['PaperType'] = dataset['PaperType'].astype('category')
dataset['Award'] = dataset['Award'].fillna('No Award').astype('category')

# 1. Enhanced Trend Analysis on Citation Counts
plt.figure(figsize=(12, 6))
sns.lineplot(
    data=dataset,
    x='Year',
    y='CitationCount_CrossRef',
    hue='Conference',
    style='PaperType',
    markers=True,
    dashes=False,
    palette='tab10'
)
plt.title('Citation Counts Over Years by Conference and Paper Type', fontsize=16)
plt.xlabel('Publication Year', fontsize=14)
plt.ylabel('Citation Count (CrossRef)', fontsize=14)
plt.xticks(rotation=45)
plt.legend(title='Conference / PaperType', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# 2. Machine Learning to Predict Awards
features = ['PaperType', 'Downloads_Xplore', 'Year', 'Conference', 'AuthorKeywords']
X = pd.get_dummies(dataset[features])
le = LabelEncoder()
y = le.fit_transform(dataset['Award'])

# Remove any possible NaNs
mask = ~np.isnan(y)
X = X[mask]
y = y[mask]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Classification Report for Award Prediction:\n", classification_report(y_test, y_pred))

# Confusion matrix plot
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=le.classes_,
    yticklabels=le.classes_
)
plt.title('Confusion Matrix for Award Prediction', fontsize=16)
plt.xlabel('Predicted Award Status', fontsize=14)
plt.ylabel('True Award Status', fontsize=14)
plt.tight_layout()
plt.show()

# 3. Regression Analysis for Download Prediction
features = ['AminerCitationCount', 'GraphicsReplicabilityStamp', 'Year', 'Conference']
X = pd.get_dummies(dataset[features])
y = dataset['Downloads_Xplore']

mask = ~np.isnan(y)
X = X[mask]
y = y[mask]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
reg_model = LinearRegression()
reg_model.fit(X_train, y_train)
y_pred = reg_model.predict(X_test)

print("Mean Squared Error for Downloads Prediction:", mean_squared_error(y_test, y_pred))

# Scatter plot with ideal line
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7, color='teal')
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal Prediction')
plt.title('Actual vs Predicted Downloads', fontsize=16)
plt.xlabel('Actual Downloads', fontsize=14)
plt.ylabel('Predicted Downloads', fontsize=14)
plt.legend()
plt.tight_layout()
plt.show()

# 4. Keyword Trend Analysis
cv = CountVectorizer(stop_words='english')
keyword_matrix = cv.fit_transform(dataset['AuthorKeywords'])
keyword_sums = np.array(keyword_matrix.sum(axis=0)).flatten()
keyword_freq = pd.DataFrame({
    'Keyword': cv.get_feature_names_out(),
    'Frequency': keyword_sums
})
top_keywords = keyword_freq.sort_values('Frequency', ascending=False).head(10)

wordcloud = WordCloud(
    width=800,
    height=400,
    background_color='white',
    colormap='viridis'
).generate_from_frequencies(top_keywords.set_index('Keyword')['Frequency'])

plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.title('Top 10 Author Keywords Word Cloud', fontsize=16)
plt.axis('off')
plt.tight_layout()
plt.show()

# 5. Impact of Awards on Citations
award_vs_citation = dataset.groupby('Award')['CitationCount_CrossRef'].mean().reset_index()

plt.figure(figsize=(10, 6))
# Sort awards by citation count for clearer bars
award_order = award_vs_citation.sort_values('CitationCount_CrossRef', ascending=False)['Award']
sns.barplot(
    x='Award',
    y='CitationCount_CrossRef',
    data=award_vs_citation,
    order=award_order,
    palette='pastel'
)
plt.title('Average Citation Count by Award Status', fontsize=16)
plt.xlabel('Award Status', fontsize=14)
plt.ylabel('Average Citation Count (CrossRef)', fontsize=14)
# Annotate bars
for idx, row in award_vs_citation.iterrows():
    plt.text(
        x=award_order.tolist().index(row['Award']),
        y=row['CitationCount_CrossRef'] + 0.5,
        s=f"{row['CitationCount_CrossRef']:.1f}",
        ha='center'
    )
plt.tight_layout()
plt.show()

# ---- NEW BLOCK ---- # 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import numpy as np

# Set a consistent style
sns.set(style='whitegrid', context='talk')

# Load the dataset
df = pd.read_csv('dataset.csv')

# Fill NaN values
df['AuthorKeywords'].fillna('', inplace=True)
df['AminerCitationCount'].fillna(df['AminerCitationCount'].mean(), inplace=True)
df['CitationCount_CrossRef'].fillna(df['CitationCount_CrossRef'].mean(), inplace=True)
df['Downloads_Xplore'].fillna(df['Downloads_Xplore'].mean(), inplace=True)
df['Award'].fillna('No Award', inplace=True)

# --- Conference Popularity Analysis ---
def analyze_conference_popularity(data):
    conference_trends = (
        data
        .groupby(['Conference', 'Year'])['Downloads_Xplore']
        .sum()
        .reset_index()
    )
    plt.figure(figsize=(14, 6), dpi=100)
    palette = sns.color_palette('tab10', n_colors=conference_trends['Conference'].nunique())
    sns.lineplot(
        x='Year', 
        y='Downloads_Xplore', 
        hue='Conference', 
        data=conference_trends, 
        marker='o', 
        palette=palette
    )
    plt.title('Conference Popularity Over Time', fontsize=18)
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Total Downloads (Xplore)', fontsize=14)
    plt.legend(title='Conference', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# --- Keyword Evolution and Impact ---
def keyword_evolution(data):
    # Build top-10 keyword frequencies
    keyword_trends = (
        data['AuthorKeywords']
        .str.get_dummies(sep=',')
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )
    plt.figure(figsize=(10, 6), dpi=100)
    sns.barplot(
        x=keyword_trends.values, 
        y=keyword_trends.index, 
        palette='viridis'
    )
    plt.title('Top 10 Keywords by Frequency', fontsize=18)
    plt.xlabel('Frequency', fontsize=14)
    plt.ylabel('Keyword', fontsize=14)
    plt.tight_layout()
    plt.show()

# --- Award Prediction Analysis ---
def award_prediction(data):
    features = pd.get_dummies(data[['PaperType', 'Downloads_Xplore']])
    labels = data['Award'].apply(lambda x: 1 if x != 'No Award' else 0)
    
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.3, random_state=42
    )
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    print("\n--- Award Prediction Classification Report ---")
    print(classification_report(y_test, preds))
    print('Accuracy:', accuracy_score(y_test, preds))
    
    # Confusion Matrix Heatmap
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(6, 5), dpi=100)
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', cbar=False,
        xticklabels=['No Award', 'Award'],
        yticklabels=['No Award', 'Award']
    )
    plt.title('Confusion Matrix: Award Prediction', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.tight_layout()
    plt.show()

# --- Temporal Trends in Citation and Download Patterns ---
def temporal_trends(data):
    trends = (
        data
        .groupby('Year')
        .agg({
            'AminerCitationCount': 'mean',
            'CitationCount_CrossRef': 'mean',
            'Downloads_Xplore': 'mean'
        })
        .reset_index()
    )
    plt.figure(figsize=(14, 6), dpi=100)
    sns.lineplot(
        data=trends, x='Year', y='AminerCitationCount', marker='o', label='Aminer Citations'
    )
    sns.lineplot(
        data=trends, x='Year', y='CitationCount_CrossRef', marker='s', label='CrossRef Citations'
    )
    sns.lineplot(
        data=trends, x='Year', y='Downloads_Xplore', marker='^', label='Downloads (Xplore)'
    )
    plt.title('Temporal Trends in Citations and Downloads', fontsize=18)
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Average Count', fontsize=14)
    plt.legend(title=None, loc='best')
    plt.grid(linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# --- Text Classification using NLP and Logistic Regression ---
def text_classification(data):
    tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
    X = tfidf.fit_transform(data['Abstract'].fillna('')).toarray()
    y = data['PaperType'].factorize()[0]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    print("\n--- Text Classification Report (PaperType) ---")
    print(classification_report(y_test, preds))
    print('Accuracy:', accuracy_score(y_test, preds))
    
    # Confusion Matrix Heatmap
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(6, 5), dpi=100)
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Greens', cbar=False
    )
    plt.title('Confusion Matrix: PaperType Classification', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.tight_layout()
    plt.show()

# Execute all analyses
analyze_conference_popularity(df)
keyword_evolution(df)
award_prediction(df)
temporal_trends(df)
text_classification(df)

# ---- NEW BLOCK ---- # 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, r2_score
from sklearn.linear_model import LinearRegression
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords

#---------------------------------------------------------------------
# 1. Load & Clean Data
#---------------------------------------------------------------------
df = pd.read_csv('dataset.csv')

# Impute numeric columns
num_cols = ['FirstPage', 'LastPage', 'AminerCitationCount',
            'CitationCount_CrossRef', 'Downloads_Xplore']
imputer = SimpleImputer(strategy='median')
df[num_cols] = imputer.fit_transform(df[num_cols])

# Fill text and categorical NaNs
df['Abstract'].fillna('', inplace=True)
df['AuthorKeywords'].fillna('', inplace=True)
df['InternalReferences'].fillna('', inplace=True)
df['GraphicsReplicabilityStamp'].fillna('No Stamp', inplace=True)
df['Award'].fillna('No Award', inplace=True)

#---------------------------------------------------------------------
# 2. Enhanced Trend Analysis on Citation Counts
#---------------------------------------------------------------------
plt.figure(figsize=(12, 6))
trend_df = (
    df.groupby(['Year', 'Conference'])['CitationCount_CrossRef']
      .sum()
      .reset_index()
)
sns.lineplot(
    data=trend_df,
    x='Year', y='CitationCount_CrossRef',
    hue='Conference',
    palette='tab10',
    estimator=None
)
plt.title('Total CrossRef Citations by Year and Conference', fontsize=14)
plt.xlabel('Publication Year', fontsize=12)
plt.ylabel('Sum of CitationCount_CrossRef', fontsize=12)
plt.legend(title='Conference', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

#---------------------------------------------------------------------
# 3. Machine Learning to Predict Awards
#---------------------------------------------------------------------
# Encode target
label_enc = LabelEncoder()
df['Award_Encoded'] = label_enc.fit_transform(df['Award'])

# Prepare features & target
X = df[['PaperType', 'InternalReferences', 'Downloads_Xplore', 'Year', 'Conference']]
X = pd.get_dummies(X)
y = df['Award_Encoded']

# Split & train
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
rfc = RandomForestClassifier(random_state=42)
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)

# Classification report (printed)
print("Classification Report for Award Prediction:\n")
print(classification_report(y_test, y_pred, target_names=label_enc.classes_))

# Confusion Matrix plot
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=label_enc.classes_)
fig, ax = plt.subplots(figsize=(6, 6))
disp.plot(cmap='Blues', ax=ax, colorbar=False)
plt.title('Confusion Matrix: Award Prediction', fontsize=14)
plt.tight_layout()
plt.show()

# Feature Importance
importances = rfc.feature_importances_
feat_names = X.columns
feat_imp_df = pd.Series(importances, index=feat_names).sort_values(ascending=False).head(10)

plt.figure(figsize=(8, 5))
sns.barplot(x=feat_imp_df.values, y=feat_imp_df.index, palette='viridis')
plt.title('Top 10 Feature Importances for Award Prediction', fontsize=14)
plt.xlabel('Importance', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.tight_layout()
plt.show()

#---------------------------------------------------------------------
# 4. Keyword Trend Analysis
#---------------------------------------------------------------------
stop_words = set(stopwords.words('english'))
df['KeywordsList'] = (
    df['AuthorKeywords']
      .str.lower()
      .str.split(',')
      .apply(lambda kw: [w.strip() for w in kw if w.strip() and w.strip() not in stop_words])
)

all_words = [w for kws in df['KeywordsList'] for w in kws]
wordcloud = WordCloud(
    width=800, height=400,
    background_color='white',
    colormap='inferno',
    max_words=100
).generate(' '.join(all_words))

plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('WordCloud of Author Keywords (Top 100)', fontsize=16)
plt.tight_layout()
plt.show()

#---------------------------------------------------------------------
# 5. Regression Analysis for Download Prediction
#---------------------------------------------------------------------
# Prepare predictors
X_reg = df[['AminerCitationCount', 'GraphicsReplicabilityStamp',
            'InternalReferences', 'Year', 'Conference']]
X_reg = pd.get_dummies(X_reg, drop_first=True)
y_reg = df['Downloads_Xplore']

# Align indices to drop any remaining NaNs
mask = y_reg.notna()
X_reg = X_reg.loc[mask, :]
y_reg = y_reg.loc[mask]

# Fit linear model
lr = LinearRegression()
lr.fit(X_reg, y_reg)
y_pred_reg = lr.predict(X_reg)

# Compute R²
r2 = r2_score(y_reg, y_pred_reg)

# Scatter plot with y=x line
plt.figure(figsize=(8, 6))
plt.scatter(y_reg, y_pred_reg, alpha=0.4, color='teal', edgecolor='k', linewidth=0.3)
lims = [min(y_reg.min(), y_pred_reg.min()), max(y_reg.max(), y_pred_reg.max())]
plt.plot(lims, lims, 'r--', linewidth=1)
plt.title('Downloads_Xplore: True vs. Predicted', fontsize=14)
plt.xlabel('True Downloads_Xplore', fontsize=12)
plt.ylabel('Predicted Downloads_Xplore', fontsize=12)
plt.text(
    0.05, 0.95,
    f'R² = {r2:.3f}',
    transform=plt.gca().transAxes,
    fontsize=12,
    verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

#---------------------------------------------------------------------
# 6. Impact of Awards on CitationCount_CrossRef
#---------------------------------------------------------------------
award_df     = df[df['Award'] != 'No Award']
non_award_df = df[df['Award'] == 'No Award']

plt.figure(figsize=(8, 6))
sns.kdeplot(
    award_df['CitationCount_CrossRef'].dropna(),
    shade=True, color='navy', label='Awarded', alpha=0.6
)
sns.kdeplot(
    non_award_df['CitationCount_CrossRef'].dropna(),
    shade=True, color='orange', label='Not Awarded', alpha=0.6
)
plt.title('Distribution of CitationCount_CrossRef by Award Status', fontsize=14)
plt.xlabel('CitationCount_CrossRef', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend(title='Status')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

