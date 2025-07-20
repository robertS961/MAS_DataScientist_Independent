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