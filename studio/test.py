import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
from wordcloud import WordCloud
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram

# Load the dataset
df = pd.read_csv('dataset.csv')

# Distribution of Paper Types
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='PaperType', palette='Set2')
plt.title("Distribution of Paper Types")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Distribution of Years
plt.figure(figsize=(10, 6))
sns.histplot(df['Year'], bins=20, kde=True, color='coral')
plt.title("Distribution of Papers by Year")
plt.xlabel("Year")
plt.ylabel("Number of Papers")
plt.tight_layout()
plt.show()

# Correlation Matrix
correlation_matrix = df[['AminerCitationCount', 'CitationCount_CrossRef', 'Downloads_Xplore']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# Trends Over Time
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='Year', y='CitationCount_CrossRef', marker='o', color='purple')
plt.title("Trends in Citation Counts Over Years")
plt.ylabel("Citation Count")
plt.xlabel("Year")
plt.tight_layout()
plt.show()

# Author Collaboration Network
author_collab_network = nx.Graph()
for _, row in df.iterrows():
    authors = row['AuthorNames-Deduped'].split(';')
    author_collab_network.add_edges_from([(authors[i], authors[j]) for i in range(len(authors)) for j in range(i+1, len(authors))])

plt.figure(figsize=(12, 12))
nx.draw(author_collab_network, node_size=20, node_color='green', with_labels=False)
plt.title("Author Collaboration Network")
plt.tight_layout()
plt.show()

# Word Cloud from Abstracts
text = " ".join(abstract for abstract in df.Abstract.dropna())
wordcloud = WordCloud(width=800, height=400, random_state=21, max_font_size=110).generate(text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.title('Word Cloud of Paper Abstracts')
plt.tight_layout()
plt.show()

# Hierarchical Clustering
features = df[['AminerCitationCount', 'CitationCount_CrossRef', 'Downloads_Xplore']].dropna()
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

linked = linkage(features_scaled, 'ward')
plt.figure(figsize=(10, 7))
dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
plt.title("Hierarchical Clustering Dendrogram")
plt.tight_layout()
plt.show()

# PCA for Dimensionality Reduction
pca = PCA(n_components=2)
pca_result = pca.fit_transform(features_scaled)
plt.figure(figsize=(10, 7))
plt.scatter(pca_result[:, 0], pca_result[:, 1], c='blue', alpha=0.5)
plt.title('PCA of Citation and Download Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.tight_layout()
plt.show()