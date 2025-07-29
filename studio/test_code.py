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







