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


