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


