import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
import statsmodels.api as sm
from textblob import TextBlob

# Load dataset
df = pd.read_csv('dataset.csv')

# Fill NaN values
imputer_num = SimpleImputer(strategy='mean')
num_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[num_cols] = imputer_num.fit_transform(df[num_cols])

imputer_cat = SimpleImputer(strategy='most_frequent')
cat_cols = df.select_dtypes(include=['object']).columns
df[cat_cols] = imputer_cat.fit_transform(df[cat_cols])

# Convert categorical variables into categories for some columns
df['Conference'] = df['Conference'].astype('category')
df['PaperType'] = df['PaperType'].astype('category')
df['Award'] = df['Award'].astype('category')

# 1. Enhanced Trend Analysis on Citation Counts
def trend_analysis():
    trend_data = df.groupby(['Year', 'Conference', 'PaperType']).agg({
        'AminerCitationCount': 'sum',
        'CitationCount_CrossRef': 'sum'
    }).reset_index()

    plt.figure(figsize=(16, 8))
    sns.lineplot(
        data=trend_data,
        x='Year',
        y='AminerCitationCount',
        hue='Conference',
        style='PaperType',
        markers=True,
        palette='tab10',
        linewidth=2
    )
    plt.title('Total Aminer Citations Over Time by Conference and Paper Type', fontsize=16)
    plt.xlabel('Publication Year', fontsize=14)
    plt.ylabel('Total Aminer Citation Count', fontsize=14)
    plt.legend(title='Conf. / PaperType', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
    plt.tight_layout()
    plt.show()

trend_analysis()

# 2. Machine Learning to Predict Awards
def predict_award():
    features = df[['PaperType', 'AuthorKeywords', 'InternalReferences', 'Downloads_Xplore', 'Year', 'Conference']]
    features = pd.get_dummies(features)
    target = df['Award'].cat.codes  # Convert category to integer

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Feature importance
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    feat_names = features.columns[indices]

    plt.figure(figsize=(16, 6))
    plt.bar(feat_names, importances[indices], color='steelblue')
    plt.xticks(rotation=90, fontsize=10)
    plt.yticks(fontsize=12)
    plt.xlabel('Feature', fontsize=14)
    plt.ylabel('Importance Score', fontsize=14)
    plt.title('Random Forest Feature Importances for Award Prediction', fontsize=16)
    plt.tight_layout()
    plt.show()

predict_award()

# 3. Regression Analysis for Download Prediction
def regression_analysis():
    X = df[['AminerCitationCount', 'GraphicsReplicabilityStamp', 'InternalReferences', 'Year', 'Conference']]
    X = pd.get_dummies(X, drop_first=True)
    y = df['Downloads_Xplore']

    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()

    # Print full summary to console
    print(model.summary())

    # Scatter + Regression line + R² annotation
    y_pred = model.predict(X)
    r2 = model.rsquared

    plt.figure(figsize=(16, 6))
    plt.scatter(y, y_pred, alpha=0.5, c='darkorange', edgecolor='k')
    # Plot y=x reference line for perfect prediction
    minv, maxv = y.min(), y.max()
    plt.plot([minv, maxv], [minv, maxv], 'b--', linewidth=2)
    plt.text(
        0.05, 0.95,
        f'$R^2$ = {r2:.3f}',
        transform=plt.gca().transAxes,
        fontsize=14,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
    )
    plt.xlabel('Actual Downloads_Xplore', fontsize=14)
    plt.ylabel('Predicted Downloads_Xplore', fontsize=14)
    plt.title('Download Prediction: Actual vs. Predicted', fontsize=16)
    plt.tight_layout()
    plt.show()

regression_analysis()

# 4. Keyword Trend Analysis
def keyword_trend_analysis():
    df['AuthorKeywords'] = df['AuthorKeywords'].fillna('').apply(lambda x: x.split(';'))
    all_keywords = [kw.strip() for sub in df['AuthorKeywords'] for kw in sub if kw.strip()]

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(all_keywords)
    keyword_freq = pd.Series(
        X.toarray().sum(axis=0),
        index=vectorizer.get_feature_names_out()
    ).sort_values(ascending=False)

    top20 = keyword_freq.head(20)
    plt.figure(figsize=(14, 6))
    sns.barplot(x=top20.values, y=top20.index, palette='Blues_d')
    plt.xlabel('Frequency', fontsize=14)
    plt.ylabel('Keyword', fontsize=14)
    plt.title('Top 20 Keywords by Frequency', fontsize=16)
    plt.tight_layout()
    plt.show()

keyword_trend_analysis()

# 5. Abstract Sentiment Analysis
def sentiment_analysis():
    df['Sentiment'] = df['Abstract'].apply(lambda text: TextBlob(text).sentiment.polarity)

    plt.figure(figsize=(16, 6))
    sc = plt.scatter(
        df['Sentiment'],
        df['CitationCount_CrossRef'],
        c=df['Year'],
        cmap='viridis',
        alpha=0.6,
        edgecolor='k'
    )
    cbar = plt.colorbar(sc)
    cbar.set_label('Publication Year', fontsize=12)
    plt.xlabel('Abstract Sentiment Polarity', fontsize=14)
    plt.ylabel('CitationCount (CrossRef)', fontsize=14)
    plt.title('Sentiment Polarity of Abstracts vs. CrossRef Citations', fontsize=16)
    plt.tight_layout()
    plt.show()

sentiment_analysis() 