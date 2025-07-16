import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('dataset.csv')

# Set the style for seaborn
sns.set(style="whitegrid")

# Plot 1: Number of Papers per Year
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Year', order=sorted(df['Year'].unique()))
plt.title('Number of Papers per Year')
plt.xticks(rotation=45)
plt.xlabel('Year')
plt.ylabel('Number of Papers')
plt.tight_layout()
plt.show()

# Plot 2: Top 10 Conferences by Number of Papers
top_conferences = df['Conference'].value_counts().nlargest(10)
plt.figure(figsize=(10, 6))
sns.barplot(x=top_conferences.values, y=top_conferences.index, palette='viridis')
plt.title('Top 10 Conferences by Number of Papers')
plt.xlabel('Number of Papers')
plt.ylabel('Conference')
plt.tight_layout()
plt.show()

# Plot 3: Distribution of Citation Counts
plt.figure(figsize=(10, 6))
sns.histplot(df['CitationCount_CrossRef'].dropna(), bins=30, kde=True, color='skyblue')
plt.title('Distribution of Citation Counts')
plt.xlabel('Citation Count')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# Plot 4: Downloads vs Citation Count
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Downloads_Xplore', y='CitationCount_CrossRef', hue='PaperType', alpha=0.7)
plt.title('Downloads vs Citation Count')
plt.xlabel('Downloads')
plt.ylabel('Citation Count')
plt.legend(title='Paper Type')
plt.tight_layout()
plt.show()

# Plot 5: Awarded Papers by Year
awarded_papers = df[df['Award'] == True]
plt.figure(figsize=(10, 6))
sns.countplot(data=awarded_papers, x='Year', order=sorted(awarded_papers['Year'].unique()))
plt.title('Awarded Papers by Year')
plt.xticks(rotation=45)
plt.xlabel('Year')
plt.ylabel('Number of Awarded Papers')
plt.tight_layout()
plt.show()

