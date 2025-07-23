import re
from helper_functions import initialize_state_from_csv, define_variables, pretty_print_messages, generate_pdf_report
from agents import vis_a, create_code
from classes import State, Configurable
from langchain.schema import AIMessage


previous = ''' Apologies for the oversight. Here are the refined ideas with suggested enhancements:

1. **Enhanced Trend Analysis on Citation Counts (Data Science)**
   - **Objective**: Utilize `Year`, `AminerCitationCount`, and `CitationCount_CrossRef` while incorporating `Conference` and `PaperType`.
   - **Approach**: Conduct time-series analysis to identify citation trends and examine whether conferences or paper types exhibit distinct patterns.

2. **Network Analysis of Authors and Affiliations (Data Science)**
   - **Objective**: Use `AuthorNames-Deduped`, `AuthorAffiliation`, merged with `InternalReferences`.
   - **Approach**: Build networks depicting author connections through shared affiliations and citation links within the dataset, identifying influential researchers.

3. **Machine Learning to Predict Awards (Data Science)**
   - **Objective**: Implement models using `PaperType`, `AuthorKeywords`, `InternalReferences`, `Downloads_Xplore`, and additionally, `Year` and `Conference`.
   - **Approach**: Develop classification algorithms to predict the likelihood of winning an award, considering temporal and conference-based patterns.

4. **Regression Analysis for Download Prediction (Statistical Learning)**
   - **Objective**: Analyze `Downloads_Xplore` with predictors like `AminerCitationCount`, `GraphicsReplicabilityStamp`, `InternalReferences`, and add `Year` and `Conference`.
   - **Approach**: Perform a multi-variable regression analysis to understand the factors impacting download numbers.

5. **Factor Analysis of Research Topics (Statistical Learning)**
   - **Objective**: Use `AuthorKeywords` and `Abstract` to delve into research themes.
   - **Approach**: Conduct factor analysis pairing with abstract analysis for a comprehensive exploration of research topics.

6. **Survival Analysis on Paper Popularity (Statistical Learning)**
   - **Objective**: Use `Year`, `Downloads_Xplore`, `CitationCount_CrossRef`, `PaperType`, and `Conference`.
   - **Approach**: Leverage survival analysis to study paper popularity lifecycles and assess how particular paper types or conferences impact popularity over time.

These ideas incorporate suggestions for using additional columns to derive more in-depth insights and improvement opportunities.
 
Here are the consolidated ideas from the agents, along with additional suggestions:

**Data Science Ideas:**

1. **Collaborative Network Analysis:**
   - Construct a network graph using `AuthorNames-Deduped` and `AuthorAffiliation` to identify collaboration patterns. Analyze centrality measures to find influential authors and explore network clustering to uncover collaboration communities across different `Conference` events, and visualize evolution over `Year`.

2. **Conference Impact Assessment:**
   - Evaluate the impact of different `Conferences` on citation metrics (`AminerCitationCount`, `CitationCount_CrossRef`) and their relationship with `PaperType` and `Year` to understand the historical significance of conferences.

3. **Keyword Trend Analysis:**
   - Use NLP to analyze the `AuthorKeywords` over `Year` for frequency and trend analysis to uncover emerging topics and shift in research interests. Visualization can illustrate these keyword evolutions.

4. **Citation Prediction Model:**
   - Develop machine learning models (e.g., regression, XGBoost) to predict `CitationCount_CrossRef` using features like `Year`, `PaperType`, `AuthorAffiliation`, `Downloads_Xplore`, and others extracted from `Abstract`.

**Statistical Learning Ideas:**

5. **Factor Analysis on Author Affiliations:**
   - Apply factor analysis on `AuthorAffiliation` to identify hidden factors influencing research output. Correlate these with `AminerCitationCount` and `Downloads_Xplore` to study institutional impact.

6. **Meta-analysis of Graphics Replicability:**
   - Perform meta-analysis on the `GraphicsReplicabilityStamp` to examine correlation with `CitationCount_CrossRef`, `Downloads_Xplore`, and `Award`, emphasizing the role of replicability.

**Additional Ideas:**

7. **Abstract Sentiment Analysis:**
   - Perform sentiment analysis on `Abstract` texts to explore if sentiment correlates with `CitationCount_CrossRef`. Positive sentiment might influence citation attractiveness.

8. **Author Influence Over Time:**
   - Analyze changes in `AminerCitationCount` over `Year` for top authors to identify how their influence grows or wanes over time.

9. **Research Content Clustering:**
   - Cluster papers based on abstracts and `AuthorKeywords` to identify thematic groups and compare them across different `Conferences`.

10. **Impact of Awards on Citations:**
    - Evaluate if papers with `Award` exhibit higher `CitationCount_CrossRef` and `Downloads_Xplore`, and explore other patterns linked with award-winning papers.

This comprehensive approach will harness both data science and statistical methodologies for valuable insights from the dataset.
 
'''

state = initialize_state_from_csv()

data = state['dataset_info']
dic, config = define_variables(thread = 1, loop_limit = 25, data = data, name = "code", input = previous)

graph = create_code(State)
print("Graph for Visualization Created! \n")

for chunk in graph.stream(input = dic, config = config):
    pretty_print_messages(chunk)

print(f"This is the chunk \n {chunk} \n")
result = chunk['code_agent']['messages'][-1]['content']

code = re.findall(r"```python\n(.*?)\n```", result, re.DOTALL)
print(f"This is the code ! \n {code} \n")
with open("extracted_code.py", "a", encoding="utf-8") as f:
    f.write("# ---- NEW BLOCK ---- # \n")
    for block in code:
        f.write(block + "\n\n")

generate_pdf_report(result, 'output.pdf')

