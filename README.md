# A comparison of text distance algorithms

[![Run nbconvert](https://github.com/micheledinelli/text-distances/actions/workflows/readme.yaml/badge.svg)](https://github.com/micheledinelli/text-distances/actions/workflows/readme.yaml)


```python
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

plt.style.use("style.mplstyle")

colors = ["#003f5c", "#d45087", "#ffa600", "#665191", "#ff7c43", "#2f4b7c", "#f95d6a", "#a05195"]
```


```python
s1 = "Obama speaks to the media in Illinois"
s2 = "The president greets the press in Chicago"
s3 = "Duck"
s4 = "Cool"

corpus = [s1, s2, s3, s4]

vectorizer = CountVectorizer()
vectorizer.fit(corpus)

matrix = vectorizer.fit_transform(corpus)

table = matrix.todense()
df = pd.DataFrame(table, 
                  columns=vectorizer.get_feature_names_out(), 
                  index=['text_1', 'text_2', 'text_3', 'text_4'])

df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>chicago</th>
      <th>cool</th>
      <th>duck</th>
      <th>greets</th>
      <th>illinois</th>
      <th>in</th>
      <th>media</th>
      <th>obama</th>
      <th>president</th>
      <th>press</th>
      <th>speaks</th>
      <th>the</th>
      <th>to</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>text_1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>text_2</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>text_3</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>text_4</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



## Text distance algorithms


```python
def euclidean_distance(s1, s2, vectorizer):
    vector1 = vectorizer.transform([s1]).toarray()
    vector2 = vectorizer.transform([s2]).toarray()
    distance = np.linalg.norm(vector1 - vector2)
    return distance

def cosine_distance(s1, s2, vectorizer):
    vector1 = vectorizer.transform([s1]).toarray()
    vector2 = vectorizer.transform([s2]).toarray()
    similarity_matrix = cosine_similarity(vector1, vector2)
    distance = 1 - similarity_matrix[0, 0]
    return distance

def manhattan_distance(s1, s2, vectorizer):
    vector1 = vectorizer.transform([s1]).toarray()
    vector2 = vectorizer.transform([s2]).toarray()
    distance = np.sum(np.abs(vector1 - vector2))
    return distance

def hamming_distance(s1, s2, vectorizer):
    vector1 = vectorizer.transform([s1]).toarray()
    vector2 = vectorizer.transform([s2]).toarray()
    distance = np.sum(vector1 != vector2)
    return distance
```


```python
def plot_distances(s1: str, s2: str, vectorizer, ax=None):
    if ax:
        # Calculate distances
        euclidean_dist = euclidean_distance(s1, s2, vectorizer)
        cosine_dist = cosine_distance(s1, s2, vectorizer)
        manhattan_dist = manhattan_distance(s1, s2, vectorizer)

        ax.scatter([s1, s2], [0, euclidean_dist], label='Euclidean', color=colors[0])
        ax.scatter([s1, s2], [0, cosine_dist], label='Cosine', color=colors[1])
        ax.scatter([s1, s2], [0, manhattan_dist], label='Manhattan', color=colors[2])

        # Connect the points with lines
        ax.plot([s1, s2], [0, euclidean_dist], linestyle='-', color=colors[0])
        ax.plot([s1, s2], [0, cosine_dist], linestyle='-', color=colors[1])
        ax.plot([s1, s2], [0, manhattan_dist], linestyle='-', color=colors[2])

        # Annotate the distance values
        ax.text(s1, 0, f'{s1}\n(0, 0)', ha='center', va='bottom')
        ax.text(s2, euclidean_dist, f'{s2}\n({euclidean_dist:.2f}, {euclidean_dist:.2f})', ha='center', va='bottom')
        ax.text(s2, cosine_dist, f'{s2}\n({cosine_dist:.2f}, {cosine_dist:.2f})', ha='center', va='bottom')
        ax.text(s2, manhattan_dist, f'{s2}\n({manhattan_dist:.2f}, {manhattan_dist:.2f})', ha='center', va='bottom')
        ax.legend()
        
    else:
        # Calculate distances
        euclidean_dist = euclidean_distance(s1, s2, vectorizer)
        cosine_dist = cosine_distance(s1, s2, vectorizer)
        manhattan_dist = manhattan_distance(s1, s2, vectorizer)

        # Create a scatter plot
        plt.scatter([s1, s2], [0, euclidean_dist], label='Euclidean', color=colors[0])
        plt.scatter([s1, s2], [0, cosine_dist], label='Cosine', color=colors[1])
        plt.scatter([s1, s2], [0, manhattan_dist], label='Manhattan', color=colors[2])

        # Connect the points with lines
        plt.plot([s1, s2], [0, euclidean_dist], linestyle='-', color=colors[0])
        plt.plot([s1, s2], [0, cosine_dist], linestyle='-', color=colors[1])
        plt.plot([s1, s2], [0, manhattan_dist], linestyle='-', color=colors[2])

        # Annotate the distance values
        plt.text(s1, 0, f'{s1}\n(0, 0)', ha='center', va='bottom')
        plt.text(s2, euclidean_dist, f'{s2}\n({euclidean_dist:.2f}, {euclidean_dist:.2f})', ha='center', va='bottom')
        plt.text(s2, cosine_dist, f'{s2}\n({cosine_dist:.2f}, {cosine_dist:.2f})', ha='center', va='bottom')
        plt.text(s2, manhattan_dist, f'{s2}\n({manhattan_dist:.2f}, {manhattan_dist:.2f})', ha='center', va='bottom')

        # Set labels and title
        plt.xlabel('Sentences')
        plt.ylabel('Distance')
        plt.title('Distances between sentences')
        plt.legend()

        # print(f"Euclidean distance: {euclidean_dist:.2f}, Cosine distance: {cosine_dist:.2f}, Manhattan distance: {manhattan_dist:.2f}, Hamming distance: {hamming_dist}")

        # Show the plot
        plt.figtext(0.5, 0.01, f'Euclidean: {euclidean_dist:.2f}, Cosine: {cosine_dist:.2f}, Manhattan: {manhattan_dist:.2f}', ha='center', va='bottom')
        plt.show()

```


```python
plot_distances(s1, s2, vectorizer=vectorizer)
```


    
![png](README_files/README_6_0.png)
    



```python
# Create a figure with subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Plot each pair on the same set of axes
plot_distances(s3, s4, vectorizer, ax=axes[0])
plot_distances(s4, s4, vectorizer, ax=axes[1])
plot_distances(s1, s3, vectorizer, ax=axes[2])

# Set labels and title for the entire figure
fig.suptitle('Distances between sentences', fontsize=16)
fig.text(0.5, 0.05, 'Sentences', ha='center', va='bottom', fontsize=14)
fig.text(0.07, 0.5, 'Distance', ha='center', va='center', rotation='vertical', fontsize=14)

plt.legend()
plt.tight_layout(rect=[0.1, 0.1, 0.9, 0.95])
plt.show()
```


    
![png](README_files/README_7_0.png)
    


### Distribution distance


```python
import re
from collections import Counter

def get_distribution(text):
    words = re.findall(r'\b\w+\b', text.lower())
    word_count = len(words)
    distribution = Counter(words)
    distribution = {word: count / word_count for word, count in distribution.items()}
    return distribution

def kl_divergence(p, q):
    return sum(p[word] * np.log2(p[word] / q[word]) for word in p.keys() if p[word] > 0 and q[word] > 0)

def js_divergence(p, q):
    m = {word: 0.5 * (p[word] + q[word]) for word in p.keys() & q.keys()}
    return 0.5 * (kl_divergence(p, m) + kl_divergence(q, m))

def js_divergence_between_sentences(sentence1, sentence2):
    distribution1 = get_distribution(sentence1)
    distribution2 = get_distribution(sentence2)
    
    # Ensure all words in both distributions
    all_words = set(list(distribution1.keys()) + list(distribution2.keys()))
    distribution1 = {word: distribution1.get(word, 0) for word in all_words}
    distribution2 = {word: distribution2.get(word, 0) for word in all_words}
    
    jsd = js_divergence(distribution1, distribution2)
    return jsd
```


```python
def plot_distributions(ax, s1: str, s2: str):
    distribution1 = get_distribution(s1)
    distribution2 = get_distribution(s2)
    
    # Get all unique words from both distributions
    all_words = list(set(distribution1.keys()) | set(distribution2.keys()))
    
    # Create arrays for word frequencies in both distributions
    freq1 = np.array([distribution1.get(word, 0) for word in all_words])
    freq2 = np.array([distribution2.get(word, 0) for word in all_words])
    
    # Plot bar charts for each distribution
    ax.bar(all_words, freq1, alpha=0.7, label=s1, color=colors[0], ec="black")
    ax.bar(all_words, freq2, alpha=0.7, label=s2, color=colors[2], bottom=freq1, ec="black")
    
    # Set labels and title
    ax.set_xlabel('Words')
    ax.set_ylabel('Word Frequency')
    ax.set_title('Distributions of Sentences')
    ax.legend()

fig, ax = plt.subplots(figsize=(14, 7))
plot_distributions(ax, s1, s2)

string = f'Js divergence {js_divergence_between_sentences(s1, s2)}'
fig.text(0.5, -0.05, string, ha='center', va='bottom', fontsize=14)
plt.show()
```


    
![png](README_files/README_10_0.png)
    


### Semantic distance

## Text Representation

### LCS


```python
def visualize_lcs_matrix(X, Y, lcs_matrix, longest_common_subsequence):
    fig, ax = plt.subplots()
    ax.set_title('Longest Common Subsequence Matrix')
    cax = ax.matshow(lcs_matrix, cmap='Blues')

    for i in range(len(X) + 1):
        for j in range(len(Y) + 1):
            ax.text(j, i, str(lcs_matrix[i, j]), ha='center', va='center', color='black')

    ax.set_xticks(np.arange(len(Y) + 1))
    ax.set_yticks(np.arange(len(X) + 1))
    ax.set_xticklabels([''] + list(Y), fontsize=20)
    ax.set_yticklabels([''] + list(X), fontsize=20)
    plt.colorbar(cax)
    fig.text(0.5, 0.05, "Longest Common Subsequence: " + longest_common_subsequence, ha='center', va='bottom', fontsize=14)
    plt.show()
```


```python
def longest_common_subsequence(X, Y):
    m = len(X)
    n = len(Y)

    # Create a matrix to store lengths of LCS
    lcs_matrix = np.zeros((m+1, n+1), dtype=int)

    # Build the matrix
    for i in range(1, m+1):
        for j in range(1, n+1):
            if X[i-1] == Y[j-1]:
                lcs_matrix[i][j] = lcs_matrix[i-1][j-1] + 1
            else:
                lcs_matrix[i][j] = max(lcs_matrix[i-1][j], lcs_matrix[i][j-1])

    # Find the length of LCS
    length_lcs = lcs_matrix[m][n]

    # Find the actual LCS
    lcs = []
    i, j = m, n
    while i > 0 and j > 0:
        if X[i-1] == Y[j-1]:
            lcs.insert(0, X[i-1])
            i -= 1
            j -= 1
        elif lcs_matrix[i-1][j] > lcs_matrix[i][j-1]:
            i -= 1
        else:
            j -= 1

    return length_lcs, ''.join(lcs), lcs_matrix

length_lcs, lcs_sequence, lcs_matrix = longest_common_subsequence(s4, "whool")
print(f"Length of Longest Common Subsequence: {length_lcs}")
print(f"Longest Common Subsequence: {lcs_sequence}")

visualize_lcs_matrix(s4, "whool", lcs_matrix, lcs_sequence)

```

    Length of Longest Common Subsequence: 3
    Longest Common Subsequence: ool



    
![png](README_files/README_15_1.png)
    


### TF-IDF


```python
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# Sample documents
documents = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?",
]

# Create the TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Fit and transform the documents
tfidf_matrix = vectorizer.fit_transform(documents)

# Get feature names (words)
feature_names = vectorizer.get_feature_names_out()

# Convert the TF-IDF matrix to a dense matrix for easier inspection
dense_matrix = tfidf_matrix.todense()

# Create a DataFrame for better visualization
df = pd.DataFrame(dense_matrix, columns=feature_names)

# Plot the heatmap
plt.figure(figsize=(14, 6))
heatmap = sns.heatmap(df, cmap='viridis', cbar_kws={'label': 'TF-IDF Value'})

# Set x-axis labels as legends
heatmap.set_xticklabels(feature_names, rotation=45, ha='right')

plt.title('TF-IDF')
plt.xlabel('Words')
plt.ylabel('Documents')
plt.show()

print(documents)
```


    
![png](README_files/README_17_0.png)
    


    ['This is the first document.', 'This document is the second document.', 'And this is the third one.', 'Is this the first document?']


### Word Mover's distance


```python
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

import gensim.downloader as api
model = api.load('word2vec-google-news-300')

def preprocess(sentence):
    return [w for w in sentence.lower().split() if w not in stop_words]
```


```python
sentence_obama = preprocess(s1)
sentence_president = preprocess(s2)

distance = model.wmdistance(sentence_obama, sentence_president)
print('distance = %.4f' % distance)
```

    distance = 1.0175


## Shallow Windows

- Word2Vec
- BERT


```python
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Sample sentences for training
sports_sentences = [
    "Soccer is the most popular sport in the world.",
    "Basketball games are fast-paced and exciting.",
    "Tennis requires precision and agility on the court.",
]

technology_sentences = [
    "Artificial intelligence is reshaping industries.",
    "Programming languages are essential for software development.",
    "The latest smartphones feature advanced technology.",
]

travel_sentences = [
    "Exploring ancient ruins can be a fascinating journey.",
    "Beaches with crystal-clear waters are ideal for relaxation.",
    "Hiking trails offer stunning views of nature's beauty.",
]

# Combine sentences for different groups
all_sentences = sports_sentences + technology_sentences + travel_sentences
labels = ['sports'] * len(sports_sentences) + ['technology'] * len(technology_sentences) + ['travel'] * len(travel_sentences)
colors_map = {'sports': colors[0], 'technology': colors[1], 'travel': colors[2]}

# Tokenize the sentences into words
tokenized_sentences = [sentence.split() for sentence in all_sentences]

# Train Word2Vec model
model = Word2Vec(sentences=tokenized_sentences, vector_size=100, window=5, min_count=1, workers=4)

# Visualize word embeddings using t-SNE
def tsne_plot(model, labels):
    tokens = []
    sentence_colors = []

    for i, word in enumerate(model.wv.index_to_key):
        tokens.append(model.wv[word])
        token_label = labels[i % len(labels)]  # Cycling through labels
        sentence_colors.append(colors_map[token_label])

    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=42)
    new_values = tsne_model.fit_transform(tokens)

    plt.figure(figsize=(16, 9))
    for i in range(len(new_values)):
        plt.scatter(new_values[i, 0], new_values[i, 1], color=sentence_colors[i], ec='black', s=55)
        plt.annotate(model.wv.index_to_key[i], xy=(new_values[i, 0], new_values[i, 1]),
                     xytext=(5, 4), textcoords='offset points', ha='right', va='bottom')

    # Add legends
    for category, color in colors_map.items():
        plt.scatter([], [], color=color, label=category)

    plt.title("Word2Vec representation in 2D space")
    plt.legend()
    plt.show()

# Plot the t-SNE visualization
tsne_plot(model, labels)

```

    /Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages/sklearn/manifold/_t_sne.py:790: FutureWarning: The default learning rate in TSNE will change from 200.0 to 'auto' in 1.2.
      warnings.warn(
    /Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages/sklearn/manifold/_t_sne.py:982: FutureWarning: The PCA initialization in TSNE will change to have the standard deviation of PC1 equal to 1e-4 in 1.2. This will ensure better convergence.
      warnings.warn(



    
![png](README_files/README_22_1.png)
    



```python
from transformers import BertTokenizer, BertModel
import torch

# Load pre-trained BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

bert_corpus = ' '.join(all_sentences)

# Encode a sentence
tokens = tokenizer(bert_corpus, return_tensors='pt')

# Obtain the BERT output
with torch.no_grad():
    outputs = model(**tokens)

# Get the contextualized embeddings for each token
embeddings = outputs.last_hidden_state
```

    Some weights of the model checkpoint at bert-base-uncased were not used when initializing BertModel: ['cls.seq_relationship.weight', 'cls.predictions.transform.LayerNorm.weight', 'cls.predictions.transform.dense.weight', 'cls.predictions.transform.LayerNorm.bias', 'cls.predictions.bias', 'cls.predictions.transform.dense.bias', 'cls.seq_relationship.bias']
    - This IS expected if you are initializing BertModel from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
    - This IS NOT expected if you are initializing BertModel from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).



```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Flatten the embeddings for t-SNE
flat_embeddings = embeddings.squeeze().numpy().reshape(-1, embeddings.shape[-1])

# Apply t-SNE for dimensionality reduction
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(flat_embeddings)

# Plot the 2D embeddings
plt.figure(figsize=(10, 8))
for i, token in enumerate(tokenizer.convert_ids_to_tokens(tokens['input_ids'][0])):
    plt.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1])
    plt.annotate(token, xy=(embeddings_2d[i, 0], embeddings_2d[i, 1]), xytext=(5, 2),
                 textcoords='offset points', ha='right', va='bottom')

plt.title('t-SNE Visualization of BERT Embeddings')
plt.show()

```

    /Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages/sklearn/manifold/_t_sne.py:780: FutureWarning: The default initialization in TSNE will change from 'random' to 'pca' in 1.2.
      warnings.warn(
    /Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages/sklearn/manifold/_t_sne.py:790: FutureWarning: The default learning rate in TSNE will change from 200.0 to 'auto' in 1.2.
      warnings.warn(



    
![png](README_files/README_24_1.png)
    


## Matrix factorization methods
Notebooks have been converted and README has been updated.
