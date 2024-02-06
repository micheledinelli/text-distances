# A comparison of text distance algorithms

[![Run nbconvert](https://github.com/micheledinelli/text-distances/actions/workflows/readme.yaml/badge.svg)](https://github.com/micheledinelli/text-distances/actions/workflows/readme.yaml)

    [nltk_data] Downloading package punkt to
    [nltk_data]     /Users/micheledinelli/nltk_data...
    [nltk_data]   Package punkt is already up-to-date!
    [nltk_data] Downloading package stopwords to
    [nltk_data]     /Users/micheledinelli/nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!
    [nltk_data] Downloading package punkt to
    [nltk_data]     /Users/micheledinelli/nltk_data...
    [nltk_data]   Package punkt is already up-to-date!


# Text distance: outline

- [Length distance](#length-distance)
  - Euclidean distance
  - Cosine distance
  - Manhattan distance
  - Hamming distance
- [Distribution distance](#distribution-distance)
  - JS divergence
  - KL divergence
  - Wasserstein distance
- [Semantic distance](#semantic-distance)
  - Word mover's distance
  - Word mover's distance extension

### Length Distance <a id="length-distance"></a>

#### Algorithms


    
![png](text-distance-analysis_files/text-distance-analysis_10_0.png)
    


Insight on euclidean vs cosine similarity




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
      <th>good</th>
      <th>refrigerators</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Q</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>D1</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>D2</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>D3</th>
      <td>2</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




    
![png](text-distance-analysis_files/text-distance-analysis_13_0.png)
    


### Distribution distance

#### Algorithms

    KL divergence between documents: 0.116
    JS divergence between documents: 0.030
    Wasserstein distance between documents: 0.035



    
![png](text-distance-analysis_files/text-distance-analysis_18_0.png)
    



    
![png](text-distance-analysis_files/text-distance-analysis_19_0.png)
    





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
      <th>Words</th>
      <th>Sentence</th>
      <th>Frequency</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>loves</td>
      <td>The cat sat mat . The cat played ball yarn . I...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>it</td>
      <td>The cat sat mat . The cat played ball yarn . I...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>the</td>
      <td>The cat sat mat . The cat played ball yarn . I...</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>cat</td>
      <td>The cat sat mat . The cat played ball yarn . I...</td>
      <td>5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>mat</td>
      <td>The cat sat mat . The cat played ball yarn . I...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>yarn</td>
      <td>The cat sat mat . The cat played ball yarn . I...</td>
      <td>2</td>
    </tr>
    <tr>
      <th>6</th>
      <td>learning</td>
      <td>The cat sat mat . The cat played ball yarn . I...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>key</td>
      <td>The cat sat mat . The cat played ball yarn . I...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>a</td>
      <td>The cat sat mat . The cat played ball yarn . I...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>advanced</td>
      <td>The cat sat mat . The cat played ball yarn . I...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>really</td>
      <td>The cat sat mat . The cat played ball yarn . I...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>11</th>
      <td>ball</td>
      <td>The cat sat mat . The cat played ball yarn . I...</td>
      <td>2</td>
    </tr>
    <tr>
      <th>12</th>
      <td>machine</td>
      <td>The cat sat mat . The cat played ball yarn . I...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>played</td>
      <td>The cat sat mat . The cat played ball yarn . I...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>14</th>
      <td>algorithms</td>
      <td>The cat sat mat . The cat played ball yarn . I...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>successful</td>
      <td>The cat sat mat . The cat played ball yarn . I...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>sat</td>
      <td>The cat sat mat . The cat played ball yarn . I...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>17</th>
      <td>happy</td>
      <td>The cat sat mat . The cat played ball yarn . I...</td>
      <td>2</td>
    </tr>
    <tr>
      <th>18</th>
      <td>loves</td>
      <td>Advanced algorithms key successful machine lea...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>it</td>
      <td>Advanced algorithms key successful machine lea...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>the</td>
      <td>Advanced algorithms key successful machine lea...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>cat</td>
      <td>Advanced algorithms key successful machine lea...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>mat</td>
      <td>Advanced algorithms key successful machine lea...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>yarn</td>
      <td>Advanced algorithms key successful machine lea...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>learning</td>
      <td>Advanced algorithms key successful machine lea...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>25</th>
      <td>key</td>
      <td>Advanced algorithms key successful machine lea...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>26</th>
      <td>a</td>
      <td>Advanced algorithms key successful machine lea...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>27</th>
      <td>advanced</td>
      <td>Advanced algorithms key successful machine lea...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>28</th>
      <td>really</td>
      <td>Advanced algorithms key successful machine lea...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>ball</td>
      <td>Advanced algorithms key successful machine lea...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>30</th>
      <td>machine</td>
      <td>Advanced algorithms key successful machine lea...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>31</th>
      <td>played</td>
      <td>Advanced algorithms key successful machine lea...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>32</th>
      <td>algorithms</td>
      <td>Advanced algorithms key successful machine lea...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>33</th>
      <td>successful</td>
      <td>Advanced algorithms key successful machine lea...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>34</th>
      <td>sat</td>
      <td>Advanced algorithms key successful machine lea...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>35</th>
      <td>happy</td>
      <td>Advanced algorithms key successful machine lea...</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



### Semantic distance


    
![png](text-distance-analysis_files/text-distance-analysis_24_0.png)
    



    
![png](text-distance-analysis_files/text-distance-analysis_25_0.png)
    


    Length of Longest Common Subsequence: 3
    Longest Common Subsequence: ool



    
![png](text-distance-analysis_files/text-distance-analysis_27_1.png)
    

# A comparison of text representation methods

# Text representation: outline

- [String based](#string-based)
  - Character based
    - LCS distance
    - Edit distance
    - Jaro similarity
  - Phrase based
    - Dice
    - Jaccard 
- [Corpus based](#corpus-based)
  - Bag of word model
    - BOW
    - TF-IDF 
  - Shallow window based
    - Word2Vec
    - GloVe
    - BERT
- [Matrix factorization methods](#matrix-factorization)
  - LSA
  - LDA
- [Graph structure](#graph-based)
  - Knowledge graph
  - Graph neural network

## String Based

### Algorithms

    Levenshtein distance between The sky is blue and The sun is bright: 7.00
    Longest common substring distance between The sky is blue and The sun is bright: 5.00
    Jaro similarity between The sky is blue and The sun is bright: 0.75


    Levenshtein distance between The sky is blue and The sky is blue: 0.00
    Longest common substring distance between The sky is blue and The sky is blue: 15.00
    Jaro similarity between The sky is blue and The sky is blue: 1.00





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
      <th>customer_id</th>
      <th>birth</th>
      <th>city</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>109238</td>
      <td>07-12-87</td>
      <td>Rome</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1132125</td>
      <td>23-08-89</td>
      <td>London</td>
    </tr>
    <tr>
      <th>2</th>
      <td>159483</td>
      <td>28-11-90</td>
      <td>Paris</td>
    </tr>
    <tr>
      <th>3</th>
      <td>198828</td>
      <td>22-12-92</td>
      <td>Bristol</td>
    </tr>
  </tbody>
</table>
</div>






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
      <th>order_numb</th>
      <th>cust_numb</th>
      <th>cost</th>
      <th>city</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>121212</td>
      <td>109238</td>
      <td>100</td>
      <td>Rome</td>
    </tr>
    <tr>
      <th>1</th>
      <td>151892</td>
      <td>1132125</td>
      <td>200</td>
      <td>London</td>
    </tr>
    <tr>
      <th>2</th>
      <td>312526</td>
      <td>159483</td>
      <td>300</td>
      <td>Paris</td>
    </tr>
    <tr>
      <th>3</th>
      <td>418825</td>
      <td>19882</td>
      <td>400</td>
      <td>Bristol</td>
    </tr>
  </tbody>
</table>
</div>




    
![png](text-representation-analysis_files/text-representation-analysis_10_0.png)
    


    Dice coefficient between The sky is blue and The sun is bright: 0.70
    Jaccard similarity between The sky is blue and The sun is bright: 0.53


## Corpus Based


    
![png](text-representation-analysis_files/text-representation-analysis_13_0.png)
    


## Shallow Window based


    
![png](text-representation-analysis_files/text-representation-analysis_15_0.png)
    


    Some weights of the model checkpoint at bert-base-uncased were not used when initializing BertModel: ['cls.predictions.transform.dense.weight', 'cls.predictions.transform.LayerNorm.weight', 'cls.predictions.transform.LayerNorm.bias', 'cls.predictions.bias', 'cls.seq_relationship.bias', 'cls.seq_relationship.weight', 'cls.predictions.transform.dense.bias']
    - This IS expected if you are initializing BertModel from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
    - This IS NOT expected if you are initializing BertModel from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).



    
![png](text-representation-analysis_files/text-representation-analysis_16_1.png)
    


## Matrix factorization

    Topic 1: space, like, don, know, year
    Topic 2: thanks, graphics, files, image, space
    Topic 3: space, nasa, launch, shuttle, orbit
    Topic 4: graphics, just, don, think, aspects
    Topic 5: thanks, mail, list, advance, space


## Semantic text matching

## Graph based


    
![png](text-representation-analysis_files/text-representation-analysis_21_0.png)
    


    Top terms per component:
    Topic 1: don, space, people, god, com
    Topic 2: morality, sgi, livesey, caltech, keith
    Topic 3: newton, apple, god, kent, sandvik
    Topic 4: toronto, apple, henry, kent, sandvik
    Topic 5: spencer, zoo, god, toronto, henry
    Topic 6: spencer, zoo, graphics, toronto, henry
    Topic 7: com, pat, alaska, digex, access
    Topic 8: values, frank, alaska, morality, objective
    Topic 9: digex, objective, livesey, access, sgi
    Topic 10: jaeger, alaska, islam, livesey, sgi

[![Run nbconvert](https://github.com/micheledinelli/text-distances/actions/workflows/readme.yaml/badge.svg?branch=main)](https://github.com/micheledinelli/text-distances/actions/workflows/readme.yaml)
