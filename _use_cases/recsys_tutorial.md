---
layout: default
title: Building Recommender Systems in Snorkel
description: Labeling text reviews for book recommendations
excerpt: Labeling text reviews for book recommendations
order: 5
---


# Recommender Systems Tutorial
In this tutorial, we'll provide a simple walkthrough of how to use Snorkel to improve recommendations. We consider a setting similar to the [Netflix challenge](https://www.kaggle.com/netflix-inc/netflix-prize-data), but with books instead of movies. We have a set of users and books, and for each user we know the set of books they have interacted with (read or marked as to-read). We don't have the user's ratings for the read books, except in a small number of cases. We also have some text reviews written by users. Our goal is to predict whether a user will read and like any given book. We represent users using the set of books they have interacted with, and train a model to predict a `rating` given the set of books the user interacted with and a new book to be rated (a `rating` of 1 means the user will read and like the book). Note that [Recommender systems](https://en.wikipedia.org/wiki/Recommender_system) is a very well studied area with a wide variety of settings and approaches, and we just focus on one of them.

We will use the [Goodreads](https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/home) dataset, from
"Item Recommendation on Monotonic Behavior Chains", RecSys'18 (Mengting Wan, Julian McAuley), and "Fine-Grained Spoiler Detection from Large-Scale Review Corpora", ACL'19 (Mengting Wan, Rishabh Misra, Ndapa Nakashole, Julian McAuley).
In this dataset, we have user interactions and reviews for Young Adult novels from the Goodreads website, along with metadata (like `title` and `authors`) for the novels.


```python
import os

if os.path.basename(os.getcwd()) == "snorkel-tutorials":
    os.chdir("recsys")
```

## Loading Data

We start by running the `download_and_process_data` function. The function returns the `df_train`, `df_test`, `df_dev`, `df_val` dataframes, which correspond to our training, test, development, and validation sets. Each of those dataframes has the following fields:
* `user_idx`: A unique identifier for a user.
* `book_idx`: A unique identifier for a book that is being rated by the user.
* `book_idxs`: The set of books that the user has interacted with (read or planned to read).
* `review_text`: Optional text review written by the user for the book.
* `rating`: Either `0` (which means the user did not read or did not like the book) or `1` (which means the user read and liked the book). The `rating` field is missing for `df_train`.
Our objective is to predict whether a given user (represented by the set of book_idxs the user has interacted with) will read and like any given book. That is, we want to train a model that takes a set of `book_idxs` (the user) and a single `book_idx` (the book to rate) and predicts the `rating`.

In addition, `download_and_process_data` also returns the `df_books` dataframe, which contains one row per book, along with metadata for that book (such as `title` and `first_author`).


```python
from utils import download_and_process_data

(df_train, df_test, df_dev, df_val), df_books = download_and_process_data()

df_books.head()
```

    /home/ubuntu/snorkel-tutorials/.tox/recsys/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:516: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_qint8 = np.dtype([("qint8", np.int8, 1)])
    /home/ubuntu/snorkel-tutorials/.tox/recsys/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:517: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_quint8 = np.dtype([("quint8", np.uint8, 1)])
    /home/ubuntu/snorkel-tutorials/.tox/recsys/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:518: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_qint16 = np.dtype([("qint16", np.int16, 1)])
    /home/ubuntu/snorkel-tutorials/.tox/recsys/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:519: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_quint16 = np.dtype([("quint16", np.uint16, 1)])
    /home/ubuntu/snorkel-tutorials/.tox/recsys/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:520: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_qint32 = np.dtype([("qint32", np.int32, 1)])
    /home/ubuntu/snorkel-tutorials/.tox/recsys/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:525: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      np_resource = np.dtype([("resource", np.ubyte, 1)])
    /home/ubuntu/snorkel-tutorials/.tox/recsys/lib/python3.6/site-packages/tensorboard/compat/tensorflow_stub/dtypes.py:541: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_qint8 = np.dtype([("qint8", np.int8, 1)])
    /home/ubuntu/snorkel-tutorials/.tox/recsys/lib/python3.6/site-packages/tensorboard/compat/tensorflow_stub/dtypes.py:542: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_quint8 = np.dtype([("quint8", np.uint8, 1)])
    /home/ubuntu/snorkel-tutorials/.tox/recsys/lib/python3.6/site-packages/tensorboard/compat/tensorflow_stub/dtypes.py:543: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_qint16 = np.dtype([("qint16", np.int16, 1)])
    /home/ubuntu/snorkel-tutorials/.tox/recsys/lib/python3.6/site-packages/tensorboard/compat/tensorflow_stub/dtypes.py:544: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_quint16 = np.dtype([("quint16", np.uint16, 1)])
    /home/ubuntu/snorkel-tutorials/.tox/recsys/lib/python3.6/site-packages/tensorboard/compat/tensorflow_stub/dtypes.py:545: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_qint32 = np.dtype([("qint32", np.int32, 1)])
    /home/ubuntu/snorkel-tutorials/.tox/recsys/lib/python3.6/site-packages/tensorboard/compat/tensorflow_stub/dtypes.py:550: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      np_resource = np.dtype([("resource", np.ubyte, 1)])
    /home/ubuntu/snorkel-tutorials/recsys/utils.py:165: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      df_interactions_nz["rating_4_5"] = df_interactions_nz.rating.map(ratings_map)





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
      <th>authors</th>
      <th>average_rating</th>
      <th>book_id</th>
      <th>country_code</th>
      <th>description</th>
      <th>is_ebook</th>
      <th>language_code</th>
      <th>ratings_count</th>
      <th>similar_books</th>
      <th>text_reviews_count</th>
      <th>title</th>
      <th>first_author</th>
      <th>book_idx</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>[293603]</td>
      <td>4.35</td>
      <td>10099492</td>
      <td>US</td>
      <td>It all comes down to this.\nVlad's running out...</td>
      <td>True</td>
      <td>eng</td>
      <td>152</td>
      <td>[25861113, 7430195, 18765937, 6120544, 3247550...</td>
      <td>9</td>
      <td>Twelfth Grade Kills (The Chronicles of Vladimi...</td>
      <td>293603</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>[4018722]</td>
      <td>3.71</td>
      <td>22642971</td>
      <td>US</td>
      <td>The future world is at peace.\nElla Shepherd h...</td>
      <td>True</td>
      <td>eng</td>
      <td>1525</td>
      <td>[20499652, 17934493, 13518102, 16210411, 17149...</td>
      <td>428</td>
      <td>The Body Electric</td>
      <td>4018722</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>[6537142]</td>
      <td>3.89</td>
      <td>31556136</td>
      <td>US</td>
      <td>A gorgeously written and deeply felt literary ...</td>
      <td>True</td>
      <td></td>
      <td>109</td>
      <td>[]</td>
      <td>45</td>
      <td>Like Water</td>
      <td>6537142</td>
      <td>2</td>
    </tr>
    <tr>
      <th>12</th>
      <td>[6455200, 5227552]</td>
      <td>3.90</td>
      <td>18522274</td>
      <td>US</td>
      <td>Zoe Vanderveen is on the run with her captor t...</td>
      <td>True</td>
      <td>en-US</td>
      <td>191</td>
      <td>[25063023, 18553080, 17567752, 18126509, 17997...</td>
      <td>6</td>
      <td>Volition (The Perception Trilogy, #2)</td>
      <td>6455200</td>
      <td>3</td>
    </tr>
    <tr>
      <th>13</th>
      <td>[187837]</td>
      <td>3.19</td>
      <td>17262776</td>
      <td>US</td>
      <td>The war is over, but for thirteen-year-old Rac...</td>
      <td>True</td>
      <td>eng</td>
      <td>248</td>
      <td>[16153997, 10836616, 17262238, 16074827, 13628...</td>
      <td>68</td>
      <td>Little Red Lies</td>
      <td>187837</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



We look at a sample of the labeled development set. As an example, we want our final recommendations model to be able to predict that a user who has interacted with `book_idxs` (25743, 22318, 7662, 6857, 83, 14495, 30664, ...) would either not read or not like the book with `book_idx` 22764 (first row), while a user who has interacted with `book_idxs` (3880, 18078, 9092, 29933, 1511, 8560, ...) would read and like the book with `book_idx` 3181 (second row).


```python
df_dev.sample(frac=1, random_state=12).head()
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
      <th>user_idx</th>
      <th>book_idxs</th>
      <th>book_idx</th>
      <th>rating</th>
      <th>review_text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>80480</th>
      <td>3115</td>
      <td>(30671, 21132, 20057, 16271, 8224, 10340, 9046...</td>
      <td>4165</td>
      <td>1</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>80483</th>
      <td>3115</td>
      <td>(30671, 21132, 20057, 16271, 8224, 10340, 9046...</td>
      <td>27119</td>
      <td>1</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>782352</th>
      <td>30443</td>
      <td>(1696, 10119, 2486, 14371, 17962, 2081, 1742, ...</td>
      <td>20471</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>538899</th>
      <td>20809</td>
      <td>(3854, 13950, 17535, 12642, 15803, 8199, 28827...</td>
      <td>27766</td>
      <td>1</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>768968</th>
      <td>29931</td>
      <td>(1325, 14324, 19639, 1511, 19219, 16980, 4363,...</td>
      <td>17807</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



## Writing Labeling Functions

If a user has interacted with several books written by an author, there is a good chance that the user will read and like other books by the same author. We express this as a labeling function, using the `first_author` field in the `df_books` dataframe.


```python
POSITIVE = 1
NEGATIVE = 0
ABSTAIN = -1
```


```python
from snorkel.labeling.lf import labeling_function

book_to_first_author = dict(zip(df_books.book_idx, df_books.first_author))
first_author_to_books_df = df_books.groupby("first_author")[["book_idx"]].agg(set)
first_author_to_books = dict(
    zip(first_author_to_books_df.index, first_author_to_books_df.book_idx)
)


@labeling_function()
def shared_first_author(x):
    author = book_to_first_author[x.book_idx]
    same_author_books = first_author_to_books[author]
    num_read = len(set(x.book_idxs).intersection(same_author_books))
    return POSITIVE if num_read > 15 else ABSTAIN
```

We can also leverage the long text reviews written by users to guess whether they liked or disliked a book. For example, the third df_dev entry above has a review with the text '4.5 STARS', which indicates that the user liked the book. We write a simple LF that looks for similar phrases to guess the user's rating of a book. We interpret >= 4 stars to indicate a positive rating, while < 4 stars is negative.


```python
low_rating_strs = [
    "one star",
    "1 star",
    "two star",
    "2 star",
    "3 star",
    "three star",
    "3.5 star",
    "2.5 star",
    "1 out of 5",
    "2 out of 5",
    "3 out of 5",
]
high_rating_strs = ["5 stars", "five stars", "four stars", "4 stars", "4.5 stars"]


@labeling_function()
def stars_in_review(x):
    if not isinstance(x.review_text, str):
        return ABSTAIN
    for low_rating_str in low_rating_strs:
        if low_rating_str in x.review_text.lower():
            return NEGATIVE
    for high_rating_str in high_rating_strs:
        if high_rating_str in x.review_text.lower():
            return POSITIVE
    return ABSTAIN
```

We can also run [TextBlob](https://textblob.readthedocs.io/en/dev/index.html), a tool that provides a pretrained sentiment analyzer, on the reviews, and use its polarity and subjectivity scores to estimate the user's rating for the book.


```python
from snorkel.preprocess import preprocessor
from textblob import TextBlob


@preprocessor(memoize=True)
def textblob_polarity(x):
    if x.review_text:
        x.blob = TextBlob(str(x.review_text))
    else:
        x.blob = None
    return x


# Label high polarity reviews as positive.
@labeling_function(pre=[textblob_polarity])
def polarity_positive(x):
    if x.blob:
        if x.blob.polarity > 0.3:
            return POSITIVE
    return ABSTAIN


# Label high subjectivity reviews as positive.
@labeling_function(pre=[textblob_polarity])
def subjectivity_positive(x):
    if x.blob:
        if x.blob.subjectivity > 0.75:
            return POSITIVE
    return ABSTAIN


# Label low polarity reviews as negative.
@labeling_function(pre=[textblob_polarity])
def polarity_negative(x):
    if x.blob:
        if x.blob.polarity < 0.0:
            return NEGATIVE
    return ABSTAIN
```


```python
from snorkel.labeling import PandasLFApplier, LFAnalysis

lfs = [
    stars_in_review,
    shared_first_author,
    polarity_positive,
    subjectivity_positive,
    polarity_negative,
]

applier = PandasLFApplier(lfs)
L_dev = applier.apply(df_dev)
LFAnalysis(L_dev, lfs).lf_summary(df_dev.rating)
```

    100%|██████████| 84237/84237 [02:15<00:00, 623.23it/s]





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
      <th>j</th>
      <th>Polarity</th>
      <th>Coverage</th>
      <th>Overlaps</th>
      <th>Conflicts</th>
      <th>Correct</th>
      <th>Incorrect</th>
      <th>Emp. Acc.</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>stars_in_review</th>
      <td>0</td>
      <td>[0, 1]</td>
      <td>0.018614</td>
      <td>0.004547</td>
      <td>0.001793</td>
      <td>1275</td>
      <td>293</td>
      <td>0.813138</td>
    </tr>
    <tr>
      <th>shared_first_author</th>
      <td>1</td>
      <td>[1]</td>
      <td>0.043591</td>
      <td>0.000807</td>
      <td>0.000249</td>
      <td>2793</td>
      <td>879</td>
      <td>0.760621</td>
    </tr>
    <tr>
      <th>polarity_positive</th>
      <td>2</td>
      <td>[1]</td>
      <td>0.045835</td>
      <td>0.013035</td>
      <td>0.000962</td>
      <td>3123</td>
      <td>738</td>
      <td>0.808858</td>
    </tr>
    <tr>
      <th>subjectivity_positive</th>
      <td>3</td>
      <td>[1]</td>
      <td>0.017249</td>
      <td>0.013082</td>
      <td>0.002766</td>
      <td>1093</td>
      <td>360</td>
      <td>0.752237</td>
    </tr>
    <tr>
      <th>polarity_negative</th>
      <td>4</td>
      <td>[0]</td>
      <td>0.015836</td>
      <td>0.003953</td>
      <td>0.003110</td>
      <td>727</td>
      <td>607</td>
      <td>0.544978</td>
    </tr>
  </tbody>
</table>
</div>



### Applying labeling functions to the training set

We apply the labeling functions to the training set, and then filter out examples unlabeled by any LF, and combine the rest with the dev set to form our final training set.


```python
from snorkel.labeling.model.label_model import LabelModel

# Train LabelModel.
L_train = applier.apply(df_train)
label_model = LabelModel(cardinality=2, verbose=True)
label_model.fit(L_train, n_epochs=5000, seed=123, log_freq=20, lr=0.01)
Y_train_preds = label_model.predict(L_train)
```

    100%|██████████| 600658/600658 [15:55<00:00, 628.48it/s]



```python
import pandas as pd
from snorkel.labeling import filter_unlabeled_dataframe

df_train_filtered, Y_train_preds_filtered = filter_unlabeled_dataframe(
    df_train, Y_train_preds, L_train
)
df_train_filtered["rating"] = Y_train_preds_filtered
combined_df_train = pd.concat([df_train_filtered, df_dev], axis=0)
```

    /home/ubuntu/snorkel-tutorials/.tox/recsys/lib/python3.6/site-packages/ipykernel_launcher.py:7: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      import sys
    /home/ubuntu/snorkel-tutorials/.tox/recsys/lib/python3.6/site-packages/ipykernel_launcher.py:8: FutureWarning: Sorting because non-concatenation axis is not aligned. A future version
    of pandas will change to not sort by default.
    
    To accept the future behavior, pass 'sort=False'.
    
    To retain the current behavior and silence the warning, pass 'sort=True'.
    
      


### Rating Prediction Model
We write a Keras model for predicting ratings given a user's book list and a book (which is being rated). The model represents the list of books the user interacted with, `books_idxs`, by learning an embedding for each idx, and averaging the embeddings in `book_idxs`. It learns another embedding for the `book_idx`, the book to be rated. Then it concatenates the two embeddings and uses an [MLP](https://en.wikipedia.org/wiki/Multilayer_perceptron) to compute the probability of the `rating` being 1.


```python
import numpy as np
import tensorflow as tf
from utils import precision, recall, f1

n_books = len(df_books)


# Keras model to predict rating given book_idxs and book_idx.
def get_model(embed_dim=64, hidden_layer_sizes=[32]):
    # Compute embedding for book_idxs.
    len_book_idxs = tf.keras.layers.Input([])
    book_idxs = tf.keras.layers.Input([None])
    book_idxs_emb = tf.keras.layers.Embedding(n_books, embed_dim)(book_idxs)
    book_idxs_emb = tf.math.divide(
        tf.keras.backend.sum(book_idxs_emb, axis=1), tf.expand_dims(len_book_idxs, 1)
    )
    # Compute embedding for book_idx.
    book_idx = tf.keras.layers.Input([])
    book_idx_emb = tf.keras.layers.Embedding(n_books, embed_dim)(book_idx)
    input_layer = tf.keras.layers.concatenate([book_idxs_emb, book_idx_emb], 1)
    # Build Multi Layer Perceptron on input layer.
    cur_layer = input_layer
    for size in hidden_layer_sizes:
        tf.keras.layers.Dense(size, activation=tf.nn.relu)(cur_layer)
    output_layer = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(cur_layer)
    # Create and compile keras model.
    model = tf.keras.Model(
        inputs=[len_book_idxs, book_idxs, book_idx], outputs=[output_layer]
    )
    model.compile(
        "Adagrad", "binary_crossentropy", metrics=["accuracy", f1, precision, recall]
    )
    return model
```

We use triples of (`book_idxs`, `book_idx`, `rating`) from our dataframes as training examples. In addition, we want to train the model to recognize when a user will not read a book. To create examples for that, we randomly sample a `book_id` not in `book_idxs` and use that with a `rating` of 0 as a _random negative_ example. We create one such _random negative_ example for every positive (`rating` 1) example in our dataframe so that positive and negative examples are roughly balanced.


```python
# Generator to turn dataframe into examples.
def get_examples_generator(df):
    def generator():
        for book_idxs, book_idx, rating in zip(df.book_idxs, df.book_idx, df.rating):
            # Remove book_idx from book_idxs so the model can't just look it up.
            book_idxs = tuple(filter(lambda x: x != book_idx, book_idxs))
            yield {
                "len_book_idxs": len(book_idxs),
                "book_idxs": book_idxs,
                "book_idx": book_idx,
                "label": rating,
            }
            if rating == 1:
                # Generate a random negative book_id not in book_idxs.
                random_negative = np.random.randint(0, n_books)
                while random_negative in book_idxs:
                    random_negative = np.random.randint(0, n_books)
                yield {
                    "len_book_idxs": len(book_idxs),
                    "book_idxs": book_idxs,
                    "book_idx": random_negative,
                    "label": 0,
                }

    return generator


def get_data_tensors(df):
    # Use generator to get examples each epoch, along with shuffling and batching.
    padded_shapes = {
        "len_book_idxs": [],
        "book_idxs": [None],
        "book_idx": [],
        "label": [],
    }
    dataset = (
        tf.data.Dataset.from_generator(
            get_examples_generator(df), {k: tf.int64 for k in padded_shapes}
        )
        .shuffle(123)
        .repeat(None)
        .padded_batch(batch_size=256, padded_shapes=padded_shapes)
    )
    tensor_dict = tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()
    return (
        tensor_dict["len_book_idxs"],
        tensor_dict["book_idxs"],
        tensor_dict["book_idx"],
        tensor_dict["label"],
    )
```

We now train the model on our combined training data (data labeled by LFs plus dev data).



```python
model = get_model()

train_data_tensors = get_data_tensors(combined_df_train)
val_data_tensors = get_data_tensors(df_val)
model.fit(
    train_data_tensors[:-1],
    train_data_tensors[-1],
    steps_per_epoch=300,
    validation_data=(val_data_tensors[:-1], val_data_tensors[-1]),
    validation_steps=40,
    epochs=30,
    verbose=1,
)
```

    WARNING: Logging before flag parsing goes to stderr.
    W0811 03:42:43.977466 140444578916160 deprecation.py:506] From /home/ubuntu/snorkel-tutorials/.tox/recsys/lib/python3.6/site-packages/tensorflow/python/keras/initializers.py:119: calling RandomUniform.__init__ (from tensorflow.python.ops.init_ops) with dtype is deprecated and will be removed in a future version.
    Instructions for updating:
    Call initializer instance with the dtype argument instead of passing it to the constructor
    W0811 03:42:44.057152 140444578916160 deprecation.py:506] From /home/ubuntu/snorkel-tutorials/.tox/recsys/lib/python3.6/site-packages/tensorflow/python/ops/init_ops.py:1251: calling VarianceScaling.__init__ (from tensorflow.python.ops.init_ops) with dtype is deprecated and will be removed in a future version.
    Instructions for updating:
    Call initializer instance with the dtype argument instead of passing it to the constructor
    W0811 03:42:44.172415 140444578916160 deprecation.py:323] From /home/ubuntu/snorkel-tutorials/.tox/recsys/lib/python3.6/site-packages/tensorflow/python/ops/nn_impl.py:180: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use tf.where in 2.0, which has the same broadcast rule as np.where
    W0811 03:42:44.194866 140444578916160 deprecation.py:323] From /home/ubuntu/snorkel-tutorials/.tox/recsys/lib/python3.6/site-packages/tensorflow/python/data/ops/dataset_ops.py:494: py_func (from tensorflow.python.ops.script_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    tf.py_func is deprecated in TF V2. Instead, there are two
        options available in V2.
        - tf.py_function takes a python function which manipulates tf eager
        tensors instead of numpy arrays. It's easy to convert a tf eager tensor to
        an ndarray (just call tensor.numpy()) but having access to eager tensors
        means `tf.py_function`s can use accelerators such as GPUs as well as
        being differentiable using a gradient tape.
        - tf.numpy_function maintains the semantics of the deprecated tf.py_func
        (it is not differentiable, and manipulates numpy arrays). It drops the
        stateful argument making all functions stateful.
        
    W0811 03:42:44.384439 140444578916160 deprecation.py:506] From /home/ubuntu/snorkel-tutorials/.tox/recsys/lib/python3.6/site-packages/tensorflow/python/keras/optimizer_v2/adagrad.py:105: calling Constant.__init__ (from tensorflow.python.ops.init_ops) with dtype is deprecated and will be removed in a future version.
    Instructions for updating:
    Call initializer instance with the dtype argument instead of passing it to the constructor


    Epoch 1/30
    300/300 [==============================] - 13s 43ms/step - loss: 0.6756 - acc: 0.6089 - f1: 0.0957 - precision: 0.5927 - recall: 0.0565 - val_loss: 0.6797 - val_acc: 0.6043 - val_f1: 0.1248 - val_precision: 0.4908 - val_recall: 0.0803
    Epoch 2/30
    300/300 [==============================] - 12s 41ms/step - loss: 0.6680 - acc: 0.6217 - f1: 0.1130 - precision: 0.6850 - recall: 0.0651 - val_loss: 0.6770 - val_acc: 0.6215 - val_f1: 0.1260 - val_precision: 0.5339 - val_recall: 0.0777
    Epoch 3/30
    300/300 [==============================] - 12s 40ms/step - loss: 0.6767 - acc: 0.6109 - f1: 0.0896 - precision: 0.5716 - recall: 0.0521 - val_loss: 0.6768 - val_acc: 0.6218 - val_f1: 0.1036 - val_precision: 0.5426 - val_recall: 0.0604
    Epoch 4/30
    300/300 [==============================] - 12s 41ms/step - loss: 0.6674 - acc: 0.6164 - f1: 0.0846 - precision: 0.7079 - recall: 0.0467 - val_loss: 0.6729 - val_acc: 0.6204 - val_f1: 0.0834 - val_precision: 0.6433 - val_recall: 0.0463
    Epoch 5/30
    300/300 [==============================] - 13s 42ms/step - loss: 0.6544 - acc: 0.6317 - f1: 0.1251 - precision: 0.8791 - recall: 0.0689 - val_loss: 0.6728 - val_acc: 0.6188 - val_f1: 0.1353 - val_precision: 0.6626 - val_recall: 0.0796
    Epoch 6/30
    300/300 [==============================] - 12s 40ms/step - loss: 0.6652 - acc: 0.6207 - f1: 0.1258 - precision: 0.7197 - recall: 0.0722 - val_loss: 0.6723 - val_acc: 0.6191 - val_f1: 0.1116 - val_precision: 0.6090 - val_recall: 0.0650
    Epoch 7/30
    300/300 [==============================] - 12s 40ms/step - loss: 0.6696 - acc: 0.6203 - f1: 0.1197 - precision: 0.6410 - recall: 0.0699 - val_loss: 0.6688 - val_acc: 0.6238 - val_f1: 0.1309 - val_precision: 0.6574 - val_recall: 0.0760
    Epoch 8/30
    300/300 [==============================] - 13s 42ms/step - loss: 0.6459 - acc: 0.6386 - f1: 0.1628 - precision: 0.9043 - recall: 0.0915 - val_loss: 0.6672 - val_acc: 0.6229 - val_f1: 0.1357 - val_precision: 0.6293 - val_recall: 0.0806
    Epoch 9/30
    300/300 [==============================] - 12s 41ms/step - loss: 0.6449 - acc: 0.6454 - f1: 0.2012 - precision: 0.8502 - recall: 0.1176 - val_loss: 0.6699 - val_acc: 0.6218 - val_f1: 0.1863 - val_precision: 0.6713 - val_recall: 0.1119
    Epoch 10/30
    300/300 [==============================] - 12s 40ms/step - loss: 0.6652 - acc: 0.6262 - f1: 0.1659 - precision: 0.6868 - recall: 0.0994 - val_loss: 0.6693 - val_acc: 0.6226 - val_f1: 0.1695 - val_precision: 0.6203 - val_recall: 0.1042
    Epoch 11/30
    300/300 [==============================] - 12s 41ms/step - loss: 0.6508 - acc: 0.6356 - f1: 0.1792 - precision: 0.7736 - recall: 0.1051 - val_loss: 0.6660 - val_acc: 0.6218 - val_f1: 0.1697 - val_precision: 0.6910 - val_recall: 0.1017
    Epoch 12/30
    300/300 [==============================] - 13s 42ms/step - loss: 0.6319 - acc: 0.6582 - f1: 0.2507 - precision: 0.9109 - recall: 0.1484 - val_loss: 0.6662 - val_acc: 0.6285 - val_f1: 0.1964 - val_precision: 0.6827 - val_recall: 0.1198
    Epoch 13/30
    300/300 [==============================] - 12s 41ms/step - loss: 0.6503 - acc: 0.6418 - f1: 0.2319 - precision: 0.7442 - recall: 0.1423 - val_loss: 0.6640 - val_acc: 0.6262 - val_f1: 0.2041 - val_precision: 0.6502 - val_recall: 0.1283
    Epoch 14/30
    300/300 [==============================] - 12s 40ms/step - loss: 0.6603 - acc: 0.6328 - f1: 0.2065 - precision: 0.6891 - recall: 0.1286 - val_loss: 0.6655 - val_acc: 0.6262 - val_f1: 0.2124 - val_precision: 0.6757 - val_recall: 0.1326
    Epoch 15/30
    300/300 [==============================] - 13s 42ms/step - loss: 0.6258 - acc: 0.6654 - f1: 0.2807 - precision: 0.8917 - recall: 0.1693 - val_loss: 0.6593 - val_acc: 0.6447 - val_f1: 0.2682 - val_precision: 0.6351 - val_recall: 0.1785
    Epoch 16/30
    300/300 [==============================] - 12s 42ms/step - loss: 0.6254 - acc: 0.6698 - f1: 0.3099 - precision: 0.8509 - recall: 0.1939 - val_loss: 0.6622 - val_acc: 0.6399 - val_f1: 0.2729 - val_precision: 0.6423 - val_recall: 0.1811
    Epoch 17/30
    300/300 [==============================] - 12s 40ms/step - loss: 0.6555 - acc: 0.6413 - f1: 0.2594 - precision: 0.7051 - recall: 0.1657 - val_loss: 0.6569 - val_acc: 0.6480 - val_f1: 0.2730 - val_precision: 0.6222 - val_recall: 0.1820
    Epoch 18/30
    300/300 [==============================] - 12s 41ms/step - loss: 0.6374 - acc: 0.6565 - f1: 0.2770 - precision: 0.7844 - recall: 0.1741 - val_loss: 0.6552 - val_acc: 0.6502 - val_f1: 0.2844 - val_precision: 0.7020 - val_recall: 0.1832
    Epoch 19/30
    300/300 [==============================] - 13s 42ms/step - loss: 0.6125 - acc: 0.6843 - f1: 0.3560 - precision: 0.9090 - recall: 0.2246 - val_loss: 0.6605 - val_acc: 0.6376 - val_f1: 0.2801 - val_precision: 0.6077 - val_recall: 0.1883
    Epoch 20/30
    300/300 [==============================] - 12s 41ms/step - loss: 0.6367 - acc: 0.6615 - f1: 0.3263 - precision: 0.7565 - recall: 0.2142 - val_loss: 0.6573 - val_acc: 0.6425 - val_f1: 0.2851 - val_precision: 0.6615 - val_recall: 0.1859
    Epoch 21/30
    300/300 [==============================] - 12s 41ms/step - loss: 0.6518 - acc: 0.6464 - f1: 0.2898 - precision: 0.7023 - recall: 0.1908 - val_loss: 0.6530 - val_acc: 0.6444 - val_f1: 0.2745 - val_precision: 0.6969 - val_recall: 0.1781
    Epoch 22/30
    300/300 [==============================] - 12s 42ms/step - loss: 0.6093 - acc: 0.6865 - f1: 0.3641 - precision: 0.8752 - recall: 0.2336 - val_loss: 0.6563 - val_acc: 0.6420 - val_f1: 0.3230 - val_precision: 0.6317 - val_recall: 0.2234
    Epoch 23/30
    300/300 [==============================] - 12s 41ms/step - loss: 0.6075 - acc: 0.6937 - f1: 0.3985 - precision: 0.8586 - recall: 0.2639 - val_loss: 0.6584 - val_acc: 0.6425 - val_f1: 0.3206 - val_precision: 0.6583 - val_recall: 0.2184
    Epoch 24/30
    300/300 [==============================] - 12s 40ms/step - loss: 0.6475 - acc: 0.6543 - f1: 0.3360 - precision: 0.7018 - recall: 0.2288 - val_loss: 0.6536 - val_acc: 0.6513 - val_f1: 0.3285 - val_precision: 0.6443 - val_recall: 0.2315
    Epoch 25/30
    300/300 [==============================] - 12s 41ms/step - loss: 0.6277 - acc: 0.6697 - f1: 0.3460 - precision: 0.7637 - recall: 0.2299 - val_loss: 0.6541 - val_acc: 0.6414 - val_f1: 0.3140 - val_precision: 0.6921 - val_recall: 0.2113
    Epoch 26/30
    300/300 [==============================] - 13s 42ms/step - loss: 0.5962 - acc: 0.7036 - f1: 0.4288 - precision: 0.8953 - recall: 0.2856 - val_loss: 0.6571 - val_acc: 0.6496 - val_f1: 0.3547 - val_precision: 0.6407 - val_recall: 0.2519
    Epoch 27/30
    300/300 [==============================] - 12s 41ms/step - loss: 0.6235 - acc: 0.6782 - f1: 0.3938 - precision: 0.7585 - recall: 0.2725 - val_loss: 0.6545 - val_acc: 0.6433 - val_f1: 0.3419 - val_precision: 0.6270 - val_recall: 0.2442
    Epoch 28/30
    300/300 [==============================] - 12s 41ms/step - loss: 0.6442 - acc: 0.6575 - f1: 0.3540 - precision: 0.6969 - recall: 0.2459 - val_loss: 0.6505 - val_acc: 0.6465 - val_f1: 0.3411 - val_precision: 0.6550 - val_recall: 0.2377
    Epoch 29/30
    300/300 [==============================] - 13s 42ms/step - loss: 0.5956 - acc: 0.7031 - f1: 0.4314 - precision: 0.8615 - recall: 0.2922 - val_loss: 0.6497 - val_acc: 0.6502 - val_f1: 0.3657 - val_precision: 0.6073 - val_recall: 0.2711
    Epoch 30/30
    300/300 [==============================] - 13s 42ms/step - loss: 0.5918 - acc: 0.7112 - f1: 0.4604 - precision: 0.8547 - recall: 0.3201 - val_loss: 0.6519 - val_acc: 0.6502 - val_f1: 0.3609 - val_precision: 0.6148 - val_recall: 0.2670





    <tensorflow.python.keras.callbacks.History at 0x7fba06ade7b8>



Finally, we evaluate the model's predicted ratings on our test data.



```python
test_data_tensors = get_data_tensors(df_test)
model.evaluate(test_data_tensors[:-1], test_data_tensors[-1], steps=30)
```

    30/30 [==============================] - 1s 31ms/step - loss: 0.6434 - acc: 0.6542 - f1: 0.3968 - precision: 0.6623 - recall: 0.2922





    [0.6434326887130737, 0.65416664, 0.39681196, 0.6622598, 0.29224512]



## Summary

In this tutorial, we showed one way to use Snorkel for recommendations. We used books' metadata and review text to create `LF`s that estimate user ratings. We used a `LabelModel` to combine the outputs of those `LF`s. Finally, we trained a model to predict what books a user will read and like (and therefore what books should be recommended to the user) based only on what books the user has interacted with in the past.

Here we demonstrated one way to use Snorkel for training a recommender system. Note, however, that this approach could easily be adapted to take advantage of additional information as it is available (e.g., user profile data, denser user ratings, and so on.)
