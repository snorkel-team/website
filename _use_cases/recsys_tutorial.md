---
layout: default
title: Building Recommender Systems in Snorkel
description: Labeling text reviews for book recommendations
excerpt: Labeling text reviews for book recommendations
order: 5
---


# Recommender Systems Tutorial
In this tutorial, we'll provide a simple walkthrough of how to use Snorkel to build a recommender system.
We consider a setting similar to the [Netflix challenge](https://www.kaggle.com/netflix-inc/netflix-prize-data), but with books instead of movies.
We have a set of users and books, and for each user we know the set of books they have interacted with (read or marked as to-read).
We don't have the user's numerical ratings for the books they read, except in a small number of cases.
We also have some text reviews written by users.

Our goal is to build a recommender system by training a classifier to predict whether a user will read and like any given book.
We'll train our model over a user-book pair to predict a `rating` (a `rating` of 1 means the user will read and like the book).
To simplify inference, we'll represent a user by the set of books they interacted with (rather than learning a specific representation for each user).
Once we have this model trained, we can use it to recommend books to a user when they visit the site.
For example, we can just predict the rating for the user paired with a book for a few thousand likely books, then pick the books with the top ten predicted ratings.

Of course, there are many other ways to approach this problem.
The field of [recommender systems](https://en.wikipedia.org/wiki/Recommender_system) is a very well studied area with a wide variety of settings and approaches, and we just focus on one of them.

We will use the [Goodreads](https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/home) dataset, from
"Item Recommendation on Monotonic Behavior Chains", RecSys'18 (Mengting Wan, Julian McAuley), and "Fine-Grained Spoiler Detection from Large-Scale Review Corpora", ACL'19 (Mengting Wan, Rishabh Misra, Ndapa Nakashole, Julian McAuley).
In this dataset, we have user interactions and reviews for Young Adult novels from the Goodreads website, along with metadata (like `title` and `authors`) for the novels.


```python
import logging
import os

logging.basicConfig(level=logging.INFO)


if os.path.basename(os.getcwd()) == "snorkel-tutorials":
    os.chdir("recsys")
```

## Loading Data

We start by running the `download_and_process_data` function.
The function returns the `df_train`, `df_test`, `df_dev`, `df_valid` dataframes, which correspond to our training, test, development, and validation sets.
Each of those dataframes has the following fields:
* `user_idx`: A unique identifier for a user.
* `book_idx`: A unique identifier for a book that is being rated by the user.
* `book_idxs`: The set of books that the user has interacted with (read or planned to read).
* `review_text`: Optional text review written by the user for the book.
* `rating`: Either `0` (which means the user did not read or did not like the book) or `1` (which means the user read and liked the book). The `rating` field is missing for `df_train`.
Our objective is to predict whether a given user (represented by the set of book_idxs the user has interacted with) will read and like any given book.
That is, we want to train a model that takes a set of `book_idxs` (the user) and a single `book_idx` (the book to rate) and predicts the `rating`.

In addition, `download_and_process_data` also returns the `df_books` dataframe, which contains one row per book, along with metadata for that book (such as `title` and `first_author`).


```python
from utils import download_and_process_data

(df_train, df_test, df_dev, df_valid), df_books = download_and_process_data()

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
    WARNING: Logging before flag parsing goes to stderr.
    I0814 17:15:39.273437 139895126660928 utils.py:213] Downloading raw data
    I0814 17:15:39.274201 139895126660928 utils.py:217] Processing book data
    I0814 17:15:53.886617 139895126660928 utils.py:219] Processing interaction data
    /home/ubuntu/snorkel-tutorials/recsys/utils.py:223: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      df_interactions_nz["rating_4_5"] = df_interactions_nz.rating.map(ratings_map)
    I0814 17:17:39.577783 139895126660928 utils.py:224] Processing review data
    I0814 17:18:19.264470 139895126660928 utils.py:226] Joining interaction data





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



We look at a sample of the labeled development set.
As an example, we want our final recommendations model to be able to predict that a user who has interacted with `book_idxs` (25743, 22318, 7662, 6857, 83, 14495, 30664, ...) would either not read or not like the book with `book_idx` 22764 (first row), while a user who has interacted with `book_idxs` (3880, 18078, 9092, 29933, 1511, 8560, ...) would read and like the book with `book_idx` 3181 (second row).


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
      <th>537461</th>
      <td>20750</td>
      <td>(16291, 9527, 810, 375, 25580, 29806, 3501, 26...</td>
      <td>26689</td>
      <td>1</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>293495</th>
      <td>11331</td>
      <td>(30780, 13579, 6387, 8652, 2462, 20361, 15624,...</td>
      <td>27340</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>728037</th>
      <td>28300</td>
      <td>(10586, 6471, 10671, 3072, 29502, 7111, 17182,...</td>
      <td>17202</td>
      <td>1</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>193870</th>
      <td>7503</td>
      <td>(16349, 14639, 5197, 21034, 30269, 27065, 2575...</td>
      <td>8448</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>100394</th>
      <td>3931</td>
      <td>(4220, 22689, 22780, 21086, 4739, 30664, 10275...</td>
      <td>25740</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



## Writing Labeling Functions


```python
POSITIVE = 1
NEGATIVE = 0
ABSTAIN = -1
```

If a user has interacted with several books written by an author, there is a good chance that the user will read and like other books by the same author.
We express this as a labeling function, using the `first_author` field in the `df_books` dataframe.
We picked the threshold 15 by plotting histograms and running error analysis using the dev set.


```python
from snorkel.labeling.lf import labeling_function

book_to_first_author = dict(zip(df_books.book_idx, df_books.first_author))
first_author_to_books_df = df_books.groupby("first_author")[["book_idx"]].agg(set)
first_author_to_books = dict(
    zip(first_author_to_books_df.index, first_author_to_books_df.book_idx)
)


@labeling_function(
    resources=dict(
        book_to_first_author=book_to_first_author,
        first_author_to_books=first_author_to_books,
    )
)
def shared_first_author(x, book_to_first_author, first_author_to_books):
    author = book_to_first_author[x.book_idx]
    same_author_books = first_author_to_books[author]
    num_read = len(set(x.book_idxs).intersection(same_author_books))
    return POSITIVE if num_read > 15 else ABSTAIN
```

We can also leverage the long text reviews written by users to guess whether they liked or disliked a book.
For example, the third `df_dev` entry above has a review with the text `'4.5 STARS'`, which indicates that the user liked the book.
We write a simple LF that looks for similar phrases to guess the user's rating of a book.
We interpret >= 4 stars to indicate a positive rating, while < 4 stars is negative.


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


@labeling_function(
    resources=dict(low_rating_strs=low_rating_strs, high_rating_strs=high_rating_strs)
)
def stars_in_review(x, low_rating_strs, high_rating_strs):
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
As usual, these thresholds were picked by analyzing the score distributions and running error analysis.


```python
from snorkel.preprocess import preprocessor
from textblob import TextBlob


@preprocessor(memoize=True)
def textblob_polarity(x):
    if isinstance(x.review_text, str):
        x.blob = TextBlob(x.review_text)
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

    100%|██████████| 7915/7915 [00:07<00:00, 994.94it/s] 





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
      <td>0.027669</td>
      <td>0.004927</td>
      <td>0.001390</td>
      <td>184</td>
      <td>35</td>
      <td>0.840183</td>
    </tr>
    <tr>
      <th>shared_first_author</th>
      <td>1</td>
      <td>[1]</td>
      <td>0.050158</td>
      <td>0.001516</td>
      <td>0.000126</td>
      <td>337</td>
      <td>60</td>
      <td>0.848866</td>
    </tr>
    <tr>
      <th>polarity_positive</th>
      <td>2</td>
      <td>[1]</td>
      <td>0.040809</td>
      <td>0.012255</td>
      <td>0.000758</td>
      <td>258</td>
      <td>65</td>
      <td>0.798762</td>
    </tr>
    <tr>
      <th>subjectivity_positive</th>
      <td>3</td>
      <td>[1]</td>
      <td>0.017814</td>
      <td>0.014277</td>
      <td>0.003538</td>
      <td>107</td>
      <td>34</td>
      <td>0.758865</td>
    </tr>
    <tr>
      <th>polarity_negative</th>
      <td>4</td>
      <td>[0]</td>
      <td>0.018193</td>
      <td>0.004927</td>
      <td>0.003917</td>
      <td>83</td>
      <td>61</td>
      <td>0.576389</td>
    </tr>
  </tbody>
</table>
</div>



### Applying labeling functions to the training set

We apply the labeling functions to the training set, and then filter out examples unlabeled by any LF to form our final training set.


```python
from snorkel.labeling.model.label_model import LabelModel

L_train = applier.apply(df_train)
label_model = LabelModel(cardinality=2, verbose=True)
label_model.fit(L_train, n_epochs=5000, seed=123, log_freq=20, lr=0.01)
preds_train = label_model.predict(L_train)
```

    100%|██████████| 797586/797586 [12:16<00:00, 1083.38it/s]
    I0814 17:30:50.074586 139895126660928 label_model.py:749] Computing O...
    I0814 17:30:50.170222 139895126660928 label_model.py:755] Estimating \mu...
    I0814 17:30:50.172878 139895126660928 logger.py:79] [0 epochs]: TRAIN:[loss=0.002]
    I0814 17:30:50.185016 139895126660928 logger.py:79] [20 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:50.197252 139895126660928 logger.py:79] [40 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:50.209349 139895126660928 logger.py:79] [60 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:50.221447 139895126660928 logger.py:79] [80 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:50.233370 139895126660928 logger.py:79] [100 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:50.245395 139895126660928 logger.py:79] [120 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:50.257663 139895126660928 logger.py:79] [140 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:50.269640 139895126660928 logger.py:79] [160 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:50.281580 139895126660928 logger.py:79] [180 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:50.293640 139895126660928 logger.py:79] [200 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:50.305543 139895126660928 logger.py:79] [220 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:50.317475 139895126660928 logger.py:79] [240 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:50.329413 139895126660928 logger.py:79] [260 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:50.341285 139895126660928 logger.py:79] [280 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:50.353212 139895126660928 logger.py:79] [300 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:50.365334 139895126660928 logger.py:79] [320 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:50.377307 139895126660928 logger.py:79] [340 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:50.389528 139895126660928 logger.py:79] [360 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:50.401491 139895126660928 logger.py:79] [380 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:50.413330 139895126660928 logger.py:79] [400 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:50.425260 139895126660928 logger.py:79] [420 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:50.437290 139895126660928 logger.py:79] [440 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:50.449555 139895126660928 logger.py:79] [460 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:50.461633 139895126660928 logger.py:79] [480 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:50.473637 139895126660928 logger.py:79] [500 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:50.485734 139895126660928 logger.py:79] [520 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:50.497795 139895126660928 logger.py:79] [540 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:50.509747 139895126660928 logger.py:79] [560 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:50.521946 139895126660928 logger.py:79] [580 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:50.534167 139895126660928 logger.py:79] [600 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:50.546195 139895126660928 logger.py:79] [620 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:50.558326 139895126660928 logger.py:79] [640 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:50.570265 139895126660928 logger.py:79] [660 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:50.582284 139895126660928 logger.py:79] [680 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:50.594533 139895126660928 logger.py:79] [700 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:50.606484 139895126660928 logger.py:79] [720 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:50.618563 139895126660928 logger.py:79] [740 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:50.630787 139895126660928 logger.py:79] [760 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:50.643077 139895126660928 logger.py:79] [780 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:50.655352 139895126660928 logger.py:79] [800 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:50.667226 139895126660928 logger.py:79] [820 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:50.679294 139895126660928 logger.py:79] [840 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:50.691244 139895126660928 logger.py:79] [860 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:50.703303 139895126660928 logger.py:79] [880 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:50.715260 139895126660928 logger.py:79] [900 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:50.727328 139895126660928 logger.py:79] [920 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:50.739200 139895126660928 logger.py:79] [940 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:50.751429 139895126660928 logger.py:79] [960 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:50.763593 139895126660928 logger.py:79] [980 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:50.775859 139895126660928 logger.py:79] [1000 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:50.788101 139895126660928 logger.py:79] [1020 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:50.800579 139895126660928 logger.py:79] [1040 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:50.812884 139895126660928 logger.py:79] [1060 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:50.825309 139895126660928 logger.py:79] [1080 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:50.837304 139895126660928 logger.py:79] [1100 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:50.849509 139895126660928 logger.py:79] [1120 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:50.861765 139895126660928 logger.py:79] [1140 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:50.874136 139895126660928 logger.py:79] [1160 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:50.886442 139895126660928 logger.py:79] [1180 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:50.898702 139895126660928 logger.py:79] [1200 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:50.910858 139895126660928 logger.py:79] [1220 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:50.923148 139895126660928 logger.py:79] [1240 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:50.935363 139895126660928 logger.py:79] [1260 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:50.947790 139895126660928 logger.py:79] [1280 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:50.960047 139895126660928 logger.py:79] [1300 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:50.972263 139895126660928 logger.py:79] [1320 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:50.984333 139895126660928 logger.py:79] [1340 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:50.996439 139895126660928 logger.py:79] [1360 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:51.008404 139895126660928 logger.py:79] [1380 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:51.020300 139895126660928 logger.py:79] [1400 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:51.032280 139895126660928 logger.py:79] [1420 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:51.044095 139895126660928 logger.py:79] [1440 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:51.056093 139895126660928 logger.py:79] [1460 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:51.068230 139895126660928 logger.py:79] [1480 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:51.080171 139895126660928 logger.py:79] [1500 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:51.092098 139895126660928 logger.py:79] [1520 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:51.104483 139895126660928 logger.py:79] [1540 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:51.116434 139895126660928 logger.py:79] [1560 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:51.128567 139895126660928 logger.py:79] [1580 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:51.140633 139895126660928 logger.py:79] [1600 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:51.152662 139895126660928 logger.py:79] [1620 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:51.164633 139895126660928 logger.py:79] [1640 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:51.176759 139895126660928 logger.py:79] [1660 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:51.188678 139895126660928 logger.py:79] [1680 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:51.200708 139895126660928 logger.py:79] [1700 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:51.212785 139895126660928 logger.py:79] [1720 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:51.224779 139895126660928 logger.py:79] [1740 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:51.236878 139895126660928 logger.py:79] [1760 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:51.248964 139895126660928 logger.py:79] [1780 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:51.260921 139895126660928 logger.py:79] [1800 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:51.273132 139895126660928 logger.py:79] [1820 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:51.285407 139895126660928 logger.py:79] [1840 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:51.297818 139895126660928 logger.py:79] [1860 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:51.310514 139895126660928 logger.py:79] [1880 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:51.322724 139895126660928 logger.py:79] [1900 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:51.335032 139895126660928 logger.py:79] [1920 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:51.347162 139895126660928 logger.py:79] [1940 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:51.359275 139895126660928 logger.py:79] [1960 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:51.371477 139895126660928 logger.py:79] [1980 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:51.383541 139895126660928 logger.py:79] [2000 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:51.395691 139895126660928 logger.py:79] [2020 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:51.407995 139895126660928 logger.py:79] [2040 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:51.420147 139895126660928 logger.py:79] [2060 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:51.432208 139895126660928 logger.py:79] [2080 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:51.444383 139895126660928 logger.py:79] [2100 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:51.456567 139895126660928 logger.py:79] [2120 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:51.468657 139895126660928 logger.py:79] [2140 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:51.480729 139895126660928 logger.py:79] [2160 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:51.492893 139895126660928 logger.py:79] [2180 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:51.504992 139895126660928 logger.py:79] [2200 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:51.517257 139895126660928 logger.py:79] [2220 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:51.529517 139895126660928 logger.py:79] [2240 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:51.541854 139895126660928 logger.py:79] [2260 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:51.554209 139895126660928 logger.py:79] [2280 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:51.566596 139895126660928 logger.py:79] [2300 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:51.579092 139895126660928 logger.py:79] [2320 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:51.591497 139895126660928 logger.py:79] [2340 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:51.603853 139895126660928 logger.py:79] [2360 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:51.616179 139895126660928 logger.py:79] [2380 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:51.628416 139895126660928 logger.py:79] [2400 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:51.640773 139895126660928 logger.py:79] [2420 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:51.652885 139895126660928 logger.py:79] [2440 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:51.665138 139895126660928 logger.py:79] [2460 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:51.677559 139895126660928 logger.py:79] [2480 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:51.690058 139895126660928 logger.py:79] [2500 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:51.702590 139895126660928 logger.py:79] [2520 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:51.714862 139895126660928 logger.py:79] [2540 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:51.727257 139895126660928 logger.py:79] [2560 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:51.739703 139895126660928 logger.py:79] [2580 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:51.752080 139895126660928 logger.py:79] [2600 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:51.764387 139895126660928 logger.py:79] [2620 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:51.776860 139895126660928 logger.py:79] [2640 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:51.789069 139895126660928 logger.py:79] [2660 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:51.801312 139895126660928 logger.py:79] [2680 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:51.813497 139895126660928 logger.py:79] [2700 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:51.825767 139895126660928 logger.py:79] [2720 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:51.838007 139895126660928 logger.py:79] [2740 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:51.850354 139895126660928 logger.py:79] [2760 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:51.862698 139895126660928 logger.py:79] [2780 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:51.874985 139895126660928 logger.py:79] [2800 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:51.887178 139895126660928 logger.py:79] [2820 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:51.899532 139895126660928 logger.py:79] [2840 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:51.911886 139895126660928 logger.py:79] [2860 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:51.924115 139895126660928 logger.py:79] [2880 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:51.936376 139895126660928 logger.py:79] [2900 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:51.948826 139895126660928 logger.py:79] [2920 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:51.961257 139895126660928 logger.py:79] [2940 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:51.973456 139895126660928 logger.py:79] [2960 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:51.985737 139895126660928 logger.py:79] [2980 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:51.998170 139895126660928 logger.py:79] [3000 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:52.010457 139895126660928 logger.py:79] [3020 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:52.023060 139895126660928 logger.py:79] [3040 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:52.035462 139895126660928 logger.py:79] [3060 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:52.047881 139895126660928 logger.py:79] [3080 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:52.060218 139895126660928 logger.py:79] [3100 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:52.072613 139895126660928 logger.py:79] [3120 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:52.084890 139895126660928 logger.py:79] [3140 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:52.097040 139895126660928 logger.py:79] [3160 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:52.109399 139895126660928 logger.py:79] [3180 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:52.121665 139895126660928 logger.py:79] [3200 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:52.133998 139895126660928 logger.py:79] [3220 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:52.146300 139895126660928 logger.py:79] [3240 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:52.158744 139895126660928 logger.py:79] [3260 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:52.171091 139895126660928 logger.py:79] [3280 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:52.183213 139895126660928 logger.py:79] [3300 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:52.195402 139895126660928 logger.py:79] [3320 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:52.207855 139895126660928 logger.py:79] [3340 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:52.220472 139895126660928 logger.py:79] [3360 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:52.232908 139895126660928 logger.py:79] [3380 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:52.245181 139895126660928 logger.py:79] [3400 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:52.257652 139895126660928 logger.py:79] [3420 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:52.269988 139895126660928 logger.py:79] [3440 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:52.282374 139895126660928 logger.py:79] [3460 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:52.294627 139895126660928 logger.py:79] [3480 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:52.307283 139895126660928 logger.py:79] [3500 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:52.319733 139895126660928 logger.py:79] [3520 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:52.332201 139895126660928 logger.py:79] [3540 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:52.344540 139895126660928 logger.py:79] [3560 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:52.357019 139895126660928 logger.py:79] [3580 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:52.369407 139895126660928 logger.py:79] [3600 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:52.381838 139895126660928 logger.py:79] [3620 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:52.394240 139895126660928 logger.py:79] [3640 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:52.406520 139895126660928 logger.py:79] [3660 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:52.418991 139895126660928 logger.py:79] [3680 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:52.431367 139895126660928 logger.py:79] [3700 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:52.443757 139895126660928 logger.py:79] [3720 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:52.455949 139895126660928 logger.py:79] [3740 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:52.468314 139895126660928 logger.py:79] [3760 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:52.480811 139895126660928 logger.py:79] [3780 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:52.493185 139895126660928 logger.py:79] [3800 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:52.505382 139895126660928 logger.py:79] [3820 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:52.517765 139895126660928 logger.py:79] [3840 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:52.529972 139895126660928 logger.py:79] [3860 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:52.542211 139895126660928 logger.py:79] [3880 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:52.554461 139895126660928 logger.py:79] [3900 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:52.566870 139895126660928 logger.py:79] [3920 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:52.579218 139895126660928 logger.py:79] [3940 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:52.591530 139895126660928 logger.py:79] [3960 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:52.603925 139895126660928 logger.py:79] [3980 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:52.616180 139895126660928 logger.py:79] [4000 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:52.628661 139895126660928 logger.py:79] [4020 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:52.641045 139895126660928 logger.py:79] [4040 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:52.653524 139895126660928 logger.py:79] [4060 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:52.665812 139895126660928 logger.py:79] [4080 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:52.678239 139895126660928 logger.py:79] [4100 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:52.690371 139895126660928 logger.py:79] [4120 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:52.702651 139895126660928 logger.py:79] [4140 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:52.715012 139895126660928 logger.py:79] [4160 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:52.727106 139895126660928 logger.py:79] [4180 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:52.739332 139895126660928 logger.py:79] [4200 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:52.751591 139895126660928 logger.py:79] [4220 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:52.763956 139895126660928 logger.py:79] [4240 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:52.776299 139895126660928 logger.py:79] [4260 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:52.788577 139895126660928 logger.py:79] [4280 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:52.801053 139895126660928 logger.py:79] [4300 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:52.813688 139895126660928 logger.py:79] [4320 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:52.826127 139895126660928 logger.py:79] [4340 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:52.838535 139895126660928 logger.py:79] [4360 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:52.851121 139895126660928 logger.py:79] [4380 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:52.863532 139895126660928 logger.py:79] [4400 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:52.875773 139895126660928 logger.py:79] [4420 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:52.887969 139895126660928 logger.py:79] [4440 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:52.900257 139895126660928 logger.py:79] [4460 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:52.912682 139895126660928 logger.py:79] [4480 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:52.925096 139895126660928 logger.py:79] [4500 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:52.937613 139895126660928 logger.py:79] [4520 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:52.949754 139895126660928 logger.py:79] [4540 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:52.962102 139895126660928 logger.py:79] [4560 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:52.974398 139895126660928 logger.py:79] [4580 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:52.986732 139895126660928 logger.py:79] [4600 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:52.999076 139895126660928 logger.py:79] [4620 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:53.011294 139895126660928 logger.py:79] [4640 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:53.023772 139895126660928 logger.py:79] [4660 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:53.035991 139895126660928 logger.py:79] [4680 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:53.048420 139895126660928 logger.py:79] [4700 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:53.060846 139895126660928 logger.py:79] [4720 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:53.073180 139895126660928 logger.py:79] [4740 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:53.085498 139895126660928 logger.py:79] [4760 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:53.097835 139895126660928 logger.py:79] [4780 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:53.110163 139895126660928 logger.py:79] [4800 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:53.122511 139895126660928 logger.py:79] [4820 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:53.134806 139895126660928 logger.py:79] [4840 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:53.147325 139895126660928 logger.py:79] [4860 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:53.159759 139895126660928 logger.py:79] [4880 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:53.172252 139895126660928 logger.py:79] [4900 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:53.184683 139895126660928 logger.py:79] [4920 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:53.197281 139895126660928 logger.py:79] [4940 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:53.209853 139895126660928 logger.py:79] [4960 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:53.221947 139895126660928 logger.py:79] [4980 epochs]: TRAIN:[loss=0.000]
    I0814 17:30:53.233542 139895126660928 label_model.py:806] Finished Training



```python
from snorkel.labeling import filter_unlabeled_dataframe

df_train_filtered, preds_train_filtered = filter_unlabeled_dataframe(
    df_train, preds_train, L_train
)
df_train_filtered["rating"] = preds_train_filtered
```

    /home/ubuntu/snorkel-tutorials/.tox/recsys/lib/python3.6/site-packages/ipykernel_launcher.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      


### Rating Prediction Model
We write a Keras model for predicting ratings given a user's book list and a book (which is being rated).
The model represents the list of books the user interacted with, `books_idxs`, by learning an embedding for each idx, and averaging the embeddings in `book_idxs`.
It learns another embedding for the `book_idx`, the book to be rated.
Then it concatenates the two embeddings and uses an [MLP](https://en.wikipedia.org/wiki/Multilayer_perceptron) to compute the probability of the `rating` being 1.
This type of model is common in large-scale recommender systems, for example, the [YouTube recommender system](https://ai.google/research/pubs/pub45530).


```python
import numpy as np
import tensorflow as tf
from utils import precision_batch, recall_batch, f1_batch

n_books = max([max(df.book_idx) for df in [df_train, df_test, df_dev, df_valid]])


# Keras model to predict rating given book_idxs and book_idx.
def get_model(embed_dim=64, hidden_layer_sizes=[32]):
    # Compute embedding for book_idxs.
    len_book_idxs = tf.keras.layers.Input([])
    book_idxs = tf.keras.layers.Input([None])
    # book_idxs % n_books is to prevent crashing if a book_idx in book_idxs is > n_books.
    book_idxs_emb = tf.keras.layers.Embedding(n_books, embed_dim)(book_idxs % n_books)
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
        "Adagrad",
        "binary_crossentropy",
        metrics=["accuracy", f1_batch, precision_batch, recall_batch],
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
        (
            tensor_dict["len_book_idxs"],
            tensor_dict["book_idxs"],
            tensor_dict["book_idx"],
        ),
        tensor_dict["label"],
    )
```

We now train the model on our combined training data (data labeled by LFs plus dev data).



```python
from utils import get_n_epochs

model = get_model()

X_train, Y_train = get_data_tensors(df_train_filtered)
X_valid, Y_valid = get_data_tensors(df_valid)
model.fit(
    X_train,
    Y_train,
    steps_per_epoch=300,
    validation_data=(X_valid, Y_valid),
    validation_steps=40,
    epochs=get_n_epochs(),
    verbose=1,
)
```

    W0814 17:31:03.636563 139895126660928 deprecation.py:506] From /home/ubuntu/snorkel-tutorials/.tox/recsys/lib/python3.6/site-packages/tensorflow/python/keras/initializers.py:119: calling RandomUniform.__init__ (from tensorflow.python.ops.init_ops) with dtype is deprecated and will be removed in a future version.
    Instructions for updating:
    Call initializer instance with the dtype argument instead of passing it to the constructor
    W0814 17:31:03.719264 139895126660928 deprecation.py:506] From /home/ubuntu/snorkel-tutorials/.tox/recsys/lib/python3.6/site-packages/tensorflow/python/ops/init_ops.py:1251: calling VarianceScaling.__init__ (from tensorflow.python.ops.init_ops) with dtype is deprecated and will be removed in a future version.
    Instructions for updating:
    Call initializer instance with the dtype argument instead of passing it to the constructor
    W0814 17:31:03.830687 139895126660928 deprecation.py:323] From /home/ubuntu/snorkel-tutorials/.tox/recsys/lib/python3.6/site-packages/tensorflow/python/ops/nn_impl.py:180: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use tf.where in 2.0, which has the same broadcast rule as np.where
    W0814 17:31:03.852228 139895126660928 deprecation.py:323] From /home/ubuntu/snorkel-tutorials/.tox/recsys/lib/python3.6/site-packages/tensorflow/python/data/ops/dataset_ops.py:494: py_func (from tensorflow.python.ops.script_ops) is deprecated and will be removed in a future version.
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
        
    W0814 17:31:04.035569 139895126660928 deprecation.py:506] From /home/ubuntu/snorkel-tutorials/.tox/recsys/lib/python3.6/site-packages/tensorflow/python/keras/optimizer_v2/adagrad.py:105: calling Constant.__init__ (from tensorflow.python.ops.init_ops) with dtype is deprecated and will be removed in a future version.
    Instructions for updating:
    Call initializer instance with the dtype argument instead of passing it to the constructor


    Epoch 1/30
    300/300 [==============================] - 13s 42ms/step - loss: 0.6739 - acc: 0.6138 - f1_batch: 0.1281 - precision_batch: 0.5828 - recall_batch: 0.0765 - val_loss: 0.6802 - val_acc: 0.5983 - val_f1_batch: 0.1564 - val_precision_batch: 0.4784 - val_recall_batch: 0.1022
    Epoch 2/30
    300/300 [==============================] - 12s 41ms/step - loss: 0.6632 - acc: 0.6325 - f1_batch: 0.1478 - precision_batch: 0.7501 - recall_batch: 0.0853 - val_loss: 0.6792 - val_acc: 0.6015 - val_f1_batch: 0.1822 - val_precision_batch: 0.5035 - val_recall_batch: 0.1222
    Epoch 3/30
    300/300 [==============================] - 12s 41ms/step - loss: 0.6535 - acc: 0.6444 - f1_batch: 0.1974 - precision_batch: 0.8309 - recall_batch: 0.1154 - val_loss: 0.6806 - val_acc: 0.6008 - val_f1_batch: 0.2149 - val_precision_batch: 0.5528 - val_recall_batch: 0.1428
    Epoch 4/30
    300/300 [==============================] - 12s 42ms/step - loss: 0.6483 - acc: 0.6580 - f1_batch: 0.2551 - precision_batch: 0.8437 - recall_batch: 0.1549 - val_loss: 0.6773 - val_acc: 0.6069 - val_f1_batch: 0.2333 - val_precision_batch: 0.5590 - val_recall_batch: 0.1614
    Epoch 5/30
    300/300 [==============================] - 12s 41ms/step - loss: 0.6371 - acc: 0.6704 - f1_batch: 0.2930 - precision_batch: 0.8500 - recall_batch: 0.1820 - val_loss: 0.6742 - val_acc: 0.6118 - val_f1_batch: 0.2661 - val_precision_batch: 0.5652 - val_recall_batch: 0.1899
    Epoch 6/30
    300/300 [==============================] - 12s 42ms/step - loss: 0.6358 - acc: 0.6754 - f1_batch: 0.3315 - precision_batch: 0.8565 - recall_batch: 0.2115 - val_loss: 0.6742 - val_acc: 0.6197 - val_f1_batch: 0.3078 - val_precision_batch: 0.5888 - val_recall_batch: 0.2236
    Epoch 7/30
    300/300 [==============================] - 12s 41ms/step - loss: 0.6241 - acc: 0.6933 - f1_batch: 0.3734 - precision_batch: 0.8568 - recall_batch: 0.2453 - val_loss: 0.6760 - val_acc: 0.6133 - val_f1_batch: 0.2942 - val_precision_batch: 0.5545 - val_recall_batch: 0.2137
    Epoch 8/30
    300/300 [==============================] - 12s 42ms/step - loss: 0.6246 - acc: 0.6930 - f1_batch: 0.4003 - precision_batch: 0.8649 - recall_batch: 0.2656 - val_loss: 0.6739 - val_acc: 0.6228 - val_f1_batch: 0.3407 - val_precision_batch: 0.5980 - val_recall_batch: 0.2532
    Epoch 9/30
    300/300 [==============================] - 12s 41ms/step - loss: 0.6138 - acc: 0.7080 - f1_batch: 0.4286 - precision_batch: 0.8597 - recall_batch: 0.2921 - val_loss: 0.6714 - val_acc: 0.6244 - val_f1_batch: 0.3368 - val_precision_batch: 0.5941 - val_recall_batch: 0.2522
    Epoch 10/30
    300/300 [==============================] - 12s 41ms/step - loss: 0.6113 - acc: 0.7121 - f1_batch: 0.4564 - precision_batch: 0.8597 - recall_batch: 0.3173 - val_loss: 0.6690 - val_acc: 0.6314 - val_f1_batch: 0.3701 - val_precision_batch: 0.5900 - val_recall_batch: 0.2852
    Epoch 11/30
    300/300 [==============================] - 12s 41ms/step - loss: 0.6051 - acc: 0.7198 - f1_batch: 0.4736 - precision_batch: 0.8550 - recall_batch: 0.3333 - val_loss: 0.6688 - val_acc: 0.6305 - val_f1_batch: 0.3829 - val_precision_batch: 0.5778 - val_recall_batch: 0.3002
    Epoch 12/30
    300/300 [==============================] - 12s 42ms/step - loss: 0.6008 - acc: 0.7251 - f1_batch: 0.4999 - precision_batch: 0.8600 - recall_batch: 0.3583 - val_loss: 0.6731 - val_acc: 0.6261 - val_f1_batch: 0.3840 - val_precision_batch: 0.5818 - val_recall_batch: 0.2981
    Epoch 13/30
    300/300 [==============================] - 12s 41ms/step - loss: 0.5972 - acc: 0.7317 - f1_batch: 0.5186 - precision_batch: 0.8573 - recall_batch: 0.3774 - val_loss: 0.6693 - val_acc: 0.6345 - val_f1_batch: 0.4087 - val_precision_batch: 0.6039 - val_recall_batch: 0.3244
    Epoch 14/30
    300/300 [==============================] - 12s 41ms/step - loss: 0.5877 - acc: 0.7395 - f1_batch: 0.5317 - precision_batch: 0.8558 - recall_batch: 0.3917 - val_loss: 0.6651 - val_acc: 0.6382 - val_f1_batch: 0.4235 - val_precision_batch: 0.6067 - val_recall_batch: 0.3380
    Epoch 15/30
    300/300 [==============================] - 12s 41ms/step - loss: 0.5899 - acc: 0.7383 - f1_batch: 0.5458 - precision_batch: 0.8525 - recall_batch: 0.4067 - val_loss: 0.6670 - val_acc: 0.6383 - val_f1_batch: 0.4376 - val_precision_batch: 0.5887 - val_recall_batch: 0.3623
    Epoch 16/30
    300/300 [==============================] - 12s 41ms/step - loss: 0.5776 - acc: 0.7509 - f1_batch: 0.5625 - precision_batch: 0.8513 - recall_batch: 0.4254 - val_loss: 0.6674 - val_acc: 0.6399 - val_f1_batch: 0.4325 - val_precision_batch: 0.5920 - val_recall_batch: 0.3525
    Epoch 17/30
    300/300 [==============================] - 12s 41ms/step - loss: 0.5824 - acc: 0.7472 - f1_batch: 0.5733 - precision_batch: 0.8519 - recall_batch: 0.4374 - val_loss: 0.6689 - val_acc: 0.6369 - val_f1_batch: 0.4416 - val_precision_batch: 0.5937 - val_recall_batch: 0.3661
    Epoch 18/30
    300/300 [==============================] - 12s 41ms/step - loss: 0.5716 - acc: 0.7560 - f1_batch: 0.5829 - precision_batch: 0.8485 - recall_batch: 0.4503 - val_loss: 0.6647 - val_acc: 0.6478 - val_f1_batch: 0.4540 - val_precision_batch: 0.6075 - val_recall_batch: 0.3751
    Epoch 19/30
    300/300 [==============================] - 12s 41ms/step - loss: 0.5713 - acc: 0.7583 - f1_batch: 0.5966 - precision_batch: 0.8508 - recall_batch: 0.4646 - val_loss: 0.6618 - val_acc: 0.6470 - val_f1_batch: 0.4739 - val_precision_batch: 0.6002 - val_recall_batch: 0.4057
    Epoch 20/30
    300/300 [==============================] - 12s 41ms/step - loss: 0.5658 - acc: 0.7622 - f1_batch: 0.6020 - precision_batch: 0.8474 - recall_batch: 0.4720 - val_loss: 0.6629 - val_acc: 0.6491 - val_f1_batch: 0.4818 - val_precision_batch: 0.5945 - val_recall_batch: 0.4171
    Epoch 21/30
    300/300 [==============================] - 12s 41ms/step - loss: 0.5647 - acc: 0.7649 - f1_batch: 0.6188 - precision_batch: 0.8500 - recall_batch: 0.4912 - val_loss: 0.6679 - val_acc: 0.6387 - val_f1_batch: 0.4646 - val_precision_batch: 0.5919 - val_recall_batch: 0.3939
    Epoch 22/30
    300/300 [==============================] - 12s 41ms/step - loss: 0.5613 - acc: 0.7667 - f1_batch: 0.6200 - precision_batch: 0.8429 - recall_batch: 0.4954 - val_loss: 0.6640 - val_acc: 0.6458 - val_f1_batch: 0.4857 - val_precision_batch: 0.5909 - val_recall_batch: 0.4248
    Epoch 23/30
    300/300 [==============================] - 12s 41ms/step - loss: 0.5520 - acc: 0.7743 - f1_batch: 0.6305 - precision_batch: 0.8461 - recall_batch: 0.5072 - val_loss: 0.6594 - val_acc: 0.6478 - val_f1_batch: 0.4846 - val_precision_batch: 0.5996 - val_recall_batch: 0.4185
    Epoch 24/30
    300/300 [==============================] - 12s 41ms/step - loss: 0.5566 - acc: 0.7709 - f1_batch: 0.6356 - precision_batch: 0.8446 - recall_batch: 0.5147 - val_loss: 0.6629 - val_acc: 0.6505 - val_f1_batch: 0.5027 - val_precision_batch: 0.5969 - val_recall_batch: 0.4454
    Epoch 25/30
    300/300 [==============================] - 12s 41ms/step - loss: 0.5447 - acc: 0.7791 - f1_batch: 0.6410 - precision_batch: 0.8427 - recall_batch: 0.5219 - val_loss: 0.6629 - val_acc: 0.6489 - val_f1_batch: 0.4971 - val_precision_batch: 0.5881 - val_recall_batch: 0.4423
    Epoch 26/30
    300/300 [==============================] - 12s 41ms/step - loss: 0.5505 - acc: 0.7750 - f1_batch: 0.6464 - precision_batch: 0.8399 - recall_batch: 0.5300 - val_loss: 0.6669 - val_acc: 0.6439 - val_f1_batch: 0.4900 - val_precision_batch: 0.5856 - val_recall_batch: 0.4320
    Epoch 27/30
    300/300 [==============================] - 12s 41ms/step - loss: 0.5418 - acc: 0.7814 - f1_batch: 0.6530 - precision_batch: 0.8410 - recall_batch: 0.5384 - val_loss: 0.6616 - val_acc: 0.6479 - val_f1_batch: 0.4973 - val_precision_batch: 0.5879 - val_recall_batch: 0.4408
    Epoch 28/30
    300/300 [==============================] - 12s 42ms/step - loss: 0.5411 - acc: 0.7823 - f1_batch: 0.6577 - precision_batch: 0.8413 - recall_batch: 0.5446 - val_loss: 0.6573 - val_acc: 0.6580 - val_f1_batch: 0.5242 - val_precision_batch: 0.6066 - val_recall_batch: 0.4755
    Epoch 29/30
    300/300 [==============================] - 13s 42ms/step - loss: 0.5369 - acc: 0.7847 - f1_batch: 0.6624 - precision_batch: 0.8388 - recall_batch: 0.5513 - val_loss: 0.6610 - val_acc: 0.6487 - val_f1_batch: 0.5172 - val_precision_batch: 0.5793 - val_recall_batch: 0.4771
    Epoch 30/30
    300/300 [==============================] - 12s 41ms/step - loss: 0.5376 - acc: 0.7843 - f1_batch: 0.6705 - precision_batch: 0.8359 - recall_batch: 0.5639 - val_loss: 0.6651 - val_acc: 0.6444 - val_f1_batch: 0.4995 - val_precision_batch: 0.5870 - val_recall_batch: 0.4454





    <tensorflow.python.keras.callbacks.History at 0x7f3a1e540f60>



Finally, we evaluate the model's predicted ratings on our test data.



```python
X_test, Y_test = get_data_tensors(df_test)
_ = model.evaluate(X_test, Y_test, steps=30)
```

    30/30 [==============================] - 1s 32ms/step - loss: 0.6568 - acc: 0.6510 - f1_batch: 0.4858 - precision_batch: 0.5550 - recall_batch: 0.4435


Our model has generalized quite well to our test set!
Note that we should additionally measure ranking metrics, like precision@10, before deploying to production.

## Summary

In this tutorial, we showed one way to use Snorkel for recommendations.
We used book metadata and review text to create LFs that estimate user ratings.
We used Snorkel's `LabelModel` to combine the outputs of those LFs.
Finally, we trained a model to predict whether a user will read and like a given book (and therefore what books should be recommended to the user) based only on what books the user has interacted with in the past.

Here we demonstrated one way to use Snorkel for training a recommender system.
Note, however, that this approach could easily be adapted to take advantage of additional information as it is available (e.g., user profile data, denser user ratings, and so on.)
