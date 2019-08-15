---
layout: default
title: Intro to Slicing Functions
description: Monitoring critical data subsets for spam classification
excerpt: Monitoring critical data subsets for spam classification
order: 3
---


# ✂️ Snorkel Intro Tutorial: _Data Slicing_

In real-world applications, some model outcomes are often more important than others — e.g. vulnerable cyclist detections in an autonomous driving task, or, in our running **spam** application, potentially malicious link redirects to external websites.

Traditional machine learning systems optimize for overall quality, which may be too coarse-grained.
Models that achieve high overall performance might produce unacceptable failure rates on critical slices of the data — data subsets that might correspond to vulnerable cyclist detection in an autonomous driving task, or in our running spam detection application, external links to potentially malicious websites.

In this tutorial, we introduce _Slicing Functions (SFs)_ as a programming interface to:
1. **Monitor** application-critical data slices
2. **Improve model performance** on slices

First, we'll set up our notebook for reproducibility and proper logging.


```python
import logging
import os
import pandas as pd
from snorkel.utils import set_seed

# For reproducibility
os.environ["PYTHONHASHSEED"] = "0"
set_seed(111)

# Make sure we're running from the spam/ directory
if os.path.basename(os.getcwd()) == "snorkel-tutorials":
    os.chdir("spam")

# To visualize logs
logger = logging.getLogger()
logger.setLevel(logging.WARNING)

# Show full columns for viewing data
pd.set_option("display.max_colwidth", -1)
```

_Note:_ this tutorial differs from the labeling tutorial in that we use ground truth labels in the train split for demo purposes.
SFs are intended to be used *after the training set has already been labeled* by LFs (or by hand) in the trainind data pipeline.


```python
from utils import load_spam_dataset

df_train, df_valid, df_test = load_spam_dataset(
    load_train_labels=True, include_dev=False
)

df_train.head()
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
      <th>author</th>
      <th>date</th>
      <th>text</th>
      <th>label</th>
      <th>video</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Alessandro leite</td>
      <td>2014-11-05T22:21:36</td>
      <td>pls http://www10.vakinha.com.br/VaquinhaE.aspx?e=313327 help me get vip gun  cross fire al﻿</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Salim Tayara</td>
      <td>2014-11-02T14:33:30</td>
      <td>if your like drones, plz subscribe to Kamal Tayara. He takes videos with  his drone that are absolutely beautiful.﻿</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Phuc Ly</td>
      <td>2014-01-20T15:27:47</td>
      <td>go here to check the views :3﻿</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>DropShotSk8r</td>
      <td>2014-01-19T04:27:18</td>
      <td>Came here to check the views, goodbye.﻿</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>css403</td>
      <td>2014-11-07T14:25:48</td>
      <td>i am 2,126,492,636 viewer :D﻿</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



## 1. Write slicing functions

We leverage *slicing functions* (SFs) — an abstraction that shares syntax with *labeling functions*, which you should already be familiar with.
If not, please see the [intro tutorial](https://github.com/snorkel-team/snorkel-tutorials/blob/master/spam/01_spam_tutorial.ipynb).
A key difference: whereas labeling functions output labels, slicing functions output binary _masks_ indicating whether an example is in the slice or not.

In the following cells, we use the [`@slicing_function()`](https://snorkel.readthedocs.io/en/master/packages/_autosummary/slicing/snorkel.slicing.slicing_function.html#snorkel.slicing.slicing_function) decorator to initialize an SF that identifies shortened links the spam dataset.
These links could redirect us to potentially dangerous websites, and we don't want our users to click them!
To select the subset of shortened links in our dataset, we write a regex that checks for the commonly-used `.ly` extension.

You'll notice that the slicing function is noisily defined — it doesn't represent the ground truth for all short links.
Instead, SFs are often heuristics to quickly measure performance over important subsets of the data.


```python
import re
from snorkel.slicing import slicing_function


@slicing_function()
def short_link(x):
    """Returns whether text matches common pattern for shortened ".ly" links."""
    return bool(re.search(r"\w+\.ly", x.text))


sfs = [short_link]
```

### Visualize slices

With a utility function, [`slice_dataframe`](https://snorkel.readthedocs.io/en/master/packages/_autosummary/slicing/snorkel.slicing.slice_dataframe.html#snorkel.slicing.slice_dataframe), we can visualize examples belonging to this slice in a `pandas.DataFrame`.


```python
from snorkel.slicing import slice_dataframe

short_link_df = slice_dataframe(df_valid, short_link)
short_link_df[["text", "label"]]
```

    100%|██████████| 120/120 [00:00<00:00, 20542.69it/s]





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
      <th>text</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>280</th>
      <td>Being paid to respond to fast paid surveys from home has enabled me to give up working and make more than 4500 bucks monthly.  To read more go to this web site bit.ly\1bSefQe</td>
      <td>1</td>
    </tr>
    <tr>
      <th>192</th>
      <td>Meet The Richest Online Marketer  NOW CLICK : bit.ly/make-money-without-adroid</td>
      <td>1</td>
    </tr>
    <tr>
      <th>301</th>
      <td>coby this USL and past :&lt;br /&gt;&lt;a href="http://adf.ly"&gt;http://adf.ly&lt;/a&gt; /1HmVtX&lt;br /&gt;delete space after y﻿</td>
      <td>1</td>
    </tr>
    <tr>
      <th>350</th>
      <td>adf.ly / KlD3Y</td>
      <td>1</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Earn money for being online with 0 efforts!    bit.ly\14gKvDo</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



## 2. Train a discriminative model

To start, we'll initialize a discriminative model using our [`MultitaskClassifier`](https://snorkel.readthedocs.io/en/master/packages/_autosummary/classification/snorkel.classification.MultitaskClassifier.html#snorkel.classification.MultitaskClassifier).
We'll assume that you are familiar with Snorkel's multitask model — if not, we'd recommend you check out our [Multitask Tutorial](https://github.com/snorkel-team/snorkel-tutorials/blob/master/multitask/multitask_tutorial.ipynb).

In this section, we're ignoring slice information for modeling purposes; slices are used solely for monitoring fine-grained performance.

### Featurize Data

First, we'll featurize the data—as you saw in the introductory Spam tutorial, we can extract simple bag of words features and store them as numpy arrays.


```python
from sklearn.feature_extraction.text import CountVectorizer
from utils import df_to_torch_features

vectorizer = CountVectorizer(ngram_range=(1, 1))
X_train, Y_train = df_to_torch_features(vectorizer, df_train, fit_train=True)
X_valid, Y_valid = df_to_torch_features(vectorizer, df_valid, fit_train=False)
X_test, Y_test = df_to_torch_features(vectorizer, df_test, fit_train=False)
```

### Create DataLoaders

Next, we'll use the extracted Tensors to initialize a [`DictDataLoader`](https://snorkel.readthedocs.io/en/master/packages/_autosummary/classification/snorkel.classification.DictDataLoader.html) — as a quick recap, this is a Snorkel-specific class that inherits from the common PyTorch class and supports multiple data fields in the `X_dict` and labels in the `Y_dict`.

In this task, we'd like to store the bag-of-words `bow_features` in our `X_dict`, and we have one set of labels (for now) correpsonding to the `spam_task`.


```python
from utils import create_dict_dataloader

BATCH_SIZE = 32


dl_train = create_dict_dataloader(
    X_train, Y_train, split="train", batch_size=BATCH_SIZE, shuffle=True
)
dl_valid = create_dict_dataloader(
    X_valid, Y_valid, split="valid", batch_size=BATCH_SIZE, shuffle=False
)
dl_test = create_dict_dataloader(
    X_test, Y_test, split="test", batch_size=BATCH_SIZE, shuffle=False
)
```

We can inspect our datasets to confirm that they have the appropriate fields.


```python
dl_valid.dataset
```




    DictDataset(name=spam_dataset, X_keys=['bow_features'], Y_keys=['spam_task'])



### Define [`MultitaskClassifier`](https://snorkel.readthedocs.io/en/master/packages/_autosummary/classification/snorkel.classification.MultitaskClassifier.html)

We define a simple Multi-Layer Perceptron (MLP) architecture to learn from the `bow_features`.
We do so by initializing a `spam_task` using Snorkel's [`Task`](https://snorkel.readthedocs.io/en/master/packages/_autosummary/classification/snorkel.classification.Task.html) API.

*We note that it's certainly possible to define an MLP in a simple framework (e.g. `sklearn`'s [`MLPClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)), but the multitask API will lend us additional flexibility later in the pipeline!*


```python
from utils import create_spam_task

bow_dim = X_train.shape[1]
spam_task = create_spam_task(bow_dim)
```

We'll initialize a  [`MultitaskClassifier`](https://snorkel.readthedocs.io/en/master/packages/_autosummary/classification/snorkel.classification.MultitaskClassifier.html) with the `spam_task` we've created, initialize a corresponding [`Trainer`](https://snorkel.readthedocs.io/en/master/packages/_autosummary/classification/snorkel.classification.Trainer.html), and `fit` to our dataloaders!


```python
from snorkel.classification import MultitaskClassifier, Trainer

model = MultitaskClassifier([spam_task])
trainer = Trainer(n_epochs=5, lr=1e-4, progress_bar=True)
trainer.fit(model, [dl_train, dl_valid])
```

    Epoch 0:: 100%|██████████| 50/50 [00:06<00:00,  7.08it/s, model/all/train/loss=0.61, model/all/train/lr=0.0001, spam_task/spam_dataset/valid/accuracy=0.9, spam_task/spam_dataset/valid/f1=0.898]
    Epoch 1:: 100%|██████████| 50/50 [00:05<00:00,  7.85it/s, model/all/train/loss=0.416, model/all/train/lr=0.0001, spam_task/spam_dataset/valid/accuracy=0.942, spam_task/spam_dataset/valid/f1=0.933]
    Epoch 2:: 100%|██████████| 50/50 [00:06<00:00,  7.46it/s, model/all/train/loss=0.242, model/all/train/lr=0.0001, spam_task/spam_dataset/valid/accuracy=0.933, spam_task/spam_dataset/valid/f1=0.923]
    Epoch 3:: 100%|██████████| 50/50 [00:06<00:00,  7.09it/s, model/all/train/loss=0.144, model/all/train/lr=0.0001, spam_task/spam_dataset/valid/accuracy=0.925, spam_task/spam_dataset/valid/f1=0.913]
    Epoch 4:: 100%|██████████| 50/50 [00:05<00:00,  8.80it/s, model/all/train/loss=0.0923, model/all/train/lr=0.0001, spam_task/spam_dataset/valid/accuracy=0.917, spam_task/spam_dataset/valid/f1=0.902]


How well does our model do?


```python
model.score([dl_train, dl_valid], as_dataframe=True)
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
      <th>label</th>
      <th>dataset</th>
      <th>split</th>
      <th>metric</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>spam_task</td>
      <td>spam_dataset</td>
      <td>train</td>
      <td>accuracy</td>
      <td>0.990542</td>
    </tr>
    <tr>
      <th>1</th>
      <td>spam_task</td>
      <td>spam_dataset</td>
      <td>train</td>
      <td>f1</td>
      <td>0.990937</td>
    </tr>
    <tr>
      <th>2</th>
      <td>spam_task</td>
      <td>spam_dataset</td>
      <td>valid</td>
      <td>accuracy</td>
      <td>0.916667</td>
    </tr>
    <tr>
      <th>3</th>
      <td>spam_task</td>
      <td>spam_dataset</td>
      <td>valid</td>
      <td>f1</td>
      <td>0.901961</td>
    </tr>
  </tbody>
</table>
</div>



## 3. Perform error analysis

In overall metrics (e.g. `f1`, `accuracy`) our model appears to perform well!
However, we emphasize we might actually be **more interested in performance for application-critical subsets,** or _slices_.

Let's perform an error analysis, using [`get_label_buckets`](https://snorkel.readthedocs.io/en/master/packages/_autosummary/analysis/snorkel.analysis.get_label_buckets.html), to see where our model makes mistakes.
We collect the predictions from the model and visualize examples in specific error buckets.


```python
from snorkel.analysis import get_label_buckets

outputs = model.predict(dl_valid, return_preds=True)
error_buckets = get_label_buckets(
    outputs["golds"]["spam_task"], outputs["preds"]["spam_task"]
)
```

For application purposes, we might care especially about false negatives (i.e. true label was $1$, but model predicted $0$) — for the spam task, this error mode might expose users to external redirects to malware!


```python
df_valid[["text", "label"]].iloc[error_buckets[(1, 0)]].head()
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
      <th>text</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>218</th>
      <td>WOW muslims are really egoistic..... 23% of the World population and not in this video or donating 1 dollar to the poor ones in Africa :( shame on those terrorist muslims</td>
      <td>1</td>
    </tr>
    <tr>
      <th>157</th>
      <td>Fuck it was the best ever 0687119038 nummber of patrik kluivert his son share !﻿</td>
      <td>1</td>
    </tr>
    <tr>
      <th>154</th>
      <td>1 753 682 421 GANGNAM STYLE ^^</td>
      <td>1</td>
    </tr>
    <tr>
      <th>31</th>
      <td>Part 5. Comforter of the afflicted, pray for us Help of Christians, pray for us Queen of Angels, pray for us Queen of Patriarchs, pray for us Queen of Prophets, pray for us Queen of Apostles, pray for us Queen of Martyrs, pray for us Queen of Confessors, pray for us Queen of Virgins, pray for us Queen of all Saints, pray for us Queen conceived without original sin, pray for us Queen of the most holy Rosary, pray for us Queen of the family, pray for us Queen of peace, pray for us</td>
      <td>1</td>
    </tr>
    <tr>
      <th>142</th>
      <td>I WILL NEVER FORGET THIS SONG IN MY LIFE LIKE THIS COMMENT OF YOUR HEARING THIS SONG FOR LIKE A YEAR!!!!!</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



In the next sections, we'll explore how we can programmatically monitor these error modes with built-in helpers from Snorkel.

## 4. Monitor slice performance

In order to monitor performance on our `short_link` slice, we add labels to an existing dataloader.
First, for our $n$ examples and $k$ slices in each split, we apply the SF to our data to create an $n \times k$ matrix. (So far, $k=1$).


```python
from snorkel.slicing import PandasSFApplier

applier = PandasSFApplier(sfs)
S_train = applier.apply(df_train)
S_valid = applier.apply(df_valid)
S_test = applier.apply(df_test)
```

    100%|██████████| 1586/1586 [00:00<00:00, 43046.06it/s]
    100%|██████████| 120/120 [00:00<00:00, 30746.27it/s]
    100%|██████████| 250/250 [00:00<00:00, 37144.03it/s]


Specifically, [`add_slice_labels`](https://snorkel.readthedocs.io/en/master/packages/_autosummary/slicing/snorkel.slicing.add_slice_labels.html#snorkel.slicing.add_slice_labels) will add two sets of labels for each slice:
* `spam_task_slice:{slice_name}_ind`: an indicator label, which corresponds to the outputs of the slicing functions.
These indicate whether each example is in the slice (`label=1`)or not (`label=0`).
* `spam_task_slice:{slice_name}_pred`: a _masked_ set of the original task labels (in this case, labeled `spam_task`) for each slice. Examples that are masked (with `label=-1`) will not contribute to loss or scoring.


```python
from snorkel.slicing import add_slice_labels

slice_names = [sf.name for sf in sfs]
add_slice_labels(dl_train, spam_task, S_train, slice_names)
add_slice_labels(dl_valid, spam_task, S_valid, slice_names)
add_slice_labels(dl_test, spam_task, S_test, slice_names)
```


```python
dl_valid.dataset
```




    DictDataset(name=spam_dataset, X_keys=['bow_features'], Y_keys=['spam_task', 'spam_task_slice:short_link_ind', 'spam_task_slice:short_link_pred', 'spam_task_slice:base_ind', 'spam_task_slice:base_pred'])



With our updated dataloader, we want to evaluate our model on the defined slice.
In the  [`MultitaskClassifier`](https://snorkel.readthedocs.io/en/master/packages/_autosummary/classification/snorkel.classification.MultitaskClassifier.html), we can call [`score`](https://snorkel.readthedocs.io/en/master/packages/_autosummary/classification/snorkel.classification.MultitaskClassifier.html#snorkel.classification.MultitaskClassifier.score) with an additional argument, `remap_labels`, to specify that the slice's prediction labels, `spam_task_slice:short_link_pred`, should be mapped to the `spam_task` for evaluation.


```python
model.score(
    dataloaders=[dl_valid, dl_test],
    remap_labels={"spam_task_slice:short_link_pred": "spam_task"},
    as_dataframe=True,
)
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
      <th>label</th>
      <th>dataset</th>
      <th>split</th>
      <th>metric</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>spam_task</td>
      <td>spam_dataset</td>
      <td>valid</td>
      <td>accuracy</td>
      <td>0.916667</td>
    </tr>
    <tr>
      <th>1</th>
      <td>spam_task</td>
      <td>spam_dataset</td>
      <td>valid</td>
      <td>f1</td>
      <td>0.901961</td>
    </tr>
    <tr>
      <th>2</th>
      <td>spam_task_slice:short_link_pred</td>
      <td>spam_dataset</td>
      <td>valid</td>
      <td>accuracy</td>
      <td>0.400000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>spam_task_slice:short_link_pred</td>
      <td>spam_dataset</td>
      <td>valid</td>
      <td>f1</td>
      <td>0.571429</td>
    </tr>
    <tr>
      <th>4</th>
      <td>spam_task</td>
      <td>spam_dataset</td>
      <td>test</td>
      <td>accuracy</td>
      <td>0.928000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>spam_task</td>
      <td>spam_dataset</td>
      <td>test</td>
      <td>f1</td>
      <td>0.918182</td>
    </tr>
    <tr>
      <th>6</th>
      <td>spam_task_slice:short_link_pred</td>
      <td>spam_dataset</td>
      <td>test</td>
      <td>accuracy</td>
      <td>0.333333</td>
    </tr>
    <tr>
      <th>7</th>
      <td>spam_task_slice:short_link_pred</td>
      <td>spam_dataset</td>
      <td>test</td>
      <td>f1</td>
      <td>0.500000</td>
    </tr>
  </tbody>
</table>
</div>



### Performance monitoring with [`SliceScorer`](https://snorkel.readthedocs.io/en/master/packages/_autosummary/slicing/snorkel.slicing.SliceScorer.html#snorkel.slicing.SliceScorer)

If you're using a model other than [`MultitaskClassifier`](https://snorkel.readthedocs.io/en/master/packages/_autosummary/classification/snorkel.classification.MultitaskClassifier.html#snorkel-classification-multitaskclassifier), you can still evaluate on slices using the more general `SliceScorer` class.

We define a `LogisticRegression` model from sklearn and show how we might visualize these slice-specific scores.


```python
from sklearn.linear_model import LogisticRegression

sklearn_model = LogisticRegression(C=0.001, solver="liblinear")
sklearn_model.fit(X=X_train, y=Y_train)
sklearn_model.score(X_test, Y_test)
```




    0.928



Now, we initialize the [`SliceScorer`](https://snorkel.readthedocs.io/en/master/packages/_autosummary/slicing/snorkel.slicing.SliceScorer.html#snorkel.slicing.SliceScorer) using 1) an existing [`Scorer`](https://snorkel.readthedocs.io/en/master/packages/_autosummary/slicing/snorkel.slicing.SliceScorer.html) and 2) desired `slice_names` to see slice-specific performance.


```python
from snorkel.utils import preds_to_probs

preds_test = sklearn_model.predict(X_test)
probs_test = preds_to_probs(preds_test, 2)
```


```python
from snorkel.analysis import Scorer
from snorkel.slicing import SliceScorer

scorer = Scorer(metrics=["accuracy", "f1"])
slice_scorer = SliceScorer(slice_names)
slice_scorer.score_slices(
    S=S_test, golds=Y_test, preds=preds_test, probs=probs_test, as_dataframe=True
)
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
      <th>f1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>overall</th>
      <td>0.925</td>
    </tr>
    <tr>
      <th>short_link</th>
      <td>0.500</td>
    </tr>
  </tbody>
</table>
</div>



### Write additional slicing functions (SFs)

We'll take inspiration from the labeling tutorial to write additional slicing functions.


```python
from snorkel.slicing import SlicingFunction, slicing_function
from snorkel.preprocess import preprocessor


# Keyword-based SFs
def keyword_lookup(x, keywords):
    return any(word in x.text.lower() for word in keywords)


def make_keyword_sf(keywords):
    return SlicingFunction(
        name=f"keyword_{keywords[0]}",
        f=keyword_lookup,
        resources=dict(keywords=keywords),
    )


keyword_subscribe = make_keyword_sf(keywords=["subscribe"])
keyword_please = make_keyword_sf(keywords=["please", "plz"])


# Regex-based SF
@slicing_function()
def regex_check_out(x):
    return bool(re.search(r"check.*out", x.text, flags=re.I))


# Heuristic-based SF
@slicing_function()
def short_comment(x):
    """Ham comments are often short, such as 'cool video!'"""
    return len(x.text.split()) < 5


# Leverage preprocessor in SF
from textblob import TextBlob


@preprocessor(memoize=True)
def textblob_sentiment(x):
    scores = TextBlob(x.text)
    x.polarity = scores.sentiment.polarity
    return x


@slicing_function(pre=[textblob_sentiment])
def textblob_polarity(x):
    return x.polarity > 0.9
```

Like we saw above, we'd like to visualize examples in the slice.
In this case, most examples with high-polarity sentiments are strong opinions about the video — hence, they are usually relevant to the video, and the corresponding labels are $0$.


```python
polarity_df = slice_dataframe(df_valid, textblob_polarity)
polarity_df[["text", "label"]].head()
```

    100%|██████████| 120/120 [00:00<00:00, 843.20it/s]





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
      <th>text</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>16</th>
      <td>Love this song !!!!!!</td>
      <td>0</td>
    </tr>
    <tr>
      <th>309</th>
      <td>One of the best song of all the time﻿</td>
      <td>0</td>
    </tr>
    <tr>
      <th>164</th>
      <td>She is perfect</td>
      <td>0</td>
    </tr>
    <tr>
      <th>310</th>
      <td>Best world cup offical song﻿</td>
      <td>0</td>
    </tr>
    <tr>
      <th>352</th>
      <td>I remember this :D</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



Like we did above, we can evaluate model performance on _all SFs_ using the `SliceScorer`.


```python
extra_sfs = [
    keyword_subscribe,
    keyword_please,
    regex_check_out,
    short_comment,
    textblob_polarity,
]

sfs = [short_link] + extra_sfs
slice_names = [sf.name for sf in sfs]
```


```python
applier = PandasSFApplier(sfs)
S_test = applier.apply(df_test)

slice_scorer = SliceScorer(slice_names)
slice_scorer.score_slices(
    S=S_test, golds=Y_test, preds=preds_test, probs=probs_test, as_dataframe=True
)
```

    100%|██████████| 250/250 [00:00<00:00, 1017.04it/s]





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
      <th>f1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>overall</th>
      <td>0.925000</td>
    </tr>
    <tr>
      <th>short_link</th>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>keyword_subscribe</th>
      <td>0.971429</td>
    </tr>
    <tr>
      <th>keyword_please</th>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>regex_check_out</th>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>short_comment</th>
      <td>0.666667</td>
    </tr>
    <tr>
      <th>textblob_polarity</th>
      <td>0.727273</td>
    </tr>
  </tbody>
</table>
</div>



## 5. Improve slice performance

In classification tasks, we might attempt to increase slice performance with techniques like _oversampling_ (i.e. with PyTorch's [`WeightedRandomSampler`](https://pytorch.org/docs/stable/data.html#torch.utils.data.WeightedRandomSampler)).
This would shift the training distribution to over-represent certain minority populations.
Intuitively, we'd like to show the model more `short_link` examples so that the representation is better suited to handle them.

A technique like upsampling might work with a small number of slices, but with hundreds or thousands or production slices, it could quickly become intractable to tune upsampling weights per slice.
In the following section, we show a modeling approach that we call _Slice-based Learning,_ which handles numerous slices using with slice-specific representation learning.

### Representation learning with slices

To cope with scale, we will attempt to learn and combine many slice-specific representations with an attention mechanism.
(For details, please see our technical report — coming soon!)

First, we update our existing dataloader with our new slices.


```python
applier = PandasSFApplier(sfs)
S_train = applier.apply(df_train)
S_valid = applier.apply(df_valid)
```

    100%|██████████| 1586/1586 [00:01<00:00, 1005.21it/s]
    100%|██████████| 120/120 [00:00<00:00, 8201.21it/s]



```python
slice_names = [sf.name for sf in sfs]
add_slice_labels(dl_train, spam_task, S_train, slice_names)
add_slice_labels(dl_valid, spam_task, S_valid, slice_names)
add_slice_labels(dl_test, spam_task, S_test, slice_names)
```

Using the helper, [`convert_to_slice_tasks`](https://snorkel.readthedocs.io/en/master/packages/_autosummary/slicing/snorkel.slicing.convert_to_slice_tasks.html), we can convert our original `spam_task` into slice-specific tasks.
These will be used to learn "expert task heads" for each slice, in the style of multi-task learning.
The original `spam_task` now contains the attention mechanism to then combine these slice experts.


```python
from snorkel.slicing import convert_to_slice_tasks

slice_tasks = convert_to_slice_tasks(spam_task, slice_names)
slice_tasks
```




    [Task(name=spam_task_slice:short_link_ind),
     Task(name=spam_task_slice:keyword_subscribe_ind),
     Task(name=spam_task_slice:keyword_please_ind),
     Task(name=spam_task_slice:regex_check_out_ind),
     Task(name=spam_task_slice:short_comment_ind),
     Task(name=spam_task_slice:textblob_polarity_ind),
     Task(name=spam_task_slice:base_ind),
     Task(name=spam_task_slice:short_link_pred),
     Task(name=spam_task_slice:keyword_subscribe_pred),
     Task(name=spam_task_slice:keyword_please_pred),
     Task(name=spam_task_slice:regex_check_out_pred),
     Task(name=spam_task_slice:short_comment_pred),
     Task(name=spam_task_slice:textblob_polarity_pred),
     Task(name=spam_task_slice:base_pred),
     Task(name=spam_task)]




```python
slice_model = MultitaskClassifier(slice_tasks)
```

We train a single model initialized with all slice tasks.
We note that we can monitor slice-specific performance during training — this is a powerful way to track especially critical subsets of the data.

*Note: This model includes more parameters (corresponding to additional slices), and therefore requires more time to train.
We set `num_epochs=1` for demonstration purposes.*


```python
trainer = Trainer(n_epochs=1, lr=1e-4, progress_bar=True)
trainer.fit(slice_model, [dl_train, dl_valid])
```

    Epoch 0::  98%|█████████▊| 49/50 [00:43<00:00,  1.13it/s, model/all/train/loss=0.357, model/all/train/lr=0.0001]/home/ubuntu/snorkel-tutorials/.tox/spam/lib/python3.6/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 due to no predicted samples.
      'precision', 'predicted', average, warn_for)
    Epoch 0:: 100%|██████████| 50/50 [00:44<00:00,  1.07it/s, model/all/train/loss=0.356, model/all/train/lr=0.0001, spam_task/spam_dataset/valid/accuracy=0.942, spam_task/spam_dataset/valid/f1=0.935, spam_task_slice:short_link_ind/spam_dataset/valid/f1=0, spam_task_slice:short_link_pred/spam_dataset/valid/accuracy=1, spam_task_slice:short_link_pred/spam_dataset/valid/f1=1, spam_task_slice:base_ind/spam_dataset/valid/f1=1, spam_task_slice:base_pred/spam_dataset/valid/accuracy=0.933, spam_task_slice:base_pred/spam_dataset/valid/f1=0.926, spam_task_slice:keyword_subscribe_ind/spam_dataset/valid/f1=0, spam_task_slice:keyword_subscribe_pred/spam_dataset/valid/accuracy=1, spam_task_slice:keyword_subscribe_pred/spam_dataset/valid/f1=1, spam_task_slice:keyword_please_ind/spam_dataset/valid/f1=0, spam_task_slice:keyword_please_pred/spam_dataset/valid/accuracy=1, spam_task_slice:keyword_please_pred/spam_dataset/valid/f1=1, spam_task_slice:regex_check_out_ind/spam_dataset/valid/f1=0.762, spam_task_slice:regex_check_out_pred/spam_dataset/valid/accuracy=1, spam_task_slice:regex_check_out_pred/spam_dataset/valid/f1=1, spam_task_slice:short_comment_ind/spam_dataset/valid/f1=0, spam_task_slice:short_comment_pred/spam_dataset/valid/accuracy=0.947, spam_task_slice:short_comment_pred/spam_dataset/valid/f1=0.5, spam_task_slice:textblob_polarity_ind/spam_dataset/valid/f1=0, spam_task_slice:textblob_polarity_pred/spam_dataset/valid/accuracy=1, spam_task_slice:textblob_polarity_pred/spam_dataset/valid/f1=1]


At inference time, the primary task head (`spam_task`) will make all final predictions.
We'd like to evaluate all the slice heads on the original task head.
To do this, we use our `remap_labels` API, as we did earlier.
Note that this time, we map each `ind` head to `None` — it doesn't make sense to evaluate these labels on the base task head.


```python
Y_dict = dl_valid.dataset.Y_dict
eval_mapping = {label: "spam_task" for label in Y_dict.keys() if "pred" in label}
eval_mapping.update({label: None for label in Y_dict.keys() if "ind" in label})
```

*Note: in this toy dataset, we might not see significant gains because our dataset is so small that (i) there are few examples the train split, giving little signal to learn over, and (ii) there are few examples in the test split, making our evaluation metrics very noisy.
For a demonstration of data slicing deployed in state-of-the-art models, please see our [SuperGLUE](https://github.com/HazyResearch/snorkel-superglue/tree/master/tutorials) tutorials.*


```python
slice_model.score([dl_valid, dl_test], remap_labels=eval_mapping, as_dataframe=True)
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
      <th>label</th>
      <th>dataset</th>
      <th>split</th>
      <th>metric</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>spam_task</td>
      <td>spam_dataset</td>
      <td>valid</td>
      <td>accuracy</td>
      <td>0.941667</td>
    </tr>
    <tr>
      <th>1</th>
      <td>spam_task</td>
      <td>spam_dataset</td>
      <td>valid</td>
      <td>f1</td>
      <td>0.934579</td>
    </tr>
    <tr>
      <th>2</th>
      <td>spam_task_slice:short_link_pred</td>
      <td>spam_dataset</td>
      <td>valid</td>
      <td>accuracy</td>
      <td>0.800000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>spam_task_slice:short_link_pred</td>
      <td>spam_dataset</td>
      <td>valid</td>
      <td>f1</td>
      <td>0.888889</td>
    </tr>
    <tr>
      <th>4</th>
      <td>spam_task_slice:base_pred</td>
      <td>spam_dataset</td>
      <td>valid</td>
      <td>accuracy</td>
      <td>0.941667</td>
    </tr>
    <tr>
      <th>5</th>
      <td>spam_task_slice:base_pred</td>
      <td>spam_dataset</td>
      <td>valid</td>
      <td>f1</td>
      <td>0.934579</td>
    </tr>
    <tr>
      <th>6</th>
      <td>spam_task_slice:keyword_subscribe_pred</td>
      <td>spam_dataset</td>
      <td>valid</td>
      <td>accuracy</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>spam_task_slice:keyword_subscribe_pred</td>
      <td>spam_dataset</td>
      <td>valid</td>
      <td>f1</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>spam_task_slice:keyword_please_pred</td>
      <td>spam_dataset</td>
      <td>valid</td>
      <td>accuracy</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>spam_task_slice:keyword_please_pred</td>
      <td>spam_dataset</td>
      <td>valid</td>
      <td>f1</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>spam_task_slice:regex_check_out_pred</td>
      <td>spam_dataset</td>
      <td>valid</td>
      <td>accuracy</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>spam_task_slice:regex_check_out_pred</td>
      <td>spam_dataset</td>
      <td>valid</td>
      <td>f1</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>12</th>
      <td>spam_task_slice:short_comment_pred</td>
      <td>spam_dataset</td>
      <td>valid</td>
      <td>accuracy</td>
      <td>0.947368</td>
    </tr>
    <tr>
      <th>13</th>
      <td>spam_task_slice:short_comment_pred</td>
      <td>spam_dataset</td>
      <td>valid</td>
      <td>f1</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>14</th>
      <td>spam_task_slice:textblob_polarity_pred</td>
      <td>spam_dataset</td>
      <td>valid</td>
      <td>accuracy</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>15</th>
      <td>spam_task_slice:textblob_polarity_pred</td>
      <td>spam_dataset</td>
      <td>valid</td>
      <td>f1</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>16</th>
      <td>spam_task</td>
      <td>spam_dataset</td>
      <td>test</td>
      <td>accuracy</td>
      <td>0.936000</td>
    </tr>
    <tr>
      <th>17</th>
      <td>spam_task</td>
      <td>spam_dataset</td>
      <td>test</td>
      <td>f1</td>
      <td>0.927928</td>
    </tr>
    <tr>
      <th>18</th>
      <td>spam_task_slice:short_link_pred</td>
      <td>spam_dataset</td>
      <td>test</td>
      <td>accuracy</td>
      <td>0.333333</td>
    </tr>
    <tr>
      <th>19</th>
      <td>spam_task_slice:short_link_pred</td>
      <td>spam_dataset</td>
      <td>test</td>
      <td>f1</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>20</th>
      <td>spam_task_slice:base_pred</td>
      <td>spam_dataset</td>
      <td>test</td>
      <td>accuracy</td>
      <td>0.936000</td>
    </tr>
    <tr>
      <th>21</th>
      <td>spam_task_slice:base_pred</td>
      <td>spam_dataset</td>
      <td>test</td>
      <td>f1</td>
      <td>0.927928</td>
    </tr>
    <tr>
      <th>22</th>
      <td>spam_task_slice:keyword_subscribe_pred</td>
      <td>spam_dataset</td>
      <td>test</td>
      <td>accuracy</td>
      <td>0.861111</td>
    </tr>
    <tr>
      <th>23</th>
      <td>spam_task_slice:keyword_subscribe_pred</td>
      <td>spam_dataset</td>
      <td>test</td>
      <td>f1</td>
      <td>0.925373</td>
    </tr>
    <tr>
      <th>24</th>
      <td>spam_task_slice:keyword_please_pred</td>
      <td>spam_dataset</td>
      <td>test</td>
      <td>accuracy</td>
      <td>0.956522</td>
    </tr>
    <tr>
      <th>25</th>
      <td>spam_task_slice:keyword_please_pred</td>
      <td>spam_dataset</td>
      <td>test</td>
      <td>f1</td>
      <td>0.977778</td>
    </tr>
    <tr>
      <th>26</th>
      <td>spam_task_slice:regex_check_out_pred</td>
      <td>spam_dataset</td>
      <td>test</td>
      <td>accuracy</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>27</th>
      <td>spam_task_slice:regex_check_out_pred</td>
      <td>spam_dataset</td>
      <td>test</td>
      <td>f1</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>28</th>
      <td>spam_task_slice:short_comment_pred</td>
      <td>spam_dataset</td>
      <td>test</td>
      <td>accuracy</td>
      <td>0.956522</td>
    </tr>
    <tr>
      <th>29</th>
      <td>spam_task_slice:short_comment_pred</td>
      <td>spam_dataset</td>
      <td>test</td>
      <td>f1</td>
      <td>0.714286</td>
    </tr>
    <tr>
      <th>30</th>
      <td>spam_task_slice:textblob_polarity_pred</td>
      <td>spam_dataset</td>
      <td>test</td>
      <td>accuracy</td>
      <td>0.916667</td>
    </tr>
    <tr>
      <th>31</th>
      <td>spam_task_slice:textblob_polarity_pred</td>
      <td>spam_dataset</td>
      <td>test</td>
      <td>f1</td>
      <td>0.800000</td>
    </tr>
  </tbody>
</table>
</div>



## Recap

This tutorial walked through the process authoring slices, monitoring model performance on specific slices, and improving model performance using slice information.
This programming abstraction provides a mechanism to heuristically identify critical data subsets.
For more technical details about _Slice-based Learning,_ stay tuned — our technical report is coming soon!
