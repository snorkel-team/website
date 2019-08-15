---
layout: default
title: Intro to Slicing Functions
description: Monitoring critical data subsets for spam classification
excerpt: Monitoring critical data subsets for spam classification
order: 3
github_link: https://github.com/snorkel-team/snorkel-tutorials/blob/master/spam/03_spam_data_slicing_tutorial.ipynb
---


# ✂️ Snorkel Intro Tutorial: _Data Slicing_

In real-world applications, some model outcomes are often more important than others — e.g. vulnerable cyclist detections in an autonomous driving task, or, in our running **spam** application, potentially malicious link redirects to external websites.

Traditional machine learning systems optimize for overall quality, which may be too coarse-grained.
Models that achieve high overall performance might produce unacceptable failure rates on critical slices of the data — data subsets that might correspond to vulnerable cyclist detection in an autonomous driving task, or in our running spam detection application, external links to potentially malicious websites.

In this tutorial, we:
1. **Introduce _Slicing Functions (SFs)_** as a programming interface
1. **Monitor** application-critical data subsets
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
SFs are intended to be used *after the training set has already been labeled* by LFs (or by hand) in the training data pipeline.


```python
from utils import load_spam_dataset

df_train, df_valid, df_test = load_spam_dataset(load_train_labels=True, split_dev=False)
```

## 1. Write slicing functions

We leverage *slicing functions* (SFs), which output binary _masks_ indicating whether an example is in the slice or not.
Each slice represents some noisily-defined subset of the data (corresponding to an SF) that we'd like to programmatically monitor.

In the following cells, we use the [`@slicing_function()`](https://snorkel.readthedocs.io/en/master/packages/_autosummary/slicing/snorkel.slicing.slicing_function.html#snorkel.slicing.slicing_function) decorator to initialize an SF that identifies shortened links the spam dataset.
These links could redirect us to potentially dangerous websites, and we don't want our users to click them!
To select the subset of shortened links in our dataset, we write a regex that checks for the commonly-used `.ly` extension.

You'll notice that the `short_link` SF is a heuristic, like the other programmatic ops we've defined, and may not fully cover the slice of interest.
That's okay — in last section, we'll show how a model can handle this in Snorkel.


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

    100%|██████████| 120/120 [00:00<00:00, 19190.78it/s]





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



## 2. Monitor slice performance with [`Scorer.score_slices`](https://snorkel.readthedocs.io/en/master/packages/_autosummary/analysis/snorkel.analysis.Scorer.html#snorkel.analysis.Scorer.score_slices)

In this section, we'll demonstrate how we might monitor slice performance on the `short_link` slice — this approach is compatible with _any modeling framework_.

### Train a simple classifier
First, we featurize the data — as you saw in the introductory Spam tutorial, we can extract simple bag-of-words features and store them as numpy arrays.


```python
from sklearn.feature_extraction.text import CountVectorizer
from utils import df_to_features

vectorizer = CountVectorizer(ngram_range=(1, 1))
X_train, Y_train = df_to_features(vectorizer, df_train, "train")
X_valid, Y_valid = df_to_features(vectorizer, df_valid, "valid")
X_test, Y_test = df_to_features(vectorizer, df_test, "test")
```

We define a `LogisticRegression` model from `sklearn` and show how we might visualize these slice-specific scores.


```python
from sklearn.linear_model import LogisticRegression

sklearn_model = LogisticRegression(C=0.001, solver="liblinear")
sklearn_model.fit(X=X_train, y=Y_train)
sklearn_model.score(X_test, Y_test)
```




    0.928




```python
from snorkel.utils import preds_to_probs

preds_test = sklearn_model.predict(X_test)
probs_test = preds_to_probs(preds_test, 2)
```

### Store slice metadata in `S`

We apply our list of `sfs` to the data using an SF applier.
For our data format, we leverage the [`PandasSFApplier`](https://snorkel.readthedocs.io/en/master/packages/_autosummary/slicing/snorkel.slicing.PandasSFApplier.html#snorkel.slicing.PandasSFApplier).
The output of the `applier` is an [`np.recarray`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.recarray.html) which stores vectors in named fields indicating whether each of $n$ examples belongs to the corresponding slice.


```python
from snorkel.slicing import PandasSFApplier

applier = PandasSFApplier(sfs)
S_test = applier.apply(df_test)
```

    100%|██████████| 250/250 [00:00<00:00, 25077.15it/s]


Now, we initialize a [`Scorer`](https://snorkel.readthedocs.io/en/master/packages/_autosummary/analysis/snorkel.analysis.Scorer.html#snorkel.analysis.Scorer) using the desired `metrics`.


```python
from snorkel.analysis import Scorer

scorer = Scorer(metrics=["accuracy", "f1"])
```

Using the [`score_slices`](https://snorkel.readthedocs.io/en/master/packages/_autosummary/analysis/snorkel.analysis.Scorer.html#snorkel.analysis.Scorer.score_slices) method, we can see both `overall` and slice-specific performance.


```python
scorer.score_slices(
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
      <th>accuracy</th>
      <th>f1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>overall</th>
      <td>0.928000</td>
      <td>0.925</td>
    </tr>
    <tr>
      <th>short_link</th>
      <td>0.333333</td>
      <td>0.500</td>
    </tr>
  </tbody>
</table>
</div>



Despite high overall performance, the `short_link` slice performs poorly here!

### Write additional slicing functions (SFs)

Slices are dynamic — as monitoring needs grow or change with new data distributions or application needs, an ML pipeline might require dozens, or even hundreds, of slices.

We'll take inspiration from the labeling tutorial to write additional slicing functions.
We demonstrate how the same powerful preprocessors and utilities available for labeling functions can be leveraged for slicing functions.


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

Again, we'd like to visualize examples in a particular slice. This time, we'll inspect the `textblob_polarity` slice.

Most examples with high-polarity sentiments are strong opinions about the video — hence, they are usually relevant to the video, and the corresponding labels are $0$.
We might define a slice here for *product and marketing reasons*, it's important to make sure that we don't misclassify very positive comments from good users.


```python
polarity_df = slice_dataframe(df_valid, textblob_polarity)
polarity_df[["text", "label"]].head()
```

    100%|██████████| 120/120 [00:00<00:00, 887.05it/s]





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



We can evaluate performance on _all SFs_ using the model-agnostic [`Scorer`](https://snorkel.readthedocs.io/en/master/packages/_autosummary/analysis/snorkel.analysis.Scorer.html#snorkel-analysis-scorer).


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

Let's see how the `sklearn` model we learned before performs on these new slices!


```python
applier = PandasSFApplier(sfs)
S_test = applier.apply(df_test)

scorer.score_slices(
    S=S_test, golds=Y_test, preds=preds_test, probs=probs_test, as_dataframe=True
)
```

    100%|██████████| 250/250 [00:00<00:00, 1100.62it/s]





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
      <th>accuracy</th>
      <th>f1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>overall</th>
      <td>0.928000</td>
      <td>0.925000</td>
    </tr>
    <tr>
      <th>short_link</th>
      <td>0.333333</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>keyword_subscribe</th>
      <td>0.944444</td>
      <td>0.971429</td>
    </tr>
    <tr>
      <th>keyword_please</th>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>regex_check_out</th>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>short_comment</th>
      <td>0.945652</td>
      <td>0.666667</td>
    </tr>
    <tr>
      <th>textblob_polarity</th>
      <td>0.875000</td>
      <td>0.727273</td>
    </tr>
  </tbody>
</table>
</div>



Looks like some do extremely well on our small test set, while others do decently.
At the very least, we may want to monitor these to make sure that as we iterate to improve certain slices like `short_link`, we don't hurt the performance of others.
Next, we'll introduce a model that helps us to do this balancing act automatically!

## 3. Improve slice performance

In the following section, we demonstrate a modeling approach that we call _Slice-based Learning,_ which improves performance by adding extra slice-specific representational capacity to whichever model we're using.
Intuitively, we'd like to model to learn *representations that are better suited to handle examples in this slice*.
In our approach, we model each slice as a separate "expert task" in the style of [multi-task learning](https://github.com/snorkel-team/snorkel-tutorials/blob/master/multitask/multitask_tutorial.ipynb); for further details of how slice-based learning works under the hood, check out the [code](https://github.com/snorkel-team/snorkel/blob/master/snorkel/slicing/utils.py) (with paper coming soon)!

In other approaches, one might attempt to increase slice performance with techniques like _oversampling_ (i.e. with PyTorch's [`WeightedRandomSampler`](https://pytorch.org/docs/stable/data.html#torch.utils.data.WeightedRandomSampler)), effectively shifting the training distribution towards certain populations.

This might work with small number of slices, but with hundreds or thousands or production slices at scale, it could quickly become intractable to tune upsampling weights per slice.

### Set up modeling pipeline with [`SlicingClassifier`](https://snorkel.readthedocs.io/en/master/packages/_autosummary/slicing/snorkel.slicing.SlicingClassifier.html)

Snorkel supports performance monitoring on slices using discriminative models from [`snorkel.slicing`](https://snorkel.readthedocs.io/en/master/packages/slicing.html).
To demonstrate this functionality, we'll first set up a the datasets + modeling pipeline in the PyTorch-based [`snorkel.classification`](https://snorkel.readthedocs.io/en/master/packages/classification.html) package.

First, we initialize a dataloaders for each split.


```python
from utils import create_dict_dataloader

BATCH_SIZE = 64


train_dl = create_dict_dataloader(
    X_train, Y_train, "train", batch_size=BATCH_SIZE, shuffle=True
)
valid_dl = create_dict_dataloader(
    X_valid, Y_valid, "valid", batch_size=BATCH_SIZE, shuffle=False
)
test_dl = create_dict_dataloader(
    X_test, Y_test, "test", batch_size=BATCH_SIZE, shuffle=True
)
```

We'll now initialize a [`SlicingClassifier`](https://snorkel.readthedocs.io/en/master/packages/_autosummary/slicing/snorkel.slicing.SlicingClassifier.html):
* `base_architecture`: We define a simple Multi-Layer Perceptron (MLP) in Pytorch to serve as the primary representation architecture. We note that the `BinarySlicingClassifier` is **agnostic to the base architecture** — you might leverage a Transformer model for text, or a ResNet for images.
* `head_dim`: identifies the final output feature dimension of the `base_architecture`
* `slice_names`: Specify the slices that we plan to train on with this classifier.


```python
from snorkel.slicing import SlicingClassifier
from utils import get_pytorch_mlp


# Define model architecture
bow_dim = X_train.shape[1]
hidden_dim = bow_dim
mlp = get_pytorch_mlp(hidden_dim=hidden_dim, num_layers=2)

# Init slice model
slice_model = SlicingClassifier(
    base_architecture=mlp, head_dim=hidden_dim, slice_names=[sf.name for sf in sfs]
)
```

### Monitor slice performance _during training_

Using Snorkel's [`Trainer`](https://snorkel.readthedocs.io/en/master/packages/_autosummary/classification/snorkel.classification.Trainer.html), we fit to `train_dl`, and validate on `valid_dl`.

We note that we can monitor slice-specific performance during training — this is a powerful way to track especially critical subsets of the data.
If logging in `Tensorboard` (i.e. [`snorkel.classification.TensorboardWritier`](https://snorkel.readthedocs.io/en/master/packages/_autosummary/classification/snorkel.classification.TensorBoardWriter.html)), we would visualize individual loss curves and validation metrics to debug convegence for specific slices.


```python
from snorkel.classification import Trainer

# For demonstration purposes, we set n_epochs=2
trainer = Trainer(lr=1e-4, n_epochs=2)
trainer.fit(slice_model, [train_dl, valid_dl])
```

    Epoch 0:: 100%|██████████| 25/25 [01:10<00:00,  2.84s/it, model/all/train/loss=0.472, model/all/train/lr=0.0001, task/SnorkelDataset/valid/accuracy=0.908, task/SnorkelDataset/valid/f1=0.893]
    Epoch 1:: 100%|██████████| 25/25 [01:10<00:00,  2.90s/it, model/all/train/loss=0.0931, model/all/train/lr=0.0001, task/SnorkelDataset/valid/accuracy=0.933, task/SnorkelDataset/valid/f1=0.926]


### Representation learning with slices

To cope with scale, we will attempt to learn and combine many slice-specific representations with an attention mechanism.
(For details about this approach, please see our technical report — coming soon!)

First, we'll generate the remaining `S` matrixes with the new set of slicing functions.


```python
applier = PandasSFApplier(sfs)
S_train = applier.apply(df_train)
S_valid = applier.apply(df_valid)
```

    100%|██████████| 1586/1586 [00:01<00:00, 1306.66it/s]
    100%|██████████| 120/120 [00:00<00:00, 6503.30it/s]


In order to train using slice information, we'd like to initialize a **slice-aware dataloader**.
To do this, we can use [`slice_model.make_slice_dataloader`](https://snorkel.readthedocs.io/en/master/packages/_autosummary/slicing/snorkel.slicing.SlicingClassifier.html#snorkel.slicing.SlicingClassifier.predict) to add slice labels to an existing dataloader.

Under the hood, this method leverages slice metadata to add slice labels to the appropriate fields such that it's compatible with the initialized [`SliceClassifier`](https://snorkel.readthedocs.io/en/master/packages/_autosummary/slicing/snorkel.slicing.SlicingClassifier.html#snorkel-slicing-slicingclassifier).


```python
train_dl_slice = slice_model.make_slice_dataloader(
    train_dl.dataset, S_train, shuffle=True, batch_size=BATCH_SIZE
)
valid_dl_slice = slice_model.make_slice_dataloader(
    valid_dl.dataset, S_valid, shuffle=False, batch_size=BATCH_SIZE
)
test_dl_slice = slice_model.make_slice_dataloader(
    test_dl.dataset, S_test, shuffle=False, batch_size=BATCH_SIZE
)
```

We train a single model initialized with all slice tasks.


```python
from snorkel.classification import Trainer

# For demonstration purposes, we set n_epochs=2
trainer = Trainer(n_epochs=2, lr=1e-4, progress_bar=True)
trainer.fit(slice_model, [train_dl_slice, valid_dl_slice])
```

    Epoch 0::  96%|█████████▌| 24/25 [01:10<00:03,  3.04s/it, model/all/train/loss=0.376, model/all/train/lr=0.0001]/Users/braden/repos/snorkel-tutorials/.tox/spam/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 due to no predicted samples.
      'precision', 'predicted', average, warn_for)
    Epoch 0:: 100%|██████████| 25/25 [01:12<00:00,  2.99s/it, model/all/train/loss=0.371, model/all/train/lr=0.0001, task/SnorkelDataset/valid/accuracy=0.933, task/SnorkelDataset/valid/f1=0.926, task_slice:short_link_ind/SnorkelDataset/valid/f1=0, task_slice:short_link_pred/SnorkelDataset/valid/accuracy=0.8, task_slice:short_link_pred/SnorkelDataset/valid/f1=0.889, task_slice:keyword_subscribe_ind/SnorkelDataset/valid/f1=0, task_slice:keyword_subscribe_pred/SnorkelDataset/valid/accuracy=1, task_slice:keyword_subscribe_pred/SnorkelDataset/valid/f1=1, task_slice:keyword_please_ind/SnorkelDataset/valid/f1=0, task_slice:keyword_please_pred/SnorkelDataset/valid/accuracy=1, task_slice:keyword_please_pred/SnorkelDataset/valid/f1=1, task_slice:regex_check_out_ind/SnorkelDataset/valid/f1=0.471, task_slice:regex_check_out_pred/SnorkelDataset/valid/accuracy=1, task_slice:regex_check_out_pred/SnorkelDataset/valid/f1=1, task_slice:short_comment_ind/SnorkelDataset/valid/f1=0, task_slice:short_comment_pred/SnorkelDataset/valid/accuracy=0.947, task_slice:short_comment_pred/SnorkelDataset/valid/f1=0.5, task_slice:textblob_polarity_ind/SnorkelDataset/valid/f1=0, task_slice:textblob_polarity_pred/SnorkelDataset/valid/accuracy=1, task_slice:textblob_polarity_pred/SnorkelDataset/valid/f1=1, task_slice:base_ind/SnorkelDataset/valid/f1=1, task_slice:base_pred/SnorkelDataset/valid/accuracy=0.933, task_slice:base_pred/SnorkelDataset/valid/f1=0.926]
    Epoch 1:: 100%|██████████| 25/25 [01:17<00:00,  3.31s/it, model/all/train/loss=0.17, model/all/train/lr=0.0001, task/SnorkelDataset/valid/accuracy=0.925, task/SnorkelDataset/valid/f1=0.914, task_slice:short_link_ind/SnorkelDataset/valid/f1=0, task_slice:short_link_pred/SnorkelDataset/valid/accuracy=0.2, task_slice:short_link_pred/SnorkelDataset/valid/f1=0.333, task_slice:keyword_subscribe_ind/SnorkelDataset/valid/f1=0.333, task_slice:keyword_subscribe_pred/SnorkelDataset/valid/accuracy=1, task_slice:keyword_subscribe_pred/SnorkelDataset/valid/f1=1, task_slice:keyword_please_ind/SnorkelDataset/valid/f1=0.5, task_slice:keyword_please_pred/SnorkelDataset/valid/accuracy=1, task_slice:keyword_please_pred/SnorkelDataset/valid/f1=1, task_slice:regex_check_out_ind/SnorkelDataset/valid/f1=0.791, task_slice:regex_check_out_pred/SnorkelDataset/valid/accuracy=1, task_slice:regex_check_out_pred/SnorkelDataset/valid/f1=1, task_slice:short_comment_ind/SnorkelDataset/valid/f1=0, task_slice:short_comment_pred/SnorkelDataset/valid/accuracy=0.947, task_slice:short_comment_pred/SnorkelDataset/valid/f1=0.5, task_slice:textblob_polarity_ind/SnorkelDataset/valid/f1=0, task_slice:textblob_polarity_pred/SnorkelDataset/valid/accuracy=1, task_slice:textblob_polarity_pred/SnorkelDataset/valid/f1=1, task_slice:base_ind/SnorkelDataset/valid/f1=1, task_slice:base_pred/SnorkelDataset/valid/accuracy=0.908, task_slice:base_pred/SnorkelDataset/valid/f1=0.893]


At inference time, the primary task head (`spam_task`) will make all final predictions.
We'd like to evaluate all the slice heads on the original task head — [`score_slices`](https://snorkel.readthedocs.io/en/master/packages/_autosummary/slicing/snorkel.slicing.SlicingClassifier.html#snorkel.slicing.SlicingClassifier.score_slices) remaps all slice-related labels, denoted `spam_task_slice:{slice_name}_pred`, to be evaluated on the `spam_task`.


```python
slice_model.score_slices([valid_dl_slice, test_dl_slice], as_dataframe=True)
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
      <td>task</td>
      <td>SnorkelDataset</td>
      <td>valid</td>
      <td>accuracy</td>
      <td>0.925000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>task</td>
      <td>SnorkelDataset</td>
      <td>valid</td>
      <td>f1</td>
      <td>0.914286</td>
    </tr>
    <tr>
      <th>2</th>
      <td>task_slice:short_link_pred</td>
      <td>SnorkelDataset</td>
      <td>valid</td>
      <td>accuracy</td>
      <td>0.400000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>task_slice:short_link_pred</td>
      <td>SnorkelDataset</td>
      <td>valid</td>
      <td>f1</td>
      <td>0.571429</td>
    </tr>
    <tr>
      <th>4</th>
      <td>task_slice:keyword_subscribe_pred</td>
      <td>SnorkelDataset</td>
      <td>valid</td>
      <td>accuracy</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>task_slice:keyword_subscribe_pred</td>
      <td>SnorkelDataset</td>
      <td>valid</td>
      <td>f1</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>task_slice:keyword_please_pred</td>
      <td>SnorkelDataset</td>
      <td>valid</td>
      <td>accuracy</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>task_slice:keyword_please_pred</td>
      <td>SnorkelDataset</td>
      <td>valid</td>
      <td>f1</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>task_slice:regex_check_out_pred</td>
      <td>SnorkelDataset</td>
      <td>valid</td>
      <td>accuracy</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>task_slice:regex_check_out_pred</td>
      <td>SnorkelDataset</td>
      <td>valid</td>
      <td>f1</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>task_slice:short_comment_pred</td>
      <td>SnorkelDataset</td>
      <td>valid</td>
      <td>accuracy</td>
      <td>0.947368</td>
    </tr>
    <tr>
      <th>11</th>
      <td>task_slice:short_comment_pred</td>
      <td>SnorkelDataset</td>
      <td>valid</td>
      <td>f1</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>12</th>
      <td>task_slice:textblob_polarity_pred</td>
      <td>SnorkelDataset</td>
      <td>valid</td>
      <td>accuracy</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>13</th>
      <td>task_slice:textblob_polarity_pred</td>
      <td>SnorkelDataset</td>
      <td>valid</td>
      <td>f1</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>14</th>
      <td>task_slice:base_pred</td>
      <td>SnorkelDataset</td>
      <td>valid</td>
      <td>accuracy</td>
      <td>0.925000</td>
    </tr>
    <tr>
      <th>15</th>
      <td>task_slice:base_pred</td>
      <td>SnorkelDataset</td>
      <td>valid</td>
      <td>f1</td>
      <td>0.914286</td>
    </tr>
    <tr>
      <th>16</th>
      <td>task</td>
      <td>SnorkelDataset</td>
      <td>test</td>
      <td>accuracy</td>
      <td>0.932000</td>
    </tr>
    <tr>
      <th>17</th>
      <td>task</td>
      <td>SnorkelDataset</td>
      <td>test</td>
      <td>f1</td>
      <td>0.922374</td>
    </tr>
    <tr>
      <th>18</th>
      <td>task_slice:short_link_pred</td>
      <td>SnorkelDataset</td>
      <td>test</td>
      <td>accuracy</td>
      <td>0.333333</td>
    </tr>
    <tr>
      <th>19</th>
      <td>task_slice:short_link_pred</td>
      <td>SnorkelDataset</td>
      <td>test</td>
      <td>f1</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>20</th>
      <td>task_slice:keyword_subscribe_pred</td>
      <td>SnorkelDataset</td>
      <td>test</td>
      <td>accuracy</td>
      <td>0.861111</td>
    </tr>
    <tr>
      <th>21</th>
      <td>task_slice:keyword_subscribe_pred</td>
      <td>SnorkelDataset</td>
      <td>test</td>
      <td>f1</td>
      <td>0.925373</td>
    </tr>
    <tr>
      <th>22</th>
      <td>task_slice:keyword_please_pred</td>
      <td>SnorkelDataset</td>
      <td>test</td>
      <td>accuracy</td>
      <td>0.956522</td>
    </tr>
    <tr>
      <th>23</th>
      <td>task_slice:keyword_please_pred</td>
      <td>SnorkelDataset</td>
      <td>test</td>
      <td>f1</td>
      <td>0.977778</td>
    </tr>
    <tr>
      <th>24</th>
      <td>task_slice:regex_check_out_pred</td>
      <td>SnorkelDataset</td>
      <td>test</td>
      <td>accuracy</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25</th>
      <td>task_slice:regex_check_out_pred</td>
      <td>SnorkelDataset</td>
      <td>test</td>
      <td>f1</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>26</th>
      <td>task_slice:short_comment_pred</td>
      <td>SnorkelDataset</td>
      <td>test</td>
      <td>accuracy</td>
      <td>0.967391</td>
    </tr>
    <tr>
      <th>27</th>
      <td>task_slice:short_comment_pred</td>
      <td>SnorkelDataset</td>
      <td>test</td>
      <td>f1</td>
      <td>0.769231</td>
    </tr>
    <tr>
      <th>28</th>
      <td>task_slice:textblob_polarity_pred</td>
      <td>SnorkelDataset</td>
      <td>test</td>
      <td>accuracy</td>
      <td>0.916667</td>
    </tr>
    <tr>
      <th>29</th>
      <td>task_slice:textblob_polarity_pred</td>
      <td>SnorkelDataset</td>
      <td>test</td>
      <td>f1</td>
      <td>0.800000</td>
    </tr>
    <tr>
      <th>30</th>
      <td>task_slice:base_pred</td>
      <td>SnorkelDataset</td>
      <td>test</td>
      <td>accuracy</td>
      <td>0.932000</td>
    </tr>
    <tr>
      <th>31</th>
      <td>task_slice:base_pred</td>
      <td>SnorkelDataset</td>
      <td>test</td>
      <td>f1</td>
      <td>0.922374</td>
    </tr>
  </tbody>
</table>
</div>



*Note: in this toy dataset, we see high variance in slice performance, because our dataset is so small that (i) there are few examples the train split, giving little signal to learn over, and (ii) there are few examples in the test split, making our evaluation metrics very noisy.
For a demonstration of data slicing deployed in state-of-the-art models, please see our [SuperGLUE](https://github.com/HazyResearch/snorkel-superglue/tree/master/tutorials) tutorials.*

---
## Recap

This tutorial walked through the process authoring slices, monitoring model performance on specific slices, and improving model performance using slice information.
This programming abstraction provides a mechanism to heuristically identify critical data subsets.
For more technical details about _Slice-based Learning,_ stay tuned — our technical report is coming soon!


```python

```
