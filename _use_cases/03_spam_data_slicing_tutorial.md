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

_Note:_ this tutorial differs from the labeling tutorial in that we use ground truth labels in the train split for demo purposes.
SFs are intended to be used *after the training set has already been labeled* by LFs (or by hand) in the training data pipeline.


```python
from utils import load_spam_dataset

df_train, df_test = load_spam_dataset(load_train_labels=True)
```

## 1. Write slicing functions

We leverage *slicing functions* (SFs), which output binary _masks_ indicating whether an data point is in the slice or not.
Each slice represents some noisily-defined subset of the data (corresponding to an SF) that we'd like to programmatically monitor.

In the following cells, we use the [`@slicing_function()`](https://snorkel.readthedocs.io/en/master/packages/_autosummary/slicing/snorkel.slicing.slicing_function.html#snorkel.slicing.slicing_function) decorator to initialize an SF that identifies short comments

You'll notice that the `short_comment` SF is a heuristic, like the other programmatic ops we've defined, and may not fully cover the slice of interest.
That's okay — in last section, we'll show how a model can handle this in Snorkel.


```python
import re
from snorkel.slicing import slicing_function


@slicing_function()
def short_comment(x):
    """Ham comments are often short, such as 'cool video!'"""
    return len(x.text.split()) < 5


sfs = [short_comment]
```

### Visualize slices

With a utility function, [`slice_dataframe`](https://snorkel.readthedocs.io/en/master/packages/_autosummary/slicing/snorkel.slicing.slice_dataframe.html#snorkel.slicing.slice_dataframe), we can visualize data points belonging to this slice in a `pandas.DataFrame`.


```python
from snorkel.slicing import slice_dataframe

short_comment_df = slice_dataframe(df_test, short_comment)
```


```python
short_comment_df[["text", "label"]].head()
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
      <th>194</th>
      <td>super music﻿</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>I like shakira..﻿</td>
      <td>0</td>
    </tr>
    <tr>
      <th>110</th>
      <td>subscribe to my feed</td>
      <td>1</td>
    </tr>
    <tr>
      <th>263</th>
      <td>Awesome ﻿</td>
      <td>0</td>
    </tr>
    <tr>
      <th>77</th>
      <td>Nice</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



## 2. Monitor slice performance with [`Scorer.score_slices`](https://snorkel.readthedocs.io/en/master/packages/_autosummary/analysis/snorkel.analysis.Scorer.html#snorkel.analysis.Scorer.score_slices)

In this section, we'll demonstrate how we might monitor slice performance on the `short_comment` slice — this approach is compatible with _any modeling framework_.

### Train a simple classifier
First, we featurize the data — as you saw in the introductory Spam tutorial, we can extract simple bag-of-words features and store them as numpy arrays.


```python
from sklearn.feature_extraction.text import CountVectorizer
from utils import df_to_features

vectorizer = CountVectorizer(ngram_range=(1, 1))
X_train, Y_train = df_to_features(vectorizer, df_train, "train")
X_test, Y_test = df_to_features(vectorizer, df_test, "test")
```

We define a `LogisticRegression` model from `sklearn`.


```python
from sklearn.linear_model import LogisticRegression

sklearn_model = LogisticRegression(C=0.001, solver="liblinear")
sklearn_model.fit(X=X_train, y=Y_train)
```




    LogisticRegression(C=0.001, class_weight=None, dual=False, fit_intercept=True,
                       intercept_scaling=1, l1_ratio=None, max_iter=100,
                       multi_class='auto', n_jobs=None, penalty='l2',
                       random_state=None, solver='liblinear', tol=0.0001, verbose=0,
                       warm_start=False)




```python
from snorkel.utils import preds_to_probs

preds_test = sklearn_model.predict(X_test)
probs_test = preds_to_probs(preds_test, 2)
```


```python
from sklearn.metrics import f1_score

print(f"Test set F1: {100 * f1_score(Y_test, preds_test):.1f}%")
```

    Test set F1: 92.5%


### Store slice metadata in `S`

We apply our list of `sfs` to the data using an SF applier.
For our data format, we leverage the [`PandasSFApplier`](https://snorkel.readthedocs.io/en/master/packages/_autosummary/slicing/snorkel.slicing.PandasSFApplier.html#snorkel.slicing.PandasSFApplier).
The output of the `applier` is an [`np.recarray`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.recarray.html) which stores vectors in named fields indicating whether each of $n$ data points belongs to the corresponding slice.


```python
from snorkel.slicing import PandasSFApplier

applier = PandasSFApplier(sfs)
S_test = applier.apply(df_test)
```

Now, we initialize a [`Scorer`](https://snorkel.readthedocs.io/en/master/packages/_autosummary/analysis/snorkel.analysis.Scorer.html#snorkel.analysis.Scorer) using the desired `metrics`.


```python
from snorkel.analysis import Scorer

scorer = Scorer(metrics=["f1"])
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
      <th>f1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>overall</th>
      <td>0.925000</td>
    </tr>
    <tr>
      <th>short_comment</th>
      <td>0.666667</td>
    </tr>
  </tbody>
</table>
</div>



Despite high overall performance, the `short_comment` slice performs poorly here!

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


keyword_please = make_keyword_sf(keywords=["please", "plz"])


# Regex-based SFs
@slicing_function()
def regex_check_out(x):
    return bool(re.search(r"check.*out", x.text, flags=re.I))


@slicing_function()
def short_link(x):
    """Returns whether text matches common pattern for shortened ".ly" links."""
    return bool(re.search(r"\w+\.ly", x.text))


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

Again, we'd like to visualize data points in a particular slice. This time, we'll inspect the `textblob_polarity` slice.

Most data points with high-polarity sentiments are strong opinions about the video — hence, they are usually relevant to the video, and the corresponding labels are $0$ (not spam).
We might define a slice here for *product and marketing reasons*, it's important to make sure that we don't misclassify very positive comments from good users.


```python
polarity_df = slice_dataframe(df_test, textblob_polarity)
```


```python
polarity_df[["text", "label"]].head()
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
      <th>263</th>
      <td>Awesome ﻿</td>
      <td>0</td>
    </tr>
    <tr>
      <th>240</th>
      <td>Shakira is the best dancer</td>
      <td>0</td>
    </tr>
    <tr>
      <th>261</th>
      <td>OMG LISTEN TO THIS ITS SOO GOOD!! :D﻿</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Shakira is very beautiful</td>
      <td>0</td>
    </tr>
    <tr>
      <th>114</th>
      <td>awesome</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



We can evaluate performance on _all SFs_ using the model-agnostic [`Scorer`](https://snorkel.readthedocs.io/en/master/packages/_autosummary/analysis/snorkel.analysis.Scorer.html#snorkel-analysis-scorer).


```python
extra_sfs = [keyword_please, regex_check_out, short_link, textblob_polarity]

sfs = [short_comment] + extra_sfs
slice_names = [sf.name for sf in sfs]
```

Let's see how the `sklearn` model we learned before performs on these new slices.


```python
applier = PandasSFApplier(sfs)
S_test = applier.apply(df_test)
```


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
      <th>f1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>overall</th>
      <td>0.925000</td>
    </tr>
    <tr>
      <th>short_comment</th>
      <td>0.666667</td>
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
      <th>short_link</th>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>textblob_polarity</th>
      <td>0.727273</td>
    </tr>
  </tbody>
</table>
</div>



Looks like some do extremely well on our small test set, while others do decently.
At the very least, we may want to monitor these to make sure that as we iterate to improve certain slices like `short_comment`, we don't hurt the performance of others.
Next, we'll introduce a model that helps us to do this balancing act automatically.

## 3. Improve slice performance

In the following section, we demonstrate a modeling approach that we call _Slice-based Learning,_ which improves performance by adding extra slice-specific representational capacity to whichever model we're using.
Intuitively, we'd like to model to learn *representations that are better suited to handle data points in this slice*.
In our approach, we model each slice as a separate "expert task" in the style of [multi-task learning](https://github.com/snorkel-team/snorkel-tutorials/blob/master/multitask/multitask_tutorial.ipynb); for further details of how slice-based learning works under the hood, check out the [code](https://github.com/snorkel-team/snorkel/blob/master/snorkel/slicing/utils.py) (with paper coming soon)!

In other approaches, one might attempt to increase slice performance with techniques like _oversampling_ (i.e. with PyTorch's [`WeightedRandomSampler`](https://pytorch.org/docs/stable/data.html#torch.utils.data.WeightedRandomSampler)), effectively shifting the training distribution towards certain populations.

This might work with small number of slices, but with hundreds or thousands or production slices at scale, it could quickly become intractable to tune upsampling weights per slice.

### Constructing a [`SliceAwareClassifier`](https://snorkel.readthedocs.io/en/v0.9.3/packages/_autosummary/slicing/snorkel.slicing.SliceAwareClassifier.html)


To cope with scale, we will attempt to learn and combine many slice-specific representations with an attention mechanism.
(Please see our [Section 3 of our technical report](https://arxiv.org/abs/1909.06349) for details on this approach).

First we'll initialize a [`SliceAwareClassifier`](https://snorkel.readthedocs.io/en/v0.9.3/packages/_autosummary/slicing/snorkel.slicing.SliceAwareClassifier.html):
* `base_architecture`: We define a simple Multi-Layer Perceptron (MLP) in Pytorch to serve as the primary representation architecture. We note that the `BinarySlicingClassifier` is **agnostic to the base architecture** — you might leverage a Transformer model for text, or a ResNet for images.
* `head_dim`: identifies the final output feature dimension of the `base_architecture`
* `slice_names`: Specify the slices that we plan to train on with this classifier.


```python
from snorkel.slicing import SliceAwareClassifier
from utils import get_pytorch_mlp

# Define model architecture
bow_dim = X_train.shape[1]
hidden_dim = bow_dim
mlp = get_pytorch_mlp(hidden_dim=hidden_dim, num_layers=2)

# Initialize slice model
slice_model = SliceAwareClassifier(
    base_architecture=mlp,
    head_dim=hidden_dim,
    slice_names=[sf.name for sf in sfs],
    scorer=scorer,
)
```

Next, we'll generate the remaining `S` matrixes with the new set of slicing functions.


```python
applier = PandasSFApplier(sfs)
S_train = applier.apply(df_train)
S_test = applier.apply(df_test)
```

In order to train using slice information, we'd like to initialize a **slice-aware dataloader**.
To do this, we can use [`slice_model.make_slice_dataloader`](https://snorkel.readthedocs.io/en/v0.9.3/packages/_autosummary/slicing/snorkel.slicing.SliceAwareClassifier.html#snorkel.slicing.SliceAwareClassifier.predict) to add slice labels to an existing dataloader.

Under the hood, this method leverages slice metadata to add slice labels to the appropriate fields such that it will be compatible with our model, a [`SliceAwareClassifier`](https://snorkel.readthedocs.io/en/v0.9.3/packages/_autosummary/slicing/snorkel.slicing.SliceAwareClassifier.html#snorkel-slicing-slicingclassifier).


```python
from utils import create_dict_dataloader

BATCH_SIZE = 64

train_dl = create_dict_dataloader(X_train, Y_train, "train")
train_dl_slice = slice_model.make_slice_dataloader(
    train_dl.dataset, S_train, shuffle=True, batch_size=BATCH_SIZE
)
test_dl = create_dict_dataloader(X_test, Y_test, "train")
test_dl_slice = slice_model.make_slice_dataloader(
    test_dl.dataset, S_test, shuffle=False, batch_size=BATCH_SIZE
)
```

### Representation learning with slices

Using Snorkel's [`Trainer`](https://snorkel.readthedocs.io/en/master/packages/_autosummary/classification/snorkel.classification.Trainer.html), we fit our classifier with the training set dataloader.


```python
from snorkel.classification import Trainer

# For demonstration purposes, we set n_epochs=2
trainer = Trainer(n_epochs=2, lr=1e-4, progress_bar=True)
trainer.fit(slice_model, [train_dl_slice])
```

    Epoch 0:: 100%|██████████| 25/25 [00:24<00:00,  1.01it/s, model/all/train/loss=0.5, model/all/train/lr=0.0001]
    Epoch 1:: 100%|██████████| 25/25 [00:25<00:00,  1.01s/it, model/all/train/loss=0.257, model/all/train/lr=0.0001]


At inference time, the primary task head (`spam_task`) will make all final predictions.
We'd like to evaluate all the slice heads on the original task head — [`score_slices`](https://snorkel.readthedocs.io/en/v0.9.3/packages/_autosummary/slicing/snorkel.slicing.SliceAwareClassifier.html#snorkel.slicing.SliceAwareClassifier.score_slices) remaps all slice-related labels, denoted `spam_task_slice:{slice_name}_pred`, to be evaluated on the `spam_task`.


```python
slice_model.score_slices([test_dl_slice], as_dataframe=True)
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
      <td>train</td>
      <td>f1</td>
      <td>0.941704</td>
    </tr>
    <tr>
      <th>1</th>
      <td>task_slice:short_comment_pred</td>
      <td>SnorkelDataset</td>
      <td>train</td>
      <td>f1</td>
      <td>0.769231</td>
    </tr>
    <tr>
      <th>2</th>
      <td>task_slice:keyword_please_pred</td>
      <td>SnorkelDataset</td>
      <td>train</td>
      <td>f1</td>
      <td>0.977778</td>
    </tr>
    <tr>
      <th>3</th>
      <td>task_slice:regex_check_out_pred</td>
      <td>SnorkelDataset</td>
      <td>train</td>
      <td>f1</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>task_slice:short_link_pred</td>
      <td>SnorkelDataset</td>
      <td>train</td>
      <td>f1</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>task_slice:textblob_polarity_pred</td>
      <td>SnorkelDataset</td>
      <td>train</td>
      <td>f1</td>
      <td>0.800000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>task_slice:base_pred</td>
      <td>SnorkelDataset</td>
      <td>train</td>
      <td>f1</td>
      <td>0.941704</td>
    </tr>
  </tbody>
</table>
</div>



*Note: in this toy dataset, we see high variance in slice performance, because our dataset is so small that (i) there are few data points in the train split, giving little signal to learn over, and (ii) there are few data points in the test split, making our evaluation metrics very noisy.
For a demonstration of data slicing deployed in state-of-the-art models, please see our [SuperGLUE](https://github.com/HazyResearch/snorkel-superglue/tree/master/tutorials) tutorials.*

---
## Recap

This tutorial walked through the process authoring slices, monitoring model performance on specific slices, and improving model performance using slice information.
This programming abstraction provides a mechanism to heuristically identify critical data subsets.
For more technical details about _Slice-based Learning,_ please see our [NeurIPS 2019 paper](https://arxiv.org/abs/1909.06349)!
