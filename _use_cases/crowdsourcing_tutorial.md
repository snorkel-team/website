---
layout: default
title: Hybrid Crowd Labeling Workflows in Snorkel
description: Mixing programmatic and crowdworker labels for sentiment analysis
excerpt: Mixing programmatic and crowdworker labels for sentiment analysis
order: 4
github_link: https://github.com/snorkel-team/snorkel-tutorials/blob/master/crowdsourcing/crowdsourcing_tutorial.ipynb
---


# Crowdsourcing Tutorial

In this tutorial, we'll provide a simple walkthrough of how to use Snorkel in conjunction with crowdsourcing to create a training set for a sentiment analysis task.
We already have crowdsourced labels for about half of the training dataset.
The crowdsourced labels are fairly accurate, but do not cover the entire training dataset, nor are they available for the test set or during inference.
To make up for their lack of training set coverage, we combine crowdsourced labels with heuristic labeling functions to increase the number of training labels we have.
Like most Snorkel labeling pipelines, we'll use the denoised labels to train a deep learning
model which can be applied to new, unseen data to automatically make predictions.

## Dataset Details

In this tutorial, we'll use the [Weather Sentiment](https://data.world/crowdflower/weather-sentiment) dataset from Figure Eight.
Our goal is to train a classifier that can label new tweets as expressing either a positive or negative sentiment.

Crowdworkers were asked to label the sentiment of a particular tweet relating to the weather.
The catch is that 20 crowdworkers graded each tweet, and in many cases crowdworkers assigned conflicting sentiment labels to the same tweet.
This is a common issue when dealing with crowdsourced labeling workloads.

Label options were positive, negative, or one of three other options saying they weren't sure if it was positive or negative; we use only the positive/negative labels.
We've also altered the dataset to reflect a realistic crowdsourcing pipeline where only a subset of our available training set has received crowd labels.

We will treat each crowdworker's labels as coming from a single labeling function (LF).
This will allow us to learn a weight for how much to trust the labels from each crowdworker.
We will also write a few heuristic labeling functions to cover the data points without crowd labels.
Snorkel's ability to build high-quality datasets from multiple noisy labeling signals makes it an ideal framework to approach this problem.

## Loading Crowdsourcing Dataset

We start by loading our data which has 287 data points in total.
We take 50 for our development set and 50 for our test set.
The remaining 187 data points form our training set.
Since the dataset is already small, we skip using a validation set.
Note that this very small dataset is primarily used for demonstration purposes here.
In a real setting, we would expect to have access to many more unlabeled tweets, which could help us to train a higher quality model.


```python
from data import load_data

crowd_labels, df_train, df_dev, df_test = load_data()
Y_dev = df_dev.sentiment.values
Y_test = df_test.sentiment.values
```

## Writing Labeling Functions
Each crowdworker can be thought of as a single labeling function,
as each worker labels a subset of data points,
and may have errors or conflicting labels with other workers / labeling functions.
So we create one labeling function per worker.
We'll simply return the label the worker submitted for a given tweet, and abstain
if they didn't submit a label for it.

### Crowdworker labeling functions


```python
labels_by_annotator = crowd_labels.groupby("worker_id")
worker_dicts = {}
for worker_id in labels_by_annotator.groups:
    worker_df = labels_by_annotator.get_group(worker_id)[["label"]]
    worker_dicts[worker_id] = dict(zip(worker_df.index, worker_df.label))

print("Number of workers:", len(worker_dicts))
```

    Number of workers: 100



```python
from snorkel.labeling import LabelingFunction

ABSTAIN = -1


def worker_lf(x, worker_dict):
    return worker_dict.get(x.tweet_id, ABSTAIN)


def make_worker_lf(worker_id):
    worker_dict = worker_dicts[worker_id]
    name = f"worker_{worker_id}"
    return LabelingFunction(name, f=worker_lf, resources={"worker_dict": worker_dict})


worker_lfs = [make_worker_lf(worker_id) for worker_id in worker_dicts]
```

Let's take a quick look at how well they do on the development set.


```python
from snorkel.labeling import PandasLFApplier

applier = PandasLFApplier(worker_lfs)
L_train = applier.apply(df_train)
L_dev = applier.apply(df_dev)
```

Note that because our dev set is so small and our LFs are relatively sparse, many LFs will appear to have zero coverage.
Fortunately, our label model learns weights for LFs based on their outputs on the training set, which is generally much larger.


```python
from snorkel.labeling import LFAnalysis

LFAnalysis(L_dev, worker_lfs).lf_summary(Y_dev).sample(5)
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
      <th>worker_18804376</th>
      <td>83</td>
      <td>[0, 1]</td>
      <td>0.04</td>
      <td>0.04</td>
      <td>0.04</td>
      <td>2</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>worker_16526185</th>
      <td>59</td>
      <td>[0, 1]</td>
      <td>0.20</td>
      <td>0.20</td>
      <td>0.18</td>
      <td>10</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>worker_10197897</th>
      <td>26</td>
      <td>[0, 1]</td>
      <td>0.06</td>
      <td>0.06</td>
      <td>0.04</td>
      <td>3</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>worker_19028457</th>
      <td>91</td>
      <td>[0, 1]</td>
      <td>0.28</td>
      <td>0.28</td>
      <td>0.26</td>
      <td>5</td>
      <td>9</td>
      <td>0.357143</td>
    </tr>
    <tr>
      <th>worker_19007283</th>
      <td>90</td>
      <td>[0, 1]</td>
      <td>0.16</td>
      <td>0.16</td>
      <td>0.10</td>
      <td>6</td>
      <td>2</td>
      <td>0.750000</td>
    </tr>
  </tbody>
</table>
</div>



So the crowd labels in general are quite good! But how much of our dev and training
sets do they cover?


```python
print(f"Training set coverage: {100 * LFAnalysis(L_train).label_coverage(): 0.1f}%")
print(f"Dev set coverage: {100 * LFAnalysis(L_dev).label_coverage(): 0.1f}%")
```

    Training set coverage:  50.3%
    Dev set coverage:  50.0%


### Additional labeling functions

To improve coverage of the training set, we can mix the crowdworker labeling functions with labeling
functions of other types.
For example, we can use [TextBlob](https://textblob.readthedocs.io/en/dev/index.html), a tool that provides a pretrained sentiment analyzer. We run TextBlob on our tweets and create some simple LFs that threshold its polarity score, similar to what we did in the spam_tutorial.


```python
from snorkel.labeling import labeling_function
from snorkel.preprocess import preprocessor
from textblob import TextBlob


@preprocessor(memoize=True)
def textblob_polarity(x):
    scores = TextBlob(x.tweet_text)
    x.polarity = scores.polarity
    return x


# Label high polarity tweets as positive.
@labeling_function(pre=[textblob_polarity])
def polarity_positive(x):
    return 1 if x.polarity > 0.3 else -1


# Label low polarity tweets as negative.
@labeling_function(pre=[textblob_polarity])
def polarity_negative(x):
    return 0 if x.polarity < -0.25 else -1


# Similar to polarity_negative, but with higher coverage and lower precision.
@labeling_function(pre=[textblob_polarity])
def polarity_negative_2(x):
    return 0 if x.polarity <= 0.3 else -1
```

### Applying labeling functions to the training set


```python
text_lfs = [polarity_positive, polarity_negative, polarity_negative_2]
lfs = text_lfs + worker_lfs

applier = PandasLFApplier(lfs)
L_train = applier.apply(df_train)
L_dev = applier.apply(df_dev)
```


```python
LFAnalysis(L_dev, lfs).lf_summary(Y_dev).head()
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
      <th>polarity_positive</th>
      <td>0</td>
      <td>[1]</td>
      <td>0.30</td>
      <td>0.16</td>
      <td>0.12</td>
      <td>15</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>polarity_negative</th>
      <td>1</td>
      <td>[0]</td>
      <td>0.10</td>
      <td>0.10</td>
      <td>0.04</td>
      <td>5</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>polarity_negative_2</th>
      <td>2</td>
      <td>[0]</td>
      <td>0.70</td>
      <td>0.40</td>
      <td>0.32</td>
      <td>26</td>
      <td>9</td>
      <td>0.742857</td>
    </tr>
    <tr>
      <th>worker_6332651</th>
      <td>3</td>
      <td>[0, 1]</td>
      <td>0.06</td>
      <td>0.06</td>
      <td>0.06</td>
      <td>1</td>
      <td>2</td>
      <td>0.333333</td>
    </tr>
    <tr>
      <th>worker_6336109</th>
      <td>4</td>
      <td>[]</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>



Using the text-based LFs, we've expanded coverage on both our training set
and dev set to 100%.
We'll now take these noisy and conflicting labels, and use the LabelModel
to denoise and combine them.


```python
print(f"Training set coverage: {100 * LFAnalysis(L_train).label_coverage(): 0.1f}%")
print(f"Dev set coverage: {100 * LFAnalysis(L_dev).label_coverage(): 0.1f}%")
```

    Training set coverage:  100.0%
    Dev set coverage:  100.0%


## Train LabelModel And Generate Probabilistic Labels


```python
from snorkel.labeling.model import LabelModel

# Train LabelModel.
label_model = LabelModel(cardinality=2, verbose=True)
label_model.fit(L_train, n_epochs=100, seed=123, log_freq=20, l2=0.1, lr=0.01)
```

As a spot-check for the quality of our LabelModel, we'll score it on the dev set.


```python
from snorkel.analysis import metric_score

preds_dev = label_model.predict(L_dev)

acc = metric_score(Y_dev, preds_dev, probs=None, metric="accuracy")
print(f"LabelModel Accuracy: {acc:.3f}")
```

    LabelModel Accuracy: 0.920


We see that we get very high accuracy on the development set.
This is due to the abundance of high quality crowdworker labels.
**Since we don't have these high quality crowdsourcing labels for the
test set or new incoming data points, we can't use the LabelModel reliably
at inference time.**
In order to run inference on new incoming data points, we need to train a
discriminative model over the tweets themselves.
Let's generate a set of labels for that training set.


```python
preds_train = label_model.predict(L_train)
```

## Use Soft Labels to Train End Model

### Getting features from BERT
Since we have very limited training data, we cannot train a complex model like an LSTM with a lot of parameters.
Instead, we use a pre-trained model, [BERT](https://github.com/google-research/bert), to generate embeddings for each our tweets, and treat the embedding values as features.
This may take 5-10 minutes on a CPU, as the BERT model is very large.


```python
import numpy as np
import torch
from pytorch_transformers import BertModel, BertTokenizer

model = BertModel.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


def encode_text(text):
    input_ids = torch.tensor([tokenizer.encode(text)])
    return model(input_ids)[0].mean(1)[0].detach().numpy()


X_train = np.array(list(df_train.tweet_text.apply(encode_text).values))
X_test = np.array(list(df_test.tweet_text.apply(encode_text).values))
```

### Model on labels
Now, we train a simple logistic regression model on the BERT features, using labels
obtained from our LabelModel.


```python
from sklearn.linear_model import LogisticRegression

sklearn_model = LogisticRegression(solver="liblinear")
sklearn_model.fit(X_train, preds_train)
```


```python
print(f"Accuracy of trained model: {sklearn_model.score(X_test, Y_test)}")
```

    Accuracy of trained model: 0.86


We now have a trained model that can be applied to future data points without requiring crowdsourced labels, and with accuracy not much lower than the `LabelModel` that _does_ have access to crowdsourced labels!

## Summary

In this tutorial, we accomplished the following:
* We demonstrated how to combine crowdsourced labels with other programmatic LFs to improve coverage.
* We used the `LabelModel` to combine inputs from crowdworkers and other LFs to generate high quality probabilistic labels.
* We used our labels to train a classifier for making predictions on new, unseen data points.
