---
layout: default
title: Hybrid Crowd Labeling Workflows in Snorkel
description: Programmatic and crowdworker labels for sentiment analysis
excerpt: Programmatic and crowdworker labels for sentiment analysis
order: 4
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

We start by loading our data which has 287 examples in total.
We take 50 for our development set and 50 for our test set.
The remaining 187 examples form our training set.
Since the dataset is already small, we skip using a validation set.
Note that this very small dataset is primarily used for demonstration purposes here.
In a real setting, we would expect to have access to many more unlabeled tweets, which could help us to train a higher quality model.


```python
import os

# Make sure we're in the right directory
if os.path.basename(os.getcwd()) == "snorkel-tutorials":
    os.chdir("crowdsourcing")
```


```python
from data import load_data

crowd_labels, df_train, df_dev, df_test = load_data()
Y_dev = df_dev.sentiment.values
Y_test = df_test.sentiment.values
```

First, let's take a look at our development set to get a sense of what the tweets look like.
We use the following label convention: 0 = Negative, 1 = Positive.


```python
import pandas as pd

# Don't truncate text fields in the display
pd.set_option("display.max_colwidth", 0)

df_dev.head()
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
      <th>tweet_id</th>
      <th>tweet_text</th>
      <th>sentiment</th>
    </tr>
    <tr>
      <th>tweet_id</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>79197834</th>
      <td>79197834</td>
      <td>@mention not in sunny dover! haha</td>
      <td>1</td>
    </tr>
    <tr>
      <th>80059939</th>
      <td>80059939</td>
      <td>It is literally pissing it down in sideways rain. I have nothing to protect me from this monstrous weather.</td>
      <td>0</td>
    </tr>
    <tr>
      <th>79196441</th>
      <td>79196441</td>
      <td>Dear perfect weather, thanks for the vest lunch hour of all time. (@ Lady Bird Lake Trail w/ 2 others) {link}</td>
      <td>1</td>
    </tr>
    <tr>
      <th>84047300</th>
      <td>84047300</td>
      <td>RT @mention: I can't wait for the storm tonight :)</td>
      <td>1</td>
    </tr>
    <tr>
      <th>83255121</th>
      <td>83255121</td>
      <td>60 degrees. And its almost the end of may. Wisconsin... I hate you.</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



Now let's take a look at the crowd labels.
We'll convert these into labeling functions.


```python
crowd_labels.head()
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
      <th>worker_id</th>
      <th>label</th>
    </tr>
    <tr>
      <th>tweet_id</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>82510997</th>
      <td>18034918</td>
      <td>1</td>
    </tr>
    <tr>
      <th>82510997</th>
      <td>7450342</td>
      <td>1</td>
    </tr>
    <tr>
      <th>82510997</th>
      <td>18465660</td>
      <td>1</td>
    </tr>
    <tr>
      <th>82510997</th>
      <td>17475684</td>
      <td>0</td>
    </tr>
    <tr>
      <th>82510997</th>
      <td>14472526</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



## Writing Labeling Functions
Each crowdworker can be thought of as a single labeling function,
as each worker labels a subset of examples,
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

    100%|██████████| 187/187 [00:00<00:00, 927.50it/s]
    100%|██████████| 50/50 [00:00<00:00, 927.47it/s]


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
      <th>worker_6453108</th>
      <td>11</td>
      <td>[0, 1]</td>
      <td>0.06</td>
      <td>0.06</td>
      <td>0.04</td>
      <td>3</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>worker_18465660</th>
      <td>80</td>
      <td>[0, 1]</td>
      <td>0.06</td>
      <td>0.06</td>
      <td>0.06</td>
      <td>3</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>worker_15847995</th>
      <td>56</td>
      <td>[1]</td>
      <td>0.02</td>
      <td>0.02</td>
      <td>0.00</td>
      <td>1</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>worker_15124755</th>
      <td>53</td>
      <td>[0, 1]</td>
      <td>0.06</td>
      <td>0.06</td>
      <td>0.06</td>
      <td>3</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>worker_17475684</th>
      <td>68</td>
      <td>[0, 1]</td>
      <td>0.30</td>
      <td>0.30</td>
      <td>0.28</td>
      <td>10</td>
      <td>5</td>
      <td>0.666667</td>
    </tr>
  </tbody>
</table>
</div>



So the crowd labels in general are quite good! But how much of our dev and training
sets do they cover?


```python
print(f"Training set coverage: {LFAnalysis(L_train).label_coverage(): 0.3f}")
print(f"Dev set coverage: {LFAnalysis(L_dev).label_coverage(): 0.3f}")
```

    Training set coverage:  0.503
    Dev set coverage:  0.500


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

    100%|██████████| 187/187 [00:00<00:00, 519.91it/s]
    100%|██████████| 50/50 [00:00<00:00, 572.53it/s]



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
print(f"Training set coverage: {LFAnalysis(L_train).label_coverage(): 0.3f}")
print(f"Dev set coverage: {LFAnalysis(L_dev).label_coverage(): 0.3f}")
```

    Training set coverage:  1.000
    Dev set coverage:  1.000


## Train LabelModel And Generate Probabilistic Labels


```python
from snorkel.labeling import LabelModel

# Train LabelModel.
label_model = LabelModel(cardinality=2, verbose=True)
label_model.fit(L_train, n_epochs=100, seed=123, log_freq=20, l2=0.1, lr=0.01)
```

As a spot-check for the quality of our LabelModel, we'll score it on the dev set.


```python
from snorkel.analysis import metric_score

Y_dev_preds = label_model.predict(L_dev)

acc = metric_score(Y_dev, Y_dev_preds, probs=None, metric="accuracy")
print(f"LabelModel Accuracy: {acc:.3f}")
```

    LabelModel Accuracy: 0.920


We see that we get very high accuracy on the development set.
This is due to the abundance of high quality crowdworker labels.
**Since we don't have these high quality crowdsourcing labels for the
test set or new incoming examples, we can't use the LabelModel reliably
at inference time.**
In order to run inference on new incoming examples, we need to train a
discriminative model over the tweets themselves.
Let's generate a set of probabilistic labels for that training set.


```python
Y_train_preds = label_model.predict(L_train)
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


train_vectors = np.array(list(df_train.tweet_text.apply(encode_text).values))
test_vectors = np.array(list(df_test.tweet_text.apply(encode_text).values))
```

### Model on soft labels
Now, we train a simple logistic regression model on the BERT features, using labels
obtained from our LabelModel.


```python
from sklearn.linear_model import LogisticRegression

sklearn_model = LogisticRegression(solver="liblinear")
sklearn_model.fit(train_vectors, Y_train_preds)
```




    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                       intercept_scaling=1, l1_ratio=None, max_iter=100,
                       multi_class='warn', n_jobs=None, penalty='l2',
                       random_state=None, solver='liblinear', tol=0.0001, verbose=0,
                       warm_start=False)




```python
print(f"Accuracy of trained model: {sklearn_model.score(test_vectors, Y_test)}")
```

    Accuracy of trained model: 0.86


We now have a trained model that can be applied to future examples without requiring crowdsourced labels, and with accuracy not much lower than the `LabelModel` that _does_ have access to crowdsourced labels!

## Summary

In this tutorial, we accomplished the following:
* We demonstrated how to combine crowdsourced labels with other programmatic LFs to improve coverage.
* We used the `LabelModel` to combine inputs from crowdworkers and other LFs to generate high quality probabilistic labels.
* We used our probabilistic labels to train a classifier for making predictions on new, unseen examples.
