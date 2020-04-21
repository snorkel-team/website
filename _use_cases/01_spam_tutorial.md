---
layout: default
title: Intro to Labeling Functions
description: Labeling data for spam classification
excerpt: Labeling data for spam classification
order: 1
github_link: https://github.com/snorkel-team/snorkel-tutorials/blob/master/spam/01_spam_tutorial.ipynb
---


# 🚀 Snorkel Intro Tutorial: Data Labeling

In this tutorial, we will walk through the process of using Snorkel to build a training set for classifying YouTube comments as spam or not spam.
The goal of this tutorial is to illustrate the basic components and concepts of Snorkel in a simple way, but also to dive into the actual process of iteratively developing real applications in Snorkel.

**Note that this is a toy dataset that helps highlight the different features of Snorkel. For examples of high-performance, real-world uses of Snorkel, see [our publications list](https://www.snorkel.org/resources/).**

Additionally:
* For an overview of Snorkel, visit [snorkel.org](https://snorkel.org)
* You can also check out the [Snorkel API documentation](https://snorkel.readthedocs.io/)

Our goal is to train a classifier over the comment data that can predict whether a comment is spam or not spam.
We have access to a large amount of *unlabeled data* in the form of YouTube comments with some metadata.
In order to train a classifier, we need to label our data, but doing so by hand for real world applications can often be prohibitively slow and expensive.

In these cases, we can turn to a _weak supervision_ approach, using **_labeling functions (LFs)_** in Snorkel: noisy, programmatic rules and heuristics that assign labels to unlabeled training data.
We'll dive into the Snorkel API and how we write labeling functions later in this tutorial, but as an example,
we can write an LF that labels data points with `"http"` in the comment text as spam since many spam
comments contain links:

```python
from snorkel.labeling import labeling_function

@labeling_function()
def lf_contains_link(x):
    # Return a label of SPAM if "http" in comment text, otherwise ABSTAIN
    return SPAM if "http" in x.text.lower() else ABSTAIN
```

The tutorial is divided into four parts:
1. **Loading Data**: We load a [YouTube comments dataset](http://www.dt.fee.unicamp.br/~tiago//youtubespamcollection/), originally introduced in ["TubeSpam: Comment Spam Filtering on YouTube"](https://ieeexplore.ieee.org/document/7424299/), ICMLA'15 (T.C. Alberto, J.V. Lochter, J.V. Almeida).

2. **Writing Labeling Functions**: We write Python programs that take as input a data point and assign labels (or abstain) using heuristics, pattern matching, and third-party models.

3. **Combining Labeling Function Outputs with the Label Model**: We model the outputs of the labeling functions over the training set using a novel, theoretically-grounded [modeling approach](https://arxiv.org/abs/1605.07723), which estimates the accuracies and correlations of the labeling functions using only their agreements and disagreements, and then uses this to reweight and combine their outputs, which we then use as _probabilistic_ training labels.

4. **Training a Classifier**: We train a classifier that can predict labels for *any* YouTube comment (not just the ones labeled by the labeling functions) using the probabilistic training labels from step 3.

### Task: Spam Detection

We use a [YouTube comments dataset](http://www.dt.fee.unicamp.br/~tiago//youtubespamcollection/) that consists of YouTube comments from 5 videos. The task is to classify each comment as being

* **`HAM`**: comments relevant to the video (even very simple ones), or
* **`SPAM`**: irrelevant (often trying to advertise something) or inappropriate messages

For example, the following comments are `SPAM`:

        "Subscribe to me for free Android games, apps.."

        "Please check out my vidios"

        "Subscribe to me and I'll subscribe back!!!"

and these are `HAM`:

        "3:46 so cute!"

        "This looks so fun and it's a good song"

        "This is a weird video."

### Data Splits in Snorkel

We split our data into two sets:
* **Training Set**: The largest split of the dataset, and the one without any ground truth ("gold") labels.
We will generate labels for these data points with weak supervision.
* **Test Set**: A small, standard held-out blind hand-labeled set for final evaluation of our classifier. This set should only be used for final evaluation, _not_ error analysis.

Note that in more advanced production settings, we will often further split up the available hand-labeled data into a _development split_, for getting ideas to write labeling functions, and a _validation split_ for e.g. checking our performance without looking at test set scores, hyperparameter tuning, etc.  These splits are used in some of the other advanced tutorials, but omitted for simplicity here.

## 1. Loading Data

We load the YouTube comments dataset and create Pandas DataFrame objects for the train and test sets.
DataFrames are extremely popular in Python data analysis workloads, and Snorkel provides native support
for several DataFrame-like data structures, including Pandas, Dask, and PySpark.
For more information on working with Pandas DataFrames, see the [Pandas DataFrame guide](https://pandas.pydata.org/pandas-docs/stable/getting_started/dsintro.html).

Each DataFrame consists of the following fields:
* **`author`**: Username of the comment author
* **`data`**: Date and time the comment was posted
* **`text`**: Raw text content of the comment
* **`label`**: Whether the comment is `SPAM` (1), `HAM` (0), or `UNKNOWN/ABSTAIN` (-1)
* **`video`**: Video the comment is associated with

We start by loading our data.
The `load_spam_dataset()` method downloads the raw CSV files from the internet, divides them into splits, converts them into DataFrames, and shuffles them.
As mentioned above, the dataset contains comments from 5 of the most popular YouTube videos during a period between 2014 and 2015.
* The first four videos' comments are combined to form the `train` set. This set has no gold labels.
* The fifth video is part of the `test` set.


```python
from utils import load_spam_dataset

df_train, df_test = load_spam_dataset()

# We pull out the label vectors for ease of use later
Y_test = df_test.label.values
```

The class distribution varies slightly between `SPAM` and `HAM`, but they're approximately class-balanced.


```python
# For clarity, we define constants to represent the class labels for spam, ham, and abstaining.
ABSTAIN = -1
HAM = 0
SPAM = 1
```

## 2. Writing Labeling Functions (LFs)

### A gentle introduction to LFs

**Labeling functions (LFs) help users encode domain knowledge and other supervision sources programmatically.**

LFs are heuristics that take as input a data point and either assign a label to it (in this case, `HAM` or `SPAM`) or abstain (don't assign any label). Labeling functions can be *noisy*: they don't have perfect accuracy and don't have to label every data point.
Moreover, different labeling functions can overlap (label the same data point) and even conflict (assign different labels to the same data point). This is expected, and we demonstrate how we deal with this later.

Because their only requirement is that they map a data point a label (or abstain), they can wrap a wide variety of forms of supervision. Examples include, but are not limited to:
* *Keyword searches*: looking for specific words in a sentence
* *Pattern matching*: looking for specific syntactical patterns
* *Third-party models*: using an pre-trained model (usually a model for a different task than the one at hand)
* *Distant supervision*: using external knowledge base
* *Crowdworker labels*: treating each crowdworker as a black-box function that assigns labels to subsets of the data

### Recommended practice for LF development

Typical LF development cycles include multiple iterations of ideation, refining, evaluation, and debugging.
A typical cycle consists of the following steps:

1. Look at examples to generate ideas for LFs
1. Write an initial version of an LF
1. Spot check its performance by looking at its output on data points in the training set (or development set if available)
1. Refine and debug to improve coverage or accuracy as necessary

Our goal for LF development is to create a high quality set of training labels for our unlabeled dataset,
not to label everything or directly create a model for inference using the LFs.
The training labels are used to train a separate discriminative model (in this case, one which just uses the comment text) in order to generalize to new, unseen data points.
Using this model, we can make predictions for data points that our LFs don't cover.

We'll walk through the development of two LFs using basic analysis tools in Snorkel, then provide a full set of LFs that we developed for this tutorial.

### a) Exploring the training set for initial ideas

We'll start by looking at 20 random data points from the `train` set to generate some ideas for LFs.


```python
df_train[["author", "text", "video"]].sample(20, random_state=2)
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
      <th>text</th>
      <th>video</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>ambareesh nimkar</td>
      <td>"eye of the tiger" "i am the champion" seems l...</td>
      <td>2</td>
    </tr>
    <tr>
      <th>87</th>
      <td>pratik patel</td>
      <td>mindblowing dance.,.,.superbbb song﻿</td>
      <td>3</td>
    </tr>
    <tr>
      <th>14</th>
      <td>RaMpAgE420</td>
      <td>Check out Berzerk video on my channel ! :D</td>
      <td>4</td>
    </tr>
    <tr>
      <th>80</th>
      <td>Jason Haddad</td>
      <td>Hey, check out my new website!! This site is a...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>104</th>
      <td>austin green</td>
      <td>Eminem is my insperasen and fav﻿</td>
      <td>4</td>
    </tr>
    <tr>
      <th>305</th>
      <td>M.E.S</td>
      <td>hey guys look im aware im spamming and it piss...</td>
      <td>4</td>
    </tr>
    <tr>
      <th>22</th>
      <td>John Monster</td>
      <td>Οh my god ... Roar is the most liked video at ...</td>
      <td>2</td>
    </tr>
    <tr>
      <th>338</th>
      <td>Alanoud Alsaleh</td>
      <td>I started hating Katy Perry after finding out ...</td>
      <td>2</td>
    </tr>
    <tr>
      <th>336</th>
      <td>Leonardo Baptista</td>
      <td>http://www.avaaz.org/po/petition/Youtube_Corpo...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>143</th>
      <td>UKz DoleSnacher</td>
      <td>Remove This video its wank﻿</td>
      <td>1</td>
    </tr>
    <tr>
      <th>163</th>
      <td>Monica Parker</td>
      <td>Check out this video on YouTube:﻿</td>
      <td>3</td>
    </tr>
    <tr>
      <th>129</th>
      <td>b0b1t.48058475</td>
      <td>i rekt ur mum last nite. cuz da haterz were 2 ...</td>
      <td>2</td>
    </tr>
    <tr>
      <th>277</th>
      <td>MeSoHornyMeLuvULongTime</td>
      <td>This video is so racist!!! There are only anim...</td>
      <td>2</td>
    </tr>
    <tr>
      <th>265</th>
      <td>HarveyIsTheBoss</td>
      <td>You gotta say its funny. well not 2 billion wo...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>214</th>
      <td>janez novak</td>
      <td>share and like this page to win a hand signed ...</td>
      <td>4</td>
    </tr>
    <tr>
      <th>76</th>
      <td>Bizzle Sperq</td>
      <td>https://www.facebook.com/nicushorbboy add mee ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>123</th>
      <td>Gaming and Stuff PRO</td>
      <td>Hello! Do you like gaming, art videos, scienti...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>268</th>
      <td>Young IncoVEVO</td>
      <td>Check out my Music Videos! and PLEASE SUBSCRIB...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>433</th>
      <td>Chris Edgar</td>
      <td>Love the way you lie - Driveshaft﻿</td>
      <td>4</td>
    </tr>
    <tr>
      <th>40</th>
      <td>rap classics</td>
      <td>check out my channel for rap and hip hop music</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



One dominant pattern in the comments that look like spam (which we might know from prior domain experience, or from inspection of a few training data points) is the use of the phrase "check out" (e.g. "check out my channel").
Let's start with that.

### b) Writing an LF to identify spammy comments that use the phrase "check out"

Labeling functions in Snorkel are created with the
[`@labeling_function` decorator](https://snorkel.readthedocs.io/en/master/packages/_autosummary/labeling/snorkel.labeling.labeling_function.html).
The [decorator](https://realpython.com/primer-on-python-decorators/) can be applied to _any Python function_ that returns a label for a single data point.

Let's start developing an LF to catch instances of commenters trying to get people to "check out" their channel, video, or website.
We'll start by just looking for the exact string `"check out"` in the text, and see how that compares to looking for just `"check"` in the text.
For the two versions of our rule, we'll write a Python function over a single data point that express it, then add the decorator.


```python
from snorkel.labeling import labeling_function


@labeling_function()
def check(x):
    return SPAM if "check" in x.text.lower() else ABSTAIN


@labeling_function()
def check_out(x):
    return SPAM if "check out" in x.text.lower() else ABSTAIN
```

To apply one or more LFs that we've written to a collection of data points, we use an
[`LFApplier`](https://snorkel.readthedocs.io/en/master/packages/_autosummary/labeling/snorkel.labeling.LFApplier.html).
Because our data points are represented with a Pandas DataFrame in this tutorial, we use the
[`PandasLFApplier`](https://snorkel.readthedocs.io/en/master/packages/_autosummary/labeling/snorkel.labeling.PandasLFApplier.html).
Correspondingly, a single data point `x` that's passed into our LFs will be a [Pandas `Series` object](https://pandas.pydata.org/pandas-docs/stable/reference/series.html).

It's important to note that these LFs will work for any object with an attribute named `text`, not just Pandas objects.
Snorkel has several other appliers for different data point collection types which you can browse in the [API documentation](https://snorkel.readthedocs.io/en/master/packages/labeling.html).

The output of the `apply(...)` method is a ***label matrix***, a fundamental concept in Snorkel.
It's a NumPy array `L` with one column for each LF and one row for each data point, where `L[i, j]` is the label that the `j`th labeling function output for the `i`th data point.
We'll create a label matrix for the `train` set.


```python
from snorkel.labeling import PandasLFApplier

lfs = [check_out, check]

applier = PandasLFApplier(lfs=lfs)
L_train = applier.apply(df=df_train)
```


```python
L_train
```




    array([[-1, -1],
           [-1, -1],
           [-1,  1],
           ...,
           [ 1,  1],
           [-1,  1],
           [ 1,  1]])



### c) Evaluate performance on training set

We can easily calculate the coverage of these LFs (i.e., the percentage of the dataset that they label) as follows:


```python
coverage_check_out, coverage_check = (L_train != ABSTAIN).mean(axis=0)
print(f"check_out coverage: {coverage_check_out * 100:.1f}%")
print(f"check coverage: {coverage_check * 100:.1f}%")
```

    check_out coverage: 21.4%
    check coverage: 25.8%


Lots of statistics about labeling functions &mdash; like coverage &mdash; are useful when building any Snorkel application.
So Snorkel provides tooling for common LF analyses using the
[`LFAnalysis` utility](https://snorkel.readthedocs.io/en/master/packages/_autosummary/labeling/snorkel.labeling.LFAnalysis.html).
We report the following summary statistics for multiple LFs at once:

* **Polarity**: The set of unique labels this LF outputs (excluding abstains)
* **Coverage**: The fraction of the dataset the LF labels
* **Overlaps**: The fraction of the dataset where this LF and at least one other LF label
* **Conflicts**: The fraction of the dataset where this LF and at least one other LF label and disagree
* **Correct**: The number of data points this LF labels correctly (if gold labels are provided)
* **Incorrect**: The number of data points this LF labels incorrectly (if gold labels are provided)
* **Empirical Accuracy**: The empirical accuracy of this LF (if gold labels are provided)

For *Correct*, *Incorrect*, and *Empirical Accuracy*, we don't want to penalize the LF for data points where it abstained.
We calculate these statistics only over those data points where the LF output a label.
**Note that in our current setup, we can't compute these statistics because we don't have any ground-truth labels (other than in the test set, which we cannot look at). Not to worry—Snorkel's `LabelModel` will estimate them without needing any ground-truth labels in the next step!**


```python
from snorkel.labeling import LFAnalysis

LFAnalysis(L=L_train, lfs=lfs).lf_summary()
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>check_out</th>
      <td>0</td>
      <td>[1]</td>
      <td>0.214376</td>
      <td>0.214376</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>check</th>
      <td>1</td>
      <td>[1]</td>
      <td>0.257881</td>
      <td>0.214376</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



We might want to pick the `check` rule, since `check` has higher coverage. Let's take a look at 10 random `train` set data points where `check` labeled `SPAM` to see if it matches our intuition or if we can identify some false positives.


```python
df_train.iloc[L_train[:, 1] == SPAM].sample(10, random_state=1)
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
      <th>305</th>
      <td>M.E.S</td>
      <td>NaN</td>
      <td>hey guys look im aware im spamming and it piss...</td>
      <td>-1.0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>265</th>
      <td>Kawiana Lewis</td>
      <td>2015-02-27T02:20:40.987000</td>
      <td>Check out this video on YouTube:opponents mm &lt;...</td>
      <td>-1.0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>89</th>
      <td>Stricker Stric</td>
      <td>NaN</td>
      <td>eminem new song check out my videos</td>
      <td>-1.0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>147</th>
      <td>TheGenieBoy</td>
      <td>NaN</td>
      <td>check out fantasy music    right here -------&amp;...</td>
      <td>-1.0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>240</th>
      <td>Made2Falter</td>
      <td>2014-09-09T23:55:30</td>
      <td>Check out our vids, our songs are awesome! And...</td>
      <td>-1.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>273</th>
      <td>Artady</td>
      <td>2014-08-11T16:27:55</td>
      <td>https://soundcloud.com/artady please check my ...</td>
      <td>-1.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>94</th>
      <td>Nick McGoldrick</td>
      <td>2014-10-27T13:19:06</td>
      <td>Check out my drum cover of E.T. here! thanks -...</td>
      <td>-1.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>139</th>
      <td>MFkin PRXPHETZ</td>
      <td>2014-01-20T09:08:39</td>
      <td>if you like raw talent, raw lyrics, straight r...</td>
      <td>-1.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>303</th>
      <td>이 정훈</td>
      <td>NaN</td>
      <td>This great Warning will happen soon. ,0\nLneaD...</td>
      <td>-1.0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>246</th>
      <td>media.uploader</td>
      <td>NaN</td>
      <td>Check out my channel to see Rihanna short mix ...</td>
      <td>-1.0</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



No clear false positives here, but many look like they could be labeled by `check_out` as well.

Let's see 10 data points where `check_out` abstained, but `check` labeled. We can use the`get_label_buckets(...)` to group data points by their predicted label and/or true labels.


```python
from snorkel.analysis import get_label_buckets

buckets = get_label_buckets(L_train[:, 0], L_train[:, 1])
df_train.iloc[buckets[(ABSTAIN, SPAM)]].sample(10, random_state=1)
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
      <th>403</th>
      <td>ownpear902</td>
      <td>2014-07-22T18:44:36.299000</td>
      <td>check it out free stuff for watching videos an...</td>
      <td>-1.0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>256</th>
      <td>PacKmaN</td>
      <td>2014-11-05T21:56:39</td>
      <td>check men out i put allot of effort into my mu...</td>
      <td>-1.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>196</th>
      <td>Angek95</td>
      <td>2014-11-03T22:28:56</td>
      <td>Check my channel, please!﻿</td>
      <td>-1.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>282</th>
      <td>CronicleFPS</td>
      <td>2014-11-06T03:10:26</td>
      <td>Check me out I'm all about gaming ﻿</td>
      <td>-1.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>352</th>
      <td>MrJtill0317</td>
      <td>NaN</td>
      <td>┏━━━┓┏┓╋┏┓┏━━━┓┏━━━┓┏┓╋╋┏┓  ┃┏━┓┃┃┃╋┃┃┃┏━┓┃┗┓┏...</td>
      <td>-1.0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>161</th>
      <td>MarianMusicChannel</td>
      <td>2014-08-24T03:57:52</td>
      <td>Hello! I'm Marian, I'm a singer from Venezuela...</td>
      <td>-1.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>270</th>
      <td>Kyle Jaber</td>
      <td>2014-01-19T00:21:29</td>
      <td>Check me out! I'm kyle. I rap so yeah ﻿</td>
      <td>-1.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>292</th>
      <td>Soundhase</td>
      <td>2014-08-19T18:59:38</td>
      <td>Hi Guys! check this awesome EDM &amp;amp; House mi...</td>
      <td>-1.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>179</th>
      <td>Nerdy Peach</td>
      <td>2014-10-29T22:44:41</td>
      <td>Hey! I'm NERDY PEACH and I'm a new youtuber an...</td>
      <td>-1.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>16</th>
      <td>zhichao wang</td>
      <td>2013-11-29T02:13:56</td>
      <td>i think about 100 millions of the views come f...</td>
      <td>-1.0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



Most of these seem like small modifications of "check out", like "check me out" or "check it out".
Can we get the best of both worlds?

### d) Balance accuracy and coverage

Let's see if we can use regular expressions to account for modifications of "check out" and get the coverage of `check` plus the accuracy of `check_out`.


```python
import re


@labeling_function()
def regex_check_out(x):
    return SPAM if re.search(r"check.*out", x.text, flags=re.I) else ABSTAIN
```

Again, let's generate our label matrices and see how we do.


```python
lfs = [check_out, check, regex_check_out]

applier = PandasLFApplier(lfs=lfs)
L_train = applier.apply(df=df_train)
```


```python
LFAnalysis(L=L_train, lfs=lfs).lf_summary()
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>check_out</th>
      <td>0</td>
      <td>[1]</td>
      <td>0.214376</td>
      <td>0.214376</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>check</th>
      <td>1</td>
      <td>[1]</td>
      <td>0.257881</td>
      <td>0.233922</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>regex_check_out</th>
      <td>2</td>
      <td>[1]</td>
      <td>0.233922</td>
      <td>0.233922</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



We've split the difference in `train` set coverage—this looks promising!
Let's verify that we corrected our false positive from before.

To understand the coverage difference between `check` and `regex_check_out`, let's take a look at 10 data points from the `train` set.
Remember: coverage isn't always good.
Adding false positives will increase coverage.


```python
buckets = get_label_buckets(L_train[:, 1], L_train[:, 2])
df_train.iloc[buckets[(SPAM, ABSTAIN)]].sample(10, random_state=1)
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
      <th>16</th>
      <td>zhichao wang</td>
      <td>2013-11-29T02:13:56</td>
      <td>i think about 100 millions of the views come f...</td>
      <td>-1.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>99</th>
      <td>Santeri Saariokari</td>
      <td>2014-09-03T16:32:59</td>
      <td>Hey guys go to check my video name "growtopia ...</td>
      <td>-1.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>21</th>
      <td>BeBe Burkey</td>
      <td>2013-11-28T16:30:13</td>
      <td>and u should.d check my channel and tell me wh...</td>
      <td>-1.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>239</th>
      <td>Cony</td>
      <td>2013-11-28T16:01:47</td>
      <td>You should check my channel for Funny VIDEOS!!﻿</td>
      <td>-1.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>288</th>
      <td>Kochos</td>
      <td>2014-01-20T17:08:37</td>
      <td>i check back often to help reach 2x10^9 views ...</td>
      <td>-1.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>65</th>
      <td>by.Ovskiy</td>
      <td>2014-10-13T17:09:46</td>
      <td>Rap from Belarus, check my channel:)﻿</td>
      <td>-1.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>196</th>
      <td>Angek95</td>
      <td>2014-11-03T22:28:56</td>
      <td>Check my channel, please!﻿</td>
      <td>-1.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>333</th>
      <td>FreexGaming</td>
      <td>2014-10-18T08:12:26</td>
      <td>want to win borderlands the pre-sequel? check ...</td>
      <td>-1.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>167</th>
      <td>Brandon Pryor</td>
      <td>2014-01-19T00:36:25</td>
      <td>I dont even watch it anymore i just come here ...</td>
      <td>-1.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>266</th>
      <td>Zielimeek21</td>
      <td>2013-11-28T21:49:00</td>
      <td>I'm only checking the views﻿</td>
      <td>-1.0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



Most of these are SPAM, but a good number are false positives.
**To keep precision high (while not sacrificing much in terms of coverage), we'd choose our regex-based rule.**

### e) Writing an LF that uses a third-party model

The LF interface is extremely flexible, and can wrap existing models.
A common technique is to use a commodity model trained for other tasks that are related to, but not the same as, the one we care about.

For example, the [TextBlob](https://textblob.readthedocs.io/en/dev/index.html) tool provides a pretrained sentiment analyzer. Our spam classification task is not the same as sentiment classification, but we may believe that `SPAM` and `HAM` comments have different distributions of sentiment scores.
We'll focus on writing LFs for `HAM`, since we identified `SPAM` comments above.

**A brief intro to `Preprocessor`s**

A [Snorkel `Preprocessor`](https://snorkel.readthedocs.io/en/master/packages/_autosummary/preprocess/snorkel.preprocess.Preprocessor.html#snorkel.preprocess.Preprocessor)
is constructed from a black-box Python function that maps a data point to a new data point.
`LabelingFunction`s can use `Preprocessor`s, which lets us write LFs over transformed or enhanced data points.
We add the [`@preprocessor(...)` decorator](https://snorkel.readthedocs.io/en/master/packages/_autosummary/preprocess/snorkel.preprocess.preprocessor.html)
to preprocessing functions to create `Preprocessor`s.
`Preprocessor`s also have extra functionality, such as memoization
(i.e. input/output caching, so it doesn't re-execute for each LF that uses it).

We'll start by creating a `Preprocessor` that runs `TextBlob` on our comments, then extracts the polarity and subjectivity scores.


```python
from snorkel.preprocess import preprocessor
from textblob import TextBlob


@preprocessor(memoize=True)
def textblob_sentiment(x):
    scores = TextBlob(x.text)
    x.polarity = scores.sentiment.polarity
    x.subjectivity = scores.sentiment.subjectivity
    return x
```

We can now pick a reasonable threshold and write a corresponding labeling function (note that it doesn't have to be perfect as the `LabelModel` will soon help us estimate each labeling function's accuracy and reweight their outputs accordingly):


```python
@labeling_function(pre=[textblob_sentiment])
def textblob_polarity(x):
    return HAM if x.polarity > 0.9 else ABSTAIN
```

Let's do the same for the subjectivity scores.
This will run faster than the last cell, since we memoized the `Preprocessor` outputs.


```python
@labeling_function(pre=[textblob_sentiment])
def textblob_subjectivity(x):
    return HAM if x.subjectivity >= 0.5 else ABSTAIN
```

Let's apply our LFs so we can analyze their performance.


```python
lfs = [textblob_polarity, textblob_subjectivity]

applier = PandasLFApplier(lfs)
L_train = applier.apply(df_train)
```


```python
LFAnalysis(L_train, lfs).lf_summary()
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>textblob_polarity</th>
      <td>0</td>
      <td>[0]</td>
      <td>0.035309</td>
      <td>0.013871</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>textblob_subjectivity</th>
      <td>1</td>
      <td>[0]</td>
      <td>0.357503</td>
      <td>0.013871</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



**Again, these LFs aren't perfect—note that the `textblob_subjectivity` LF has fairly high coverage and could have a high rate of false positives. We'll rely on Snorkel's `LabelModel` to estimate the labeling function accuracies and reweight and combine their outputs accordingly.**

## 3. Writing More Labeling Functions

If a single LF had high enough coverage to label our entire test dataset accurately, then we wouldn't need a classifier at all.
We could just use that single simple heuristic to complete the task.
But most problems are not that simple.
Instead, we usually need to **combine multiple LFs** to label our dataset, both to increase the size of the generated training set (since we can't generate training labels for data points that no LF voted on) and to improve the overall accuracy of the training labels we generate by factoring in multiple different signals.

In the following sections, we'll show just a few of the many types of LFs that you could write to generate a training dataset for this problem.

### a) Keyword LFs

For text applications, some of the simplest LFs to write are often just keyword lookups.
These will often follow the same execution pattern, so we can create a template and use the `resources` parameter to pass in LF-specific keywords.
Similar to the [`labeling_function` decorator](https://snorkel.readthedocs.io/en/master/packages/_autosummary/labeling/snorkel.labeling.labeling_function.html#snorkel.labeling.labeling_function),
the [`LabelingFunction` class](https://snorkel.readthedocs.io/en/master/packages/_autosummary/labeling/snorkel.labeling.LabelingFunction.html#snorkel.labeling.LabelingFunction)
wraps a Python function (the `f` parameter), and we can use the `resources` parameter to pass in keyword arguments (here, our keywords to lookup) to said function.


```python
from snorkel.labeling import LabelingFunction


def keyword_lookup(x, keywords, label):
    if any(word in x.text.lower() for word in keywords):
        return label
    return ABSTAIN


def make_keyword_lf(keywords, label=SPAM):
    return LabelingFunction(
        name=f"keyword_{keywords[0]}",
        f=keyword_lookup,
        resources=dict(keywords=keywords, label=label),
    )


"""Spam comments talk about 'my channel', 'my video', etc."""
keyword_my = make_keyword_lf(keywords=["my"])

"""Spam comments ask users to subscribe to their channels."""
keyword_subscribe = make_keyword_lf(keywords=["subscribe"])

"""Spam comments post links to other channels."""
keyword_link = make_keyword_lf(keywords=["http"])

"""Spam comments make requests rather than commenting."""
keyword_please = make_keyword_lf(keywords=["please", "plz"])

"""Ham comments actually talk about the video's content."""
keyword_song = make_keyword_lf(keywords=["song"], label=HAM)
```

### b) Pattern-matching LFs (regular expressions)

If we want a little more control over a keyword search, we can look for regular expressions instead.
The LF we developed above (`regex_check_out`) is an example of this.

### c)  Heuristic LFs

There may other heuristics or "rules of thumb" that you come up with as you look at the data.
So long as you can express it in a function, it's a viable LF!


```python
@labeling_function()
def short_comment(x):
    """Ham comments are often short, such as 'cool video!'"""
    return HAM if len(x.text.split()) < 5 else ABSTAIN
```

### d) LFs with Complex Preprocessors

Some LFs rely on fields that aren't present in the raw data, but can be derived from it.
We can enrich our data (providing more fields for the LFs to refer to) using `Preprocessor`s.

For example, we can use the fantastic NLP (natural language processing) tool [spaCy](https://spacy.io/) to add lemmas, part-of-speech (pos) tags, etc. to each token.
Snorkel provides a prebuilt preprocessor for spaCy called `SpacyPreprocessor` which adds a new field to the
data point containing a [spaCy `Doc` object](https://spacy.io/api/doc).
For more info, see the [`SpacyPreprocessor` documentation](https://snorkel.readthedocs.io/en/master/packages/_autosummary/preprocess/snorkel.preprocess.nlp.SpacyPreprocessor.html#snorkel.preprocess.nlp.SpacyPreprocessor).


If you prefer to use a different NLP tool, you can also wrap that as a `Preprocessor` and use it in the same way.
For more info, see the [`preprocessor` documentation](https://snorkel.readthedocs.io/en/master/packages/_autosummary/preprocess/snorkel.preprocess.preprocessor.html#snorkel.preprocess.preprocessor).


```python
from snorkel.preprocess.nlp import SpacyPreprocessor

# The SpacyPreprocessor parses the text in text_field and
# stores the new enriched representation in doc_field
spacy = SpacyPreprocessor(text_field="text", doc_field="doc", memoize=True)
```


```python
@labeling_function(pre=[spacy])
def has_person(x):
    """Ham comments mention specific people and are short."""
    if len(x.doc) < 20 and any([ent.label_ == "PERSON" for ent in x.doc.ents]):
        return HAM
    else:
        return ABSTAIN
```

Because spaCy is such a common preprocessor for NLP applications, we also provide a
[prebuilt `labeling_function`-like decorator that uses spaCy](https://snorkel.readthedocs.io/en/master/packages/_autosummary/labeling/snorkel.labeling.lf.nlp.nlp_labeling_function.html#snorkel.labeling.lf.nlp.nlp_labeling_function).
This resulting LF is identical to the one defined manually above.


```python
from snorkel.labeling.lf.nlp import nlp_labeling_function


@nlp_labeling_function()
def has_person_nlp(x):
    """Ham comments mention specific people and are short."""
    if len(x.doc) < 20 and any([ent.label_ == "PERSON" for ent in x.doc.ents]):
        return HAM
    else:
        return ABSTAIN
```

**Adding new domain-specific preprocessors and LF types is a great way to contribute to Snorkel!
If you have an idea, feel free to reach out to the maintainers or submit a PR!**

### e) Third-party Model LFs

We can also utilize other models, including ones trained for other tasks that are related to, but not the same as, the one we care about.
The TextBlob-based LFs we created above are great examples of this!

## 4. Combining Labeling Function Outputs with the Label Model

This tutorial demonstrates just a handful of the types of LFs that one might write for this task.
One of the key goals of Snorkel is _not_ to replace the effort, creativity, and subject matter expertise required to come up with these labeling functions, but rather to make it faster to write them, since **in Snorkel the labeling functions are assumed to be noisy, i.e. innaccurate, overlapping, etc.**
Said another way: the LF abstraction provides a flexible interface for conveying a huge variety of supervision signals, and the `LabelModel` is able to denoise these signals, reducing the need for painstaking manual fine-tuning.


```python
lfs = [
    keyword_my,
    keyword_subscribe,
    keyword_link,
    keyword_please,
    keyword_song,
    regex_check_out,
    short_comment,
    has_person_nlp,
    textblob_polarity,
    textblob_subjectivity,
]
```

With our full set of LFs, we can now apply these once again with `LFApplier` to get the label matrices.
The Pandas format provides an easy interface that many practitioners are familiar with, but it is also less optimized for scale.
For larger datasets, more compute-intensive LFs, or larger LF sets, you may decide to use one of the other data formats
that Snorkel supports natively, such as Dask DataFrames or PySpark DataFrames, and their corresponding applier objects.
For more info, check out the [Snorkel API documentation](https://snorkel.readthedocs.io/en/master/packages/labeling.html).


```python
applier = PandasLFApplier(lfs=lfs)
L_train = applier.apply(df=df_train)
L_test = applier.apply(df=df_test)
```


```python
LFAnalysis(L=L_train, lfs=lfs).lf_summary()
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>keyword_my</th>
      <td>0</td>
      <td>[1]</td>
      <td>0.198613</td>
      <td>0.185372</td>
      <td>0.109710</td>
    </tr>
    <tr>
      <th>keyword_subscribe</th>
      <td>1</td>
      <td>[1]</td>
      <td>0.127364</td>
      <td>0.108449</td>
      <td>0.068726</td>
    </tr>
    <tr>
      <th>keyword_http</th>
      <td>2</td>
      <td>[1]</td>
      <td>0.119168</td>
      <td>0.100252</td>
      <td>0.080706</td>
    </tr>
    <tr>
      <th>keyword_please</th>
      <td>3</td>
      <td>[1]</td>
      <td>0.112232</td>
      <td>0.109710</td>
      <td>0.056747</td>
    </tr>
    <tr>
      <th>keyword_song</th>
      <td>4</td>
      <td>[0]</td>
      <td>0.141866</td>
      <td>0.109710</td>
      <td>0.043506</td>
    </tr>
    <tr>
      <th>regex_check_out</th>
      <td>5</td>
      <td>[1]</td>
      <td>0.233922</td>
      <td>0.133039</td>
      <td>0.087011</td>
    </tr>
    <tr>
      <th>short_comment</th>
      <td>6</td>
      <td>[0]</td>
      <td>0.225725</td>
      <td>0.145019</td>
      <td>0.074401</td>
    </tr>
    <tr>
      <th>has_person_nlp</th>
      <td>7</td>
      <td>[0]</td>
      <td>0.071879</td>
      <td>0.056747</td>
      <td>0.030895</td>
    </tr>
    <tr>
      <th>textblob_polarity</th>
      <td>8</td>
      <td>[0]</td>
      <td>0.035309</td>
      <td>0.032156</td>
      <td>0.005044</td>
    </tr>
    <tr>
      <th>textblob_subjectivity</th>
      <td>9</td>
      <td>[0]</td>
      <td>0.357503</td>
      <td>0.252837</td>
      <td>0.160151</td>
    </tr>
  </tbody>
</table>
</div>



Our goal is now to convert the labels from our LFs into a single _noise-aware_ probabilistic (or confidence-weighted) label per data point.
A simple baseline for doing this is to take the majority vote on a per-data point basis: if more LFs voted SPAM than HAM, label it SPAM (and vice versa).
We can test this with the
[`MajorityLabelVoter` baseline model](https://snorkel.readthedocs.io/en/master/packages/_autosummary/labeling/snorkel.labeling.model.baselines.MajorityLabelVoter.html).


```python
from snorkel.labeling import MajorityLabelVoter

majority_model = MajorityLabelVoter()
preds_train = majority_model.predict(L=L_train)
```


```python
preds_train
```




    array([ 1,  1, -1, ...,  1,  1,  1])



However, as we can see from the summary statistics of our LFs in the previous section, they have varying properties and should not be treated identically. In addition to having varied accuracies and coverages, LFs may be correlated, resulting in certain signals being overrepresented in a majority-vote-based model. To handle these issues appropriately, we will instead use a more sophisticated Snorkel `LabelModel` to combine the outputs of the LFs.

This model will ultimately produce a single set of noise-aware training labels, which are probabilistic or confidence-weighted labels. We will then use these labels to train a classifier for our task. For more technical details of this overall approach, see our [NeurIPS 2016](https://arxiv.org/abs/1605.07723) and [AAAI 2019](https://arxiv.org/abs/1810.02840) papers. For more info on the API, see the [`LabelModel` documentation](https://snorkel.readthedocs.io/en/master/packages/_autosummary/labeling/snorkel.labeling.model.label_model.LabelModel.html).

Note that no gold labels are used during the training process.
The only information we need is the label matrix, which contains the output of the LFs on our training set.
The `LabelModel` is able to learn weights for the labeling functions using only the label matrix as input.
We also specify the `cardinality`, or number of classes.


```python
from snorkel.labeling import LabelModel

label_model = LabelModel(cardinality=2, verbose=True)
label_model.fit(L_train=L_train, n_epochs=500, log_freq=100, seed=123)
```


```python
majority_acc = majority_model.score(L=L_test, Y=Y_test, tie_break_policy="random")[
    "accuracy"
]
print(f"{'Majority Vote Accuracy:':<25} {majority_acc * 100:.1f}%")

label_model_acc = label_model.score(L=L_test, Y=Y_test, tie_break_policy="random")[
    "accuracy"
]
print(f"{'Label Model Accuracy:':<25} {label_model_acc * 100:.1f}%")
```

    Majority Vote Accuracy:   84.0%
    Label Model Accuracy:     86.0%


The majority vote model or more sophisticated `LabelModel` could in principle be used directly as a classifier if the outputs of our labeling functions were made available at test time.
However, these models (i.e. these re-weighted combinations of our labeling function's votes) will abstain on the data points that our labeling functions don't cover (and additionally, may require slow or unavailable features to execute at test time).
In the next section, we will instead use the outputs of the `LabelModel` as training labels to train a discriminative classifier **which can generalize beyond the labeling function outputs** to see if we can improve performance further.
This classifier will also only need the text of the comment to make predictions, making it much more suitable for inference over unseen comments.
For more information on the properties of the label model, see the [Snorkel documentation](https://snorkel.readthedocs.io/en/master/packages/_autosummary/labeling/snorkel.labeling.model.label_model.LabelModel.html).

### Filtering out unlabeled data points

As we saw earlier, some of the data points in our `train` set received no labels from any of our LFs.
These data points convey no supervision signal and tend to hurt performance, so we filter them out before training using a
[built-in utility](https://snorkel.readthedocs.io/en/master/packages/_autosummary/labeling/snorkel.labeling.filter_unlabeled_dataframe.html#snorkel.labeling.filter_unlabeled_dataframe).


```python
from snorkel.labeling import filter_unlabeled_dataframe

df_train_filtered, probs_train_filtered = filter_unlabeled_dataframe(
    X=df_train, y=probs_train, L=L_train
)
```

## 5. Training a Classifier

In this final section of the tutorial, we'll use the probabilistic training labels we generated in the last section to train a classifier for our task.
**The output of the Snorkel `LabelModel` is just a set of labels which can be used with most popular libraries for performing supervised learning, such as TensorFlow, Keras, PyTorch, Scikit-Learn, Ludwig, and XGBoost.**
In this tutorial, we use the well-known library [Scikit-Learn](https://scikit-learn.org).
**Note that typically, Snorkel is used (and really shines!) with much more complex, training data-hungry models, but we will use Logistic Regression here for simplicity of exposition.**

### Featurization

For simplicity and speed, we use a simple "bag of n-grams" feature representation: each data point is represented by a one-hot vector marking which words or 2-word combinations are present in the comment text.


```python
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(ngram_range=(1, 5))
X_train = vectorizer.fit_transform(df_train_filtered.text.tolist())
X_test = vectorizer.transform(df_test.text.tolist())
```

### Scikit-Learn Classifier

As we saw in Section 4, the `LabelModel` outputs probabilistic (float) labels.
If the classifier we are training accepts target labels as floats, we can train on these labels directly (see describe the properties of this type of "noise-aware" loss in our [NeurIPS 2016 paper](https://arxiv.org/abs/1605.07723)).

If we want to use a library or model that doesn't accept probabilistic labels (such as Scikit-Learn), we can instead replace each label distribution with the label of the class that has the maximum probability.
This can easily be done using the
[`probs_to_preds` helper method](https://snorkel.readthedocs.io/en/master/packages/_autosummary/utils/snorkel.utils.probs_to_preds.html#snorkel.utils.probs_to_preds).
We do note, however, that this transformation is lossy, as we no longer have values for our confidence in each label.


```python
from snorkel.utils import probs_to_preds

preds_train_filtered = probs_to_preds(probs=probs_train_filtered)
```

We then use these labels to train a classifier as usual.


```python
from sklearn.linear_model import LogisticRegression

sklearn_model = LogisticRegression(C=1e3, solver="liblinear")
sklearn_model.fit(X=X_train, y=preds_train_filtered)
```


```python
print(f"Test Accuracy: {sklearn_model.score(X=X_test, y=Y_test) * 100:.1f}%")
```

    Test Accuracy: 94.4%


**We observe an additional boost in accuracy over the `LabelModel` by multiple points! This is in part because the discriminative model generalizes beyond the labeling function's labels and makes good predictions on all data points, not just the ones covered by labeling functions.
By using the label model to transfer the domain knowledge encoded in our LFs to the discriminative model,
we were able to generalize beyond the noisy labeling heuristics**.

## Summary

In this tutorial, we accomplished the following:
* We introduced the concept of Labeling Functions (LFs) and demonstrated some of the forms they can take.
* We used the Snorkel `LabelModel` to automatically learn how to combine the outputs of our LFs into strong probabilistic labels.
* We showed that a classifier trained on a weakly supervised dataset can outperform an approach based on the LFs alone as it learns to generalize beyond the noisy heuristics we provide.

### Next Steps

If you enjoyed this tutorial and you've already checked out the [Getting Started](https://snorkel.org/get-started/) tutorial, check out the [Tutorials](https://snorkel.org/use-cases/) page for other tutorials that you may find interesting, including demonstrations of how to use Snorkel

* As part of a [hybrid crowdsourcing pipeline](https://snorkel.org/use-cases/crowdsourcing-tutorial)
* For [visual relationship detection over images](https://snorkel.org/use-cases/visual-relation-tutorial)
* For [information extraction over text](https://snorkel.org/use-cases/spouse-demo)
* For [data augmentation](https://snorkel.org/use-cases/02-spam-data-augmentation-tutorial)

and more!
You can also visit the [Snorkel website](https://snorkel.org) or [Snorkel API documentation](https://snorkel.readthedocs.io) for more info!
