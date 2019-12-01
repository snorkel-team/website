---
layout: default
title: Visual Relation Detection in Snorkel
description: Multi-class labeling for visual relationships in images
excerpt: Multi-class labeling for visual relationships in images
order: 6
github_link: https://github.com/snorkel-team/snorkel-tutorials/blob/master/visual_relation/visual_relation_tutorial.ipynb
---


# Visual Relationship Detection

In this tutorial, we focus on the task of classifying visual relationships between objects in an image. For any given image, there might be many such relationships, defined formally as a `subject <predictate> object` (e.g. `person <riding> bike`). As an example, in the relationship `man riding bicycle`), "man" and "bicycle" are the subject and object, respectively, and "riding" is the relationship predicate.

![Visual Relationships](https://cs.stanford.edu/people/ranjaykrishna/vrd/dataset.png)

In the examples of the relationships shown above, the red box represents the _subject_ while the green box represents the _object_. The _predicate_ (e.g. kick) denotes what relationship connects the subject and the object.

For the purpose of this tutorial, we operate over the [Visual Relationship Detection (VRD) dataset](https://cs.stanford.edu/people/ranjaykrishna/vrd/) and focus on action relationships. We define our classification task as **identifying which of three relationships holds between the objects represented by a pair of bounding boxes.**

### 1. Load Dataset
We load the VRD dataset and filter images with at least one action predicate in it, since these are more difficult to classify than geometric relationships like `above` or `next to`. We load the train, valid, and test sets as Pandas `DataFrame` objects with the following fields:
- `label`: The relationship between the objects. 0: `RIDE`, 1: `CARRY`, 2: `OTHER` action predicates
- `object_bbox`: coordinates of the bounding box for the object `[ymin, ymax, xmin, xmax]`
- `object_category`: category of the object
- `source_img`: filename for the corresponding image the relationship is in
- `subject_bbox`: coordinates of the bounding box for the object `[ymin, ymax, xmin, xmax]`
- `subject_category`: category of the subject

If you are running this notebook for the first time, it will take ~15 mins to download all the required sample data.

The sampled version of the dataset **uses the same 26 data points across the train, dev, and test sets.
This setting is meant to demonstrate quickly how Snorkel works with this task, not to demonstrate performance.**


```python
from utils import load_vrd_data

# setting sample=False will take ~3 hours to run (downloads full VRD dataset)
sample = True
is_test = os.environ.get("TRAVIS") == "true" or os.environ.get("IS_TEST") == "true"
df_train, df_valid, df_test = load_vrd_data(sample, is_test)

print("Train Relationships: ", len(df_train))
print("Dev Relationships: ", len(df_valid))
print("Test Relationships: ", len(df_test))
```

    Train Relationships:  26
    Dev Relationships:  26
    Test Relationships:  26


Note that the training `DataFrame` will have a labels field with all -1s. This denotes the lack of labels for that particular dataset. In this tutorial, we will assign probabilistic labels to the training set by writing labeling functions over attributes of the subject and objects!

## 2. Writing Labeling Functions
We now write labeling functions to detect what relationship exists between pairs of bounding boxes. To do so, we can encode various intuitions into the labeling functions:
* _Categorical_ intution: knowledge about the categories of subjects and objects usually involved in these relationships (e.g., `person` is usually the subject for predicates like `ride` and `carry`)
* _Spatial_ intuition: knowledge about the relative positions of the subject and objects (e.g., subject is usually higher than the object for the predicate `ride`)


```python
RIDE = 0
CARRY = 1
OTHER = 2
ABSTAIN = -1
```

We begin with labeling functions that encode categorical intuition: we use knowledge about common subject-object category pairs that are common for `RIDE` and `CARRY` and also knowledge about what subjects or objects are unlikely to be involved in the two relationships.


```python
from snorkel.labeling import labeling_function

# Category-based LFs
@labeling_function()
def lf_ride_object(x):
    if x.subject_category == "person":
        if x.object_category in [
            "bike",
            "snowboard",
            "motorcycle",
            "horse",
            "bus",
            "truck",
            "elephant",
        ]:
            return RIDE
    return ABSTAIN


@labeling_function()
def lf_carry_object(x):
    if x.subject_category == "person":
        if x.object_category in ["bag", "surfboard", "skis"]:
            return CARRY
    return ABSTAIN


@labeling_function()
def lf_carry_subject(x):
    if x.object_category == "person":
        if x.subject_category in ["chair", "bike", "snowboard", "motorcycle", "horse"]:
            return CARRY
    return ABSTAIN


@labeling_function()
def lf_not_person(x):
    if x.subject_category != "person":
        return OTHER
    return ABSTAIN
```

We now encode our spatial intuition, which includes measuring the distance between the bounding boxes and comparing their relative areas.


```python
YMIN = 0
YMAX = 1
XMIN = 2
XMAX = 3
```


```python
import numpy as np

# Distance-based LFs
@labeling_function()
def lf_ydist(x):
    if x.subject_bbox[XMAX] < x.object_bbox[XMAX]:
        return OTHER
    return ABSTAIN


@labeling_function()
def lf_dist(x):
    if np.linalg.norm(np.array(x.subject_bbox) - np.array(x.object_bbox)) <= 1000:
        return OTHER
    return ABSTAIN


def area(bbox):
    return (bbox[YMAX] - bbox[YMIN]) * (bbox[XMAX] - bbox[XMIN])


# Size-based LF
@labeling_function()
def lf_area(x):
    if area(x.subject_bbox) / area(x.object_bbox) <= 0.5:
        return OTHER
    return ABSTAIN
```

Note that the labeling functions have varying empirical accuracies and coverages. Due to class imbalance in our chosen relationships, labeling functions that label the `OTHER` class have higher coverage than labeling functions for `RIDE` or `CARRY`. This reflects the distribution of classes in the dataset as well.


```python
from snorkel.labeling import PandasLFApplier

lfs = [
    lf_ride_object,
    lf_carry_object,
    lf_carry_subject,
    lf_not_person,
    lf_ydist,
    lf_dist,
    lf_area,
]

applier = PandasLFApplier(lfs)
L_train = applier.apply(df_train)
L_valid = applier.apply(df_valid)
```


```python
from snorkel.labeling import LFAnalysis

Y_valid = df_valid.label.values
LFAnalysis(L_valid, lfs).lf_summary(Y_valid)
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
      <th>lf_ride_object</th>
      <td>0</td>
      <td>[0]</td>
      <td>0.230769</td>
      <td>0.230769</td>
      <td>0.230769</td>
      <td>5</td>
      <td>1</td>
      <td>0.833333</td>
    </tr>
    <tr>
      <th>lf_carry_object</th>
      <td>1</td>
      <td>[1]</td>
      <td>0.076923</td>
      <td>0.076923</td>
      <td>0.076923</td>
      <td>2</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>lf_carry_subject</th>
      <td>2</td>
      <td>[1]</td>
      <td>0.038462</td>
      <td>0.038462</td>
      <td>0.038462</td>
      <td>1</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>lf_not_person</th>
      <td>3</td>
      <td>[2]</td>
      <td>0.307692</td>
      <td>0.307692</td>
      <td>0.038462</td>
      <td>5</td>
      <td>3</td>
      <td>0.625000</td>
    </tr>
    <tr>
      <th>lf_ydist</th>
      <td>4</td>
      <td>[2]</td>
      <td>0.576923</td>
      <td>0.576923</td>
      <td>0.307692</td>
      <td>7</td>
      <td>8</td>
      <td>0.466667</td>
    </tr>
    <tr>
      <th>lf_dist</th>
      <td>5</td>
      <td>[2]</td>
      <td>1.000000</td>
      <td>0.846154</td>
      <td>0.346154</td>
      <td>13</td>
      <td>13</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>lf_area</th>
      <td>6</td>
      <td>[2]</td>
      <td>0.346154</td>
      <td>0.346154</td>
      <td>0.153846</td>
      <td>5</td>
      <td>4</td>
      <td>0.555556</td>
    </tr>
  </tbody>
</table>
</div>



## 3. Train Label Model
We now train a multi-class `LabelModel` to assign training labels to the unalabeled training set.


```python
from snorkel.labeling import LabelModel

label_model = LabelModel(cardinality=3, verbose=True)
label_model.fit(L_train, seed=123, lr=0.01, log_freq=10, n_epochs=100)
```

We use [F1](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html) Micro average for the multiclass setting, which calculates metrics globally across classes, by counting the total true positives, false negatives and false positives.


```python
label_model.score(L_valid, Y_valid, metrics=["f1_micro"])
```

    WARNING:root:Metrics calculated over data points with non-abstain labels only





    {'f1_micro': 0.5769230769230769}



## 4. Train a Classifier
You can then use these training labels to train any standard discriminative model, such as [an off-the-shelf ResNet](https://github.com/KaimingHe/deep-residual-networks), which should learn to generalize beyond the LF's we've developed!

#### Create DataLoaders for Classifier


```python
from snorkel.classification import DictDataLoader
from model import SceneGraphDataset, create_model

df_train["labels"] = label_model.predict(L_train)

if sample:
    TRAIN_DIR = "data/VRD/sg_dataset/samples"
else:
    TRAIN_DIR = "data/VRD/sg_dataset/sg_train_images"

dl_train = DictDataLoader(
    SceneGraphDataset("train_dataset", "train", TRAIN_DIR, df_train),
    batch_size=16,
    shuffle=True,
)

dl_valid = DictDataLoader(
    SceneGraphDataset("valid_dataset", "valid", TRAIN_DIR, df_valid),
    batch_size=16,
    shuffle=False,
)
```

#### Define Model Architecture


```python
import torchvision.models as models

# initialize pretrained feature extractor
cnn = models.resnet18(pretrained=True)
model = create_model(cnn)
```

### Train and Evaluate Model


```python
from snorkel.classification import Trainer

trainer = Trainer(
    n_epochs=1,  # increase for improved performance
    lr=1e-3,
    checkpointing=True,
    checkpointer_config={"checkpoint_dir": "checkpoint"},
)
trainer.fit(model, [dl_train])
```


```python
model.score([dl_valid])
```




    {'visual_relation_task/valid_dataset/valid/f1_micro': 0.6153846153846154}



## Recap
We have successfully trained a visual relationship detection model! Using categorical and spatial intuition about how objects in a visual relationship interact with each other, we are able to assign high quality training labels to object pairs in the VRD dataset in a multi-class classification setting.

For more on how Snorkel can be used for visual relationship tasks, please see our [ICCV 2019 paper](https://arxiv.org/abs/1904.11622)!
