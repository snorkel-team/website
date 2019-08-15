---
layout: default
title: Visual Relation Detection in Snorkel
description: Multi-class labeling for visual relationships in images
excerpt: Multi-class labeling for visual relationships in images
order: 6
---


# Visual Relationship Detection

In this tutorial, we focus on the task of classifying visual relationships. For a given image, there might be many such relationships, defined formally as a `subject <predictate> object` (e.g. `person <riding> bike`).

These are relationships among a pair of objects in images (e.g. "man riding bicycle"), where "man" and "bicycle" are the subject and object, respectively, and "riding" is the relationship predicate.

![Visual Relationships](https://cs.stanford.edu/people/ranjaykrishna/vrd/dataset.png)

For the purpose of this tutorial, we operate over the [Visual Relationship Detection (VRD) dataset](https://cs.stanford.edu/people/ranjaykrishna/vrd/) and focus on action relationships. We define our three class classification task as **identifying whether a pair of bounding boxes represents a particular relationship.**

In the examples of the relationships shown below, the red box represents the _subject_ while the green box represents the _object_. The _predicate_ (e.g. kick) denotes what relationship connects the subject and the object.


```python
import os

if os.path.basename(os.getcwd()) == "visual_relation":
    os.chdir("..")
```

### 1. Load Dataset
We load the VRD dataset and filter images with at least one action predicate in it, since these are more difficult to classify than geometric relationships like `above` or `next to`. We load the train, valid, and test sets as Pandas `DataFrame` objects with the following fields:
- `label`: The relationship between the objects. 0: `RIDE`, 1: `CARRY`, 2: `OTHER` action predicates
- `object_bbox`: coordinates of the bounding box for the object `[ymin, ymax, xmin, xmax]`
- `object_category`: category of the object
- `source_img`: filename for the corresponding image the relationship is in
- `subject_bbox`: coordinates of the bounding box for the object `[ymin, ymax, xmin, xmax]`
- `subject_category`: category of the subject


```python
%load_ext autoreload
%autoreload 2

import numpy as np

```

If you are running this notebook for the first time, it will take ~15 mins to download all the required sample data.

The sampled version of the dataset **uses the same 26 examples across the train, dev, and test sets. This setting is meant to demonstrate how Snorkel works with this task, not to demonstrate performance.**


```python
from visual_relation.utils import load_vrd_data

# setting sample=False will take ~3 hours to run (downloads full VRD dataset)
sample = True
is_travis = os.environ.get("TRAVIS") == "true"
df_train, df_valid, df_test = load_vrd_data(sample, is_travis)

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
        if x.object_category in ["bike", "snowboard", "motorcycle", "horse"]:
            return RIDE
    return ABSTAIN


@labeling_function()
def lf_ride_rare_object(x):
    if x.subject_category == "person":
        if x.object_category in ["bus", "truck", "elephant"]:
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
def lf_person(x):
    if x.subject_category != "person":
        return OTHER
    return ABSTAIN
```

We now encode our spatial intuition, which includes measuring the distance between the bounding boxes and comparing their relative areas.


```python
# Distance-based LFs
@labeling_function()
def lf_ydist(x):
    if x.subject_bbox[3] < x.object_bbox[3]:
        return OTHER
    return ABSTAIN


@labeling_function()
def lf_dist(x):
    if np.linalg.norm(np.array(x.subject_bbox) - np.array(x.object_bbox)) <= 1000:
        return OTHER
    return ABSTAIN


# Size-based LF
@labeling_function()
def lf_area(x):
    subject_area = (x.subject_bbox[1] - x.subject_bbox[0]) * (
        x.subject_bbox[3] - x.subject_bbox[2]
    )
    object_area = (x.object_bbox[1] - x.object_bbox[0]) * (
        x.object_bbox[3] - x.object_bbox[2]
    )

    if subject_area / object_area <= 0.5:
        return OTHER
    return ABSTAIN
```

Note that the labeling functions have varying empirical accuracies and coverages. Due to class imbalance in our chosen relationships, labeling functions that label the `OTHER` class have higher coverage than labeling functions for `RIDE` or `CARRY`. This reflects the distribution of classes in the dataset as well.


```python
from snorkel.labeling import PandasLFApplier

lfs = [
    lf_ride_object,
    lf_ride_rare_object,
    lf_carry_object,
    lf_carry_subject,
    lf_person,
    lf_ydist,
    lf_dist,
    lf_area,
]

applier = PandasLFApplier(lfs)
L_train = applier.apply(df_train)
L_valid = applier.apply(df_valid)
```

    100%|██████████| 26/26 [00:00<00:00, 2297.47it/s]
    100%|██████████| 26/26 [00:00<00:00, 3016.98it/s]



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
      <th>lf_ride_rare_object</th>
      <td>1</td>
      <td>[]</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>lf_carry_object</th>
      <td>2</td>
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
      <td>3</td>
      <td>[1]</td>
      <td>0.038462</td>
      <td>0.038462</td>
      <td>0.038462</td>
      <td>1</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>lf_person</th>
      <td>4</td>
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
      <td>5</td>
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
      <td>6</td>
      <td>[2]</td>
      <td>1.000000</td>
      <td>0.846154</td>
      <td>0.346154</td>
      <td>13</td>
      <td>6</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>lf_area</th>
      <td>7</td>
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




    {'f1_micro': 0.5769230769230769}



## 4. Train a Classifier
You can then use these training labels to train any standard discriminative model, such as [an off-the-shelf ResNet](https://github.com/KaimingHe/deep-residual-networks), which should learn to generalize beyond the LF's we've developed!

#### Create DataLoaders for Classifier


```python
from snorkel.classification import DictDataLoader
from visual_relation.model import FlatConcat, SceneGraphDataset, WordEmb, init_fc

df_train["labels"] = label_model.predict(L_train)

if sample:
    TRAIN_DIR = "visual_relation/data/VRD/sg_dataset/samples"
else:
    TRAIN_DIR = "visual_relation/data/VRD/sg_dataset/sg_train_images"

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
import torch.nn as nn

from snorkel.analysis import Scorer
from snorkel.classification import Task


# initialize pretrained feature extractor
cnn = models.resnet18(pretrained=True)

# freeze the resnet weights
for param in cnn.parameters():
    param.requires_grad = False

# define input features
in_features = cnn.fc.in_features
feature_extractor = nn.Sequential(*list(cnn.children())[:-1])

# initialize FC layer: maps 3 sets of image features to class logits
WEMB_SIZE = 100
fc = nn.Linear(in_features * 3 + 2 * WEMB_SIZE, 3)
init_fc(fc)

# define layers
module_pool = nn.ModuleDict(
    {
        "feat_extractor": feature_extractor,
        "prediction_head": fc,
        "feat_concat": FlatConcat(),
        "word_emb": WordEmb(),
    }
)

# %%
from visual_relation.model import get_op_sequence

# define task flow through modules
op_sequence = get_op_sequence()
pred_cls_task = Task(
    name="visual_relation_task",
    module_pool=module_pool,
    op_sequence=op_sequence,
    scorer=Scorer(metrics=["f1_micro"]),
)
```

### Train and Evaluate Model


```python
from snorkel.classification import MultitaskClassifier, Trainer

model = MultitaskClassifier([pred_cls_task])
trainer = Trainer(
    n_epochs=1,  # increase for improved performance
    lr=1e-3,
    checkpointing=True,
    checkpointer_config={"checkpoint_dir": "checkpoint"},
)
trainer.fit(model, [dl_train])
```

    Epoch 0::   0%|          | 0/2 [00:00<?, ?it/s]/home/ubuntu/snorkel-tutorials/visual_relation/model.py:134: FutureWarning: Method .as_matrix will be removed in a future version. Use .values instead.
      return self.word_embs.loc[word].as_matrix()
    Epoch 0:: 100%|██████████| 2/2 [00:02<00:00,  1.29s/it, model/all/train/loss=1.07, model/all/train/lr=0.001]



```python
model.score([dl_valid])
```




    {'visual_relation_task/valid_dataset/valid/f1_micro': 0.6153846153846154}



We have successfully trained a visual relationship detection model! Using categorical and spatial intuition about how objects in a visual relationship interact with each other, we are able to assign high quality training labels to object pairs in the VRD dataset in a multi-class classification setting.
