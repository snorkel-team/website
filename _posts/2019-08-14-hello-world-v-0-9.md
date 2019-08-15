---
layout: default
title: Introducing the New Snorkel
author: The Snorkel Team
description: Introducing the new Snorkel v0.9.
excerpt: Introducing the new Snorkel v0.9.
show_title_author: True
---

Snorkel was started in 2016 at Stanford as a project that, as suggested by an advisor to his graduate student at the time, should probably take "an afternoon".
That afternoon ended up (happily) being a long one.
Over the last few years Snorkel has been deployed in industry (e.g. [Google](#), [Intel](#), [IBM](#)), medicine (e.g. [Stanford](#), [VA](#)), government, and science; has been the focus of over  twenty four scientific publications, including six NeurIPS and ICML papers, two Nature Communications papers, and a "Best Of" VLDB paper; and most rewarding of all, has benefited from the feedback and support of a vibrant and generous user community.

And, it's far from over.
Today we're excited to announce what we hope will end up being the most impactful step yet: the release of a new, more mature version of the open source codebase, v0.9, that advances Snorkel towards being a much more **general purpose system for programmatically building and managing training datasets**, and much more mature as a **modern python library**.

In this release we:
- Pull together several lines of our research on programmatic training data management and weak supervision, previously scattered across various codebases
- Move to a much more modular and data-agnostic configuration of Snorkel that is appropriate for a range of tasks beyond text information extraction
- Add a new set of revamped and expanded tutorials (on a snazzy new website!) that we plan to add to regularly
- Attempt to atone for our past research code sins by adding the basic elements of a modern, well-maintained Python library, e.g. proper installation, extensive unit and integration testing, typing, documentation, and more.

We know we'll have a lot more to do, but we're excited for the feedback and engagement around this next step for Snorkel!


## New Ways to Build & Manage Training Data

Snorkel is motivated by the observation that as modern machine learning models have become increasingly performant and easy-to-use---but also massively data-hungry---building and managing _training datasets_ has increasingly become the key development bottleneck and limiting reagant to actually building real-world ML applications.
The goal of Snorkel is to take the operations that practitioners employ over the training data, which are often _most_ critical to ML model success, but also most often relegated to ad hoc and manual processes---e.g. labeling, augmenting, and managing training data---and make them the first class citizens of a programmatic _development_ process.

<figure align="center">
        <img style="width: 100%; max-width: 580px;" src="/doks-theme/assets/images/layout/Overview.png"/>
</figure>

When we started the Snorkel project, we chose to tackle one aspect of this process first: _labeling_ training data.
In the new version of Snorkel, we both expand the support for this operation, and introduce two new operations: _transforming_ or _augmenting_ data, and _slicing_ or partitioning data.
For an overview of how these all fit together in the new version of Snorkel, we recommend checking out the new [getting started guide](https://snorkel.org/get-started/)!

We now give a brief preview of what these operations look like in the new Snorkel:

### Labeling Functions

The core operator in Snorkel for the majority of the project has been the _labeling function (LF)_, which provides an abstracted way to represent various heuristic or noisy programmatic labeling strategies, which Snorkel then models and combines into clean training labels.
For example, in a spam classification problem, a labeling function could employ a regular expression:

```python
@labeling_function()
def lf_regex_check_out(x):
    """Spam comments say 'check out my video', 'check it out', etc."""
    return SPAM if re.search(r"check.*out", x.text, flags=re.I) else ABSTAIN
```

In the new version of Snorkel, we include a new labeling function class that handles flexible preprocessors, memoization, efficient execution over a variety of generic data types, and much more.
For more on LFs, see the [getting started guide](https://snorkel.org/get-started/), and then the more advanced [intro tutorial to LFs](https://www.snorkel.org/use-cases/01-spam-tutorial).

### Transformation Functions

Another key training data management technique that has emerged over the last several years as especially crucial to model performance is _data augmentation_, the strategy of creating transformed copies of labeled data points to effectively inject knowledge of invariances into the model.
Data augmentation is traditionally done in ad hoc ways and hand-tuned ways, buried in data loader preprocessing scripts... but it is [absolutely critical to model performance](https://www.snorkel.org/tanda/).
It's also a perfect fit for the overall philosophy of Snorkel: enable users to _program machine learning models via the training data._

In Snorkel, data augmentation is now supported as a first class citizen, represented by the _transformation function (TF)_, which takes in a data point and returns a transformed copy of it, building on our [NeurIPS 2017 work](https://arxiv.org/abs/1709.01643) here on autoamtically learning data augmentation policies.
The canonical example of a transformation would be rotating, stretching, or shearing an image (all of which Snorkel supports), but TFs can also be used over text data, for example randomly replacing words with synonyms:

```python
@transformation_function()
def tf_replace_word_with_synonym(x):
    """Try to replace a random word with a synonym."""
    words = x.text.lower().split()
    idx = random.choice(range(len(words)))
    synonyms = get_synonyms(words[idx])
    if len(synonyms) > 0:
        x.text = " ".join(words[:idx] + [synonyms[0]] + words[idx + 1 :])
        return x
```

For more on TFs, see the [getting started guide](https://snorkel.org/get-started/), and then the more advanced [intro tutorial to TFs](https://www.snorkel.org/use-cases/02-spam-data-augmentation-tutorial).


### Slicing Functions

```python
@slicing_function()
def short_link(x):
    """Return whether text matches common pattern for shortened ".ly" links."""
    return int(bool(re.search(r"\w+\.ly", x.text)))
```

## Upgraded Labeling Pipeline

### New Matrix Completion-Style Modeling Approach

<figure align="center">
        <img style="width: 100%; max-width: 580px;" src="/doks-theme/assets/images/2019-08-14-hello-world-v-0-9/matrix_completion.png"/>
</figure>

### Big Data Operators

TODO

## Moving to a Modular, General Purpose Snorkel

TODO

## Tutorials

This is a great way to contribute...!  Link to spectrum for voting!

## Becoming a Modern Python Library

### Installation

### Unit and Integration Testing

### Documentation



## Moving Forwards

TODO: Commitment to proper maintenance, new tutorials, new integrations, community forum and mailing list