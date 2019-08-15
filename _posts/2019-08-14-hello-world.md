---
layout: default
title: Introducing the New Snorkel
author: The Snorkel Team
description: Introducing the new Snorkel v0.9.
excerpt: Introducing the new Snorkel v0.9.
---

Snorkel was started in 2016 at Stanford as a project that, as suggested by an advisor to his graduate student at the time, should probably take "an afternoon".
That long afternoon is (happily) still far from over.
Over the last few years Snorkel has been deployed in industry (at places like [Google](#), [Intel](#), [IBM](#)), medicine ([Stanford](#)), government, and science; has been the focus of over  twenty four scientific publications, including six NeurIPS and ICML papers, two Nature Communications papers, and a "Best Of" VLDB paper; and most rewarding of all, has benefited from the feedback and support of a vibrant user community.

However, today we're excited to announce what we hope will end up being the most impactful step yet: the release of a new, more mature version of Snorkel, v0.9, that represents our most significant set of changes to the open source Snorkel repo to date.

In this release, we finally pull together several lines of research around ways of **programmatically building and managing training datasets** for machine learning---including new ways of modeling and combining _weak labels_, learning and executing _data augmentation_ strategies, and _slicing_ training datasets into critical subsets---into one core framework and codebase.
In addition:
 - We move to a more modular and general purpose configuration of Snorkel that is appropriate for the range of tasks beyond information extraction from text that Snorkel users have spread out into
 - We add a new set of revamped and expanded tutorials, which we plan to add to regularly
 - We make attempts to atone for our past sins of poorly-maintained research code, and move Snorkel towards the standards of a modern Python library: pip and conda installation, extensive unit and integration testing, typing, documentation, and more




## New Ways to Build & Manage Training Data

TODO: Intro bit here

<figure>
	<img style="width: 100%; max-width: 580px;" src="/doks-theme/assets/images/layout/Overview.png"/>
</figure>


### Labeling Functions
```python
@labeling_function()
def lf_keyword_my(x):
    """Many spam comments talk about 'my channel', 'my video', etc."""
    return SPAM if "my" in x.text.lower() else ABSTAIN
```

### Transformation Functions

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


### Slicing Functions

```python
@slicing_function()
def short_link(x):
    """Return whether text matches common pattern for shortened ".ly" links."""
    return int(bool(re.search(r"\w+\.ly", x.text)))
```

## Upgraded Labeling Pipeline

### New Matrix Completion-Style Modeling Approach

TODO: Picture here!

### Big Data Operators

### New Structure Learning Approach

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