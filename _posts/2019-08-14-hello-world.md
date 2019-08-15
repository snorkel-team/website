---
layout: default
title: Introducing the New Snorkel
description: Introducing the new Snorkel v0.9.
excerpt: Introducing the new Snorkel v0.9.
---

# Introducing the New Snorkel

TODO: Intro bit here

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