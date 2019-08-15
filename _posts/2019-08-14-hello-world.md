---
layout: default
title: Introducing the New Snorkel
author: The Snorkel Team
description: Introducing the new Snorkel v0.9.
excerpt: Introducing the new Snorkel v0.9.
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