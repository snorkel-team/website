---
layout: default
category: blog post
title: June 2019 Workshop
author: The Snorkel Team
description: Recap of June 2019 Snorkel workshop.
excerpt: Recap of June 2019 Snorkel workshop.
show_title_author: True
redirect_from: /june-workshop
---


# Recap of June 2019 Snorkel Workshop

<figure align="center">
	  <img style="width: 80%; ;" src="/doks-theme/assets/images/2019-06-30-workshop/workshop.png">
</figure>

On June 24th and 25th, we hosted a workshop at Stanford to talk about recent research around Snorkel, discuss applications and extensions, and most importantly from our end, get feedback on the upcoming v0.9 release of Snorkel. 
**This upcoming release is our most significant upgrade yet, integrating research work from the last two years ([VLDBJ 2019](https://arxiv.org/abs/1711.10160), [ICML 2019](https://arxiv.org/abs/1903.05844), [SIGMOD 2019](https://arxiv.org/abs/1812.00417), [AAAI 2019](https://arxiv.org/abs/1810.02840), [DEEM 2018](https://ajratner.github.io/assets/papers/deem-metal-prototype.pdf), [NeurIPS 2017](https://arxiv.org/abs/1709.01643))**, new operators for building and managing training data, scripts demonstrating how to reproduce our recent state-of-the-art results on the public benchmark [SuperGLUE](/superglue), and a full redesign of the core library!

We were lucky enough to have 55 of our collaborators and users attend this workshop, including members of industry (Google, Microsoft, Facebook, and others), government, and medicine.
In this blog post, we’ll describe some of the upcoming changes in v0.9 we’re most excited about, summarize and share the workshop contents, and most importantly invite your feedback as we prepare for the v0.9 release this August!

## Snorkel v0.9 Teaser
Snorkel v0.9 integrates our recent research advances and Snorkel-based open source projects into **one modern Python library for building and managing training datasets**. In this release, we are:
* Adding two new core data set operators — *transformation functions* and *slicing functions* — alongside labeling functions
* Upgrading the labeling pipeline using the latest frameworks and algorithmic research
* Redesigning the codebase to be a more general purpose and modular engine

We previewed these changes at the workshop, and are now sharing the workshop contents below to get your feedback as well leading up to the release in August!

### New core data set operations: transforming and slicing
We started building Snorkel in 2016, motivated by the increasing commoditization of ML models and infrastructure, and the increasing primacy of training data. 
We set out to create a framework where building and managing training data in programmatic, heuristic, or otherwise noisy ways was the primary mode of interaction with machine learning.
To begin, we focused on one aspect of this broader set of activities: labeling training data, by writing heuristic labeling functions ([NeurIPS 2016](https://arxiv.org/abs/1605.07723), [blog](https://hazyresearch.github.io/snorkel/blog/weak_supervision.html)).

**Our upcoming release adds two new core operations as first class citizens in addition to programmatic labeling.** The first is transforming data points with transformation functions (TFs), our general [framework for data augmentation](https://hazyresearch.github.io/snorkel/blog/tanda.html). The second is [slicing training data](https://dawn.cs.stanford.edu/2019/06/15/superglue/) with slicing functions (SFs), in order to monitor and focus model attention on subsets of the training dataset where classification is more critical or difficult. 

<figure align="center">
	  <img style="width: 80%; ;" src="/doks-theme/assets/images/2019-06-30-workshop/fig_abstractions.png">
</figure>

We’re excited to bring these concepts to the forefront of programmatic data set creation. Snorkel provides users with a flexible interface for applying state-of-the-art automated data augmentation techniques like [TANDA (NeurIPS 2017)](https://hazyresearch.github.io/snorkel/blog/tanda.html) to their problems. Using this interface, attendees were able to reproduce state-of-the-art data augmentation techniques for text applications — as well as create original ones — in a single work session. 

Slicing is a more recent research advance, but has already powered state-of-the-art results like our SuperGLUE system. Workshop participants were interested in more academic literature on slicing, which we’ll be releasing in the coming months. In the meantime, check out our [blog post on slicing in SuperGLUE](/superglue) and the slicing workshop materials below.

### Upgrade labeling pipeline
The programmatic labeling engine is getting updated to the latest and greatest. The new core generative label model uses the matrix-completion approach studied in our [AAAI 2019](https://arxiv.org/abs/1810.02840) and [ICML 2019](https://arxiv.org/abs/1903.05844) papers and implemented in [Snorkel MeTaL](https://github.com/HazyResearch/metal/). The label model is currently implemented in PyTorch, but we’re starting work on a TensorFlow-based and TFX servable version that many workshop participants requested. 

**We're also providing native support for applying labeling functions using Spark and Dask**, making it easier to work with massive data sets. Spark integration in particular has been a common feature request, and this was echoed by the workshop attendees.

## Overview of Workshop Sessions

<figure align="center">
	  <img style="width: 80%; ;" src="/doks-theme/assets/images/2019-06-30-workshop/workshop-2.png">
</figure>

The workshop was structured around the three core Snorkel operations: labeling, transforming, and slicing. Labeling functions were covered on the first day, and transformation and slicing functions covered on the second day. We also explored code from our state-of-the-art SuperGLUE submission, demonstrating transformation and slicing on several of the tasks. 

The first half of each day was divided between presentations (both high-level and technical or theoretical deep dives for those interested) and Jupyter notebook-based tutorials. We had two hours of open "office hours"-style sessions in the afternoon during which participants could discuss potential use cases of Snorkel with us and each other, as well as explore the tutorial notebooks and ask questions. You can find workshop materials below!

* Introduction to Snorkel ([slides](https://www.dropbox.com/sh/ipxmm6twu4p2qo1/AACztdxm-GTWxOkA7PfX2ooaa/Day%201?dl=0&preview=02_Snorkel.pdf&subfolder_nav_tracking=1))
* Designing and Iterating on Labeling Functions ([slides](https://www.dropbox.com/sh/ipxmm6twu4p2qo1/AACztdxm-GTWxOkA7PfX2ooaa/Day%201?dl=0&preview=03_Tutorial.pdf&subfolder_nav_tracking=1), [tutorial](https://github.com/snorkel-team/snorkel-extraction/tree/master/tutorials/workshop))
* Theory Introduction and Designing Snorkel Applications ([slides](https://www.dropbox.com/sh/ipxmm6twu4p2qo1/AACztdxm-GTWxOkA7PfX2ooaa/Day%201?dl=0&preview=04_Theory_Apps.pdf&subfolder_nav_tracking=1))
* Introduction to Slicing Functions ([slides](https://www.dropbox.com/sh/ipxmm6twu4p2qo1/AABUdQ0i0UOt46q11Ldgy6z7a/Day%202?dl=0&preview=02_Slicing.pdf&subfolder_nav_tracking=1), [tutorial](https://github.com/HazyResearch/snorkel-superglue/blob/master/tutorials/WiC_slicing_tutorial.ipynb))
* Introduction to Transformation Functions ([slides](https://www.dropbox.com/sh/ipxmm6twu4p2qo1/AABUdQ0i0UOt46q11Ldgy6z7a/Day%202?dl=0&preview=03_Augmentation.pdf&subfolder_nav_tracking=1), [tutorial](https://github.com/HazyResearch/snorkel-superglue/blob/master/tutorials/WiC_augmentation_tutorial.ipynb))
* Multi-task and Transfer Learning ([slides](https://www.dropbox.com/sh/ipxmm6twu4p2qo1/AABUdQ0i0UOt46q11Ldgy6z7a/Day%202?dl=0&preview=04_MTL.pdf&subfolder_nav_tracking=1), [tutorial](https://github.com/snorkel-team/snorkel-tutorials/blob/master/multitask/multitask_tutorial.ipynb))
* Beating GLUE and SuperGLUE with Training Data ([slides](https://www.dropbox.com/sh/ipxmm6twu4p2qo1/AABUdQ0i0UOt46q11Ldgy6z7a/Day%202?dl=0&preview=05_SuperGLUE.pdf&subfolder_nav_tracking=1), [tutorial 1](https://github.com/HazyResearch/snorkel-superglue/blob/master/tutorials/COPA_pretrain_tutorial.ipynb), [tutorial 2](https://github.com/HazyResearch/snorkel-superglue/blob/master/tutorials/RTE_slicing_tutorial.ipynb))

Many of the tutorials as posted here were under active development. The workshop attendees asked for updated versions with the initial Snorkel v0.9 release along with full GLUE and SuperGLUE code, which we’ll be providing in the new tutorials repo.

## Looking Ahead
We’ve been working hard on Snorkel v0.9, and we hope this blog post has convinced you to try it out when it's **released this August**. And we want to give our workshop participants a massive thank you for their enthusiasm and feedback. **If you’re interested in attending the next workshop following the release or staying up-to-date on the latest developments, we encourage you to [join our mailing list](https://groups.google.com/forum/#!forum/snorkel-ml)** (infrequent posts, we promise). We’re excited to share Snorkel v0.9 with you and even more excited to hear your feedback, so stay tuned for our full release announcement! 
