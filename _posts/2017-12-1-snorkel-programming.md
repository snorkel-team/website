---
layout: default
title: Programming Training Data
author: Alex Ratner, Stephen Bach, Chris Ré
description: The new interface for ML.
excerpt: The new interface for ML.
---

Machine learning today is both far more and far less accessible than ever before. On the one hand, without any manual feature engineering or custom algorithm development, a developer can have a deep learning model downloaded and running near state-of-the-art within minutes. However, in other ways, machine learning has never been so opaque and inaccessible. Modern deep learning models admit one primary input type---training data---and other than that, are largely black boxes. Given some knowledge of a new domain or task, how do we inject this into our model? Given some modification to our objectives, how do we quickly modify our model? How does one *program* the modern machine learning stack?

One answer, of course, is that today’s ML systems don’t need to be programmed at all---and, given large volumes of training data, this is more true than ever before. However, in practice, these training sets have to be assembled, cleaned, and debugged---a prohibitively expensive and slow task, especially when domain expertise is required. Even more importantly, in the real world, tasks iteratively change and evolve. For example, labeling guidelines, granularities, or downstream use cases often change, necessitating re-labeling. For all these reasons, practitioners have increasingly been turning to [weaker forms of supervision](https://hazyresearch.github.io/snorkel/blog/ws_blog_post.html), such as heuristically generating training data with external knowledge bases, patterns or rules, or other classifiers. Essentially, these are all ways of programmatically generating training data---or, more succinctly, *programming training data*.


## Code as Supervision: Training ML by Programming

Our system, Snorkel---which we report on in **a new VLDB 2018 paper posted [here](https://arxiv.org/abs/1711.10160)**---is one attempt to build a system around this new type of interaction with ML. In [Snorkel](http://snorkel.stanford.edu/), we use **no hand-labeled training data**, but instead ask users to write *labeling functions (LFs)*, bits of black-box code which label subsets of unlabeled data. For example, suppose we were trying to train a machine learning model to [extract mentions of adverse drug reactions](https://github.com/snorkel-team/snorkel-extraction/tree/master/tutorials/cdr) from the scientific literature. To encode a heuristic about negation, for example, we could try writing the LF below:

<figure>
	<img src="/doks-theme/assets/images/2017-12-1-snorkel-programming/snorkel_lf.png" style="width: 100%; max-width: 500px;" alt="Example LF in Snorkel"/>
</figure>

We could then use a set of LFs to label training data for our machine learning model. Since labeling functions are just arbitrary bits of code, they can encode arbitrary signals: patterns, heuristics, external data resources, noisy labels from crowd workers, weak classifiers, and more. And, **as code**, we can reap all the other associated benefits like modularity, reusability, debuggability. If our modeling goals change, for example, we can just tweak our labeling functions to quickly adapt!

<figure>
	<img src="/doks-theme/assets/images/2017-12-1-snorkel-programming/dp.png" style="width: 100%; max-width: 1000px;" alt="The data programming pipeline in Snorkel"/>
</figure>

The problem, of course, is that the labeling functions will produce noisy outputs which may overlap and conflict, producing less-than-ideal training labels. In Snorkel, we de-noise these labels using our *[data programming](https://arxiv.org/abs/1605.07723)* approach, which comprises three steps:

 -   First, we apply the labeling functions to unlabeled data.
 -   Next, we use a **generative model** to learn the accuracies of the labeling functions *without 	any labeled data*, and weight their outputs accordingly. We can even learn the [structure of 	 their correlations](https://arxiv.org/abs/1703.00854) automatically, avoiding e.g. 
	 double-counting problems.
 -   Finally, the end output is a set of ***probabilistic*** training labels, which we can use to 
	 train a powerful, flexible **discriminative model** that will generalize beyond the signal expressed in our LFs.

This whole pipeline can be seen as providing a simple, robust, and model-agnostic approach to “programming” an ML model!


## A New Take on an Old Project: Injecting Domain Knowledge into AI

<figure>
	<img src="/doks-theme/assets/images/2017-12-1-snorkel-programming/gofai.png" style="width: 100%; max-width: 750px;" alt="Injecting domain knowledge in AI"/>
</figure>

From a historical perspective, trying to “program” AI (i.e., inject domain knowledge) is nothing new---the change is that AI has never before been so powerful, nor such a difficult black box to interact with.

In the 1980’s, the focus in AI was on *expert systems*, which combined manually-curated *knowledge bases* of facts and rules from domain experts with *inference engines* to apply them. The port of input was simple: just enter new facts or rules into the knowledge base. However, this very simplicity also belied the brittleness of these systems. Entering rules and facts by hand was neither sufficiently exhaustive nor scalable enough to handle the long-tail, high-dimensional data (e.g. text, images, speech, etc.) present in many real world applications.

In the 1990’s, machine learning began to take off as the vehicle for integrating knowledge into AI systems, promising to do so automatically from labeled *training data* in powerful and flexible ways. Classical (non-representation-learning) machine learning approaches generally had two ports of domain expert input. First, these models were generally of much lower complexity than modern ones, meaning that smaller amounts of hand-labeled data could be used. Second, these models relied on hand-engineered features, which provided a direct way to encode, modify, and interact with the model’s base representation of the data. However, feature engineering was and is generally considered a task for ML experts, who often would spend entire PhDs crafting features for a particular task.

Enter deep learning models: due to their impressive ability to automatically learn representations across many domains and tasks, they have largely obviated the task of feature engineering. However, they are for the most part complete black boxes, with very few knobs for the average developer to play with other than labeling massive training sets. In many senses, they represent the opposite extreme of the brittle but easily-modifiable rules of old expert systems. This leads us back to our original question from a slightly different angle: How do we leverage our domain knowledge or task expertise to program modern deep learning models? Is there any way to combine the directness of the old rules-based expert systems with the flexibility and power of these modern machine learning methods?


## Snorkel: Notes from a Year in the Field

Snorkel is our ongoing attempt to build a system that combines the best of these worlds: the directness of writing code with the flexibility and power of modern machine learning models under the hood. In the Snorkel workflow, no labeled training data is used. Instead, users write *labeling functions (LFs)* which serve as the programming interface to generate weak supervision, which is then automatically modeled and used to train an end model, such as a DNN.

<figure>
	<img src="/doks-theme/assets/images/2017-12-1-snorkel-programming/snorkel_system.png" style="width: 100%; max-width: 1000px;" alt="The Snorkel system workflow diagram"/>
</figure>

In our [recent VLDB paper on Snorkel](https://arxiv.org/abs/1711.10160), we find that in a variety of real-world applications, this new approach to interacting with modern machine learning models seems to work well! Some highlights include:

 -   In a user study, conducted as part of a two-day [workshop on Snorkel](http://mobilize.stanford.edu/events/snorkelworkshop2017/) hosted by the Mobilize center, we compared the productivity of teaching subject matter experts to use Snorkel, versus spending the equivalent time just hand-labeling data. We found that they were able to build models 2.8x faster and with 45.5% better predictive performance on average.
 -   On two real-world text relation extraction tasks--in collaboration with researchers from Stanford, the U.S. Dept. of Veterans Affairs, and the U.S. Food and Drug Administration--and four other benchmark text and image tasks, we found that Snorkel leads to an average 132% improvement over baseline techniques and comes within an average 3.6% of the predictive performance of large hand-curated training sets.
 -   We explored the novel tradeoff space of whether and at what complexity to model the user-provided labeling functions, leading to a rule-based optimizer for accelerating iterative development cycles.


## Next Steps

Various efforts in our lab are already underway to extend the weak supervision interaction model envisioned in Snorkel to other modalities such as [richly-formatted data](https://hazyresearch.github.io/snorkel/blog/fonduer.html), modalities or settings where labeling functions are [difficult to directly write over the raw data](http://dawn.cs.stanford.edu/2017/09/14/coral/), and more! On the technical front, we’re interested in both extending the core data programming model at the heart of Snorkel, making it easier to specify labeling functions with higher-level interfaces such as [natural language](https://hazyresearch.github.io/snorkel/blog/babble_labble.html), as well as combining with other types of weak supervision such as [data augmentation](https://hazyresearch.github.io/snorkel/blog/tanda.html).

Snorkel is an active and ongoing project, so for code, tutorials, blog posts, and more, please check it out at [snorkel.stanford.edu](snorkel.stanford.edu)!