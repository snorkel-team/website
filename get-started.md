---
layout: homepage
title: Get Started
---
<div class="hero-subheader">
  <div class="container">
    <div class="row row-spacing vertical-align mobile-padding">
      <div class="col-sm-5 mobile-margin">
        <p class="subheadline">INSTALL NOW</p>
        <h1>Get Started</h1>
        <p>
          Run the following to install:
        </p>
        <div class="code-block">
          <p># pip install snorkel</p>
          <span style="color: #9D3FA7;">import</span><span style="color: #18171C;"> snorkel</span>
        </div>
        <a class="btn" href="/use-cases/">Tutorials</a>
        <a class="btn" href="https://github.com/snorkel-team/snorkel-tutorials">GitHub</a>
      </div>
      <div class="col-sm-1"></div>
      <div class="col-sm-6">
        <img src="/doks-theme/assets/images/layout/Pattern 1.png" alt="Pattern 1" />
      </div>
    </div>

<div markdown="1">
# Snorkel 101

## Install Jekyll

### Requirements

Installing Jekyll should be straight-forward if all requirements are met. Before you start, make sure your system has the following:

- GNU/Linux, Unix, or macOS
- Ruby version 2.0 or above, including all development headers
- RubyGems
- GCC and Make (in case your system doesn’t have them installed, which you can check by running `gcc -v` and `make -v` in your system’s command line interface)

### Install with RubyGems

The best way to install Jekyll is via RubyGems. At the terminal prompt, simply run the following command to install Jekyll:

```bash
$ gem install jekyll
```

All of Jekyll’s gem dependencies are automatically installed by the above command, so you won’t have to worry about them at all.

> ##### Full Jekyll documentation
> You can find full Jekyll documentation [here](https://jekyllrb.com).
</div>

    <div class="row row-spacing mobile-padding">
      <h1>Tutorials</h1>
      <div class="light-blue-card-container">
        {% for tutorial in site.use_cases limit:3 %}
        <a class="light-blue-card" href="{{ tutorial.url }}">
          <p class="purple-numbers">{{ forloop.index | prepend: '00' | slice: -2, 2 }}</p>
          <h4>{{ tutorial.title }}</h4>
          <p>{{ tutorial.excerpt }}</p>
          <span class="card-cta">
            LEARN MORE <i class="icon icon--arrow-right"></i>
          </span>
        </a>
        {% endfor %}
      </div>
    </div>
  </div>
</div>
