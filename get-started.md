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
          Run any of the following to install:
        </p>
        <div class="code-block">
         <p>pip install snorkel</p>
          <p>conda install snorkel -c conda-forge</p>
          <!-- <span style="color: #9D3FA7;">import</span><span style="color: #18171C;"> snorkel</span> -->
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
    {% include_relative _use_cases/getting_started.md %}
  </div>
  <br>

    <div class="row row-spacing mobile-padding">
      <h1>Tutorials</h1>
      <div class="light-blue-card-container">
        {% assign cases = site.use_cases | sort: 'order' %}
        {% for tutorial in cases limit:3 %}
        <a class="light-blue-card" href="{{ tutorial.url }}">
          <p class="purple-numbers">{{ forloop.index | prepend: '00' | slice: -2, 2 }}</p>
          <h4>{{ tutorial.title }}</h4>
          <p>{{ tutorial.excerpt }}</p>
          <span class="card-cta">
            READ MORE <i class="icon icon--arrow-right"></i>
          </span>
        </a>
        {% endfor %}
      </div>
    </div>
  </div>
</div>
