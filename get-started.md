---
layout: homepage
title: Get Started
---

<div class="hero-subheader">
  <div class="container">
    <div class="row double-row-spacing vertical-align mobile-padding">
      <div class="col-sm-5 mobile-margin">
        <p class="subheadline">INSTALL NOW</p>
        <h1>Get Started</h1>
        <div class="code-block">
          <p># For pip users<br>pip install https://github.com/snorkel-team/snorkel/releases/download/v0.9.0/snorkel-0.9.0-py3-none-any.whl</p>
          <p># For conda users<br>conda install snorkel -c conda-forge</p>
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
    {% include_relative _getting_started/getting_started.md %}
  </div>
  <br>
  <br>
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


      <div class="col-sm-12 all-tweets">
        <a href="/use-cases/">SEE ALL TUTORIALS <i
            class="icon icon--arrow-right"></i></a>
      </div>
  </div>
</div>
