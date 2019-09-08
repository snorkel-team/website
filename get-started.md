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
        <div class="code-block">
          <p># For pip users<br>pip install snorkel</p>
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

      <div class="nav-grid-light-blue">
        <div class="row">
          {% assign cases = site.use_cases | sort: 'order' %}
          {% for tutorial in cases limit:3 %}
            <div class="col-sm-6 col-lg-4">
              <a href="{% if jekyll.environment == 'production' %}{{
                  site.doks.baseurl
                }}{% endif %}{{ tutorial.url }}" class="nav-grid__item_light_blue">
                <div class="nav-grid__content" data-mh>
                  <p class="purple-numbers">{{ forloop.index | prepend: '00' | slice: -2, 2 }}</p>
                  {% if tutorial.category %}
                  <p class="purple">{{ tutorial.category }}</p>
                  {% endif %}
                  <h2 class="nav-grid__title">{{ tutorial.title }}</h2>
                  <p>{{ tutorial.excerpt }}</p>
                </div>
                <p class="nav-grid__btn_light_blue">
                  {{ tutorial.cta | default: "READ MORE" }}
                  <i class="icon icon--arrow-right"></i>
                </p>
              </a>
            </div>
          {% endfor %}
        </div>
      </div>



      <div class="col-sm-12 all-tweets">
        <a href="/use-cases/">SEE ALL TUTORIALS <i
            class="icon icon--arrow-right"></i></a>
      </div>
  </div>
</div>
