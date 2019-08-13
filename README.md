# Snorkel Docs

### Getting Started

* [Install Jekyll.](https://jekyllrb.com/docs/installation/macos/) Note the special instructions for macOS Mojave!
* Clone this repo
* Run the following
```bash
$ bundle install
$ jekyll serve
```

If the above fails to work, try 
```bash
$ bundle install
$ bundle exec jekyll serve
```

If the above (still) fails to work, edit the `Gemfile` to include 
`gem 'github-pages', group: :jekyll_plugins`
in the second line.
