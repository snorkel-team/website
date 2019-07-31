# Snorkel Docs

### Getting Started

Set up environment as outlined below, then run:

```bash
$ jekyll contentful --config _config.yml,_config.dev.yml
$ jekyll serve
```

### Environment

Create this file before building, replacing [VALUES] with actual values.

###### _config.dev.yml
```yaml
contentful:
  spaces:
    - snorkel:
        space: [CONTENTFUL_SPACE_ID]
        access_token: [CONTENTFUL_ACCESS_TOKEN]
```
