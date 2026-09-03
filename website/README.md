# NEST website

Static site for nest-eeg.org, deployed on Cloudflare Pages (build output directory: `website`).

## Structure

```
website/
├── index.html            # Turkish landing page (primary)
├── pages/                # Turkish pages: technology, research, demo, docs, about, download, contributing
├── en/                   # English mirror (en/index.html, en/pages/*)
├── css/site.css          # design system (Inter, square corners, light/dark tokens, one accent)
├── js/site.js            # theme toggle, nav, reveal, illustrative EEG canvas, demo typing, QA hook
├── images/               # logo SVGs (mark, wordmark, lockup)
├── favicon.svg, favicon-32.png, apple-touch-icon.png, icon-192.png, icon-512.png, og.png
├── scripts/qa.py         # layout QA: renders every page at 3 viewports x 2 themes in headless Chrome
├── _headers              # Cloudflare Pages headers
└── wrangler.toml
```

## Local preview

```bash
cd website
python3 -m http.server 8000
# open http://localhost:8000
```

## Layout QA

Requires Google Chrome. Checks horizontal overflow, clipped text, broken images, single h1,
font loading and local links on every page.

```bash
python3 scripts/qa.py
```

## Content rules

Every number on the site must trace to the repository (`docs/STATUS.md`) or a cited source.
No performance figure is published until it is measured with free decoding on held-out
subjects with a noise-input control. Where a value is not measured, the site says "pending".
