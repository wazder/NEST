# NEST Website

NEST (Neural EEG Sequence Transducer) project website.

## Local Development

```bash
cd website
python -m http.server 8000
```

Then visit http://localhost:8000

## Deployment to Cloudflare Pages

### Option 1: Cloudflare Dashboard (Recommended)

1. Go to [Cloudflare Dashboard](https://dash.cloudflare.com)
2. Select your account → Pages → Create a project
3. Connect to Git repository (`wazder/NEST`)
4. Configure:
   - **Build command:** (leave empty)
   - **Build output directory:** `website`
   - **Root directory:** `/`
5. Deploy

### Option 2: Wrangler CLI

```bash
# Install wrangler
npm install -g wrangler

# Login
wrangler login

# Deploy
wrangler pages deploy website --project-name=nest-eeg
```

## Custom Domain Setup

1. In Cloudflare Pages project settings → Custom domains
2. Add `nest-eeg.org`
3. DNS will be automatically configured if domain is on Cloudflare

## Structure

```
website/
├── index.html          # Landing page
├── css/
│   └── style.css       # Main stylesheet
├── js/
│   └── main.js         # Interactive functionality
├── images/             # (placeholder for images)
└── pages/
    ├── demo.html       # Interactive demo
    ├── docs.html       # Documentation
    ├── about.html      # About/Architecture
    ├── research.html   # Research paper
    ├── download.html   # Downloads/Models
    └── contributing.html # Contributing guide
```
