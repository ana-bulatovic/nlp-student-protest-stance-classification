# Phase 1: Data Collection

Scripts for collecting comments from social media posts related to student protests in Serbia.

## Folder structure

```
phase1_data_collection/
  common.py                        # shared output format
  download_instagram_comments.py
  download_x_comments.py
  download_tiktok_comments.py
  download_facebook_comments.py
  urls_instagram.txt               # default URL list for Instagram script
  urls_x.txt                       # default URL list for X script
  urls_tiktok.txt                  # default URL list for TikTok script
  urls_facebook.txt                # default URL list for Facebook script
  output/                          # collected data (gitignored)
    instagram/
    x/
    tiktok/
    facebook/
  sessions/                        # login sessions / cookies (gitignored)
```

## Install

From the project root:

```bash
pip install -r requirements.txt
playwright install chromium
```

## Output format

Each run creates two UTF-8 files per post:

- `.txt` — pipe-separated: `id|source|url|text|author|created_at|is_reply|parent_id`
- `.json` — same data with metadata

## Quick start

Each script automatically reads **all URLs** from its own file when run without arguments:

| Script | URL file |
|---|---|
| `download_instagram_comments.py` | `urls_instagram.txt` |
| `download_x_comments.py` | `urls_x.txt` |
| `download_tiktok_comments.py` | `urls_tiktok.txt` |
| `download_facebook_comments.py` | `urls_facebook.txt` |

1. Open the matching `urls_*.txt` file
2. Add one post URL per line
3. Run the script — it processes every URL in the file

```bash
cd phase1_data_collection

# Add URLs to urls_instagram.txt, then:
python download_instagram_comments.py --username YOUR_USER

# Add URLs to urls_x.txt, then:
python download_x_comments.py --username YOUR_USER

# Add URLs to urls_tiktok.txt, then:
python download_tiktok_comments.py --ms-token TOKEN

# Add URLs to urls_facebook.txt, then:
python download_facebook_comments.py --cookies sessions/facebook/cookies.txt
```

Single URL (optional, skips the file):

```bash
python download_x_comments.py "https://x.com/user/status/1234567890" --username YOUR_USER
```

## Where comments come from (per platform)

### Instagram

| What you provide | What the script does |
|---|---|
| Post URL (`/p/` or `/reel/`) | Extracts shortcode, loads comments via iPhone API |
| **sessionid** cookie (recommended) | Logs in without password — enough on its own |

**How to get sessionid (recommended on Windows):**
1. Log in to [instagram.com](https://www.instagram.com) in your browser
2. Press F12 → Application → Cookies → `instagram.com` → `sessionid`
3. Copy the value and save it to `sessions/instagram/sessionid.txt` (one line, no prefix)
4. Run: `python download_instagram_comments.py`

Or pass it directly:
```bash
python download_instagram_comments.py --sessionid "VREDNOST"
```

**Source of comments:** Instagram iPhone API (direct, no GraphQL).

**Saved to:** `output/instagram/`  
**Session:** `sessions/instagram/session-USERNAME` (auto-saved after first login)

If Instagram rate-limits you, increase delays:
```bash
python download_instagram_comments.py --sleep 8 --request-sleep 2
```

---

### X (Twitter)

| What you provide | What the script does |
|---|---|
| Post URL (`/status/TWEET_ID`) | Extracts tweet ID |
| X username + password (or saved cookies) | Logs in via **twikit** |

**Source of comments:** replies to the tweet (X treats replies as the comment thread).

**Saved to:** `output/x/`  
**Session:** `sessions/x/cookies.json`

---

### TikTok

| What you provide | What the script does |
|---|---|
| Video URL | Resolves video ID |
| `ms_token` from browser cookies | Opens a browser session via **TikTokApi + Playwright** |

**How to get `ms_token`:**
1. Log in to [tiktok.com](https://www.tiktok.com) in Chrome
2. Press F12 → Application → Cookies → `tiktok.com`
3. Copy the value of `ms_token`
4. Set env: `TIKTOK_MS_TOKEN=...` or pass `--ms-token`

**Source of comments:** TikTok comment API, accessed through an automated browser session.

**Saved to:** `output/tiktok/`

---

### Facebook

| What you provide | What the script does |
|---|---|
| Public post URL or post ID | Fetches the post page |
| Cookies file (recommended) | Uses **facebook-scraper** with your logged-in session |

**How to get cookies:**
1. Log in to Facebook in the browser
2. Export cookies as Netscape format (browser extension, e.g. "Get cookies.txt")
3. Save to `sessions/facebook/cookies.txt` (needs `c_user` and `xs` values)

**Source of comments:** public comment section of the Facebook post HTML/API.

**Saved to:** `output/facebook/`

---

## What you need to find manually

For each platform, you first collect **post URLs** manually:

1. Search news portals / social media for posts about student protests
2. Copy the link to the post (not the profile)
3. Paste URLs into the matching `urls_*.txt` file (one per line)
4. Run the matching download script (no `--url-file` needed)

## Notes

- Only **public** posts are supported
- Respect platform terms of service and rate limits
- Use `--sleep 3` between posts to reduce blocking
- News portal comments are not included yet (manual copy or separate scraper)
