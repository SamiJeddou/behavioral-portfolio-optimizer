# Google sign-in → save to your own Drive — setup guide

This connects the **Live Portfolio** "Save / load" feature to Google so each visitor can save
their portfolio to **their own Google Drive**. The app uses the least-privilege **`drive.file`**
scope: it can only see and modify files **it** created — never the rest of anyone's Drive.

You do the Google Cloud Console part once (Anthropic's assistant is not allowed to create OAuth
apps or handle your client secret). The app code only *reads* the credentials from Streamlit
secrets. Total time ≈ 10 minutes.

> The two web addresses the app runs at — register **both** as redirect URIs:
> - `http://localhost:8501/`  (local development)
> - `https://sami-jeddou-behavioral-portfolio-optimizer.streamlit.app/`  (public deployment)

---

## 1 · Create / pick a Google Cloud project
1. Go to https://console.cloud.google.com/ and sign in.
2. Top bar → project dropdown → **New Project** (or use the existing one) → select it.

## 2 · Enable the Google Drive API
1. Left menu → **APIs & Services → Library**.
2. Search **Google Drive API** → open it → **Enable**.

## 3 · Configure the consent screen ("Google Auth Platform")
Google redesigned this area into a **"Google Auth Platform"** section with separate left-menu
pages — **Branding**, **Audience**, **Clients**, **Data access** — instead of one wizard. Open it
via **APIs & Services → OAuth consent screen** (it redirects there). Do these three pages:

**3a · Branding** (`/auth/branding`)
- **App name** → `Beyond Mean-Variance Portfolio Optimiser`. **This is exactly the name users see on
  the Google consent dialog** ("**<App name>** wants access to your Google Account") and on the
  unverified-app screen — keep it identical to the site's public name for trust. Fill **User support
  email** and **Developer contact** email → **Save**.

**3b · Audience** (`/auth/audience`)
- **User type → External.**
- **Publishing status → click "Publish app" to move it to "In production."** This lets *anyone* sign
  in with their own Google account. Because the app uses only the non-sensitive `drive.file` scope
  (see 3c), Google does **not** require a verification review — production goes live immediately.
  (Prefer to keep it private? Leave it in **Testing** and add specific Google addresses under
  **Test users** — only those can sign in, up to 100.)

**3c · Data access** (`/auth/scopes`) — this is where **Scopes** now live.
- Click **Add or remove scopes** → in the panel, use **"Manually add scopes"** and paste these three
  (one per line), then **Add to table → Update**:
  - `openid`
  - `https://www.googleapis.com/auth/userinfo.email`
  - `https://www.googleapis.com/auth/drive.file`  ← the only Drive scope; app-created files only.
- Click **Save**. (If a scope won't add, the Drive API from step 2 isn't enabled yet — enable it and
  retry.) `drive.file` is **non-sensitive** — it reaches only files the app itself creates — so it
  does **not** trigger Google's verification review, even in production.

> **Testing vs In production.** In **Testing**, only the Google accounts you list as test users can
> sign in (up to 100). Setting publishing status to **In production** lets anyone sign in. Because
> this app requests only the non-sensitive `drive.file` scope, **no verification review is required**
> to publish — it goes live straight away. (Verification is only needed for the *sensitive* or
> *restricted* Drive scopes, which this app deliberately avoids.)

## 4 · Create the OAuth client ID
1. Left menu → **Clients** (under Google Auth Platform), or **APIs & Services → Credentials** →
   **Create credentials → OAuth client ID**.
2. Application type **Web application**, name `bmv-streamlit` (this internal name is *not* shown to
   users — only the App name from step 3a is).
3. Under **Authorised redirect URIs → Add URI**, add **both** (exactly, including the trailing `/`):
   - `http://localhost:8501/`
   - `https://sami-jeddou-behavioral-portfolio-optimizer.streamlit.app/`
4. **Create**. Copy the **Client ID** and **Client secret** (you can re-open them any time).

## 5 · Put the credentials in Streamlit secrets
The redirect URI must match the address the app is *currently* served at, so it differs by
environment. Register both URIs in step 4, but set the matching one per environment here.

**Local** — create `C:\PortfolioApp_Phase2\.streamlit\secrets.toml` (this file is git-ignored —
never commit it):
```toml
[google_oauth]
client_id     = "PASTE-CLIENT-ID.apps.googleusercontent.com"
client_secret = "PASTE-CLIENT-SECRET"
redirect_uri  = "http://localhost:8501/"
```

**Streamlit Community Cloud** — app → **Manage app → Settings → Secrets**, paste the same block but
with the deployed redirect URI:
```toml
[google_oauth]
client_id     = "PASTE-CLIENT-ID.apps.googleusercontent.com"
client_secret = "PASTE-CLIENT-SECRET"
redirect_uri  = "https://sami-jeddou-behavioral-portfolio-optimizer.streamlit.app/"
```

## 6 · Test
1. Restart the app (local: stop/start `streamlit run`; Cloud: it redeploys on secret save).
2. Live Portfolio → **Save / load this portfolio** → **Sign in with Google** → approve.
   You should see "**Beyond Mean-Variance Portfolio Optimiser** wants access…".
3. You return signed in. Compute a portfolio → **Save to my Drive**. Check
   https://drive.google.com — you'll see a file like `My portfolio.bmv.json`.
4. **Load from your Drive** lists your saved portfolios; pick one → **Load selected from Drive**.

---

### Notes
- The consent-screen name is **only** the "App name" on the **Branding** page — change it there any
  time (changes can take a few minutes to propagate; a major rename may re-trigger verification).
- If sign-in errors with **redirect_uri_mismatch**, the `redirect_uri` in secrets doesn't exactly
  match a URI registered in step 4 (check the trailing `/` and http vs https).
- The app stores **no** personal data of its own — everything lives in the signed-in user's Drive.
- If `[google_oauth]` is absent, the app silently falls back to the manual JSON download/upload.
- `requests` (already pulled in by `yfinance`) is the only library used for the Drive calls — no
  new dependency. If your `requirements.txt` doesn't list it explicitly, add `requests`.
