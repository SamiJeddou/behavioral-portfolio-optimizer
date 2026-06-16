# © 2026 Sami Jeddou. All rights reserved.
# Published publicly for demonstration and evaluation only — no license is granted.
# Copying, modification, redistribution, or reuse (in whole or in part) without the
# author's prior written permission is prohibited.
"""Google sign-in + Drive persistence for the Live Portfolio — UI-free.

Implements an OAuth 2.0 Authorization-Code flow (with PKCE) against Google, and the
minimal Drive v3 REST calls needed to save / list / load the user's own portfolio files.

Scope is the least-privileged **drive.file**: the app can only see and modify files it
created itself — never the rest of the user's Drive. No Streamlit, no secrets handling
here (the caller passes client_id / client_secret / redirect_uri, read from st.secrets).

The module-level ``_PENDING`` dict holds the short-lived per-sign-in state (PKCE verifier,
return view) keyed by the OAuth ``state``. Because this module is imported once and cached,
``_PENDING`` survives Streamlit reruns *and* the full-page redirect to Google and back
(same server process), which a Streamlit ``session_state`` does not.
"""
import base64
import hashlib
import json
import os
import time
import urllib.parse

import requests

__all__ = [
    "SCOPES", "make_pkce", "build_auth_url", "exchange_code", "refresh_token",
    "userinfo", "list_portfolios", "save_portfolio", "load_portfolio",
    "pending_put", "pending_pop", "new_state",
]

AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
USERINFO_URL = "https://openidconnect.googleapis.com/v1/userinfo"
DRIVE_FILES = "https://www.googleapis.com/drive/v3/files"
DRIVE_UPLOAD = "https://www.googleapis.com/upload/drive/v3/files"
SCOPES = "openid email profile https://www.googleapis.com/auth/drive.file"

# files this app creates are tagged so we can list only our own portfolios
_APP_KEY = "bmv_live_portfolio"
_TIMEOUT = 25

# state -> {"verifier": str, "return_view": str, "ts": float}; survives reruns + the redirect
_PENDING = {}


def _gc_pending(ttl=1800):
    now = time.time()
    for s in [k for k, v in list(_PENDING.items()) if now - v.get("ts", 0) > ttl]:
        _PENDING.pop(s, None)


def pending_put(state, verifier, return_view="portfolio"):
    _gc_pending()
    _PENDING[state] = {"verifier": verifier, "return_view": return_view, "ts": time.time()}


def pending_pop(state):
    return _PENDING.pop(state, None)


def new_state():
    """A random anti-CSRF state token for the OAuth round-trip."""
    return base64.urlsafe_b64encode(os.urandom(18)).rstrip(b"=").decode()


def make_pkce():
    """Return (code_verifier, code_challenge) for PKCE (S256)."""
    verifier = base64.urlsafe_b64encode(os.urandom(40)).rstrip(b"=").decode()
    digest = hashlib.sha256(verifier.encode()).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
    return verifier, challenge


def build_auth_url(client_id, redirect_uri, state, challenge):
    params = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "scope": SCOPES,
        "state": state,
        "access_type": "offline",      # ask for a refresh token
        "prompt": "consent",
        "include_granted_scopes": "true",
        "code_challenge": challenge,
        "code_challenge_method": "S256",
    }
    return AUTH_URL + "?" + urllib.parse.urlencode(params)


def exchange_code(client_id, client_secret, code, redirect_uri, verifier):
    """Exchange an authorization code for tokens. Returns the token dict (+ obtained_at)."""
    data = {
        "grant_type": "authorization_code",
        "code": code,
        "client_id": client_id,
        "client_secret": client_secret,
        "redirect_uri": redirect_uri,
        "code_verifier": verifier,
    }
    r = requests.post(TOKEN_URL, data=data, timeout=_TIMEOUT)
    r.raise_for_status()
    tok = r.json()
    tok["obtained_at"] = time.time()
    return tok


def refresh_token(client_id, client_secret, refresh):
    data = {
        "grant_type": "refresh_token",
        "refresh_token": refresh,
        "client_id": client_id,
        "client_secret": client_secret,
    }
    r = requests.post(TOKEN_URL, data=data, timeout=_TIMEOUT)
    r.raise_for_status()
    tok = r.json()
    tok["obtained_at"] = time.time()
    return tok


def userinfo(access_token):
    """Identity claims (email, name) for display. Returns {} on failure."""
    try:
        r = requests.get(USERINFO_URL, headers=_hdr(access_token), timeout=_TIMEOUT)
        return r.json() if r.ok else {}
    except Exception:
        return {}


def _hdr(access_token):
    return {"Authorization": "Bearer " + access_token}


def _fname(name):
    safe = (name or "portfolio").strip().replace("/", "-").replace("\\", "-")
    return safe + ".bmv.json"


def list_portfolios(access_token):
    """List the portfolio files this app created in the user's Drive (drive.file scope)."""
    params = {
        "q": "appProperties has { key='%s' and value='1' } and trashed=false" % _APP_KEY,
        "spaces": "drive",
        "fields": "files(id,name,modifiedTime)",
        "orderBy": "modifiedTime desc",
        "pageSize": "100",
    }
    r = requests.get(DRIVE_FILES, headers=_hdr(access_token), params=params, timeout=_TIMEOUT)
    r.raise_for_status()
    return r.json().get("files", [])


def save_portfolio(access_token, name, obj, file_id=None):
    """Create (or update, if file_id given) a JSON portfolio file in the user's Drive.
    Returns the Drive file id."""
    content = json.dumps(obj, indent=2)
    if file_id:
        r = requests.patch(
            "%s/%s?uploadType=media" % (DRIVE_UPLOAD, file_id),
            headers={**_hdr(access_token), "Content-Type": "application/json"},
            data=content.encode("utf-8"), timeout=_TIMEOUT,
        )
        r.raise_for_status()
        return r.json().get("id", file_id)
    metadata = {
        "name": _fname(name),
        "mimeType": "application/json",
        "appProperties": {_APP_KEY: "1"},
    }
    boundary = "bmv-drive-boundary-8f2a1c"
    body = (
        "--%s\r\nContent-Type: application/json; charset=UTF-8\r\n\r\n%s\r\n"
        "--%s\r\nContent-Type: application/json\r\n\r\n%s\r\n--%s--"
        % (boundary, json.dumps(metadata), boundary, content, boundary)
    )
    r = requests.post(
        "%s?uploadType=multipart&fields=id" % DRIVE_UPLOAD,
        headers={**_hdr(access_token), "Content-Type": "multipart/related; boundary=%s" % boundary},
        data=body.encode("utf-8"), timeout=_TIMEOUT,
    )
    r.raise_for_status()
    return r.json().get("id")


def load_portfolio(access_token, file_id):
    """Download and parse a portfolio JSON file by id."""
    r = requests.get("%s/%s?alt=media" % (DRIVE_FILES, file_id),
                     headers=_hdr(access_token), timeout=_TIMEOUT)
    r.raise_for_status()
    return r.json()
