"""One-time YouTube OAuth authorization.

Run this once after dropping client_secret.json into secrets/. It will:
  1. Open your browser to Google's OAuth consent page
  2. Ask you to log in as your YouTube channel's Google account
       (for @BanksytheQuant: log in as the Google account that owns that channel)
  3. Ask you to grant the youtube.upload scope
  4. Save the resulting refresh token to secrets/youtube_token.json

After this finishes, the Gradio app + the upload module use the saved token
automatically (with auto-refresh of access tokens) and never prompt again
unless the refresh token is revoked.

USAGE:
    python scripts/youtube_auth.py

PREREQUISITES (one-time setup in Google Cloud Console):
    1. Create or select a project at https://console.cloud.google.com
    2. APIs & Services -> Library -> enable "YouTube Data API v3"
    3. APIs & Services -> OAuth consent screen:
         - User type: External
         - App name: anything (e.g., "Banksy Bill Analyzer")
         - User support email: your email
         - Developer contact: your email
         - Scopes: add ".../auth/youtube.upload"
         - Test users: add the Gmail account that owns the @BanksytheQuant channel
    4. APIs & Services -> Credentials -> Create Credentials -> OAuth client ID:
         - Application type: Desktop app
         - Name: anything
         - Download the JSON, rename to "client_secret.json"
         - Place it in: secrets/client_secret.json (next to this script's dir)
    5. Run this script.

The browser flow may say "Google hasn't verified this app" - that's fine,
the app is in test mode. Click Advanced -> "Go to <App Name> (unsafe)" to
proceed. Only test users (which now includes you) can authorize while in
test mode. To remove the warning later, push the OAuth consent screen
through Google's verification process - not necessary for personal use.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.tools.youtube_uploader import (
    SCOPES, CLIENT_SECRET_PATH, TOKEN_PATH, SECRETS_DIR
)


def main():
    SECRETS_DIR.mkdir(exist_ok=True, parents=True)

    if not CLIENT_SECRET_PATH.exists():
        print(f'\nERROR: {CLIENT_SECRET_PATH} does not exist.\n')
        print('Follow the one-time setup steps in the docstring of this file.')
        print('Short version:')
        print('  1. Create a Google Cloud project + enable YouTube Data API v3')
        print('  2. Create OAuth client ID (Desktop app) in APIs & Services -> Credentials')
        print('  3. Download the JSON, rename to client_secret.json')
        print(f'  4. Move it to {CLIENT_SECRET_PATH}')
        print('  5. Re-run this script.')
        sys.exit(1)

    # Lazy import so the file at least PARSES without google libs installed
    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError:
        print('\nERROR: required Google libraries not installed.\n')
        print('Install:  pip install google-api-python-client google-auth-oauthlib google-auth-httplib2')
        sys.exit(1)

    print(f'Loading client secrets from: {CLIENT_SECRET_PATH}')
    print(f'Requesting scopes: {SCOPES}')
    print('Opening browser for OAuth consent...')
    print('  (Sign in as the Google account that owns the @BanksytheQuant YouTube channel)')

    flow = InstalledAppFlow.from_client_secrets_file(str(CLIENT_SECRET_PATH), SCOPES)
    creds = flow.run_local_server(port=0, prompt='consent', open_browser=True)

    TOKEN_PATH.write_text(creds.to_json())
    print(f'\n  Saved refresh token to: {TOKEN_PATH}')
    print('\nFrom now on the Gradio app uploads to YouTube without prompting.')
    print('To force re-auth (e.g. switching to a different channel), delete that file and re-run.')


if __name__ == '__main__':
    main()
