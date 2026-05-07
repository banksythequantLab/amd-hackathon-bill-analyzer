"""YouTube uploader for the Podcast Studio.

Uses Google's official google-api-python-client + google-auth-oauthlib stack
with a resumable upload and a refresh-token cached to secrets/youtube_token.json.

The first time you use this you must run scripts/youtube_auth.py to do the
one-time OAuth dance (browser pops, you log in as the channel's Google account,
grant the youtube.upload scope, the token is saved). After that this module
loads the saved token + auto-refreshes as needed and never asks again.

ONLY the youtube.upload scope is requested - this lets us upload videos and
edit their metadata, but cannot delete videos or read private channel data.

Quota: videos.insert costs 1,600 units. The default daily quota is 10,000
units, so ~6 uploads/day per Google Cloud project. To increase, request a
quota bump on the YouTube Data API v3 page in the Cloud Console.
"""

from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Callable, Optional

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCOPES = ['https://www.googleapis.com/auth/youtube.upload']
REPO_ROOT = Path(__file__).resolve().parents[2]
SECRETS_DIR = REPO_ROOT / 'secrets'
CLIENT_SECRET_PATH = SECRETS_DIR / 'client_secret.json'
TOKEN_PATH = SECRETS_DIR / 'youtube_token.json'

# ---------------------------------------------------------------------------
# Imports are lazy so the module can be imported even if the Google libraries
# aren't installed yet (the Gradio app should still boot).
# ---------------------------------------------------------------------------
def _import_google_libs():
    try:
        from google.oauth2.credentials import Credentials  # noqa
        from google_auth_oauthlib.flow import InstalledAppFlow  # noqa
        from google.auth.transport.requests import Request  # noqa
        from googleapiclient.discovery import build  # noqa
        from googleapiclient.http import MediaFileUpload  # noqa
        from googleapiclient.errors import HttpError  # noqa
        return True
    except ImportError as e:
        return False


def is_available() -> tuple[bool, str]:
    """Return (ok, reason). UI uses this to show whether upload is wired up."""
    if not _import_google_libs():
        return False, ('google-api-python-client / google-auth-oauthlib not installed. '
                       'Run: pip install google-api-python-client google-auth-oauthlib google-auth-httplib2')
    if not CLIENT_SECRET_PATH.exists():
        return False, (f'Missing {CLIENT_SECRET_PATH} - download OAuth client credentials '
                       f'from Google Cloud Console (Desktop app type).')
    if not TOKEN_PATH.exists():
        return False, (f'Missing {TOKEN_PATH} - run "python scripts/youtube_auth.py" once '
                       f'to authorize uploads to your channel.')
    return True, 'ok'


# ---------------------------------------------------------------------------
# Credential loading (assumes auth dance already happened)
# ---------------------------------------------------------------------------
def load_credentials():
    """Load the saved refresh token, refreshing the access token if expired."""
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request

    if not TOKEN_PATH.exists():
        raise RuntimeError(
            f'No saved YouTube token at {TOKEN_PATH}. Run scripts/youtube_auth.py first.'
        )
    creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        # Persist the refreshed token so we don't re-refresh every time
        TOKEN_PATH.write_text(creds.to_json())
    if not creds.valid:
        raise RuntimeError('Loaded credentials are not valid - re-run scripts/youtube_auth.py')
    return creds


# ---------------------------------------------------------------------------
# Main upload function
# ---------------------------------------------------------------------------
def upload_video(
    video_path: str | Path,
    title: str,
    description: str = '',
    tags: Optional[list[str]] = None,
    privacy: str = 'private',
    category_id: str = '25',  # 25 = News & Politics (suitable for legislative analysis)
    made_for_kids: bool = False,
    notify_subscribers: bool = True,
    log: Callable[[str], None] = print,
) -> dict:
    """Upload a video to YouTube via resumable upload.

    Args:
        video_path: path to the .mp4 file to upload.
        title: max 100 chars; YouTube truncates anything longer.
        description: max 5000 chars.
        tags: list of strings; total tag length max 500 chars.
        privacy: 'private' (default), 'unlisted', or 'public'.
        category_id: YouTube category ID. 25 = News & Politics, 22 = People & Blogs,
                     27 = Education, 28 = Science & Tech.
        made_for_kids: COPPA flag. Set False for political/news content.
        notify_subscribers: send notification to channel subscribers on publish.
        log: callable for streaming progress messages.

    Returns:
        dict with keys: video_id, watch_url, title, privacy, status.
        Raises RuntimeError on auth/setup failures.
    """
    if not _import_google_libs():
        raise RuntimeError(
            'Google API libraries not installed. Run: pip install google-api-python-client '
            'google-auth-oauthlib google-auth-httplib2'
        )

    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
    from googleapiclient.errors import HttpError

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f'Video not found: {video_path}')
    file_size_mb = video_path.stat().st_size / 1024 / 1024
    log(f'[YT] Uploading {video_path.name} ({file_size_mb:.1f} MB) -> "{title}"')

    # Validate inputs
    if privacy not in ('private', 'unlisted', 'public'):
        raise ValueError(f'privacy must be private/unlisted/public, got {privacy!r}')
    title = (title or 'Bill Analysis').strip()[:100]
    description = (description or '')[:5000]
    tags = (tags or [])[:30]

    # Build authenticated service
    creds = load_credentials()
    youtube = build('youtube', 'v3', credentials=creds, cache_discovery=False)

    body = {
        'snippet': {
            'title': title,
            'description': description,
            'tags': tags,
            'categoryId': category_id,
        },
        'status': {
            'privacyStatus': privacy,
            'selfDeclaredMadeForKids': made_for_kids,
        },
    }

    media = MediaFileUpload(
        str(video_path),
        mimetype='video/mp4',
        chunksize=4 * 1024 * 1024,  # 4 MB chunks for fine-grained progress
        resumable=True,
    )

    log(f'[YT] privacy={privacy} category={category_id} '
        f'tags={len(tags)} title_len={len(title)} desc_len={len(description)}')

    request = youtube.videos().insert(
        part='snippet,status',
        body=body,
        media_body=media,
        notifySubscribers=notify_subscribers,
    )

    response = None
    last_progress = -1
    t0 = time.time()
    while response is None:
        try:
            status, response = request.next_chunk()
            if status:
                pct = int(status.progress() * 100)
                if pct != last_progress:  # only log on meaningful change
                    elapsed = time.time() - t0
                    log(f'[YT]   upload {pct}% ({elapsed:.1f}s elapsed)')
                    last_progress = pct
        except HttpError as e:
            # Per Google docs, retryable status codes are 500, 502, 503, 504
            if e.resp.status in (500, 502, 503, 504):
                log(f'[YT]   transient error {e.resp.status}, retrying...')
                time.sleep(2)
                continue
            raise

    elapsed = time.time() - t0
    video_id = response.get('id')
    watch_url = f'https://www.youtube.com/watch?v={video_id}'
    log(f'[YT] Upload complete in {elapsed:.1f}s')
    log(f'[YT]   video_id: {video_id}')
    log(f'[YT]   watch:    {watch_url}')
    log(f'[YT]   privacy:  {privacy} (visible to {"only you" if privacy == "private" else "anyone with link" if privacy == "unlisted" else "the public"})')

    return {
        'video_id': video_id,
        'watch_url': watch_url,
        'title': title,
        'privacy': privacy,
        'status': response.get('status', {}),
        'snippet': response.get('snippet', {}),
        'elapsed_s': round(elapsed, 1),
        'file_size_mb': round(file_size_mb, 1),
    }


# ---------------------------------------------------------------------------
# Convenience: build metadata from a canonical bill report + headline
# ---------------------------------------------------------------------------
def build_metadata_from_report(
    canonical_path: str | Path,
    headline: str,
    creative_direction: str = '',
) -> dict:
    """Run the YouTubeMetadataGenerator agent on a canonical report and return
    a dict of {title, description, tags} ready to feed into upload_video().

    This is a thin wrapper around the agent that exists in src/agents/
    but was never wired into the main pipeline. Now it earns its keep.
    """
    from src.agents.youtube_metadata_generator import YouTubeMetadataGenerator
    canonical_path = Path(canonical_path)
    report = json.loads(canonical_path.read_text())

    agent = YouTubeMetadataGenerator()
    payload = {
        'bill_short': report.get('bill_short', ''),
        'bill_label': report.get('bill_label', ''),
        'headline': headline,
        'creative_direction': creative_direction,
        'summarizer': report.get('agents', {}).get('summarizer', {}).get('output'),
        'pork_finder': report.get('agents', {}).get('pork_finder', {}).get('output'),
        'conflict_spotter': report.get('agents', {}).get('conflict_spotter', {}).get('output'),
        'usc_cross_ref': report.get('agents', {}).get('usc_cross_ref', {}).get('output'),
    }
    result = agent.run(payload)
    out = result.output if hasattr(result, 'output') else result
    return {
        'title': out.get('title', headline)[:100],
        'description': out.get('description', '')[:5000],
        'tags': out.get('tags', [])[:30],
    }
