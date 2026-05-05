# Custom Domain Setup: bills.nota.lawyer -> GitHub Pages

This wires the `bills` subdomain on `nota.lawyer` (Cloudflare-managed) to the
GitHub Pages site at `banksythequantlab.github.io/amd-hackathon-bill-analyzer/`.

End state: `https://bills.nota.lawyer/` serves the same landing page that
currently lives at `https://banksythequantlab.github.io/amd-hackathon-bill-analyzer/`.

## Three steps total. Estimated 5 minutes.

### Step 1: Add CNAME record in Cloudflare

1. Open https://dash.cloudflare.com -> select `nota.lawyer` zone
2. Sidebar -> DNS -> Records -> "Add record"
3. Fill in:
    Type:    CNAME
    Name:    bills
    Target:  banksythequantlab.github.io
    Proxy:   DNS only (gray cloud, NOT orange)
    TTL:     Auto
4. Save.

IMPORTANT: Proxy must be DNS-only (gray cloud) for the initial GH Pages
TLS handshake to work. After GH issues the cert (usually within a few
minutes), you can flip the proxy to orange if you want Cloudflare's
caching/DDoS layer in front. Doing it orange-first will fail with a
TLS handshake error.

### Step 2: Set custom domain in GitHub repo settings

1. Open https://github.com/banksythequantLab/amd-hackathon-bill-analyzer/settings/pages
2. "Custom domain" field -> enter: `bills.nota.lawyer`
3. Click Save
4. Wait 1-2 minutes for the DNS check to pass. GitHub will then
   automatically issue a Let's Encrypt cert (takes another 5-10 min).
5. Once both checks pass (green), tick "Enforce HTTPS"

The `docs/CNAME` file in this repo is already populated with
`bills.nota.lawyer`, which tells GitHub Pages which custom domain to
serve from. No further code changes needed.

### Step 3: Verify

After ~5-10 minutes total:

    curl -I https://bills.nota.lawyer/
    -> should return HTTP 200 with `server: GitHub.com` header

If it fails, run the same against the GH Pages URL to confirm the
underlying site is still serving:

    curl -I https://banksythequantlab.github.io/amd-hackathon-bill-analyzer/

## Updating the BiP #2 post

Once the custom domain is live, update the LinkedIn post copy in
`docs/bip-post-2-FINAL.md` to use `https://bills.nota.lawyer/` instead
of the GH Pages URL. The rest of the post stays identical.

## Reverting

To remove the custom domain:
1. GitHub repo Settings -> Pages -> Custom domain: clear the field
2. Cloudflare DNS -> delete the `bills` CNAME record
3. Delete `docs/CNAME` from the repo

The site keeps serving from the GH Pages URL the whole time.

## Why subdomain not path

The two ways to put GH Pages content on `nota.lawyer` were:
  (a) Subdomain: bills.nota.lawyer  (this approach)
  (b) Path:      nota.lawyer/bills/* via a Cloudflare Worker proxy

Subdomain is cleaner because:
- Single CNAME record vs Worker code + route
- No proxy invocation costs
- Doesn't conflict with whatever lives at nota.lawyer root
- GitHub Pages issues the TLS cert directly (no cert juggling)

Worker proxy is the right answer if you ever need both the legal
practice site AND a hackathon demo to share `nota.lawyer/*`. For now,
subdomain wins on simplicity.
