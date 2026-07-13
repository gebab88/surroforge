# SurroForge licensing (vendor side)

Issues the signed `SFL1.…` license tokens that `surroforge_professional`
validates offline (M3). This directory is dev-tooling — it is never packaged.

## One-time setup

    .venv/bin/python tools/licensing/issue.py keygen

Writes the private signing key to `~/.surroforge-vendor/signing.key`
(hex seed, chmod 600) and prints the public key that is embedded in
`surroforge_professional/license.py` (`_TRUSTED_KEYS`).

**BACK UP THE PRIVATE KEY.** It never enters the repository. If it is lost, no
new licenses can be issued for existing installations — recovery means key
rotation: generate a new pair, append the new public key to `_TRUSTED_KEYS`,
and ship a new professional release.

## Issue a license

    .venv/bin/python tools/licensing/issue.py issue --holder "ACME GmbH" \
        --tier professional --expires 2027-07-13

Prints the token (one line). Omit `--expires` for a perpetual license. The
customer activates it with `surroforge license install <token>` (or the
`SURROFORGE_PRO_LICENSE` env var, which takes precedence).

## Inspect a token

    .venv/bin/python tools/licensing/issue.py inspect <token-or-.lic-file>

Decodes the payload and checks the signature against the installed package's
trusted keys (`--pub HEX` to override). Expiry is judged customer-side by
`surroforge license status`.
