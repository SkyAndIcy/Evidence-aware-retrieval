"""Scrub AppIds from config.yaml before submission.

Replaces the 10-AppId pool with a single ${LLM_APPID} placeholder so the
config can be safely included in supplementary materials without leaking
AppIds (which would de-anonymize the submission and violate the gateway's
internal-only usage policy).

Usage:
    python tools/scrub_appids.py            # rewrites config.yaml in place
    python tools/scrub_appids.py --check    # exit 1 if any AppId still present

Run this BEFORE packaging any code supplement for OpenReview.
"""
from __future__ import annotations
import sys
import re
from pathlib import Path

CONFIG = Path(__file__).resolve().parent.parent / "config.yaml"
# The 10 AppIds currently in the pool (zpl11..zpl20).
APPIDS = []  # populate with your AppIds if you want --check to verify scrubbing


def main():
    check = "--check" in sys.argv
    text = CONFIG.read_text()
    remaining = [a for a in APPIDS if a in text]
    if check:
        if remaining:
            print(f"FAIL: {len(remaining)} AppId(s) still present in {CONFIG}: "
                  f"{remaining}")
            sys.exit(1)
        print("OK: no known AppIds in config.yaml")
        sys.exit(0)

    # Replace the _appid_pool block (the list of 10 ids) with a single placeholder.
    # Match the YAML anchor block:
    #   _appid_pool: &appid_pool
    #     - "id1"
    #     ...
    #     - "id10"
    pattern = re.compile(
        r"_appid_pool:\s*&appid_pool\n(?:\s*-\s*\"\d+\"\s*\n)+",
        re.MULTILINE,
    )
    replacement = (
        "_appid_pool: &appid_pool\n"
        '    - "${LLM_APPID}"   # set env var; pool optional\n'
    )
    new_text, n = pattern.subn(replacement, text, count=1)
    if n == 0:
        # Already scrubbed or structure changed — fall back to per-id replace.
        for a in APPIDS:
            new_text = new_text.replace(f'"{a}"', '"${LLM_APPID}"')
    CONFIG.write_text(new_text)
    still = [a for a in APPIDS if a in new_text]
    if still:
        print(f"WARNING: could not remove {len(still)} AppId(s): {still}")
        sys.exit(1)
    print(f"Scrubbed {CONFIG}. AppIds replaced with ${{LLM_APPID}}.")


if __name__ == "__main__":
    main()
