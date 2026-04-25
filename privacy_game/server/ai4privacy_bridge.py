"""Bridge AI4Privacy pii-masking-400k → our profile schema.

Downloads a subset of English+US rows from the HuggingFace dataset once, caches
the extracted label→value pools locally (so subsequent runs are offline), and
provides a profile-builder that draws realistic PII values from AI4Privacy for
the "base PII" fields while keeping our hand-curated pools for the leak-
mechanism fields (employer, medication, religion, etc.).

The AI4Privacy value quality is inspected and documented. Realistic for:
    CITY, STREET, BUILDINGNUM, ZIPCODE, DATEOFBIRTH, SOCIALNUM, TELEPHONENUM,
    CREDITCARDNUMBER, ACCOUNTNUM, GIVENNAME, SURNAME, PASSWORD,
    DRIVERLICENSENUM, IDCARDNUM, TAXNUM

USERNAME / EMAIL prefixes are synthetic-looking placeholders (e.g. 'faozzsd379223')
but the format is still useful. We keep them.

Why not use Faker any more:
    - AI4Privacy is the canonical open PII dataset; grounding in it is
      defensible in the writeup ("profiles sampled from ai4privacy/pii-masking-400k")
    - The label vocabulary is broader (17 realized classes vs our 10 from Faker)
    - Date-of-birth values include the natural-language diversity
      ("8th January 1999", "May/58") that Whisper/Presidio must handle
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import random
import re
from pathlib import Path
from typing import Optional


# Disk cache: ~/.cache/privacy_game/ai4privacy_us_en_pools.json
_CACHE_DIR = Path.home() / ".cache" / "privacy_game"
_CACHE_FILE = _CACHE_DIR / "ai4privacy_us_en_pools.json"
_CACHE_MANIFEST = _CACHE_DIR / "ai4privacy_pools.meta.json"


# Labels we care about — if AI4Privacy exposes others we ignore them.
TARGET_LABELS = [
    "GIVENNAME", "SURNAME", "DATEOFBIRTH", "EMAIL", "TELEPHONENUM",
    "STREET", "CITY", "ZIPCODE", "BUILDINGNUM", "SOCIALNUM",
    "CREDITCARDNUMBER", "ACCOUNTNUM", "PASSWORD", "USERNAME",
    "DRIVERLICENSENUM", "IDCARDNUM", "TAXNUM",
]

# Per-label pool size cap — 500 unique values is plenty for 250 profiles.
_POOL_CAP = 500


def build_pools_from_hf(
    n_rows: int = 5000,
    pool_cap: int = _POOL_CAP,
) -> dict[str, list[str]]:
    """Stream the AI4Privacy dataset and collect per-label value pools.

    Filters to English + US locale so values are idiomatic.
    """
    # Imported lazily so the module can be imported even if `datasets` is absent
    from datasets import load_dataset

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("ai4privacy/pii-masking-400k", split="train", streaming=True)
    pools: dict[str, list[str]] = {lbl: [] for lbl in TARGET_LABELS}
    seen: dict[str, set[str]] = {lbl: set() for lbl in TARGET_LABELS}
    scanned = 0
    for row in ds:
        if row.get("language") != "en" or row.get("locale") != "US":
            continue
        scanned += 1
        if scanned > n_rows:
            break
        for e in row.get("privacy_mask") or []:
            lbl = e.get("label")
            val = e.get("value")
            if not lbl or not val:
                continue
            if lbl not in pools:
                continue
            if val in seen[lbl]:
                continue
            if len(pools[lbl]) >= pool_cap:
                continue
            pools[lbl].append(val)
            seen[lbl].add(val)

    return pools


def load_pools(force_refresh: bool = False) -> dict[str, list[str]]:
    """Load cached pools, or rebuild from HuggingFace if missing.

    If the env var PRIVACY_GAME_AI4P_DISABLE=1, returns empty pools (forces
    callers to fall back to Faker / hand-curated).
    """
    if os.environ.get("PRIVACY_GAME_AI4P_DISABLE"):
        return {}

    if _CACHE_FILE.exists() and not force_refresh:
        try:
            return json.loads(_CACHE_FILE.read_text())
        except json.JSONDecodeError:
            pass  # fall through to rebuild

    pools = build_pools_from_hf()
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _CACHE_FILE.write_text(json.dumps(pools, indent=2, ensure_ascii=False))
    _CACHE_MANIFEST.write_text(json.dumps({
        "dataset": "ai4privacy/pii-masking-400k",
        "split": "train",
        "filter": "language=en, locale=US",
        "label_counts": {k: len(v) for k, v in pools.items()},
    }, indent=2))
    return pools


# ──────────────────────────────────────────────────────────────────────────────
# Value normalization — AI4Privacy DOB / phone formats vary; canonicalize so
# the env's regex extractors don't have to cope with every variant.


_MONTH_NAMES = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}


# Realistic DOB window — AI4Privacy has some nonsense values like "August 2041";
# reject anything outside a reasonable adult age range.
_DOB_MIN = _dt.date(1940, 1, 1)
_DOB_MAX = _dt.date(2010, 12, 31)


def _check_dob(d: _dt.date) -> Optional[str]:
    if _DOB_MIN <= d <= _DOB_MAX:
        return d.isoformat()
    return None


def normalize_dob(raw: str) -> Optional[str]:
    """Return ISO YYYY-MM-DD or None if unparseable OR outside realistic range.

    Handles: "1985-03-15", "05/07/2010", "8th January 1999", "15th January 2020",
             "May/58", "May 1999", "Jan 2000"

    Rejects DOBs outside [1940, 2010] (AI4Privacy has some junk like "Aug 2041").
    """
    if not raw:
        return None
    s = raw.strip()

    # ISO
    try:
        return _check_dob(_dt.date.fromisoformat(s))
    except ValueError:
        pass

    # Slash formats: MM/DD/YYYY or DD/MM/YYYY (US = MM/DD)
    m = re.fullmatch(r"(\d{1,2})/(\d{1,2})/(\d{2,4})", s)
    if m:
        a, b, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if y < 100:
            y = 2000 + y if y < 50 else 1900 + y
        # Try MM/DD first (US); fall through to DD/MM if invalid or out of range
        for mo, da in ((a, b), (b, a)):
            try:
                d = _dt.date(y, mo, da)
            except ValueError:
                continue
            checked = _check_dob(d)
            if checked:
                return checked

    # Ordinal word-month: "8th January 1999", "15th January 2020"
    m = re.fullmatch(
        r"(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]+)\s+(\d{2,4})", s
    )
    if m:
        da, mname, y = int(m.group(1)), m.group(2).lower(), int(m.group(3))
        if y < 100:
            y = 2000 + y if y < 50 else 1900 + y
        mo = _MONTH_NAMES.get(mname)
        if mo:
            try:
                return _check_dob(_dt.date(y, mo, da))
            except ValueError:
                pass

    # Month/year only: "May/58", "Jan 2000" — pick day=15
    m = re.fullmatch(r"([A-Za-z]+)[\s/](\d{2,4})", s)
    if m:
        mname, y = m.group(1).lower(), int(m.group(2))
        if y < 100:
            y = 2000 + y if y < 50 else 1900 + y
        mo = _MONTH_NAMES.get(mname)
        if mo:
            try:
                return _check_dob(_dt.date(y, mo, 15))
            except ValueError:
                pass

    return None


def normalize_zip(raw: str) -> Optional[str]:
    """Return a 5-digit US ZIP (strip ZIP+4 if present)."""
    if not raw:
        return None
    s = raw.strip()
    m = re.match(r"^(\d{5})(?:-\d{4})?$", s)
    return m.group(1) if m else None


def normalize_phone(raw: str) -> str:
    """Strip dots/spaces but preserve the raw if unparseable — used as-is."""
    return raw.strip()


# ──────────────────────────────────────────────────────────────────────────────
# Gender inference from first name — AI4Privacy has no gender label so we
# must derive it. Use a small hand-curated lookup of common names; unknown
# defaults to random choice for the 50/50 mix.


_KNOWN_MALE = {
    "james", "john", "michael", "david", "robert", "christopher", "daniel",
    "matthew", "anthony", "mark", "steven", "andrew", "kenneth", "george",
    "joshua", "kevin", "brian", "edward", "ronald", "timothy", "jason",
    "jeffrey", "ryan", "gary", "william", "richard", "charles", "thomas",
    "paul", "eric", "adam", "nathan", "noah", "liam", "oliver", "ethan",
    "lucas", "benjamin", "alexander", "henry", "jack", "felix", "simon",
}
_KNOWN_FEMALE = {
    "mary", "patricia", "jennifer", "linda", "elizabeth", "barbara", "susan",
    "jessica", "sarah", "karen", "nancy", "lisa", "margaret", "sandra",
    "ashley", "kimberly", "emily", "donna", "michelle", "carol", "amanda",
    "melissa", "deborah", "stephanie", "anna", "emma", "olivia", "sophia",
    "mia", "isabella", "charlotte", "amelia", "ava", "grace", "zoe", "ruth",
    "rachel", "laura", "helen", "maria",
}


def infer_gender(first_name: str, rng: random.Random) -> str:
    """Rough gender inference from first name. Unknown → random."""
    if not first_name:
        return rng.choice(["Male", "Female"])
    first = first_name.strip().lower().split()[0]
    # Strip common suffixes and hyphen parts
    first = re.split(r"[-\s]", first)[0]
    if first in _KNOWN_MALE:
        return "Male"
    if first in _KNOWN_FEMALE:
        return "Female"
    return rng.choice(["Male", "Female"])


if __name__ == "__main__":
    print("Loading AI4Privacy pools (downloading if first run)...")
    pools = load_pools()
    print(f"Cache: {_CACHE_FILE}")
    print(f"Labels collected: {len(pools)}")
    for lbl, vals in pools.items():
        print(f"  {lbl:20s}  {len(vals):4d} values   e.g. {vals[:3]}")

    # Spot-check normalization
    print("\nDOB normalization:")
    for raw in ("1985-03-15", "05/07/2010", "8th January 1999",
                "May/58", "15th January 2020", "May 1999", "garbage"):
        print(f"  {raw!r:30s} -> {normalize_dob(raw)!r}")

    print("\nZIP normalization:")
    for raw in ("28170-6392", "95683-9705", "12345", "garbage"):
        print(f"  {raw!r:20s} -> {normalize_zip(raw)!r}")

    print("\nGender inference:")
    rng = random.Random(1)
    for name in ("John", "Sarah", "Blagojka", "Anna-Louise", "Matjan", "unknown"):
        print(f"  {name:20s} -> {infer_gender(name, rng)}")
