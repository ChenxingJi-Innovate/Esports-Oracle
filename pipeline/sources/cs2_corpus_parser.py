#!/usr/bin/env python3
"""
Parse DATED completed CS2 matches from a Liquipedia event page's rendered HTML.

This extends results_parser (which returns teams/scores/winner only) with the
one thing a leakage-free case base needs: each match's UTC date. Liquipedia
renders every matchlist match with a timer-object carrying a unix
`data-timestamp`, so we pull it from inside each match block.

We parse only `brkts-matchlist-match` blocks (Swiss / group-stage matchlists),
which carry the bulk of tier-1 games. Bracket (playoff) cards use a different
`brkts-popup` structure and are intentionally skipped: they are few, and mixing
two fragile parsers would cost reliability for little corpus volume.

Returns, per completed match:
    {"ts": int, "date": "YYYY-MM-DD", "teams": [a, b],
     "scores": [sa, sb], "winner_name": str}
Upcoming matches (no winner slot, or a future/zero timestamp) are dropped.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone

_MATCH = re.compile(r'brkts-matchlist-match\b.*?(?=brkts-matchlist-match\b|$)', re.S)
_OPP = re.compile(
    r'class="(?P<cls>brkts-matchlist-cell brkts-matchlist-opponent[^"]*)"'
    r'\s*aria-label="(?P<name>[^"]+)"', re.S)
_SCORE = re.compile(
    r'matchlist-score[^>]*>\s*<div class="brkts-matchlist-cell-content">([^<]*)</div>', re.S)
_TS = re.compile(r'data-timestamp="(\d+)"')


def _to_int(s: str):
    s = s.strip()
    return int(s) if s.isdigit() else None


# League / group-stage matches render not as brackets but as `wikitable
# match-card` rows: <tr class="Match"> with TeamLeft / Score / TeamRight cells.
# This is the bulk of regular-season tier-1 data (VCT leagues, LoL splits), so
# we parse it too and merge with the bracket results.
_MCARD_ROW = re.compile(r'<tr class="Match">.*?</tr>', re.S)
_HL = re.compile(r'data-highlighting-class="([^"]+)"')
_MCARD_SCORE = re.compile(
    r'line-height:1\.1">\s*(?:<b>)?(\d+)(?:</b>)?\s*:\s*(?:<b>)?(\d+)(?:</b>)?')


def parse_league_matches(html: str) -> list[dict]:
    """Completed, dated matches from a Liquipedia `match-card` wikitable."""
    out = []
    for row in _MCARD_ROW.findall(html):
        ts_m = _TS.search(row)
        teams = _HL.findall(row)
        sc = _MCARD_SCORE.search(row)
        if not ts_m or len(teams) < 2 or not sc:
            continue
        sa, sb = int(sc.group(1)), int(sc.group(2))
        if sa == sb:
            continue  # tie/unplayed placeholder -> not a finished BoX
        ts = int(ts_m.group(1))
        if ts <= 0:
            continue
        a, b = teams[0].strip(), teams[1].strip()
        out.append({
            "ts": ts,
            "date": datetime.fromtimestamp(ts, tz=timezone.utc).date().isoformat(),
            "teams": [a, b],
            "scores": [sa, sb],
            "winner_name": a if sa > sb else b,
        })
    return out


def parse_event_dated(html: str) -> list[dict]:
    """Completed, dated matchlist matches from one event page's HTML."""
    out = []
    for block in _MATCH.findall(html):
        opps = list(_OPP.finditer(block))
        if len(opps) < 2:
            continue
        winner_idx = next((i for i, o in enumerate(opps[:2])
                           if "slot-winner" in o.group("cls")), None)
        if winner_idx is None:
            continue  # not finished

        ts_m = _TS.search(block)
        ts = int(ts_m.group(1)) if ts_m else 0
        if ts <= 0:
            continue  # undated card; cannot place in time, so skip

        names = [o.group("name").strip() for o in opps[:2]]
        scores = _SCORE.findall(block)
        sc = [_to_int(s) for s in scores[:2]] if len(scores) >= 2 else [None, None]
        d = datetime.fromtimestamp(ts, tz=timezone.utc).date()
        out.append({
            "ts": ts,
            "date": d.isoformat(),
            "teams": names,
            "scores": sc,
            "winner_name": names[winner_idx],
        })
    return out
