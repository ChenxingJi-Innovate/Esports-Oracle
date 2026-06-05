#!/usr/bin/env python3
"""
Parse completed-match results from a Liquipedia event page's rendered HTML.

Liquipedia's bracket/Swiss matches render as `brkts-matchlist-match` blocks:
each has two opponent cells (team name in <span class="name">, the winner
carrying `brkts-matchlist-slot-winner`) and two score cells. We extract, per
completed match: both team names, both scores (map count for BoX, round count
for Bo1), and which side won. Upcoming matches (no winner slot) are skipped.

This is intentionally tolerant: Liquipedia markup drifts, so we parse leniently
and return only matches where we can identify a winner.
"""
from __future__ import annotations

import re

_MATCH = re.compile(r'brkts-matchlist-match\b.*?(?=brkts-matchlist-match\b|$)', re.S)
# capture the FULL class string (for slot-winner) + the aria-label, which holds
# the canonical full team name ("Lynn Vision Gaming"), not the short span code.
_OPP = re.compile(
    r'class="(?P<cls>brkts-matchlist-cell brkts-matchlist-opponent[^"]*)"'
    r'\s*aria-label="(?P<name>[^"]+)"', re.S)
_SCORE = re.compile(r'matchlist-score[^>]*>\s*<div class="brkts-matchlist-cell-content">([^<]*)</div>', re.S)


def parse_event(html: str) -> list[dict]:
    """Return completed matches: [{teams:[a,b], scores:[sa,sb], winner_name}]."""
    out = []
    for block in _MATCH.findall(html):
        opps = list(_OPP.finditer(block))
        scores = _SCORE.findall(block)
        if len(opps) < 2:
            continue
        names = [o.group("name").strip() for o in opps[:2]]
        winner_idx = next((i for i, o in enumerate(opps[:2])
                           if "slot-winner" in o.group("cls")), None)
        if winner_idx is None:
            continue  # not finished
        sc = [_to_int(s) for s in scores[:2]] if len(scores) >= 2 else [None, None]
        out.append({
            "teams": names,
            "scores": sc,
            "winner_name": names[winner_idx],
        })
    return out


def _to_int(s: str):
    s = s.strip()
    return int(s) if s.isdigit() else None


def _norm(name: str) -> str:
    return re.sub(r'[^a-z0-9]', '', name.lower())


def winner_for(team_a: str, team_b: str, parsed: list[dict]) -> dict | None:
    """Match a prediction's two teams against parsed results (order-insensitive,
    fuzzy on punctuation/case). Returns {result:'a'|'b', score_a, score_b} or None."""
    na, nb = _norm(team_a), _norm(team_b)
    for m in parsed:
        pn = [_norm(t) for t in m["teams"]]
        if {na, nb} != set(pn) and not ({na, nb} <= set(pn)):
            # allow substring matches (e.g. "Lynn Vision" vs "Lynn Vision Gaming")
            if not (_fuzzy(na, pn) and _fuzzy(nb, pn)):
                continue
        # map our a/b onto the parsed order
        a_is_first = _fuzzy(na, [pn[0]])
        winner_is_a = (_norm(m["winner_name"]) == pn[0]) == a_is_first
        sa, sb = m["scores"] if a_is_first else m["scores"][::-1]
        return {"result": "a" if winner_is_a else "b", "score_a": sa, "score_b": sb}
    return None


def _fuzzy(n: str, candidates: list[str]) -> bool:
    return any(n and (n == c or n in c or c in n) for c in candidates)
