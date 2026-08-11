#!/usr/bin/env python3
"""Fail-closed static checks for the RRCL TMLR manuscript.

This audit does not validate numerical artifacts or compile LaTeX.  It catches
claim-discipline regressions that can be checked from the tracked manuscript:
placeholders, banned overclaims, unresolved citations and missing figures.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "paper" / "main.tex"
BIB = ROOT / "paper" / "refs.bib"
FIG_DIR = ROOT / "paper" / "figs"


def main() -> int:
    manuscript = MAIN.read_text(encoding="utf-8")
    bibliography = BIB.read_text(encoding="utf-8")
    failures: list[str] = []

    placeholder_patterns = {
        "Pending": r"\bpending\b",
        "Placeholder": r"\bplaceholder\b",
        "TBD": r"\bTBD\b",
    }
    for label, pattern in placeholder_patterns.items():
        if re.search(pattern, manuscript, flags=re.IGNORECASE):
            failures.append(f"placeholder remains: {label}")

    banned_claims = {
        "f=1 upper bound": r"(?:\$f=1\$|f=1)\s+(?:is\s+an?\s+)?upper bound",
        "universal train-only impossibility": r"we\s+(?:show|prove|establish)[^.\n]{0,100}all\s+train-only[^.\n]{0,100}impossible",
        "validated adaptive method": r"we\s+(?:present|propose|validate)[^.\n]{0,80}validated adaptive (?:method|algorithm)",
        "SOTA claim": r"we\s+(?:achieve|obtain|report)[^.\n]{0,80}(?:state[- ]of[- ]the[- ]art|\bSOTA\b)",
        "negative learnability verdict": r"(?:joint|overall)\s+learnability\s+verdict[^.\n]{0,40}(?:negative|fails?)",
        "unqualified FDST opportunity absence": r"FDST[^.\n]{0,100}(?:shows|finds|establishes)\s+(?:that\s+)?(?:there is\s+)?no\s+opportunity(?![^.\n]{0,80}(?:scalar|grid|family))",
    }
    for label, pattern in banned_claims.items():
        if re.search(pattern, manuscript, flags=re.IGNORECASE):
            failures.append(f"banned claim detected: {label}")

    citation_keys: set[str] = set()
    for group in re.findall(r"\\cite(?:p|t)?\{([^}]*)\}", manuscript):
        citation_keys.update(key.strip() for key in group.split(","))
    bib_keys = set(re.findall(r"@\w+\{\s*([^,\s]+)", bibliography))
    for key in sorted(citation_keys - bib_keys):
        failures.append(f"missing bibliography key: {key}")

    for graphic in re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", manuscript):
        candidate = FIG_DIR / graphic
        if not candidate.exists():
            failures.append(f"missing figure: {candidate.relative_to(ROOT)}")

    if manuscript.count("{") != manuscript.count("}"):
        failures.append("unbalanced braces in paper/main.tex")
    if bibliography.count("{") != bibliography.count("}"):
        failures.append("unbalanced braces in paper/refs.bib")

    required_scope_markers = (
        "development diagnostics",
        "does not support a universal",
        "overall C4 gate",
        "manifest-bearing component archives",
        "Class-conditional opportunity",
        "Single-Shot FDST Scalar-Grid Boundary",
        "composite success criterion is not met",
        "These five dependent splits have low decision resolution",
        "all natural-data conclusions are conditional",
        "does not fairly test high-dimensional directional forgetting",
        "opportunity\\_absent=true",
    )
    normalized_manuscript = " ".join(manuscript.split())
    for marker in required_scope_markers:
        if marker not in normalized_manuscript:
            failures.append(f"missing scope marker: {marker!r}")

    stale_fdst_patterns = {
        "FDST not yet executed": r"FDST[^.\n]{0,120}(?:not yet executed|has not been run|model has not been run)",
        "all natural evidence is development-only": r"restricts its empirical conclusion to development data",
    }
    for label, pattern in stale_fdst_patterns.items():
        if re.search(pattern, manuscript, flags=re.IGNORECASE):
            failures.append(f"stale post-FDST claim detected: {label}")

    if failures:
        for failure in failures:
            print(f"FAIL {failure}")
        return 1

    print(
        "PASS TMLR static audit: "
        f"{len(citation_keys)} citations resolved; placeholders/overclaims absent"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
