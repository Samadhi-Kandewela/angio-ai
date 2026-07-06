"""
ECG <-> QCA concordance engine.

Cross-references a patient's ECG diagnosis (territory-based ST elevation/
depression, from the 12-lead pipeline's report.json shape) against their
already-saved QCA angiogram view results (per-lesion artery/DS_percent/
severity, from analysis_results_store.list_view_results) to flag whether an
ECG-implicated territory is corroborated by a real angiographic lesion in the
matching vessel.

This is corroborating evidence for culprit-vessel review, not a functional-
significance score -- FFR/iFR remain the gold standard for that.

No Qt dependency -- reusable from the clinical app's UI, a CLI, or tests.
"""
from __future__ import annotations

from typing import List, Optional

SEVERE_DS = 70.0
MODERATE_DS = 50.0

# Standard ECG-territory -> culprit-vessel mapping used for STEMI localisation.
# Inferior defaults to the RCA family (~85% right-dominant circulation); the
# LCX family is the left-dominant alternative. Lateral/high_lat involve the
# LCX/diagonal branches. Keys must match ecg_segment.py's CONTIGUOUS territory
# dict exactly: anterior, septal, lateral, inferior, high_lat.
TERRITORY_TO_VESSELS = {
    "anterior": ["LAD"],
    "septal": ["LAD"],
    "lateral": ["LCX", "LAD"],   # LCX (obtuse marginal) or a LAD diagonal branch
    "high_lat": ["LCX", "LAD"],
    "inferior": ["RCA", "LCX"],
}


def _vessel_matches(candidate: str, artery: str) -> bool:
    """True if a saved lesion's artery string (e.g. "RCA/PDA", "LCX") should
    count as the candidate vessel family (e.g. "RCA") -- substring containment,
    not exact match, since compound segments like "RCA/PDA" or "LCX/PDA" are
    real values in the app's SYNTAX-derived artery taxonomy."""
    if not artery or artery == "unknown":
        return False
    return candidate.upper() in artery.upper()


def classify_ds(ds: float) -> str:
    if ds >= SEVERE_DS:
        return "SEVERE"
    if ds >= MODERATE_DS:
        return "MODERATE"
    return "MILD/NONE"


def _ecg_findings(ecg_report: dict) -> dict:
    dx = ecg_report.get("diagnosis") or {}
    return {
        "elevation": dx.get("st_elevation", {}) or {},
        "depression": dx.get("st_depression", {}) or {},
        "severity": dx.get("severity"),
        "mi_label": (ecg_report.get("classification") or {}).get("label"),
        "mi_confidence": (ecg_report.get("classification") or {}).get("confidence"),
    }


def _vessel_status(view_results: List[dict]) -> dict:
    """
    Aggregates every lesion across every saved angiogram view for the case
    into a per-vessel-family max-severity summary:
        {"RCA": {"max_ds": 93.2, "n_lesions": 2, "views": {...}}, ...}
    A lesion contributes to every candidate vessel family its artery string
    matches (e.g. an "RCA/PDA" lesion counts for both "RCA" candidates).
    """
    status = {}
    all_candidates = sorted({v for vessels in TERRITORY_TO_VESSELS.values() for v in vessels})

    for view in view_results:
        view_label = view.get("view_label", "?")
        for lesion in view.get("lesions", []):
            artery = lesion.get("artery", "unknown")
            ds = lesion.get("DS_percent")
            if ds is None:
                continue
            for candidate in all_candidates:
                if not _vessel_matches(candidate, artery):
                    continue
                entry = status.setdefault(candidate, {"max_ds": 0.0, "n_lesions": 0, "views": set()})
                entry["max_ds"] = max(entry["max_ds"], float(ds))
                entry["n_lesions"] += 1
                entry["views"].add(view_label)

    for entry in status.values():
        entry["views"] = sorted(entry["views"])
    return status


def has_any_known_artery(view_results: List[dict]) -> bool:
    """False if every lesion across every saved view has artery == "unknown"
    (i.e. no view for this case was analyzed with a localization model
    loaded) -- used to show a helpful hint instead of an empty comparison."""
    for view in view_results:
        for lesion in view.get("lesions", []):
            if lesion.get("artery", "unknown") != "unknown":
                return True
    return False


def compare(ecg_report: dict, view_results: List[dict]) -> List[dict]:
    """
    Cross-references one ECG report against every saved QCA view result for
    the same case. Returns a list of verdict dicts, each with:
        territory, vessel, signal, verdict, detail
    verdict is one of: CONCORDANT, PARTIALLY CONCORDANT, DISCORDANT,
    INCONCLUSIVE, UNCONFIRMED_BY_ECG, UNMAPPED.
    """
    ecg = _ecg_findings(ecg_report)
    vessel_status = _vessel_status(view_results)
    results = []

    vessel_signals = {}  # vessel -> {"signal": str, "territories": set}
    unmapped_territories = set()

    for signal_key, is_primary in (("elevation", True), ("depression", False)):
        for territory in ecg[signal_key].keys():
            vessels = TERRITORY_TO_VESSELS.get(territory)
            if vessels is None:
                unmapped_territories.add(territory)
                continue
            for vessel in vessels:
                entry = vessel_signals.setdefault(vessel, {"signal": None, "territories": set()})
                entry["territories"].add(territory)
                if is_primary or entry["signal"] is None:
                    entry["signal"] = "ST elevation" if is_primary else "reciprocal ST depression"

    for territory in sorted(unmapped_territories):
        results.append({
            "territory": territory, "vessel": None, "signal": "?",
            "verdict": "UNMAPPED",
            "detail": f"No known vessel mapping for ECG territory '{territory}'.",
        })

    checked_vessels = set(vessel_signals.keys())

    for vessel, info in sorted(vessel_signals.items()):
        territory_label = "+".join(sorted(info["territories"]))
        signal = info["signal"]
        v_status = vessel_status.get(vessel)

        if v_status is None:
            results.append({
                "territory": territory_label, "vessel": vessel, "signal": signal,
                "verdict": "INCONCLUSIVE",
                "detail": f"ECG implicates {vessel} via {territory_label} territory, but no saved "
                          f"angiogram view has a lesion identified in {vessel}.",
            })
            continue

        ds = v_status["max_ds"]
        views_str = ", ".join(v_status["views"])
        if ds >= SEVERE_DS:
            verdict = "CONCORDANT"
            detail = (f"ECG {territory_label} {signal} matches a SEVERE ({ds:.1f}%) QCA lesion in "
                      f"{vessel} (view: {views_str}) — supports {vessel} as the culprit vessel.")
        elif ds >= MODERATE_DS:
            verdict = "PARTIALLY CONCORDANT"
            detail = (f"ECG {territory_label} {signal} correlates with a MODERATE ({ds:.1f}%) QCA "
                      f"lesion in {vessel} (view: {views_str}) — borderline support, consider "
                      f"clinical correlation.")
        else:
            verdict = "DISCORDANT"
            detail = (f"ECG {territory_label} {signal} implicates {vessel}, but the saved angiogram "
                      f"view(s) show no significant stenosis there (max DS {ds:.1f}%, "
                      f"{v_status['n_lesions']} lesion(s) detected). Flag for manual review — "
                      f"diameter-profile QCA can under-detect near-total/total occlusions.")
        results.append({
            "territory": territory_label, "vessel": vessel, "signal": signal,
            "verdict": verdict, "detail": detail,
        })

    for vessel, v_status in vessel_status.items():
        if vessel in checked_vessels:
            continue
        ds = v_status["max_ds"]
        if ds >= MODERATE_DS:
            views_str = ", ".join(v_status["views"])
            results.append({
                "territory": None, "vessel": vessel, "signal": "QCA-only",
                "verdict": "UNCONFIRMED_BY_ECG",
                "detail": (f"A saved angiogram view (view: {views_str}) found a {classify_ds(ds)} "
                           f"lesion ({ds:.1f}%) in {vessel}, but no ECG territory in this reading "
                           f"implicates {vessel}. Could be a non-culprit/chronic lesion, or this ECG "
                           f"reading missed the corroborating territory."),
            })

    return results
