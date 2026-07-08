"""
Assign conservative doctor-facing coronary anatomy labels to reconstructed branches.

This stage does not change the 3D geometry. It turns branch IDs into anatomy-aware
metadata using the weak coronary template, segmentation geometry, confidence
reports, and final reprojection validation.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from coronary_anatomy_prior import CoronaryAnatomyPrior, DEFAULT_TEMPLATE_PATH  # noqa: E402


LCA_ORDER = ["LM", "LAD_prox", "LCX_prox", "LAD_mid", "LCX_distal", "D1", "OM1", "LAD_distal", "D2", "OM2", "LPL", "LPDA"]
RCA_ORDER = ["RCA_prox", "RCA_mid", "RCA_distal", "PDA", "RPL"]

# src/localization_labels.py SYNTAX groups -> this script's label vocabulary.
# "Diagonal"/"OM/intermediate" need an ordinal (D1/D2, OM1/OM2) resolved per
# case since several branches can share that group.
LEARNED_GROUP_TO_LABEL = {
    "LM": "LM",
    "LAD proximal": "LAD_prox",
    "LAD mid": "LAD_mid",
    "LAD distal": "LAD_distal",
    "LCX proximal": "LCX_prox",
    "LCX distal": "LCX_distal",
    "PL branch": "LPL",
    "PDA": "LPDA",
}
LEARNED_ORDINAL_GROUPS = {"Diagonal": "D", "OM/intermediate": "OM"}
LEARNED_MIN_CONFIDENCE = 0.55


def load_learned_branch_anatomy(epipolar_dir: Path | None) -> Dict[int, Dict[str, object]]:
    """Loads view A's learned per-branch anatomy (artery/group/confidence)
    from epipolar_optimized_centerline.py's anatomy_anchor_candidates.json,
    if that stage produced one (it's optional -- the localization model may
    not be installed)."""
    if epipolar_dir is None:
        return {}
    path = Path(epipolar_dir) / "anatomy_anchor_candidates.json"
    if not path.exists():
        return {}
    data = load_json(path)
    raw = data.get("view_a", {}).get("learned_branch_anatomy", {})
    return {int(branch_id): info for branch_id, info in raw.items()}


def load_json(path: Path) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_csv(path: Path, key: str) -> Dict[int, Dict[str, str]]:
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {int(row[key]): row for row in csv.DictReader(f)}


def load_validation(path: Path) -> Dict[int, Dict[str, str]]:
    by_branch: Dict[int, Dict[str, str]] = {}
    if not path.exists():
        return by_branch
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            branch_id = int(row["branch_id"])
            view = str(row["view"])
            item = by_branch.setdefault(branch_id, {})
            item[f"{view}_validation_status"] = row.get("validation_status", "")
            item[f"{view}_median_error_px"] = row.get("median_error_px", "")
            item[f"{view}_p90_error_px"] = row.get("p90_error_px", "")
    return by_branch


def branch_geometry(path: Path) -> Dict[int, Dict[str, object]]:
    if not path.exists():
        return {}
    payload = load_json(path)
    return {int(row["branch_id"]): row for row in payload.get("branches", [])}


def infer_major_tree(pipeline_summary: Dict[str, object], branches: Iterable[Dict[str, object]]) -> Dict[str, object]:
    branch_list = list(branches)
    num_a = int(pipeline_summary.get("num_view_a_branches", len(branch_list)))
    num_b = int(pipeline_summary.get("num_view_b_branches", 0))
    mean_branch_count = (num_a + max(num_b, 1)) / 2.0

    # LCA angiograms usually produce a richer bifurcating tree in a single frame.
    # RCA runs are often more continuous and less branch-dense in these masks.
    if num_a >= 6 or (num_a >= 5 and num_b >= 3):
        return {
            "major_tree": "LCA",
            "confidence": "moderate",
            "rationale": f"branch-rich bifurcating mask: view A {num_a} branches, view B {num_b} branches",
        }
    if mean_branch_count <= 4:
        return {
            "major_tree": "RCA",
            "confidence": "low",
            "rationale": f"lower branch count: view A {num_a} branches, view B {num_b} branches",
        }
    return {
        "major_tree": "unknown",
        "confidence": "low",
        "rationale": f"ambiguous branch count: view A {num_a} branches, view B {num_b} branches",
    }


def endpoint_leftness(row: Dict[str, object]) -> float:
    start = row.get("start_yx", [0.0, 0.0])
    end = row.get("end_yx", [0.0, 0.0])
    return min(float(start[1]), float(end[1]))


def mean_y(row: Dict[str, object]) -> float:
    return float(row.get("mean_yx", [256.0, 256.0])[0])


def assign_learned_labels(
    rows: List[Dict[str, object]], learned_by_branch: Dict[int, Dict[str, object]]
) -> Dict[int, Dict[str, object]]:
    """Assigns labels straight from the trained localization model's
    per-branch vote, for branches it's confident about. Direct groups (LM,
    LAD proximal/mid/distal, LCX proximal/distal, PL, PDA) map one-to-one;
    Diagonal/OM branches get an ordinal (D1/D2, OM1/OM2) by branch length,
    since several branches can share that group."""
    assignments: Dict[int, Dict[str, object]] = {}
    ordinal_pool: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        branch_id = int(row["branch_id"])
        learned = learned_by_branch.get(branch_id)
        if not learned or float(learned.get("confidence", 0.0)) < LEARNED_MIN_CONFIDENCE:
            continue
        group = learned.get("group", "")
        rationale = f"trained anatomical-localization model (group={group}, confidence={float(learned['confidence']):.2f})"
        if group in LEARNED_GROUP_TO_LABEL:
            assignments[branch_id] = {"label": LEARNED_GROUP_TO_LABEL[group], "confidence": "moderate", "rationale": rationale}
        elif group in LEARNED_ORDINAL_GROUPS:
            ordinal_pool.setdefault(group, []).append(row)

    for group, prefix in LEARNED_ORDINAL_GROUPS.items():
        candidates = sorted(ordinal_pool.get(group, []), key=lambda r: -float(r.get("length_px", 0.0)))
        for ordinal, row in enumerate(candidates[:2], start=1):
            branch_id = int(row["branch_id"])
            learned = learned_by_branch[branch_id]
            assignments[branch_id] = {
                "label": f"{prefix}{ordinal}",
                "confidence": "moderate",
                "rationale": f"trained anatomical-localization model (group={group}, confidence={float(learned['confidence']):.2f})",
            }
    return assignments


def assign_lca_labels(
    rows: List[Dict[str, object]], learned_by_branch: Dict[int, Dict[str, object]] | None = None
) -> Dict[int, Dict[str, object]]:
    assignments: Dict[int, Dict[str, object]] = {}
    if not rows:
        return assignments

    assignments.update(assign_learned_labels(rows, learned_by_branch or {}))

    ranked = sorted(rows, key=lambda r: (-float(r.get("length_px", 0.0)), -float(r.get("mean_diameter_px", 0.0))))
    unassigned = {int(row["branch_id"]) for row in rows} - set(assignments)
    used_labels = {a["label"] for a in assignments.values()}

    # The likely left main is a thick branch nearest the inlet/left side. Keep
    # confidence low because 2D branch tracing can split the ostium strangely.
    # Skipped if the localization model already placed an LM branch above.
    if "LM" not in used_labels:
        lm_pool = sorted(
            [row for row in rows if int(row["branch_id"]) in unassigned],
            key=lambda r: (endpoint_leftness(r), -float(r.get("mean_diameter_px", 0.0))),
        )
        if lm_pool:
            lm = int(lm_pool[0]["branch_id"])
            assignments[lm] = {"label": "LM", "confidence": "low", "rationale": "nearest inlet-side branch in the LCA tree"}
            unassigned.discard(lm)
            used_labels.add("LM")

    if "LAD_prox" not in used_labels and "LCX_prox" not in used_labels:
        main_candidates = [row for row in ranked if int(row["branch_id"]) in unassigned][:2]
        if len(main_candidates) == 2:
            upper, lower = sorted(main_candidates, key=mean_y)
            assignments[int(upper["branch_id"])] = {
                "label": "LAD_prox",
                "confidence": "low",
                "rationale": "long superior-running LCA branch candidate",
            }
            assignments[int(lower["branch_id"])] = {
                "label": "LCX_prox",
                "confidence": "low",
                "rationale": "long inferior/lateral-running LCA branch candidate",
            }
            unassigned.discard(int(upper["branch_id"]))
            unassigned.discard(int(lower["branch_id"]))
            used_labels.update({"LAD_prox", "LCX_prox"})

    remaining_labels = [
        label
        for label in ["LAD_mid", "LCX_distal", "D1", "OM1", "LAD_distal", "D2", "OM2", "LPL", "LPDA"]
        if label not in used_labels
    ]
    for row, label in zip([r for r in ranked if int(r["branch_id"]) in unassigned], remaining_labels):
        assignments[int(row["branch_id"])] = {
            "label": label,
            "confidence": "very_low",
            "rationale": "ranked by branch length and diameter; requires doctor review",
        }
    return assignments


def assign_rca_labels(rows: List[Dict[str, object]]) -> Dict[int, Dict[str, object]]:
    assignments: Dict[int, Dict[str, object]] = {}
    ranked = sorted(rows, key=lambda r: (-float(r.get("length_px", 0.0)), endpoint_leftness(r)))
    for row, label in zip(ranked, RCA_ORDER):
        assignments[int(row["branch_id"])] = {
            "label": label,
            "confidence": "low" if label in {"RCA_prox", "RCA_mid"} else "very_low",
            "rationale": "ranked along the likely RCA continuation; requires doctor review",
        }
    return assignments


def evidence_level(branch_status: str, validation: Dict[str, str]) -> str:
    statuses = {validation.get("view_a_validation_status"), validation.get("view_b_validation_status")}
    if branch_status == "reliable" and statuses <= {"good", "review"}:
        return "two_view_validated"
    if branch_status in {"reliable", "usable"}:
        return "two_view_supported"
    if "bad" in statuses:
        return "visual_estimate_review"
    return "visual_estimate"


def write_outputs(rows: List[Dict[str, object]], output_dir: Path, summary: Dict[str, object]):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "anatomy_branch_labels.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(output_dir / "anatomy_branch_labels.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "branches": rows}, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline-dir", type=Path, required=True)
    parser.add_argument("--branch-report", type=Path, required=True)
    parser.add_argument("--validation-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE_PATH)
    parser.add_argument(
        "--epipolar-dir",
        type=Path,
        default=None,
        help="Optional epipolar_optimized_centerline.py output dir, used to prefer the trained localization model's per-branch labels when confident.",
    )
    args = parser.parse_args()

    prior = CoronaryAnatomyPrior.load(args.template)
    pipeline_summary = load_json(args.pipeline_dir / "pipeline_summary.json")
    geometry = branch_geometry(args.pipeline_dir / "view_a_branches.json")
    branch_report = load_csv(args.branch_report, "branch_id")
    validation = load_validation(args.validation_report)
    learned_by_branch = load_learned_branch_anatomy(args.epipolar_dir)
    tree = infer_major_tree(pipeline_summary, geometry.values())

    if tree["major_tree"] == "LCA":
        assignments = assign_lca_labels(list(geometry.values()), learned_by_branch)
        order = LCA_ORDER
    elif tree["major_tree"] == "RCA":
        assignments = assign_rca_labels(list(geometry.values()))
        order = RCA_ORDER
    else:
        assignments = {}
        order = []

    rows = []
    all_branch_ids = sorted(set(branch_report) | set(geometry))
    for branch_id in all_branch_ids:
        report = branch_report.get(branch_id, {})
        validation_row = validation.get(branch_id, {})
        assignment = assignments.get(branch_id, {"label": "unknown", "confidence": "none", "rationale": "not enough anatomy evidence"})
        label = assignment["label"]
        branch = prior.branch(label)
        rows.append(
            {
                "branch_id": branch_id,
                "major_tree": tree["major_tree"],
                "anatomy_label": label,
                "display_name": branch.display_name if branch else "unknown branch",
                "label_confidence": assignment["confidence"],
                "evidence_level": evidence_level(report.get("status", ""), validation_row),
                "branch_status": report.get("status", ""),
                "view_a_validation": validation_row.get("view_a_validation_status", ""),
                "view_b_validation": validation_row.get("view_b_validation_status", ""),
                "estimated_diameter_stenosis_pct": report.get("estimated_diameter_stenosis_pct", ""),
                "min_radius_mm": report.get("min_radius_mm", ""),
                "ref_radius_mm_p80": report.get("ref_radius_mm_p80", ""),
                "rationale": assignment["rationale"],
            }
        )

    summary = {
        "major_tree": tree,
        "template": prior.summary(),
        "label_order": order,
        "safety_note": "Labels are conservative metadata. Geometry is still reconstructed from DICOM evidence, not from the template alone.",
        "outputs": {
            "csv": str(args.output_dir / "anatomy_branch_labels.csv"),
            "json": str(args.output_dir / "anatomy_branch_labels.json"),
        },
    }
    write_outputs(rows, args.output_dir, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
