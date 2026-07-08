"""
Reads back saved per-image ECG analysis results from a patient case's
ecg_results/ folder. Nothing here writes results -- pipeline_composite.run_pipeline
already writes report.json (+ result_card.png, diagnosis_report.txt, etc.)
directly into whatever output_dir it's given, so saving is just "call
run_pipeline with output_dir pointed at ecg_results/<label>/".

No Qt dependency -- reusable from any UI layer, a CLI, or a batch tool.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import List


def list_ecg_results(ecg_dir: Path) -> List[dict]:
    """
    Loads every saved per-image report.json under a case's ecg_results/
    folder. Each returned dict carries a "_view_dir" Path so callers (e.g.
    the compare-with-angiogram feature) can locate that result's saved
    result_card.png / diagnosis_report.txt.
    """
    ecg_dir = Path(ecg_dir)
    results = []
    if not ecg_dir.exists():
        return results
    for report_json in sorted(ecg_dir.glob("*/report.json")):
        try:
            data = json.loads(report_json.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        data["_view_dir"] = report_json.parent
        results.append(data)
    return results


def delete_ecg_result(view_dir: Path) -> None:
    """Deletes one saved ECG result folder (ecg_results/<label>/) and everything
    in it (report.json, result_card.png, diagnosis_report.txt, source image, etc.)."""
    view_dir = Path(view_dir)
    if view_dir.exists():
        shutil.rmtree(view_dir)
