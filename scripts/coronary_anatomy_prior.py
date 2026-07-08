"""
Utilities for the weak coronary anatomy prior.

The prior is intentionally a graph/template, not a patient-specific model. It is
used to guide branch labels, matching penalties, graph connectivity, and report
metadata while DICOM evidence remains the source of truth.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEMPLATE_PATH = ROOT / "anatomy_prior" / "coronary_template.json"


@dataclass(frozen=True)
class TemplateBranch:
    tree: str
    label: str
    display_name: str
    syntax_segment_ids: tuple[int, ...]
    parent: Optional[str]
    children: tuple[str, ...]
    relative_diameter: float
    expected_role: str
    required_for_tree: bool


class CoronaryAnatomyPrior:
    def __init__(self, payload: Dict[str, object], source_path: Optional[Path] = None):
        self.payload = payload
        self.source_path = source_path
        self.branches: Dict[str, TemplateBranch] = {}
        self.syntax_to_labels: Dict[int, list[str]] = {}
        self._load_branches()
        self.validate()

    @classmethod
    def load(cls, path: Path = DEFAULT_TEMPLATE_PATH) -> "CoronaryAnatomyPrior":
        with open(path, "r", encoding="utf-8") as f:
            return cls(json.load(f), source_path=path)

    def _load_branches(self):
        for tree_name, tree in self.payload.get("artery_trees", {}).items():
            for label, branch in tree.get("branches", {}).items():
                item = TemplateBranch(
                    tree=tree_name,
                    label=label,
                    display_name=str(branch.get("display_name", label)),
                    syntax_segment_ids=tuple(int(v) for v in branch.get("syntax_segment_ids", [])),
                    parent=branch.get("parent"),
                    children=tuple(str(v) for v in branch.get("children", [])),
                    relative_diameter=float(branch.get("relative_diameter", 1.0)),
                    expected_role=str(branch.get("expected_role", "")),
                    required_for_tree=bool(branch.get("required_for_tree", False)),
                )
                self.branches[label] = item
                for segment_id in item.syntax_segment_ids:
                    self.syntax_to_labels.setdefault(segment_id, []).append(label)

    def validate(self):
        if not self.payload.get("artery_trees"):
            raise ValueError("Coronary template has no artery_trees.")
        for label, branch in self.branches.items():
            if branch.parent and branch.parent not in self.branches:
                raise ValueError(f"{label} references missing parent {branch.parent}")
            for child in branch.children:
                if child not in self.branches:
                    raise ValueError(f"{label} references missing child {child}")
                child_parent = self.branches[child].parent
                if child_parent != label:
                    raise ValueError(f"{label}->{child} is inconsistent with child parent {child_parent}")

    def labels_for_syntax_segment(self, segment_id: int) -> list[str]:
        return list(self.syntax_to_labels.get(int(segment_id), []))

    def branch(self, label: str) -> Optional[TemplateBranch]:
        return self.branches.get(label)

    def same_tree(self, label_a: str, label_b: str) -> bool:
        a = self.branch(label_a)
        b = self.branch(label_b)
        return bool(a and b and a.tree == b.tree)

    def is_parent_child(self, parent_label: str, child_label: str) -> bool:
        parent = self.branch(parent_label)
        return bool(parent and child_label in parent.children)

    def ancestors(self, label: str) -> list[str]:
        out = []
        current = self.branch(label)
        while current and current.parent:
            out.append(current.parent)
            current = self.branch(current.parent)
        return out

    def topology_distance(self, label_a: str, label_b: str) -> Optional[int]:
        if label_a == label_b:
            return 0
        if label_a not in self.branches or label_b not in self.branches:
            return None
        frontier = [(label_a, 0)]
        seen = {label_a}
        while frontier:
            label, distance = frontier.pop(0)
            branch = self.branches[label]
            neighbors = list(branch.children)
            if branch.parent:
                neighbors.append(branch.parent)
            for neighbor in neighbors:
                if neighbor == label_b:
                    return distance + 1
                if neighbor not in seen:
                    seen.add(neighbor)
                    frontier.append((neighbor, distance + 1))
        return None

    def major_tree_for_labels(self, labels: Iterable[str]) -> str:
        counts: Dict[str, int] = {}
        for label in labels:
            branch = self.branch(label)
            if branch:
                counts[branch.tree] = counts.get(branch.tree, 0) + 1
        if not counts:
            return "unknown"
        return max(counts, key=counts.get)

    def matching_penalty(self, label_a: Optional[str], label_b: Optional[str]) -> float:
        penalties = self.payload.get("matching_penalties", {})
        if not label_a or not label_b or label_a == "unknown" or label_b == "unknown":
            return float(penalties.get("unknown_label", 5.0))
        if label_a == label_b:
            return 0.0
        branch_a = self.branch(label_a)
        branch_b = self.branch(label_b)
        if not branch_a or not branch_b:
            return float(penalties.get("unknown_label", 5.0))
        if branch_a.tree != branch_b.tree:
            return float(penalties.get("different_major_artery", 100.0))
        distance = self.topology_distance(label_a, label_b)
        if distance == 1:
            return float(penalties.get("parent_child_inconsistent", 20.0))
        return float(penalties.get("different_template_branch", 35.0))

    def summary(self) -> Dict[str, object]:
        return {
            "schema_version": self.payload.get("schema_version"),
            "source_path": str(self.source_path) if self.source_path else "",
            "trees": {
                tree_name: {
                    "root": tree.get("root"),
                    "branch_count": len(tree.get("branches", {})),
                    "required_branches": [
                        label
                        for label, branch in tree.get("branches", {}).items()
                        if branch.get("required_for_tree")
                    ],
                }
                for tree_name, tree in self.payload.get("artery_trees", {}).items()
            },
        }


def copy_template_snapshot(output_dir: Path, template_path: Path = DEFAULT_TEMPLATE_PATH) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "coronary_template_used.json"
    shutil.copyfile(template_path, target)
    return target


def main():
    prior = CoronaryAnatomyPrior.load()
    print(json.dumps(prior.summary(), indent=2))


if __name__ == "__main__":
    main()
