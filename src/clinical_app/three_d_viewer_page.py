"""
PySide6 3D Viewer page for case-level coronary reconstruction outputs.

Loads patient_data/<case_id>/reconstruction_3d/hybrid_smoothed_confidence_colored.obj
plus lesions_3d.json created by case_analysis_workflow.py.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

import vtk
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

import patient_store
from case_analysis_workflow import lesions_3d_path, reconstruction_obj


def _card() -> QFrame:
    frame = QFrame()
    frame.setProperty("card", "true")
    return frame


def _hex_to_rgb01(color: str) -> tuple[float, float, float]:
    color = color.strip().lstrip("#")
    if len(color) != 6:
        return (0.73, 0.93, 0.57)
    return tuple(int(color[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


class ThreeDViewerPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.case_id: Optional[str] = None
        self.package: dict = {}
        self.last_error = ""
        self.mesh_actor: Optional[vtk.vtkActor] = None
        self.lesion_actors: list[vtk.vtkProp3D] = []
        self.branch_surfaces: dict[int, dict] = {}

        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(28, 24, 28, 20)
        root.setSpacing(14)

        title = QLabel("3D Coronary Viewer")
        title.setProperty("role", "pageTitle")
        root.addWidget(title)

        self.lbl_status = QLabel("Open a completed case from DICOM Analysis.")
        self.lbl_status.setProperty("role", "pageSubtitle")
        self.lbl_status.setWordWrap(True)
        root.addWidget(self.lbl_status)

        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter, stretch=1)

        view_card = _card()
        view_layout = QVBoxLayout(view_card)
        view_layout.setContentsMargins(12, 12, 12, 12)

        self.vtk_widget = QVTKRenderWindowInteractor(view_card)
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.04, 0.045, 0.05)
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        self.interactor = self.vtk_widget.GetRenderWindow().GetInteractor()
        self.interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        view_layout.addWidget(self.vtk_widget)
        splitter.addWidget(view_card)

        side = _card()
        side.setMinimumWidth(360)
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(16, 16, 16, 16)
        side_layout.setSpacing(10)

        lesions_header = QLabel("Lesions")
        lesions_header.setProperty("role", "sectionHeader")
        side_layout.addWidget(lesions_header)

        self.list_lesions = QListWidget()
        self.list_lesions.currentRowChanged.connect(self._on_lesion_selected)
        side_layout.addWidget(self.list_lesions, stretch=2)

        self.lbl_lesion_detail = QLabel("Select a lesion marker or list item.")
        self.lbl_lesion_detail.setProperty("role", "hint")
        self.lbl_lesion_detail.setWordWrap(True)
        side_layout.addWidget(self.lbl_lesion_detail)

        self.ds_bar = QProgressBar()
        self.ds_bar.setRange(0, 100)
        self.ds_bar.setFormat("Diameter stenosis: %p%")
        side_layout.addWidget(self.ds_bar)

        self.confidence_bar = QProgressBar()
        self.confidence_bar.setRange(0, 100)
        self.confidence_bar.setFormat("Confidence: %p%")
        side_layout.addWidget(self.confidence_bar)

        self.btn_open_view_report = QPushButton("Open Source View Report")
        self.btn_open_view_report.setProperty("variant", "ghost")
        self.btn_open_view_report.clicked.connect(self._open_selected_view_report)
        side_layout.addWidget(self.btn_open_view_report)

        views_header = QLabel("Views Used")
        views_header.setProperty("role", "sectionHeader")
        side_layout.addWidget(views_header)

        self.list_views = QListWidget()
        side_layout.addWidget(self.list_views, stretch=1)

        self.btn_open_final_report = QPushButton("Open Final Clinical Report")
        self.btn_open_final_report.setProperty("variant", "primary")
        self.btn_open_final_report.clicked.connect(self._open_final_report)
        side_layout.addWidget(self.btn_open_final_report)

        splitter.addWidget(side)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

    def load_case(self, case_id: str) -> bool:
        self.last_error = ""
        self.case_id = case_id
        obj_path = reconstruction_obj(case_id)
        package_path = lesions_3d_path(case_id)
        if not obj_path.exists() or not package_path.exists():
            self._clear_view()
            self.last_error = (
                f"3D reconstruction is not ready for case {case_id} yet. "
                f"Missing: {obj_path.name if not obj_path.exists() else package_path.name}"
            )
            self.lbl_status.setText(self.last_error)
            return False

        try:
            self.package = json.loads(package_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            self._clear_view()
            self.last_error = f"Failed to load 3D lesion package: {exc}"
            self.lbl_status.setText(self.last_error)
            return False

        try:
            self.lbl_status.setText(f"Loaded case {case_id}: {obj_path.name}")
            self._load_mesh(obj_path)
            self._load_lesions()
            self._load_views()
            self.renderer.ResetCamera()
            self.interactor.Initialize()
            QTimer.singleShot(0, self.vtk_widget.GetRenderWindow().Render)
            return True
        except Exception as exc:
            self._clear_view()
            self.last_error = f"Failed to render 3D reconstruction for case {case_id}: {exc}"
            self.lbl_status.setText(self.last_error)
            return False

    def _clear_view(self):
        self.package = {}
        self.renderer.RemoveAllViewProps()
        self.list_lesions.clear()
        self.list_views.clear()
        self.lbl_lesion_detail.setText("Select a lesion marker or list item.")
        self.ds_bar.setValue(0)
        self.confidence_bar.setValue(0)
        self.vtk_widget.GetRenderWindow().Render()

    def _load_mesh(self, obj_path: Path):
        self.renderer.RemoveAllViewProps()
        self.lesion_actors = []
        self.branch_surfaces = self._parse_obj_branch_surfaces(obj_path)

        reader = vtk.vtkOBJReader()
        reader.SetFileName(str(obj_path))
        reader.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(reader.GetOutputPort())

        self.mesh_actor = vtk.vtkActor()
        self.mesh_actor.SetMapper(mapper)
        self.mesh_actor.GetProperty().SetColor(0.78, 0.82, 0.86)
        self.mesh_actor.GetProperty().SetOpacity(0.72)
        self.mesh_actor.GetProperty().SetSpecular(0.25)
        self.mesh_actor.GetProperty().SetSpecularPower(20)
        self.renderer.AddActor(self.mesh_actor)

    def _load_lesions(self):
        self.list_lesions.clear()
        lesions = self.package.get("lesions", [])
        for idx, lesion in enumerate(lesions):
            actor = self._make_lesion_actor(lesion)
            if actor is None:
                continue
            self.renderer.AddActor(actor)
            self.lesion_actors.append(actor)

            item = QListWidgetItem(
                f"{lesion.get('lesion_id')} | {lesion.get('severity')}\n"
                f"{lesion.get('artery')} | narrowing {float(lesion.get('DS_percent') or 0):.1f}%"
            )
            item.setData(Qt.UserRole, lesion)
            self.list_lesions.addItem(item)

        if lesions:
            self.list_lesions.setCurrentRow(0)
        else:
            self.lbl_lesion_detail.setText("No stenosis lesions were saved for this case.")

    def _parse_obj_branch_surfaces(self, obj_path: Path) -> dict[int, dict]:
        vertices: list[tuple[float, float, float]] = []
        surfaces: dict[int, dict] = {}
        current_branch = None

        with obj_path.open("r", encoding="utf-8", errors="ignore") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                if line.startswith("o "):
                    current_branch = None
                    name = line.split(maxsplit=1)[1]
                    if name.startswith("branch_"):
                        try:
                            current_branch = int(name.split("_")[1])
                            surfaces.setdefault(current_branch, {"faces": []})
                        except (IndexError, ValueError):
                            current_branch = None
                elif line.startswith("v "):
                    parts = line.split()
                    if len(parts) >= 4:
                        vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
                elif current_branch is not None and line.startswith("f "):
                    face = []
                    for token in line.split()[1:]:
                        try:
                            idx = int(token.split("/")[0])
                        except ValueError:
                            continue
                        if idx < 0:
                            idx = len(vertices) + 1 + idx
                        face.append(idx - 1)
                    if len(face) >= 3:
                        surfaces[current_branch]["faces"].append(face)

        for surface in surfaces.values():
            surface["vertices"] = vertices
        return surfaces

    def _make_lesion_actor(self, lesion: dict) -> Optional[vtk.vtkActor]:
        surface = self.branch_surfaces.get(int(lesion.get("branch_id", -1)))
        if not surface:
            return None

        x, y, z = lesion.get("position_3d", [0, 0, 0])
        ds = float(lesion.get("DS_percent") or 0.0)
        vessel_radius = max(0.3, float(lesion.get("vessel_radius_3d") or 1.0))
        ds_norm = max(0.0, min(ds, 100.0)) / 100.0
        lesion_length = max(vessel_radius * (2.2 + 3.8 * ds_norm), 2.2)
        radial_coverage = 0.65 + 0.95 * ds_norm

        tangent = lesion.get("tangent_3d") or [0.0, 0.0, 1.0]
        tx, ty, tz = (float(tangent[0]), float(tangent[1]), float(tangent[2]))
        norm = max(math.sqrt(tx * tx + ty * ty + tz * tz), 1e-9)
        tx, ty, tz = tx / norm, ty / norm, tz / norm
        center = (float(x), float(y), float(z))
        tangent_vec = (tx, ty, tz)

        source_vertices = surface["vertices"]
        selected_faces = []
        for face in surface["faces"]:
            pts = [source_vertices[idx] for idx in face]
            cx = sum(p[0] for p in pts) / len(pts)
            cy = sum(p[1] for p in pts) / len(pts)
            cz = sum(p[2] for p in pts) / len(pts)
            vx, vy, vz = cx - center[0], cy - center[1], cz - center[2]
            axial = vx * tangent_vec[0] + vy * tangent_vec[1] + vz * tangent_vec[2]
            if abs(axial) > lesion_length / 2.0:
                continue
            radial_sq = (
                (vx - axial * tangent_vec[0]) ** 2
                + (vy - axial * tangent_vec[1]) ** 2
                + (vz - axial * tangent_vec[2]) ** 2
            )
            if radial_sq <= (vessel_radius * radial_coverage) ** 2:
                selected_faces.append(face)

        if not selected_faces:
            return None

        points = vtk.vtkPoints()
        cells = vtk.vtkCellArray()
        remap: dict[int, int] = {}
        for face in selected_faces:
            cells.InsertNextCell(len(face))
            for idx in face:
                if idx not in remap:
                    remap[idx] = points.InsertNextPoint(source_vertices[idx])
                cells.InsertCellPoint(remap[idx])

        poly = vtk.vtkPolyData()
        poly.SetPoints(points)
        poly.SetPolys(cells)

        normals = vtk.vtkPolyDataNormals()
        normals.SetInputData(poly)
        normals.ConsistencyOn()
        normals.SplittingOff()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(normals.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        prop = actor.GetProperty()
        if ds >= 90.0:
            color = (0.78, 0.0, 0.0)
        elif ds >= 80.0:
            color = (0.95, 0.04, 0.02)
        else:
            color = (1.0, 0.24, 0.10)
        prop.SetColor(*color)
        prop.SetAmbient(0.38)
        prop.SetDiffuse(0.78)
        prop.SetSpecular(0.18)
        prop.SetOpacity(0.62 + 0.34 * ds_norm)
        return actor

    @staticmethod
    def _set_prop_opacity(prop: vtk.vtkProp3D, opacity: float):
        if isinstance(prop, vtk.vtkActor):
            prop.GetProperty().SetOpacity(opacity)
            return
        parts = prop.GetParts() if hasattr(prop, "GetParts") else None
        if parts is None:
            return
        parts.InitTraversal()
        for _ in range(parts.GetNumberOfItems()):
            child = parts.GetNextProp3D()
            if isinstance(child, vtk.vtkActor):
                child.GetProperty().SetOpacity(opacity)

    def _load_views(self):
        self.list_views.clear()
        for view in self.package.get("views_used", []):
            self.list_views.addItem(QListWidgetItem(
                f"{view.get('view_label', 'View')}\n"
                f"{view.get('lesion_count', 0)} lesion(s) | {view.get('n_frames_analyzed', 0)} frames"
            ))

    def _on_lesion_selected(self, row: int):
        if row < 0:
            return
        for i, actor in enumerate(self.lesion_actors):
            self._set_prop_opacity(actor, 1.0 if i == row else 0.42)

        item = self.list_lesions.item(row)
        lesion = item.data(Qt.UserRole)
        detail = (
            f"{lesion.get('lesion_id')} - {lesion.get('severity')}\n"
            f"Artery: {lesion.get('artery')} / {lesion.get('label')}\n"
            f"DS: {float(lesion.get('DS_percent') or 0):.1f}%\n"
            f"MLD: {lesion.get('MLD_mm') or lesion.get('MLD_px')} | "
            f"RVD: {lesion.get('RVD_mm') or lesion.get('RVD_px')}\n"
            f"Confidence: {lesion.get('confidence')}\n"
            f"Source view: {lesion.get('view_label')}"
        )
        self.lbl_lesion_detail.setText(detail)
        self.ds_bar.setValue(max(0, min(100, int(round(float(lesion.get("DS_percent") or 0))))))
        conf = lesion.get("confidence")
        self.confidence_bar.setValue(max(0, min(100, int(round(float(conf or 0) * 100)))))
        self.vtk_widget.GetRenderWindow().Render()

    def _open_selected_view_report(self):
        item = self.list_lesions.currentItem()
        if item is None:
            return
        lesion = item.data(Qt.UserRole)
        path = lesion.get("view_report")
        if path and hasattr(os, "startfile"):
            try:
                os.startfile(path)
            except OSError:
                pass

    def _open_final_report(self):
        if not self.case_id:
            return
        path = patient_store.get_case_dir(self.case_id) / "clinical_diagnosis_report.pdf"
        if path.exists() and hasattr(os, "startfile"):
            try:
                os.startfile(path)
            except OSError:
                pass
