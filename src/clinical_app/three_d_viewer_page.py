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

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

import vtk
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

import patient_store
from case_analysis_workflow import is_3d_ready, lesion_panel_path, lesions_3d_path, reconstruction_dir, reconstruction_obj


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
        self.pipeline_summary: dict = {}
        self.lesion_panel_entries: list[dict] = []
        self.last_error = ""
        self.mesh_actor: Optional[vtk.vtkActor] = None
        self.lesion_actors: list[vtk.vtkProp3D] = []
        self._lesion_id_to_actor_index: dict[str, int] = {}
        self.branch_surfaces: dict[int, dict] = {}

        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(28, 24, 28, 20)
        root.setSpacing(14)

        header_row = QHBoxLayout()
        title_col = QVBoxLayout()
        title_col.setSpacing(4)
        title = QLabel("3D Coronary Viewer")
        title.setProperty("role", "pageTitle")
        title_col.addWidget(title)

        self.lbl_status = QLabel("Select a case with a completed 3D reconstruction.")
        self.lbl_status.setProperty("role", "pageSubtitle")
        self.lbl_status.setWordWrap(True)
        title_col.addWidget(self.lbl_status)
        header_row.addLayout(title_col, stretch=1)

        self.btn_change_case = QPushButton("Change Case")
        self.btn_change_case.setProperty("variant", "ghost")
        self.btn_change_case.clicked.connect(self._show_case_select)
        self.btn_change_case.setVisible(False)
        header_row.addWidget(self.btn_change_case, alignment=Qt.AlignTop)
        root.addLayout(header_row)

        self.content_stack = QStackedWidget()
        root.addWidget(self.content_stack, stretch=1)

        self.content_stack.addWidget(self._build_case_select_page())

        viewer_page = QWidget()
        viewer_layout = QVBoxLayout(viewer_page)
        viewer_layout.setContentsMargins(0, 0, 0, 0)
        splitter = QSplitter(Qt.Horizontal)
        viewer_layout.addWidget(splitter)
        self.content_stack.addWidget(viewer_page)

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

        lesions_caption = QLabel(
            "Moderate/severe stenoses from the clinical report. "
            "● shown on the 3D model — ○ not in this reconstruction."
        )
        lesions_caption.setProperty("role", "captionText")
        lesions_caption.setWordWrap(True)
        side_layout.addWidget(lesions_caption)

        self.list_lesions = QListWidget()
        self.list_lesions.currentRowChanged.connect(self._on_lesion_selected)
        side_layout.addWidget(self.list_lesions, stretch=2)

        self.lbl_lesion_detail = QLabel("Select a lesion marker or list item.")
        self.lbl_lesion_detail.setProperty("role", "detailText")
        self.lbl_lesion_detail.setWordWrap(True)
        side_layout.addWidget(self.lbl_lesion_detail)

        # Percentage is drawn in a separate label rather than the progress
        # bar's own overlaid text: that text is one fixed color across a bar
        # that's part dark track / part light-green fill, so at high values
        # (common for DS%) it sat on the green fill and was unreadable.
        self.ds_bar, self.lbl_ds_value = self._build_metric_row(side_layout, "Diameter stenosis")
        self.confidence_bar, self.lbl_confidence_value = self._build_metric_row(side_layout, "Confidence")

        self.btn_open_view_report = QPushButton("Open 2D View Report")
        self.btn_open_view_report.setProperty("variant", "ghost")
        self.btn_open_view_report.setToolTip(
            "Opens the original per-view QCA report PDF for the selected lesion's own angiographic view."
        )
        self.btn_open_view_report.setEnabled(False)
        self.btn_open_view_report.clicked.connect(self._open_selected_view_report)
        side_layout.addWidget(self.btn_open_view_report)
        view_report_caption = QLabel("Report for this lesion's own 2D angiographic view.")
        view_report_caption.setProperty("role", "captionText")
        side_layout.addWidget(view_report_caption)

        side_layout.addWidget(self._divider())

        source_views_header = QLabel("Source Views")
        source_views_header.setProperty("role", "sectionHeader")
        side_layout.addWidget(source_views_header)

        source_views_row = QHBoxLayout()
        source_views_row.setSpacing(10)
        self.lbl_view_a_thumb = QLabel()
        self.lbl_view_b_thumb = QLabel()
        self.btn_view_a = QPushButton("View A")
        self.btn_view_b = QPushButton("View B")
        self.view_button_group = QButtonGroup(self)
        self.view_button_group.setExclusive(True)
        for key, thumb, button in (
            ("view_a", self.lbl_view_a_thumb, self.btn_view_a),
            ("view_b", self.lbl_view_b_thumb, self.btn_view_b),
        ):
            column = QVBoxLayout()
            column.setSpacing(6)
            thumb.setFixedSize(150, 150)
            thumb.setAlignment(Qt.AlignCenter)
            thumb.setProperty("role", "hint")
            thumb.setText("Not available")
            column.addWidget(thumb)
            button.setCheckable(True)
            button.setEnabled(False)
            button.clicked.connect(lambda _checked, k=key: self._set_camera_to_view(k))
            self.view_button_group.addButton(button)
            column.addWidget(button)
            source_views_row.addLayout(column)
        side_layout.addLayout(source_views_row)

        side_layout.addWidget(self._divider())

        views_header = QLabel("Views Used")
        views_header.setProperty("role", "sectionHeader")
        side_layout.addWidget(views_header)

        self.list_views = QListWidget()
        side_layout.addWidget(self.list_views, stretch=1)

        self.btn_open_final_report = QPushButton("Open Full Clinical Report")
        self.btn_open_final_report.setProperty("variant", "primary")
        self.btn_open_final_report.setToolTip(
            "Opens the case's combined clinical report covering every analyzed angiographic view."
        )
        self.btn_open_final_report.clicked.connect(self._open_final_report)
        side_layout.addWidget(self.btn_open_final_report)
        final_report_caption = QLabel("Combined report covering every analyzed view for this case.")
        final_report_caption.setProperty("role", "captionText")
        side_layout.addWidget(final_report_caption)

        splitter.addWidget(side)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

    def _build_case_select_page(self) -> QWidget:
        """The page's default screen. Deliberately holds no VTK widget: the
        QVTKRenderWindowInteractor is a native child window, and on Windows
        an unrendered one left mounted behind a QStackedWidget page can show
        stale content from whatever page was underneath it instead of going
        blank. Not creating it until a case is actually chosen avoids that."""
        page = _card()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(12)

        header = QLabel("Open a case")
        header.setProperty("role", "sectionHeader")
        layout.addWidget(header)

        caption = QLabel("Cases with a completed 3D reconstruction are listed below.")
        caption.setProperty("role", "captionText")
        caption.setWordWrap(True)
        layout.addWidget(caption)

        self.list_case_select = QListWidget()
        self.list_case_select.itemDoubleClicked.connect(lambda _item: self._on_view_3d_from_list())
        self.list_case_select.currentRowChanged.connect(
            lambda row: self.btn_view_3d_from_list.setEnabled(row >= 0)
        )
        layout.addWidget(self.list_case_select, stretch=1)

        self.lbl_case_select_empty = QLabel(
            "No cases have a completed 3D reconstruction yet. Run one from DICOM Analysis first."
        )
        self.lbl_case_select_empty.setProperty("role", "captionText")
        self.lbl_case_select_empty.setWordWrap(True)
        self.lbl_case_select_empty.setVisible(False)
        layout.addWidget(self.lbl_case_select_empty)

        self.btn_view_3d_from_list = QPushButton("View 3D")
        self.btn_view_3d_from_list.setProperty("variant", "primary")
        self.btn_view_3d_from_list.setEnabled(False)
        self.btn_view_3d_from_list.clicked.connect(self._on_view_3d_from_list)
        layout.addWidget(self.btn_view_3d_from_list)

        return page

    def showEvent(self, event):
        super().showEvent(event)
        if self.content_stack.currentIndex() == 0:
            self._refresh_case_select_list()

    def _refresh_case_select_list(self):
        self.list_case_select.clear()
        cases = [c for c in patient_store.list_cases() if is_3d_ready(c["case_id"])]
        for case in cases:
            label = case.get("full_name") or case.get("patient_id") or case["case_id"]
            study_date = case.get("study_date") or ""
            item = QListWidgetItem(f"{label}\n{case['case_id']}" + (f" | {study_date}" if study_date else ""))
            item.setData(Qt.UserRole, case["case_id"])
            self.list_case_select.addItem(item)
        self.lbl_case_select_empty.setVisible(not cases)
        self.list_case_select.setVisible(bool(cases))
        self.btn_view_3d_from_list.setEnabled(False)

    def _show_case_select(self):
        self.content_stack.setCurrentIndex(0)
        self.btn_change_case.setVisible(False)
        self.lbl_status.setText("Select a case with a completed 3D reconstruction.")
        self._refresh_case_select_list()

    def _on_view_3d_from_list(self):
        item = self.list_case_select.currentItem()
        if item is None:
            return
        case_id = item.data(Qt.UserRole)
        if not self.load_case(case_id):
            return

    def _divider(self) -> QFrame:
        line = QFrame()
        line.setProperty("role", "divider")
        line.setFixedHeight(1)
        return line

    def _build_metric_row(self, layout: QVBoxLayout, label: str) -> tuple[QProgressBar, QLabel]:
        row = QHBoxLayout()
        caption = QLabel(label)
        caption.setProperty("role", "captionText")
        value = QLabel("--")
        value.setProperty("role", "metricValue")
        row.addWidget(caption)
        row.addStretch(1)
        row.addWidget(value)
        layout.addLayout(row)

        bar = QProgressBar()
        bar.setRange(0, 100)
        bar.setTextVisible(False)
        bar.setFixedHeight(8)
        layout.addWidget(bar)
        return bar, value

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
            self._show_case_select()
            self.lbl_status.setText(self.last_error)
            return False

        try:
            self.package = json.loads(package_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            self._clear_view()
            self.last_error = f"Failed to load 3D lesion package: {exc}"
            self._show_case_select()
            self.lbl_status.setText(self.last_error)
            return False

        summary_path = reconstruction_dir(case_id) / "pipeline_summary.json"
        try:
            self.pipeline_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            self.pipeline_summary = {}

        panel_path = lesion_panel_path(case_id)
        try:
            self.lesion_panel_entries = json.loads(panel_path.read_text(encoding="utf-8")).get("entries", [])
        except (json.JSONDecodeError, OSError):
            # Case was reconstructed before this feature existed -- fall back
            # to exactly today's behavior (every lesions_3d.json entry, all
            # enabled) instead of showing an empty panel.
            self.lesion_panel_entries = [
                {**lesion, "in_3d": True, "lesion_3d_id": lesion.get("lesion_id")}
                for lesion in self.package.get("lesions", [])
            ]

        try:
            self.content_stack.setCurrentIndex(1)
            self.btn_change_case.setVisible(True)
            self.lbl_status.setText(f"Loaded case {case_id}: {obj_path.name}")
            self._load_mesh(obj_path)
            self._load_lesions()
            self._load_lesion_panel()
            self._load_views()
            self._load_source_views(reconstruction_dir(case_id))
            self.renderer.ResetCamera()
            self.interactor.Initialize()
            QTimer.singleShot(0, self.vtk_widget.GetRenderWindow().Render)
            return True
        except Exception as exc:
            self._clear_view()
            self.last_error = f"Failed to render 3D reconstruction for case {case_id}: {exc}"
            self._show_case_select()
            self.lbl_status.setText(self.last_error)
            return False

    def _clear_view(self):
        self.package = {}
        self.pipeline_summary = {}
        self.lesion_panel_entries = []
        self._lesion_id_to_actor_index = {}
        self.renderer.RemoveAllViewProps()
        self.list_lesions.clear()
        self.list_views.clear()
        self.lbl_lesion_detail.setText("Select a lesion marker or list item.")
        self.ds_bar.setValue(0)
        self.confidence_bar.setValue(0)
        self.lbl_ds_value.setText("--")
        self.lbl_confidence_value.setText("--")
        self.btn_open_view_report.setEnabled(False)
        for thumb, button in ((self.lbl_view_a_thumb, self.btn_view_a), (self.lbl_view_b_thumb, self.btn_view_b)):
            thumb.setPixmap(QPixmap())
            thumb.setText("Not available")
            button.setEnabled(False)
            button.setChecked(False)
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
        """Builds the 3D lesion actors from lesions_3d.json (the definitive
        "what's drawn on the mesh" list). List-widget population is handled
        separately by _load_lesion_panel(), which shows the broader,
        all-views MODERATE/SEVERE panel."""
        self.lesion_actors = []
        self._lesion_id_to_actor_index = {}
        lesions = self.package.get("lesions", [])
        for lesion in lesions:
            actor = self._make_lesion_actor(lesion)
            if actor is None:
                continue
            self.renderer.AddActor(actor)
            lesion_id = lesion.get("lesion_id")
            if lesion_id is not None:
                self._lesion_id_to_actor_index[lesion_id] = len(self.lesion_actors)
            self.lesion_actors.append(actor)

    def _load_lesion_panel(self):
        self.list_lesions.clear()
        for entry in self.lesion_panel_entries:
            in_3d = bool(entry.get("in_3d"))
            marker = "●" if in_3d else "○"
            item = QListWidgetItem(
                f"{marker} {entry.get('view_label', 'View')} | {entry.get('severity')}\n"
                f"{entry.get('artery')} | narrowing {float(entry.get('DS_percent') or 0):.1f}%"
                + ("" if in_3d else "  (not in 3D view)")
            )
            item.setData(Qt.UserRole, entry)
            if not in_3d:
                item.setFlags(item.flags() & ~Qt.ItemIsEnabled & ~Qt.ItemIsSelectable)
                item.setForeground(QColor("#5C6067"))
            self.list_lesions.addItem(item)

        first_enabled = next((i for i, e in enumerate(self.lesion_panel_entries) if e.get("in_3d")), -1)
        if first_enabled >= 0:
            self.list_lesions.setCurrentRow(first_enabled)
        else:
            self.lbl_lesion_detail.setText(
                "No stenosis lesions were saved for this case."
                if not self.lesion_panel_entries
                else "No lesions from the selected views are represented in this 3D reconstruction."
            )

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

    @staticmethod
    def _find_source_image(recon_dir: Path, view_key: str) -> Optional[Path]:
        """Prefers the plain angiogram frame over the segmentation overlay.
        The original is written by scripts/dicom_3d_pipeline.py into the
        02_pipeline/ stage subfolder, which is retained under
        reconstruction_3d/ -- not copied to the top level (unlike the
        overlay/mask images), so look there too."""
        candidates = [
            recon_dir / f"{view_key}_original.png",
            recon_dir / "02_pipeline" / f"{view_key}_original.png",
            recon_dir / f"{view_key}_overlay.png",
        ]
        return next((path for path in candidates if path.exists()), None)

    def _load_source_views(self, recon_dir: Path):
        for key, thumb, button in (
            ("view_a", self.lbl_view_a_thumb, self.btn_view_a),
            ("view_b", self.lbl_view_b_thumb, self.btn_view_b),
        ):
            has_angle = isinstance(self.pipeline_summary.get(key), dict)
            image_path = self._find_source_image(recon_dir, key)
            pixmap = QPixmap(str(image_path)) if image_path is not None else QPixmap()
            if pixmap.isNull():
                thumb.setPixmap(QPixmap())
                thumb.setText("Not available")
                button.setEnabled(False)
                button.setChecked(False)
                continue
            thumb.setPixmap(pixmap.scaled(150, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            thumb.setText("")
            button.setEnabled(has_angle)
            button.setChecked(False)

    @staticmethod
    def _rotation_from_angles(primary_deg: float, secondary_deg: float) -> np.ndarray:
        """Same DICOM positioner-angle rotation used to build the mesh
        (scripts/dicom_3d_pipeline.py::rotation_from_angles) and to jump the
        standalone viewer's camera to a saved angle
        (src/artery_3d_vtk_viewer.py::rotation_from_angles)."""
        primary = math.radians(primary_deg)
        secondary = math.radians(secondary_deg)
        cp, sp = math.cos(primary), math.sin(primary)
        cs, ss = math.cos(secondary), math.sin(secondary)
        rotate_primary = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
        rotate_secondary = np.array([[1.0, 0.0, 0.0], [0.0, cs, -ss], [0.0, ss, cs]])
        return rotate_primary @ rotate_secondary

    def _set_camera_to_view(self, key: str):
        view = self.pipeline_summary.get(key)
        if not isinstance(view, dict) or self.mesh_actor is None:
            return
        try:
            primary = float(view.get("primary_angle_deg", 0.0))
            secondary = float(view.get("secondary_angle_deg", 0.0))
        except (TypeError, ValueError):
            return

        bounds = [0.0] * 6
        self.renderer.ComputeVisiblePropBounds(bounds)
        if not all(math.isfinite(v) for v in bounds) or bounds[0] > bounds[1] or bounds[2] > bounds[3] or bounds[4] > bounds[5]:
            bounds = [-50.0, 50.0, -50.0, 50.0, -50.0, 50.0]
        center = np.array([(bounds[0] + bounds[1]) * 0.5, (bounds[2] + bounds[3]) * 0.5, (bounds[4] + bounds[5]) * 0.5])
        span = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4], 1.0)

        rotation = self._rotation_from_angles(primary, secondary)
        # Local +Z is the X-ray propagation direction (source -> detector) and
        # local +Y is the detector's own row axis (positive = further DOWN the
        # image, per scripts/epipolar_optimized_centerline.py::project_points,
        # the same trusted formula used to validate reconstruction accuracy).
        # A VTK camera's "up" is a screen-space-up vector and its "direction"
        # is source->detector, so both need the opposite sign from local
        # +Z/+Y: verified against project_points on real cases -- using
        # local +Z/+Y directly (no negation) renders the model vertically
        # flipped relative to the original angiogram.
        direction = rotation @ np.array([0.0, 0.0, 1.0])
        up = rotation @ np.array([0.0, -1.0, 0.0])
        direction = direction / max(np.linalg.norm(direction), 1e-9)
        up = up - direction * float(np.dot(up, direction))
        up_norm = np.linalg.norm(up)
        up = up / up_norm if up_norm > 1e-6 else np.array([0.0, 1.0, 0.0])

        camera_position = center - direction * span * 3.0
        camera = self.renderer.GetActiveCamera()
        camera.ParallelProjectionOn()
        camera.SetFocalPoint(*center)
        camera.SetPosition(*camera_position)
        camera.SetViewUp(*up)
        camera.SetParallelScale(span * 0.62)
        self.renderer.ResetCameraClippingRange()
        self.vtk_widget.GetRenderWindow().Render()

    def _on_lesion_selected(self, row: int):
        if row < 0:
            return
        item = self.list_lesions.item(row)
        if item is None:
            return
        entry = item.data(Qt.UserRole)
        if not entry or not entry.get("in_3d"):
            # Disabled panel rows shouldn't normally become current, but
            # guard anyway rather than highlight a stale actor selection.
            self.btn_open_view_report.setEnabled(False)
            return

        actor_index = self._lesion_id_to_actor_index.get(entry.get("lesion_3d_id"))
        for i, actor in enumerate(self.lesion_actors):
            self._set_prop_opacity(actor, 1.0 if i == actor_index else 0.42)

        detail = (
            f"{entry.get('lesion_3d_id')} - {entry.get('severity')}\n"
            f"Artery: {entry.get('artery')} / {entry.get('label')}\n"
            f"DS: {float(entry.get('DS_percent') or 0):.1f}%\n"
            f"MLD: {entry.get('MLD_mm') or entry.get('MLD_px')} | "
            f"RVD: {entry.get('RVD_mm') or entry.get('RVD_px')}\n"
            f"Confidence: {entry.get('confidence')}\n"
            f"Source view: {entry.get('view_label')}"
        )
        self.lbl_lesion_detail.setText(detail)
        ds_value = max(0, min(100, int(round(float(entry.get("DS_percent") or 0)))))
        self.ds_bar.setValue(ds_value)
        self.lbl_ds_value.setText(f"{ds_value}%")
        conf = entry.get("confidence")
        conf_value = max(0, min(100, int(round(float(conf or 0) * 100))))
        self.confidence_bar.setValue(conf_value)
        self.lbl_confidence_value.setText(f"{conf_value}%" if conf is not None else "--")
        self.btn_open_view_report.setEnabled(bool(entry.get("view_report")))
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
