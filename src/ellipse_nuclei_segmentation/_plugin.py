import json
import os
import time
import numpy as np

from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QFileDialog,
    QSpinBox,
    QGroupBox,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
)
from qtpy.QtCore import Qt
from qtpy.QtGui import QMouseEvent

from ._nd2_loader import ND2Loader
from ._annotations import (
    AnnotationManager,
    Ellipse2D,
    LineSegment,
)
from ._visualization import ellipse_points_from_lines


ANNOTATION_COLORS = {
    "click_point": "cyan",
    "line_done": "yellow",
    "ellipse_2d": "magenta",
    "ellipse_contour": "magenta",
    "ellipse_contour_labeled": "blue",
}


class FrameSpinBox(QSpinBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._commit_callback = None
        self.editingFinished.connect(self._commit)

    def set_commit_callback(self, cb):
        self._commit_callback = cb

    def _commit(self):
        if self._commit_callback is not None:
            self._commit_callback(self.value())

    def stepBy(self, steps):
        super().stepBy(steps)
        self._commit()


class DeselectableListWidget(QListWidget):
    def mousePressEvent(self, e: QMouseEvent | None):
        if e is None:
            return
        item = self.itemAt(e.pos())
        if item is None:
            self.clearSelection()
            self.setCurrentRow(-1)
        super().mousePressEvent(e)


class NucleiAnnotatorWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        self.loader: ND2Loader | None = None
        self.annotation_mgr = AnnotationManager()
        self.current_frame = 0

        self._drawing_mode = None
        self._current_annotation = None
        self._collected_points: list = []
        self._segments_done: list = []
        self._n_points_before = 0
        self._undo_history: list = []  # ('add', ann) or ('delete', ann)

        self._ann_layer_indices: dict = {}

        self._channel_layers: list = []
        self._click_pts_layer = None
        self._lines_layer = None
        self._ellipse_layer = None

        self._build_ui()
        self.viewer.bind_key("g", self._shortcut_new_annotation, overwrite=True)
        self.viewer.bind_key("Escape", self._shortcut_escape, overwrite=True)
        self.viewer.bind_key("b", self._shortcut_toggle_bf, overwrite=True)
        self.viewer.bind_key("Meta-z", self._shortcut_undo_last, overwrite=True)
        self.viewer.bind_key(
            "Meta-BackSpace", self._shortcut_delete_selected, overwrite=True
        )
        self.viewer.mouse_double_click_callbacks = [self._on_viewer_mouse_press]

    def hideEvent(self, a0):
        for key in ("g", "Escape", "b", "Meta-z", "Meta-BackSpace"):
            try:
                self.viewer.bind_key(key, None)
            except Exception:
                pass
        self.viewer.mouse_double_click_callbacks = []
        layers_to_remove = list(self._channel_layers) + [
            self._click_pts_layer,
            self._lines_layer,
            self._ellipse_layer,
        ]
        for layer in layers_to_remove:
            if layer is not None and layer in self.viewer.layers:
                try:
                    self.viewer.layers.remove(layer)
                except Exception:
                    pass

        self._channel_layers = []
        self._click_pts_layer = None
        self._lines_layer = None
        self._ellipse_layer = None
        super().hideEvent(a0)

    @property
    def _viewer_z(self) -> float:
        try:
            return float(self.viewer.dims.current_step[0])
        except Exception:
            return 0.0

    def _build_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(6)

        # ND2 file
        grp = QGroupBox("ND2 File")
        g = QVBoxLayout()
        row = QHBoxLayout()
        self.btn_load = QPushButton("Load ND2")
        self.btn_load.clicked.connect(self._on_load_nd2)
        row.addWidget(self.btn_load)
        g.addLayout(row)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Frame:"))
        self.spin_frame = FrameSpinBox()
        self.spin_frame.setMinimum(0)
        self.spin_frame.setMaximum(0)
        self.spin_frame.setEnabled(False)
        self.spin_frame.set_commit_callback(self._on_frame_changed)
        row2.addWidget(self.spin_frame)
        self.lbl_frame_total = QLabel("/ —")
        row2.addWidget(self.lbl_frame_total)
        row2.addStretch()
        g.addLayout(row2)

        grp.setLayout(g)
        layout.addWidget(grp)

        # Save / Load
        grp = QGroupBox("Save / Load")
        g = QVBoxLayout()
        row = QHBoxLayout()
        self.btn_save = QPushButton("Save annotations")
        self.btn_save.clicked.connect(self._on_save)
        row.addWidget(self.btn_save)
        self.btn_load_ann = QPushButton("Load annotations")
        self.btn_load_ann.clicked.connect(self._on_load_annotations)
        row.addWidget(self.btn_load_ann)
        g.addLayout(row)
        row = QHBoxLayout()
        g.addLayout(row)
        grp.setLayout(g)
        layout.addWidget(grp)
        self.setLayout(layout)

        # Annotations
        grp = QGroupBox("Annotations")
        g = QVBoxLayout()
        row = QHBoxLayout()
        self.btn_2d = QPushButton("Add  [G]")
        self.btn_2d.setCheckable(True)
        self.btn_2d.clicked.connect(lambda: self._start_annotation("ellipse_2d"))
        row.addWidget(self.btn_2d)
        g.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Label:"))
        self.txt_label = QLineEdit()
        self.txt_label.setPlaceholderText("")
        row.addWidget(self.txt_label)
        g.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Notes:"))
        self.txt_notes = QLineEdit()
        self.txt_notes.setPlaceholderText("")
        row.addWidget(self.txt_notes)
        g.addLayout(row)

        self.txt_label.textChanged.connect(self._on_label_notes_changed)
        self.txt_notes.textChanged.connect(self._on_label_notes_changed)

        g.addWidget(QLabel("Annotations"))
        self.list_annotations = DeselectableListWidget()
        self.list_annotations.currentRowChanged.connect(self._on_annotation_selected)
        self.list_annotations.itemDoubleClicked.connect(
            self._on_annotation_double_clicked
        )
        g.addWidget(self.list_annotations)

        row = QHBoxLayout()
        self.btn_delete = QPushButton("Delete selected  [\u2318\u232b]")
        self.btn_delete.clicked.connect(self._on_delete_annotation)
        row.addWidget(self.btn_delete)
        self.btn_undo = QPushButton("Undo  [\u2318Z]")
        self.btn_undo.clicked.connect(lambda: self._shortcut_undo_last())
        row.addWidget(self.btn_undo)
        g.addLayout(row)

        grp.setLayout(g)
        layout.addWidget(grp)

    def _on_load_nd2(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open ND2 file", "", "ND2 files (*.nd2);;All files (*)"
        )
        if not path:
            return
        try:
            self.loader = ND2Loader(path)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load ND2:\n{e}")
            return

        self.annotation_mgr.nd2_path = path
        self.spin_frame.blockSignals(True)
        self.spin_frame.setMaximum(self.loader.n_frames - 1)
        self.spin_frame.setValue(0)
        self.spin_frame.setEnabled(True)
        self.spin_frame.blockSignals(False)
        self.lbl_frame_total.setText(f"/ {self.loader.n_frames}")
        self.current_frame = 0
        self._load_frame(0)

    def _load_frame(self, frame_id: int):
        if self.loader is None:
            return

        channels = self.loader.get_all_channels(frame_id)

        if self._channel_layers and len(self._channel_layers) == len(channels):
            # Update data in-place to preserve visibility and colormap settings
            for layer, ch_data in zip(self._channel_layers, channels):
                if layer in self.viewer.layers:
                    layer.data = ch_data
        else:
            # First load or channel count changed — create layers from scratch
            for layer in self._channel_layers:
                if layer in self.viewer.layers:
                    self.viewer.layers.remove(layer)
            self._channel_layers = []
            for i, ch_data in enumerate(channels):
                layer = self.viewer.add_image(
                    ch_data,
                    name=str(i),
                    colormap="gray",
                    blending="additive",
                    opacity=0.8,
                )
                self._channel_layers.append(layer)

        self._ensure_overlay_layers()
        self.current_frame = frame_id
        self._refresh_annotation_visuals()

    def _on_frame_changed(self, value):
        self._load_frame(value)

    def _ensure_overlay_layers(self):
        ndim = 3 if self.loader else 2

        if self._lines_layer is None or self._lines_layer not in self.viewer.layers:
            self._lines_layer = self.viewer.add_shapes(
                name="Lines",
                shape_type="line",
                edge_color=ANNOTATION_COLORS["line_done"],
                edge_width=1.0,
                ndim=ndim,
            )
            self._lines_layer.mode = "pan_zoom"

        if self._ellipse_layer is None or self._ellipse_layer not in self.viewer.layers:
            self._ellipse_layer = self.viewer.add_shapes(
                name="Ellipses",
                shape_type="polygon",
                edge_color=ANNOTATION_COLORS["ellipse_contour"],
                face_color="transparent",
                edge_width=1.0,
                ndim=ndim,
            )
            self._ellipse_layer.mode = "pan_zoom"

        if (
            self._click_pts_layer is None
            or self._click_pts_layer not in self.viewer.layers
        ):
            self._click_pts_layer = self.viewer.add_points(
                name="Points",
                size=1.0,
                face_color=ANNOTATION_COLORS["click_point"],
                border_color="white",
                border_width=0.5,
                ndim=ndim,
            )
            self._click_pts_layer.mode = "pan_zoom"
            self._click_pts_layer.events.data.connect(self._on_point_added)

    def _set_overlay_layers_visible(self, visible: bool):
        for layer in (self._click_pts_layer, self._lines_layer):
            if layer is not None and layer in self.viewer.layers:
                layer.visible = visible

    def _start_annotation(self, mode: str):
        if not self.btn_2d.isChecked():
            self._cancel_annotation()
            return

        self._drawing_mode = mode
        self.btn_2d.setText("Cancel  [Esc]")
        self._collected_points = []
        self._segments_done = []

        self._current_annotation = Ellipse2D(
            frame=self.current_frame,
            z_slice=self._viewer_z,
            label=self.txt_label.text(),
            metadata={"notes": self.txt_notes.text()},
        )

        self._ensure_overlay_layers()
        self._set_overlay_layers_visible(True)
        self._n_points_before = len(self._click_pts_layer.data)
        self._click_pts_layer.mode = "add"
        self.viewer.layers.selection.active = self._click_pts_layer

        ann_id = self._current_annotation.id
        self._ann_layer_indices[ann_id] = {
            "point_indices": [],
            "shape_indices": [],
            "ellipse_indices": [],
        }

    def _snap_point_to_perp(self, raw_pt: list) -> list:
        pt = list(raw_pt)
        line_idx = len(self._segments_done)  # 0 = drawing line 1, 1 = drawing line 2
        pts_in_seg = len(self._collected_points)  # 0 = first click, 1 = second click

        if pts_in_seg == 0:
            if self._segments_done:
                pt[0] = self._segments_done[0].start[0]
            return pt

        p1 = list(self._collected_points[0])

        pt[0] = p1[0]

        if line_idx == 1:
            seg1 = self._segments_done[0]
            s1 = np.array(seg1.start[1:], dtype=float)  # (y, x)
            e1 = np.array(seg1.end[1:], dtype=float)
            dir1 = e1 - s1
            norm1 = np.linalg.norm(dir1)
            if norm1 > 1e-9:
                dir1 /= norm1
                perp = np.array([-dir1[1], dir1[0]])

                anchor = np.array(p1[1:], dtype=float)
                v = np.array(pt[1:], dtype=float) - anchor
                proj_len = float(np.dot(v, perp))
                new_yx = anchor + proj_len * perp
                pt[1] = float(new_yx[0])
                pt[2] = float(new_yx[1])

        return pt

    def _on_point_added(self, event):
        if self._drawing_mode is None:
            return

        layer = self._click_pts_layer
        n_now = len(layer.data)
        if n_now <= self._n_points_before:
            return

        raw_pt = layer.data[-1].tolist()
        snapped_pt = self._snap_point_to_perp(raw_pt)

        if snapped_pt != raw_pt:
            layer.events.data.disconnect(self._on_point_added)
            new_data = layer.data.copy()
            new_data[-1] = snapped_pt
            layer.data = new_data
            layer.events.data.connect(self._on_point_added)

        self._collected_points.append(snapped_pt)
        self._n_points_before = n_now

        ann_id = self._current_annotation.id
        self._ann_layer_indices[ann_id]["point_indices"].append(n_now - 1)

        if len(self._collected_points) == 2:
            p1, p2 = self._collected_points
            seg = LineSegment(start=p1, end=p2)
            self._segments_done.append(seg)
            self._collected_points = []

            shape_idx = len(self._lines_layer.data)
            self._lines_layer.add(
                [np.array([p1, p2])],
                shape_type=["line"],
                edge_color=[ANNOTATION_COLORS["line_done"]],
                edge_width=[2.5],
            )
            self._ann_layer_indices[ann_id]["shape_indices"].append(shape_idx)

            idx = len(self._segments_done)
            ann = self._current_annotation
            if idx == 1:
                ann.line1 = seg
            elif idx == 2:
                ann.line2 = seg

            if len(self._segments_done) >= 2:
                self._finalize_annotation()
                return

    def _snap_line2_to_center(self, ann) -> None:
        if ann.line1 is None or ann.line2 is None:
            return
        s1 = np.array(ann.line1.start[1:], dtype=float)
        e1 = np.array(ann.line1.end[1:], dtype=float)
        dir1 = e1 - s1
        norm1 = np.linalg.norm(dir1)
        if norm1 < 1e-9:
            return
        dir1 /= norm1
        perp = np.array([-dir1[1], dir1[0]])
        center = (s1 + e1) / 2.0

        s2 = np.array(ann.line2.start[1:], dtype=float)
        e2 = np.array(ann.line2.end[1:], dtype=float)
        half_len2 = np.linalg.norm(e2 - s2) / 2.0

        new_s2 = center - half_len2 * perp
        new_e2 = center + half_len2 * perp
        z = ann.line1.start[0]
        ann.line2 = LineSegment(
            start=[z, float(new_s2[0]), float(new_s2[1])],
            end=[z, float(new_e2[0]), float(new_e2[1])],
        )

    def _finalize_annotation(self):
        ann = self._current_annotation
        ann.label = self.txt_label.text()
        ann.metadata["notes"] = self.txt_notes.text()

        if ann.line1 is not None:
            ann.z_slice = float(
                int(round((ann.line1.start[0] + ann.line1.end[0]) / 2.0))
            )

        self._snap_line2_to_center(ann)
        self.annotation_mgr.add(ann)
        self._undo_history.append(("add", ann))

        self._drawing_mode = None
        self._current_annotation = None
        self._collected_points = []
        self._segments_done = []
        self.btn_2d.setChecked(False)
        self.btn_2d.setText("Add  [G]")
        self.txt_notes.blockSignals(True)
        self.txt_notes.clear()
        self.txt_notes.blockSignals(False)
        self._set_pan_zoom()
        self._set_overlay_layers_visible(False)

        self._refresh_annotation_list()
        self._refresh_annotation_visuals()

    def _cancel_annotation(self):
        if self._current_annotation is not None:
            self._ann_layer_indices.pop(self._current_annotation.id, None)

        self._drawing_mode = None
        self._current_annotation = None
        self._collected_points = []
        self._segments_done = []
        self.btn_2d.setChecked(False)
        self.btn_2d.setText("Add  [G]")
        self._set_pan_zoom()
        self._set_overlay_layers_visible(False)
        self._refresh_annotation_visuals()

    def _shortcut_escape(self, viewer=None):
        if self._drawing_mode is not None:
            self._cancel_annotation()

    def _shortcut_undo_last(self, viewer=None):
        if self._drawing_mode is not None:
            self._cancel_annotation()
        elif self._undo_history:
            action, ann = self._undo_history.pop()
            if action == "add":
                self._remove_annotation_by_id(ann.id, push_undo=False)
            elif action == "delete":
                self.annotation_mgr.add(ann)
                self._refresh_annotation_list()
                self._refresh_annotation_visuals()

    def _shortcut_delete_selected(self, viewer=None):
        self._on_delete_annotation()

    def _set_pan_zoom(self):
        if self._click_pts_layer is not None:
            self._click_pts_layer.mode = "pan_zoom"
        if self._lines_layer is not None:
            self._lines_layer.mode = "pan_zoom"

    @staticmethod
    def _resnap_line2_perp(line1: LineSegment, line2: LineSegment) -> LineSegment:
        s1 = np.array(line1.start[1:], dtype=float)
        e1 = np.array(line1.end[1:], dtype=float)
        dir1 = e1 - s1
        norm1 = np.linalg.norm(dir1)
        if norm1 < 1e-9:
            return line2

        dir1 /= norm1
        perp = np.array([-dir1[1], dir1[0]])

        s2 = np.array(line2.start[1:], dtype=float)
        e2 = np.array(line2.end[1:], dtype=float)
        mid2 = (s2 + e2) / 2.0
        half_len2 = np.linalg.norm(e2 - s2) / 2.0

        new_s2 = mid2 - half_len2 * perp
        new_e2 = mid2 + half_len2 * perp

        z = line2.start[0]
        return LineSegment(
            start=[z, float(new_s2[0]), float(new_s2[1])],
            end=[z, float(new_e2[0]), float(new_e2[1])],
        )

    def _redraw_annotation(self, ann):
        ann_id = ann.id
        rec = self._ann_layer_indices.get(ann_id, {})

        if self._lines_layer is not None and self._lines_layer in self.viewer.layers:
            for idx in sorted(rec.get("shape_indices", []), reverse=True):
                try:
                    self._lines_layer.selected_data = {idx}
                    self._lines_layer.remove_selected()
                except Exception:
                    pass
        if (
            self._ellipse_layer is not None
            and self._ellipse_layer in self.viewer.layers
        ):
            for idx in sorted(rec.get("ellipse_indices", []), reverse=True):
                try:
                    self._ellipse_layer.selected_data = {idx}
                    self._ellipse_layer.remove_selected()
                except Exception:
                    pass

        self._ann_layer_indices[ann_id] = {
            "point_indices": rec.get("point_indices", []),
            "shape_indices": [],
            "ellipse_indices": [],
        }

        if ann.is_complete:
            new_rec = self._ann_layer_indices[ann_id]

            for seg in [ann.line1, ann.line2]:
                shape_idx = len(self._lines_layer.data)
                self._lines_layer.add(
                    [np.array([seg.start, seg.end])],
                    shape_type=["line"],
                    edge_color=[ANNOTATION_COLORS["line_done"]],
                    edge_width=[2.5],
                )
                new_rec["shape_indices"].append(shape_idx)

            self._add_ellipse_contour(ann, new_rec)

    def _ellipse_color_for(self, ann) -> str:
        has_label = bool((ann.label or "").strip())
        has_notes = bool((ann.metadata.get("notes", "") or "").strip())
        if has_label or has_notes:
            return ANNOTATION_COLORS["ellipse_contour_labeled"]
        return ANNOTATION_COLORS["ellipse_contour"]

    def _add_ellipse_contour(self, ann, rec):
        if not ann.is_complete:
            return

        pts_2d = ellipse_points_from_lines(
            ann.line1.start[-2:],
            ann.line1.end[-2:],
            ann.line2.start[-2:],
            ann.line2.end[-2:],
            n_points=64,
        )
        z_val = ann.z_slice
        pts_3d = np.column_stack([np.full(len(pts_2d), z_val), pts_2d])

        ell_idx = len(self._ellipse_layer.data)
        self._ellipse_layer.add(
            [pts_3d],
            shape_type=["polygon"],
            edge_color=[self._ellipse_color_for(ann)],
            face_color=["transparent"],
            edge_width=[1.5],
        )
        rec["ellipse_indices"].append(ell_idx)

    def _refresh_annotation_visuals(self):
        if (
            self._lines_layer is None
            or self._lines_layer not in self.viewer.layers
            or self._ellipse_layer is None
            or self._ellipse_layer not in self.viewer.layers
            or self._click_pts_layer is None
            or self._click_pts_layer not in self.viewer.layers
        ):
            return

        self._lines_layer.data = []
        self._ellipse_layer.data = []

        anns = [
            a
            for a in self.annotation_mgr.get_for_frame(self.current_frame)
            if a.is_complete
        ]

        all_pts = []
        for ann in anns:
            for seg in [ann.line1, ann.line2]:
                all_pts.append(seg.start)
                all_pts.append(seg.end)

        self._click_pts_layer.selected_data = set()
        self._click_pts_layer.events.data.disconnect(self._on_point_added)
        if all_pts:
            self._click_pts_layer.data = np.array(all_pts, dtype=float)
        else:
            ndim = 3 if self.loader else 2
            self._click_pts_layer.data = np.empty((0, ndim), dtype=float)
        self._click_pts_layer.events.data.connect(self._on_point_added)

        self._ann_layer_indices = {}
        pt_idx = 0
        for ann in anns:
            rec = {
                "point_indices": [pt_idx, pt_idx + 1, pt_idx + 2, pt_idx + 3],
                "shape_indices": [],
                "ellipse_indices": [],
            }
            self._ann_layer_indices[ann.id] = rec
            pt_idx += 4

            for seg in [ann.line1, ann.line2]:
                shape_idx = len(self._lines_layer.data)
                self._lines_layer.add(
                    [np.array([seg.start, seg.end])],
                    shape_type=["line"],
                    edge_color=[ANNOTATION_COLORS["line_done"]],
                    edge_width=[2.5],
                )
                rec["shape_indices"].append(shape_idx)

            self._add_ellipse_contour(ann, rec)

    def _remove_annotation_by_id(self, ann_id: str, push_undo: bool = True):
        if push_undo:
            ann = self.annotation_mgr.get(ann_id)
            if ann is not None:
                self._undo_history.append(("delete", ann))
        self._ann_layer_indices.pop(ann_id, None)
        self.annotation_mgr.remove(ann_id)
        self._refresh_annotation_list()
        self._refresh_annotation_visuals()

    def _refresh_annotation_list(self):
        self.list_annotations.clear()
        for ann in self.annotation_mgr.annotations:
            notes = ann.metadata.get("notes", "").strip()
            text = f"{ann.id}  (frame {ann.frame}, z={ann.z_slice:.1f})"
            if ann.label:
                text += f"  {ann.label}"
            if notes:
                text += f"  [{notes}]"
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, ann.id)  # type: ignore
            self.list_annotations.addItem(item)

    def _on_label_notes_changed(self):
        if self._drawing_mode is not None or self._current_annotation is not None:
            return
        row = self.list_annotations.currentRow()
        if row < 0:
            return
        item = self.list_annotations.item(row)
        if item is None:
            return
        ann_id = item.data(Qt.UserRole)  # type: ignore
        ann = self.annotation_mgr.get(ann_id)
        if ann is None:
            return
        ann.label = self.txt_label.text()
        ann.metadata["notes"] = self.txt_notes.text()
        self._redraw_annotation(ann)
        self.list_annotations.blockSignals(True)
        self._refresh_annotation_list()
        for i in range(self.list_annotations.count()):
            if self.list_annotations.item(i).data(Qt.UserRole) == ann_id:  # type: ignore
                self.list_annotations.setCurrentRow(i)
                break
        self.list_annotations.blockSignals(False)

    def _on_annotation_selected(self, row):
        if row < 0:
            return
        item = self.list_annotations.item(row)
        ann_id = item.data(Qt.UserRole)  # type: ignore
        ann = self.annotation_mgr.get(ann_id)
        if ann is None:
            return
        self.txt_label.blockSignals(True)
        self.txt_notes.blockSignals(True)
        self.txt_label.setText(ann.label or "")
        self.txt_notes.setText(ann.metadata.get("notes", "") or "")
        self.txt_label.blockSignals(False)
        self.txt_notes.blockSignals(False)
        if ann.frame != self.current_frame:
            self.spin_frame.blockSignals(True)
            self.spin_frame.setValue(ann.frame)
            self.spin_frame.blockSignals(False)
            self._load_frame(ann.frame)

    def _on_annotation_double_clicked(self, item):
        ann_id = item.data(Qt.UserRole)  # type: ignore
        ann = self.annotation_mgr.get(ann_id)
        if ann is None:
            return
        if ann.frame != self.current_frame:
            self.spin_frame.blockSignals(True)
            self.spin_frame.setValue(ann.frame)
            self.spin_frame.blockSignals(False)
            self._load_frame(ann.frame)
        try:
            z_step = int(round(ann.z_slice))
            self.viewer.dims.set_current_step(0, z_step)
        except Exception:
            pass
        try:
            center = ann.center
            if center is not None:
                cy, cx = float(center[-2]), float(center[-1])
                scale = self.loader.scale if self.loader else (1, 1, 1)
                self.viewer.camera.center = (
                    float(ann.z_slice) * scale[0],
                    cy * scale[1],
                    cx * scale[2],
                )
        except Exception:
            pass

    def _on_delete_annotation(self):
        row = self.list_annotations.currentRow()
        if row < 0:
            return
        item = self.list_annotations.item(row)
        if item is None:
            return
        ann_id = item.data(Qt.UserRole)  # type: ignore[attr-defined]
        self._remove_annotation_by_id(ann_id)

    def _on_save(self):
        ts = time.strftime("%Y%m%d_%H%M%S")
        default_name = f"annotations_{ts}.json"
        base_dir = self.annotation_mgr.output_dir or ""
        default_path = (
            os.path.join(base_dir, default_name) if base_dir else default_name
        )
        path, _ = QFileDialog.getSaveFileName(
            self, "Save annotations", default_path, "JSON files (*.json)"
        )
        if not path:
            return
        # Snapshot current colormaps for each channel layer
        colormaps = {}
        for i, layer in enumerate(self._channel_layers):
            if layer in self.viewer.layers:
                colormaps[str(i)] = layer.colormap.name
        self.annotation_mgr.layer_colormaps = colormaps
        filename = os.path.basename(path)
        self.annotation_mgr.output_dir = os.path.dirname(path)
        self.annotation_mgr.save(filename)

    def _on_load_annotations(self):
        if self.loader is None:
            QMessageBox.warning(self, "No ND2 file", "Please load an ND2 file first.")
            return

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load annotations",
            self.annotation_mgr.output_dir or "",
            "JSON files (*.json)",
        )
        if not path:
            return

        try:
            with open(path, "r") as f:
                file_data = json.load(f)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to read file:\n{e}")
            return

        file_nd2 = file_data.get("nd2_path", "")
        current_nd2 = self.annotation_mgr.nd2_path
        if file_nd2 and current_nd2:
            if os.path.basename(file_nd2) != os.path.basename(current_nd2):
                msg = (
                    f"Annotation file was created for:\n"
                    f"  {os.path.basename(file_nd2)}\n\n"
                    f"Currently loaded:\n"
                    f"  {os.path.basename(current_nd2)}\n\n"
                    f"Load anyway?"
                )
                reply = QMessageBox.warning(
                    self,
                    "File Mismatch",
                    msg,
                    QMessageBox.Ok | QMessageBox.Cancel,  # type: ignore[attr-defined]
                )
                if reply != QMessageBox.Ok:  # type: ignore[attr-defined]
                    return

        dirname, filename = os.path.dirname(path), os.path.basename(path)
        old = self.annotation_mgr.output_dir
        self.annotation_mgr.output_dir = dirname
        self.annotation_mgr.load(filename)
        self.annotation_mgr.output_dir = old

        # Apply saved colormaps to channel layers
        colormaps = self.annotation_mgr.layer_colormaps
        for i, layer in enumerate(self._channel_layers):
            key = str(i)
            if key in colormaps and layer in self.viewer.layers:
                try:
                    layer.colormap = colormaps[key]
                except Exception:
                    pass

        self._ann_layer_indices = {}
        self._refresh_annotation_list()
        self._set_overlay_layers_visible(True)
        self._refresh_annotation_visuals()

    def _shortcut_toggle_bf(self, viewer=None):
        if len(self._channel_layers) > 1:
            layer = self._channel_layers[1]
            if layer in self.viewer.layers:
                layer.visible = not layer.visible

    def _shortcut_new_annotation(self, viewer=None):
        if self._drawing_mode is not None:
            # Already drawing – cancel and restart
            self._cancel_annotation()
        self.btn_2d.setChecked(True)
        self._start_annotation("ellipse_2d")

    def _on_viewer_mouse_press(self, viewer, event):
        if self._drawing_mode is not None:
            return
        if event.button != 1:
            return
        if not self.annotation_mgr.annotations:
            return

        pos = np.array(event.position)
        if len(pos) < 2:
            return

        click_y = float(pos[-2])
        click_x = float(pos[-1])

        best_ann = None
        best_dist = float("inf")

        for ann in self.annotation_mgr.get_for_frame(self.current_frame):
            if not ann.is_complete:
                continue
            center = ann.center
            if center is None:
                continue

            cy = float(center[-2])
            cx = float(center[-1])

            s1 = np.array(ann.line1.start[-2:], dtype=float)
            e1 = np.array(ann.line1.end[-2:], dtype=float)
            s2 = np.array(ann.line2.start[-2:], dtype=float)
            e2 = np.array(ann.line2.end[-2:], dtype=float)

            axis1 = (e1 - s1) / 2.0
            axis2 = (e2 - s2) / 2.0

            a = np.linalg.norm(axis1)
            b = np.linalg.norm(axis2)
            if a < 1e-9 or b < 1e-9:
                continue

            u1 = axis1 / a
            u2 = axis2 / b

            dv = np.array([click_y - cy, click_x - cx])
            coord1 = np.dot(dv, u1) / a
            coord2 = np.dot(dv, u2) / b

            if coord1**2 + coord2**2 <= 1.0:
                dist = float(np.linalg.norm(dv))
                if dist < best_dist:
                    best_dist = dist
                    best_ann = ann

        if best_ann is not None:
            for i in range(self.list_annotations.count()):
                item = self.list_annotations.item(i)
                if item is not None and item.data(Qt.UserRole) == best_ann.id:  # type: ignore[attr-defined]
                    self.list_annotations.setCurrentRow(i)
                    # Navigate to the annotation's z slice
                    try:
                        z_step = int(round(best_ann.z_slice))
                        self.viewer.dims.set_current_step(0, z_step)
                    except Exception:
                        pass
                    break
