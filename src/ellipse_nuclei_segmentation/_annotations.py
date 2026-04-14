import json
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class LineSegment:
    start: list
    end: list

    @property
    def midpoint(self):
        s = np.array(self.start)
        e = np.array(self.end)
        return ((s + e) / 2.0).tolist()

    @property
    def length(self):
        s = np.array(self.start)
        e = np.array(self.end)
        return float(np.linalg.norm(e - s))

    @property
    def half_length(self):
        return self.length / 2.0

    def to_dict(self):
        return {"start": self.start, "end": self.end}

    @classmethod
    def from_dict(cls, d):
        return cls(start=d["start"], end=d["end"])


@dataclass
class Ellipse2D:
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    frame: int = 0
    z_slice: float = 0.0
    line1: Optional[LineSegment] = None
    line2: Optional[LineSegment] = None
    label: str = ""
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)

    @property
    def center(self):
        if self.line1 is None:
            return None
        if self.line2 is None:
            return self.line1.midpoint
        m1 = np.array(self.line1.midpoint)
        m2 = np.array(self.line2.midpoint)
        return ((m1 + m2) / 2.0).tolist()

    @property
    def semi_axes(self):
        """Return (semi_axis_1, semi_axis_2) lengths."""
        a = self.line1.half_length if self.line1 else 0
        b = self.line2.half_length if self.line2 else 0
        return (a, b)

    @property
    def orientation_deg(self):
        if self.line1 is None:
            return 0.0
        s = np.array(self.line1.start)
        e = np.array(self.line1.end)
        diff = e - s
        angle = np.degrees(np.arctan2(diff[-2], diff[-1]))
        return float(angle)

    @property
    def is_complete(self):
        return self.line1 is not None and self.line2 is not None

    def to_dict(self):
        d = {
            "id": self.id,
            "frame": self.frame,
            "z_slice": self.z_slice,
            "line1": self.line1.to_dict() if self.line1 else None,
            "line2": self.line2.to_dict() if self.line2 else None,
            "label": self.label,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }
        return d

    @classmethod
    def from_dict(cls, d):
        obj = cls(
            id=d["id"],
            frame=d.get("frame", 0),
            z_slice=d.get("z_slice", 0.0),
            line1=LineSegment.from_dict(d["line1"]) if d.get("line1") else None,
            line2=LineSegment.from_dict(d["line2"]) if d.get("line2") else None,
            label=d.get("label", ""),
            timestamp=d.get("timestamp", time.time()),
            metadata=d.get("metadata", {}),
        )
        return obj


class AnnotationManager:
    def __init__(self, output_dir: Optional[str] = None, checkpoint_interval: int = 30):
        self.output_dir = output_dir
        self.checkpoint_interval = checkpoint_interval
        self.annotations: list = []
        self.nd2_path: str = ""
        self._last_checkpoint = time.time()

    def add(self, annotation):
        self.annotations.append(annotation)
        self.save_checkpoint()

    def remove(self, annotation_id: str):
        self.annotations = [a for a in self.annotations if a.id != annotation_id]

    def update(self, annotation_id: str, **kwargs):
        for ann in self.annotations:
            if ann.id == annotation_id:
                for k, v in kwargs.items():
                    if hasattr(ann, k):
                        setattr(ann, k, v)
                ann.timestamp = time.time()
                break

    def get(self, annotation_id: str):
        for ann in self.annotations:
            if ann.id == annotation_id:
                return ann
        return None

    def get_for_frame(self, frame: int):
        return [a for a in self.annotations if a.frame == frame]

    def to_dict_list(self):
        return [a.to_dict() for a in self.annotations]

    def save(self, filename: str = "annotations.json"):
        if self.output_dir is None:
            return None
        os.makedirs(self.output_dir, exist_ok=True)
        filepath = os.path.join(self.output_dir, filename)
        data = {
            "version": "1.0",
            "timestamp": time.time(),
            "nd2_path": self.nd2_path,
            "n_annotations": len(self.annotations),
            "annotations": self.to_dict_list(),
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        return filepath

    def load(self, filename: str = "annotations.json"):
        if self.output_dir is None:
            return
        filepath = os.path.join(self.output_dir, filename)
        if not os.path.exists(filepath):
            return

        with open(filepath, "r") as f:
            data = json.load(f)

        self.annotations = []
        for d in data.get("annotations", []):
            self.annotations.append(Ellipse2D.from_dict(d))

    def save_checkpoint(self):
        if self.output_dir is None:
            return None
        ckpt_dir = os.path.join(self.output_dir, ".napari_checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        filename = f"checkpoint_{ts}.json"
        filepath = os.path.join(ckpt_dir, filename)

        data = {
            "version": "1.0",
            "timestamp": time.time(),
            "nd2_path": self.nd2_path,
            "n_annotations": len(self.annotations),
            "annotations": self.to_dict_list(),
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        self._last_checkpoint = time.time()
        return filepath

    def load_latest_checkpoint(self):
        if self.output_dir is None:
            return False
        ckpt_dir = os.path.join(self.output_dir, ".napari_checkpoints")
        if not os.path.exists(ckpt_dir):
            return False

        files = sorted(
            [f for f in os.listdir(ckpt_dir) if f.startswith("checkpoint_")],
            reverse=True,
        )
        if not files:
            return False

        filepath = os.path.join(ckpt_dir, files[0])
        with open(filepath, "r") as f:
            data = json.load(f)

        self.annotations = []
        for d in data.get("annotations", []):
            self.annotations.append(Ellipse2D.from_dict(d))

        return True
