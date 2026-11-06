from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class Config:
    video: str = 'videos/cows.mp4'
    output_dir: str = 'outputs/demo'
    weights: str = 'yolo11n.pt'
    tracker: str = 'bytetrack.yaml'
    device: str = 'cpu'
    cow_class: int = 19
    conf: float = 0.12
    imgsz: int = 960
    max_side: int = 960
    max_seconds: float = 10.0
    stride: int = 1
    enhancement: str = 'auto'  # auto, always, off
    use_identity_recovery: bool = True
    use_zones: bool = True
    use_safety: bool = True
    use_behavior: bool = True
    # Polygon coordinates in normalized image coordinates (0..1)
    zones: dict = field(default_factory=lambda: {
        'feeding_zone': [[.05,.35],[.52,.35],[.52,.92],[.05,.92]],
        'resting_zone': [[.55,.25],[.96,.25],[.96,.85],[.55,.85]],
    })
    robot_path: list = field(default_factory=lambda: [[.10,.72],[.90,.72]])
    safety_radius_px: float = 70.0  # demo image pixels, NOT physical distance
    warning_horizon_s: float = 2.0
    posture_model_path: str | None = None  # optional custom classifier
