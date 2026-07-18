from dataclasses import dataclass

@dataclass
class TrackingConfig:
    threshold: int = 30
    min_area_px: int = 100
    max_area_px: int = 100000
    blur_kernel: int = 7
    morphology_kernel: int = 3
    method: str = "dark"
    max_jump_px: float = 80.0
    reference_frames: int = 30
    resize_factor: float = 0.5
    analyze_only_inside_rois: bool = True
    crop_to_roi_bounds: bool = True
    roi_crop_padding_px: int = 20
    draw_trajectory: bool = True
    trajectory_tail_frames: int = 300
