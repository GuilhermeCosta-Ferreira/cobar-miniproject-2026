# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np

from dataclasses import dataclass, field
from miniproject.simulation import MiniprojectSimulation

from .visualize import produce_human_view
from .hsv import get_hsv_mask
from .obstacles import get_obstacles_by_height_fast
from .velocity import get_velocity_vector
from .signal_processing import get_smooth_vision


# ================================================================
# 1. Section: Class
# ================================================================
@dataclass
class Vision:
    config: dict
    vision_smooth: np.ndarray = field(default_factory=lambda: np.zeros(2))
    current_signal: np.ndarray = field(default_factory=lambda: np.zeros(2))

    min_height: float = field(init=False)
    scare_height: float = field(init=False)
    gain: float = field(init=False)
    slow_down_rate: float = field(init=False)
    alpha: float = field(init=False)
    leaf_hue: float = field(init=False)
    leaf_saturation: float = field(init=False)
    leaf_value: float = field(init=False)
    tolerance_hue: float = field(init=False)
    tolerance_value: float = field(init=False)
    tolerance_saturation: float = field(init=False)

    current_dragonfly_score: float = 0.0
    current_dragonfly_attack: bool = False

    _velocity_history: list = field(default_factory=list)
    _centroid_history: list = field(default_factory=list)
    _mask_history: list = field(default_factory=list)
    _frame_history: list = field(default_factory=list)
    _picture_idx_history: list = field(default_factory=list)

    def __post_init__(self):
        self.min_height = float(self._get_required("min_height"))
        self.scare_height = float(self._get_required("scare_height"))
        self.gain = float(self._get_required("gain"))
        self.slow_down_rate = float(self._get_required("slow_down_rate"))
        self.alpha = float(self._get_required("alpha"))
        self.leaf_hue = float(self._get_required("leaf_hue"))
        self.leaf_saturation = float(self._get_required("leaf_saturation"))
        self.leaf_value = float(self._get_required("leaf_value"))
        self.tolerance_hue = float(self._get_required("tolerance_hue"))
        self.tolerance_value = float(self._get_required("tolerance_value"))
        self.tolerance_saturation = float(self._get_required("tolerance_saturation"))

    def _get_required(self, key: str):
        if key not in self.config:
            raise KeyError(f"Missing Vision config key: {key}")
        return self.config[key]

    # ================================================================
    # 2. Section: Properties
    # ================================================================
    @property
    def velocity_hist(self):
        return np.asarray(self._velocity_history)

    @property
    def centroid_hist(self):
        return np.asarray(self._centroid_history)

    @property
    def mask_hist(self):
        return np.asarray(self._mask_history)

    @property
    def frame_hist(self):
        return np.asarray(self._frame_history)

    @property
    def picture_idx_hist(self):
        return np.asarray(self._picture_idx_history)

    @property
    def is_active(self) -> bool:
        last_few = self.velocity_hist[-100:]
        return np.sum(last_few) != 0

    # ================================================================
    # 3. Section: Methods
    # ================================================================
    def obstacle_to_velocity(
        self,
        sim: MiniprojectSimulation,
        current_forward_vel: float,
    ) -> np.ndarray:
        step = sim._curr_step

        # 1. Get the frame as concatenated eyes
        frame = produce_human_view(sim)

        # 2. Builds a hsv dependent mask (isolate bright leafs)
        mask = get_hsv_mask(
            image=frame,
            target_hue=self.leaf_hue,
            tolerance_hue=self.tolerance_hue,
            tolerance_value=self.tolerance_value,
            tolerance_saturation=self.tolerance_saturation,
            target_saturation=self.leaf_saturation,
            target_value=self.leaf_value,
        )

        # 3. Extract the tall objects (x, y, height)
        obstacle_centroids = get_obstacles_by_height_fast(
            mask=mask, height_threshold=self.min_height
        )

        if step % 500 == 0:
            self._frame_history.append(frame)
            self._mask_history.append(mask)
            closest_centroid = (
                max(obstacle_centroids, key=lambda c: c[2])
                if len(obstacle_centroids) != 0
                else []
            )
            self._centroid_history.append(closest_centroid)
            self._picture_idx_history.append(step)

        vision_velocity = get_velocity_vector(
            image = frame,
            centroids = obstacle_centroids,
            current_forward_velocity = current_forward_vel,
            slow_down_rate=self.slow_down_rate,
            gain = self.gain,
        )

        self.vision_smooth = get_smooth_vision(
            self.vision_smooth, vision_velocity, self.alpha
        )
        self._velocity_history.append(self.vision_smooth)
        self.current_signal = self.vision_smooth

        return self.vision_smooth

    def update_dragonfly_state(self, score: float, attack: bool) -> None:
        self.current_dragonfly_score = float(score)
        self.current_dragonfly_attack = bool(attack)
