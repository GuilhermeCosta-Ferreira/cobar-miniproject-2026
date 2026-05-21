# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np

from dataclasses import dataclass, field
from miniproject.simulation import MiniprojectSimulation

from .visualize import produce_human_view
from .hsv import get_hsv_mask_fast
from .obstacles import get_obstacles_by_height_fast
from .velocity import get_velocity_vector
from .signal_processing import get_smooth_vision


# ================================================================
# 1. Section: Class
# ================================================================
@dataclass
class Vision:
    max_size: int = 100
    vision_smooth: np.ndarray = field(default_factory=lambda: np.zeros(2))
    current_signal: np.ndarray = field(default_factory=lambda: np.zeros(2))
    alpha: float = 0.05

    target_hue: float = 120.0
    tolerance_hue: float = 5.0
    min_saturation: float = 0.3
    min_value: float = 0.79
    height_threshold: int = 200

    current_dragonfly_score: float = 0.0
    current_dragonfly_attack: bool = False

    _velocity_history: list = field(default_factory=list)
    _centroid_history: list = field(default_factory=list)
    _mask_history: list = field(default_factory=list)
    _frame_history: list = field(default_factory=list)



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
        mask = get_hsv_mask_fast(
            image = frame,
            target_hue = self.target_hue,
            tolerance_hue = self.tolerance_hue,
            min_saturation = self.min_saturation,
            min_value = self.min_value,
        )

        # 3. Extract the tall objects (x, y, height)
        obstacle_centroids = get_obstacles_by_height_fast(
            mask = mask,
            height_threshold = self.height_threshold
        )

        if step % 5000 == 0:
            self._frame_history.append(frame)
            self._mask_history.append(mask)
            closest_centroid = max(obstacle_centroids, key=lambda c: c[2]) if len(obstacle_centroids) != 0 else []
            self._centroid_history.append(closest_centroid)

        vision_velocity = get_velocity_vector(
            current_forward_velocity = current_forward_vel,
            turn_speed=1.5,
            image = frame,
            centroids = obstacle_centroids,
            scary_height = 300
        )

        self.vision_smooth = get_smooth_vision(self.vision_smooth, vision_velocity, self.alpha)
        self.current_signal = self.vision_smooth

        self._velocity_history.append(self.vision_smooth)

        return self.vision_smooth

    def update_dragonfly_state(self, score: float, attack: bool) -> None:
        self.current_dragonfly_score = float(score)
        self.current_dragonfly_attack = bool(attack)
