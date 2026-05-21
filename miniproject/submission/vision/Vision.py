import numpy as np


class Vision:
    def __init__(self, max_size: int = 100):
        self.signal_history: list[np.ndarray] = []
        self.max_size = max_size
        self.current_signal = [0.0, 0.0]

        self.current_dragonfly_score: float = 0.0
        self.current_dragonfly_attack: bool = False

    @property
    def history_size(self) -> int:
        return len(self.signal_history)

    @property
    def history_sum(self) -> float:
        if self.history_size == 0:
            return 0.0

        return float(np.sum(self.signal_history))

    def add_signal(self, signal: np.ndarray) -> None:
        self.signal_history.append(signal)
        self.current_signal = signal

        if self.history_size > self.max_size:
            self.signal_history.pop(0)

    def update_dragonfly_state(self, score: float, attack: bool) -> None:
        self.current_dragonfly_score = float(score)
        self.current_dragonfly_attack = bool(attack)

    @property
    def is_active(self) -> bool:
        if self.history_size == self.max_size:
            return self.history_sum > 0

        return False
