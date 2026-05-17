import numpy as np


class Vision:
    def __init__(self, max_size: int = 100):
        self.signal_history: list[np.ndarray] = []
        self.max_size = max_size
        self.current_signal = [0.0, 0.0]

    @property
    def history_size(self) -> int:
        return len(self.signal_history)

    @property
    def history_sum(self) -> float:
        return np.sum(self.signal_history)

    def add_signal(self, signal: np.ndarray) -> None:
        self.signal_history.append(signal)
        self.current_signal = signal

        if self.history_size > self.max_size:
            self.signal_history.pop(0)

    @property
    def is_active(self) -> bool:
        if self.history_size == self.max_size:
            return self.history_sum > 0

        return False
