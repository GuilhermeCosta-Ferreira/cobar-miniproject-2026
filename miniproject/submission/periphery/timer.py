# ================================================================
# 0. Section: IMPORTS
# ================================================================
import time
from contextlib import contextmanager


# ================================================================
# 1. Section: Functions
# ================================================================
@contextmanager
def timer(name: str, timings: dict[str, float]):
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    timings[name] = end - start


# ──────────────────────────────────────────────────────
# 1.1 Subsection: Helper Functions
# ──────────────────────────────────────────────────────
def print_timings(timings: dict[str, float]) -> None:
    total = sum(timings.values())

    print("\nTiming breakdown:")
    for name, seconds in timings.items():
        percentage = 100 * seconds / total if total > 0 else 0
        print(f"{name}: {seconds * 1000:.3f} ms ({percentage:.1f}%)")

    print(f"Total: {total * 1000:.3f} ms\n")
