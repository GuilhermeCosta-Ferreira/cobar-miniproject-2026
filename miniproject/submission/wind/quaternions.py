import numpy as np


def mujoco_to_numpy_quaternion(mujoco_quat: np.ndarray) -> np.ndarray:
    """Convert a Mujoco quaternion (w, x, y, z) to a numpy quaternion (x, y, z, w)."""
    w, x, y, z = mujoco_quat
    return np.array([x, y, z, w])


def numpy_to_mujoco_quaternion(numpy_quat: np.ndarray) -> np.ndarray:
    """Convert a numpy quaternion (x, y, z, w) to a Mujoco quaternion (w, x, y, z)."""
    x, y, z, w = numpy_quat
    return np.array([w, x, y, z])


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions. Assumes MuJoco convention (w, x, y, z)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def quat_inv(q: np.ndarray) -> np.ndarray:
    """Compute the inverse of a quaternion. Assumes MuJoco convention (w, x, y, z)."""
    w, x, y, z = q
    norm_sq = w * w + x * x + y * y + z * z
    return np.array([w, -x, -y, -z]) / norm_sq


def quat_rotate(q, v):
    """Rotate a vector v by a quaternion q. Assumes MuJoco convention (w, x, y, z)."""
    # sandwich product: q * [0,v] * q_inv
    v_quat = np.array([0, *v])
    return quat_mul(quat_mul(q, v_quat), quat_inv(q))[1:]
