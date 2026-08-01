"""
    Noise_Value = Original_Value + Laplace(0, beta)
    beta = NOISE_SCALE * |Original_Value|
    NOISE_SCALE = 0.05  (5%)
"""

import numpy as np

NOISE_SCALE = 0.05  


def add_laplace_noise(value: float, noise_scale: float = NOISE_SCALE, rng: np.random.Generator = None) -> float:
    """Add Laplace-distributed noise proportional to the magnitude of `value`."""
    if rng is None:
        rng = np.random.default_rng()
    beta = noise_scale * abs(value)
    if beta == 0:
        
        beta = 1e-6
    noise = rng.laplace(0, beta)
    return value + noise


def noisy_position(x: float, y: float, noise_scale: float = NOISE_SCALE, rng: np.random.Generator = None):
    return (
        add_laplace_noise(x, noise_scale, rng),
        add_laplace_noise(y, noise_scale, rng),
    )


def noisy_speed(speed: float, noise_scale: float = NOISE_SCALE, rng: np.random.Generator = None) -> float:
    return max(0.0, add_laplace_noise(speed, noise_scale, rng))


if __name__ == "__main__":
    
    rng = np.random.default_rng(1)
    x, y = 98.4, 181.7
    nx, ny = noisy_position(x, y, rng=rng)
    print(f"Original position: ({x}, {y})")
    print(f"Noisy position:    ({nx:.1f}, {ny:.1f})")

    speed = 15.3
    ns = noisy_speed(speed, rng=rng)
    print(f"Original speed: {speed} km/h")
    print(f"Noisy speed:    {ns:.1f} km/h")
