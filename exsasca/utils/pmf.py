import numpy as np


def add_noise(pmfs: np.ndarray, alpha: float) -> np.ndarray:
    """
    pmfs: (batch_size, N, 256)
    Adds noise to a batch of PMFs .

    Let u = (1/256, 1/256, ..., 1/256).
    Then for a given pmf, we compute the noisy pmf as
            pmf' = (1-alpha) * pmf + alpha * u.
    i.e., we compute a convex combination of the original pmf
    and the uniform distribution.
    """
    assert pmfs.shape[2] == 256, "pmfs must be of shape (batch_size, N, 256)"
    assert 0 <= alpha <= 1, "alpha must be in [0,1]"

    return (1 - alpha) * pmfs + alpha * np.ones_like(pmfs) / 256.
