import numpy as np

from iab import IAB
from constants import RNG, ALPHA, C

def ucb_at(iab: IAB, t: int) -> int:
  return ucb(iab.u_x, iab.n_x, t)

def ucb_itt(iab: IAB, t: int) -> int:
  return ucb(iab.u_z, iab.n_z, t)

def sls_decay(iab: IAB, t: int, alpha: float = ALPHA) -> int:
  epsilon = iab.n_arms / (alpha * t)
  P = iab.n_zx / (iab.n_z[:, None] + 1e-8)
  P = (1 - 1e-8) * P + 1e-8 * np.eye(iab.n_arms) # add small constant to avoid singular matrix
  effect = np.linalg.inv(P) @ iab.u_z
  return epsilon_greedy(effect, epsilon, iab.n_arms)

def ucb(v: np.ndarray, n: np.ndarray, t: int) -> int:
  return np.argmax(v + np.sqrt(C * np.log(t) / (n + 1e-5)))

def epsilon_greedy(v: np.ndarray, epsilon: float, n_arms: int) -> int:
  if RNG.random() <= epsilon:
    return RNG.integers(0, n_arms - 1)
  return np.argmax(v)