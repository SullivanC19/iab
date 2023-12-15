from typing import Tuple
import numpy as np
from constants import F_EPS, RNG

class IAB:
  # compliance types: C x M
  # p_k: C
  # mu_kx: C x M (mean of uniform distributions conditioned on compliance type k)
  def __init__(
      self,
      n_arms: int,
      compliance_types: np.ndarray,
      p_k: np.ndarray,
      mu_kx: np.ndarray):
    self.n_arms = n_arms
    self._n_compliance_types = compliance_types.shape[0]
    self._compliance_types = compliance_types
    self._p_k = p_k
    self._mu_kx = mu_kx
    self._mu_x = np.zeros((self.n_arms))
    for x in range(self.n_arms):
      for k in range(self._n_compliance_types):
        self._mu_x[x] += self._p_k[k] * self._mu_kx[k, x]
    self._mu_z = np.zeros((self.n_arms))
    for z in range(self.n_arms):
      for k in range(self._n_compliance_types):
        self._mu_z[z] += self._p_k[k] * self._mu_kx[k, compliance_types[k, z]]
    self.reset()

  def reset(self):
    self.n_zx = np.zeros((self.n_arms, self.n_arms))
    self.s_z = np.zeros((self.n_arms))
    self.s_x = np.zeros((self.n_arms))
    self.itt_regret = []
    self.st_regret = []
    self.c_regret = []

  def pull(self, z: int) -> Tuple[int, float]:
    eps = F_EPS(self.n_arms)
    k = RNG.choice(np.arange(self._n_compliance_types), p=self._p_k)

    x = self._compliance_types[k]
    y = self._mu_kx[k] + eps

    self.n_zx[z, x[z]] += 1
    self.s_z[z] += y[x[z]]
    self.s_x[x[z]] += y[x[z]]

    z_itt = np.argmax(self._mu_z)
    self.itt_regret.append(y[x[z_itt]] - y[x[z]])

    x_st = np.argmax(self._mu_x)
    self.st_regret.append(y[x_st] - y[x[z]])

    if (self._compliance_types[k] == np.arange(self.n_arms)).all():
      x_c = np.argmax(self._mu_kx[k])
      self.c_regret.append(y[x_c] - y[x[z]])
    else:
      self.c_regret.append(0)

    return x[z], y[x[z]]
  
  @property
  def n_x(self) -> np.ndarray:
    return np.sum(self.n_zx, axis=0)
  
  @property
  def n_z(self) -> np.ndarray:
    return np.sum(self.n_zx, axis=1)
  
  @property
  def u_x(self) -> float:
    return self.s_x / (self.n_x + 1e-8)
  
  @property
  def u_z(self) -> float:
    return self.s_z / (self.n_z + 1e-8)
