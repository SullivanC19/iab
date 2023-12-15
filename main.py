import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple

from iab import IAB
from constants import T, CDICT
from algs import ucb_itt, ucb_at, sls_decay

EXAMPLE_1 = IAB(
  3,
  np.array([[0, 1, 2], [0, 0, 1]]),
  np.array([5/8, 3/8]),
  np.array([[1, -1, 0], [-4, 0, -2]]),
)

EXAMPLE_2 = IAB(
  3,
  np.array([[0, 1, 2], [1, 1, 0]]),
  np.array([1/4, 3/4]),
  np.array([[-2, -4, -3], [3, 1, 2]]),
)

def run_examples() -> Dict[Tuple[int, str, str], np.ndarray]:
  cum_regrets = dict()
  for iab in [EXAMPLE_1, EXAMPLE_2]:
    for alg in [ucb_itt, ucb_at, sls_decay]:
      for t in range(T):
        z = alg(iab, t + 1)
        iab.pull(z)
      alg_name = alg.__name__
      iab_name = 1 if iab == EXAMPLE_1 else 2
      cum_regrets[(iab_name, alg_name, 'itt')] = np.cumsum(iab.itt_regret)
      cum_regrets[(iab_name, alg_name, 'st')] = np.cumsum(iab.st_regret)
      cum_regrets[(iab_name, alg_name, 'c')] = np.cumsum(iab.c_regret)
      print(f"Alg: {alg_name}")
      print(f"Total ITT Regret: {np.sum(iab.itt_regret)}")
      print(f"Total ST Regret: {np.sum(iab.st_regret)}")
      print(f"Total C Regret: {np.sum(iab.c_regret)}")
      iab.reset()
  return cum_regrets

def plot_example1(cum_regrets: Dict[Tuple[int, str, str], np.ndarray]):
  fig, ax = plt.subplots(1, 3, figsize=(20, 4))
  for (iab_name, alg_name, reg_name) in cum_regrets.keys():
    if iab_name != 1:
      continue

    # ITT Regret
    if reg_name == 'itt' and alg_name in ['ucb_at', 'ucb_itt']:
      ax[0].plot(cum_regrets[(iab_name, alg_name, reg_name)], label=alg_name, color=CDICT[alg_name])
      ax[0].legend()
      ax[0].set_xlabel("T")
      ax[0].set_ylabel("ITTRegret")
    
    # ST Regret
    if reg_name == 'st' and alg_name in ['ucb_at', 'ucb_itt']:
      ax[1].plot(cum_regrets[(iab_name, alg_name, reg_name)], label=alg_name, color=CDICT[alg_name])
      ax[1].legend()
      ax[1].set_xlabel("T")
      ax[1].set_ylabel("STRegret")

    # C Regret
    if reg_name == 'c' and alg_name in ['ucb_at', 'ucb_itt']:
      ax[2].plot(cum_regrets[(iab_name, alg_name, reg_name)], label=alg_name, color=CDICT[alg_name])
      ax[2].legend()
      ax[2].set_xlabel("T")
      ax[2].set_ylabel("CRegret")

  fig.savefig("res/example1.png")

def plot_example2(cum_regrets: Dict[Tuple[int, str, str], np.ndarray]):
  fig, ax = plt.subplots(1, 4, figsize=(24, 4))
  for (iab_name, alg_name, reg_name) in cum_regrets.keys():
    if iab_name != 2:
      continue

    # CRegret (ITT, AT)
    if reg_name == 'c' and alg_name in ['ucb_at', 'ucb_itt']:
      ax[0].plot(cum_regrets[(iab_name, alg_name, reg_name)], label=alg_name, color=CDICT[alg_name])
      ax[0].legend()
      ax[0].set_xlabel("T")
      ax[0].set_ylabel("CRegret")

    # CRegret (SLS)
    if reg_name == 'c' and alg_name in ['sls_decay']:
      ax[1].plot(cum_regrets[(iab_name, alg_name, reg_name)], label=alg_name, color=CDICT[alg_name])
      ax[1].legend()
      ax[1].set_xlabel("T")
      ax[1].set_ylabel("CRegret")

    # ITT Regret (SLS, AT)
    if reg_name == 'itt' and alg_name in ['sls_decay', 'ucb_at']:
      ax[2].plot(cum_regrets[(iab_name, alg_name, reg_name)], label=alg_name, color=CDICT[alg_name])
      ax[2].legend()
      ax[2].set_xlabel("T")
      ax[2].set_ylabel("ITT Regret")

    # ITT Regret (ITT)
    if reg_name == 'itt' and alg_name in ['ucb_itt']:
      ax[3].plot(cum_regrets[(iab_name, alg_name, reg_name)], label=alg_name, color=CDICT[alg_name])
      ax[3].legend()
      ax[3].set_xlabel("T")
      ax[3].set_ylabel("ITT Regret")

  fig.savefig("res/example2.png")

if __name__ == '__main__':
  cum_regrets = run_examples()
  plot_example1(cum_regrets)
  plot_example2(cum_regrets)