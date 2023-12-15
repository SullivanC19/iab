import numpy as np

SEED = 35
RNG = np.random.Generator(np.random.PCG64(SEED))

# only support centered uniform dists [-1, 1]
F_EPS = lambda n_arms: RNG.uniform(-1, 1, size=n_arms)

C = 20
ALPHA = 0.01
T = 10000

CDICT = {
  'ucb_at': 'orange',
  'ucb_itt': 'blue',
  'sls_decay': 'green',
}