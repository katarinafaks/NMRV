import numpy as np
import math
from ex4_utils import kalman_step, gaussian_prob, sample_gauss

N = 40
v = np.linspace(5 * math.pi, 0, N)
x = np.cos(v) * v
y = np.sin(v) * v

sx = np.zeros((x.size, 1), dtype=np.float32).flatten()
sy = np.zeros((y.size, 1), dtype=np.float32).flatten()
sx[0] = x[0]
sy[0] = y[0]
state = np.zeros((A.shape[0], 1), dtype=np.float32).flatten()
state[0] = x[0]
state[1] = y[0]
covariance = np.eye(A.shape[0], dtype=np.float32)
for j in range(1, x.size):
	state, covariance, _, _ = kalman_step(
		A, C, Q_i, R_i,
		np.reshape(np.array([x[j], y[j]]), (-1, 1)),
		np.reshape(state, (-1, 1)),
		covariance
	)
	sx[j] = state[0]
	sy[j] = state[1]


# particles = . . . # v e c t o r o f N p a r t i c l e s ( rows ) and d dimen si on s ( columns )
# weights = . . . # v e c t o r o f w ei g h t s f o r N p a r t i c l e s
# # re s ample p a r t i c l e s
# weights_norm = weights / np . sum( weights ) # n o rm ali z e w ei g h t s
# weights_cumsumed = np . cumsum ( weights_norm ) # cumul a ti ve d i s t r i b u t i o n
# rand_samples = np . random . rand ( N , 1 )
# sampled_idxs = np . digitize ( rand_samples , weights_cumsumed ) # randomly s e l e c t N i n d i c e s
# particles_new = particles [ sampled_idxs . flatten ( ) , : ] # s e l e c t the c o r r e s p o n di n g ←-
# sample s