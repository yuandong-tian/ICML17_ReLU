import numpy as np
import math
from utils import construct_M, split

d = 2
n = 2

ngrid = 10000
angles = np.linspace(0, math.pi, ngrid)
w_star_angles = np.linspace(0, 2*math.pi, 2*ngrid + 2)
w_star_angles = w_star_angles[1:-1]
es = np.zeros((d, n))
es[:,0] = [1, 0]

es_star = np.zeros((d, w_star_angles.shape[0]))
es_star[0,:] = np.cos(w_star_angles)
es_star[1,:] = np.sin(w_star_angles)

all_diffs = np.zeros((2*ngrid, ngrid))

for i, angle in enumerate(angles):
    #ws_star = np.eye(d)
    es[:,1] = [np.cos(angle), np.sin(angle)]
    M = construct_M(es, es)
    M_r, M_off = split(M)

    try:
        coeffs = np.linalg.solve(M_r, M_off.T)
        M_star = construct_M(es_star, es)
        M_star_r, M_star_off = split(M_star)

        # L_12 and L_21
        diff = M_star_off.T - np.dot(M_star_r.T, coeffs)
        all_diffs[:,i] = diff[:,0]
    except:
        all_diffs[:,i] = float('nan')

plt.imshow(all_diffs.T, extent=[0,2*math.pi, math.pi, 0])
plt.xlabel('phi')
plt.ylabel('theta')
plt.plot(angles, angles, linewidth=2, color='b')
plt.axis([0,2*math.pi, 0, math.pi])
plt.savefig("l12verify.pdf")
