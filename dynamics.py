import numpy as np
import math
import matplotlib.pylab as plt

def run_dynamics2(ratio, K, eta=0.01, nIter=10000):
    x = np.zeros((2))
    x[0] = ratio
    eta = eta / 2 / math.pi

    xs = np.zeros((2, nIter))
    res = dict(
        term1 = np.zeros((nIter)),
        term2A = np.zeros((nIter)),
        term2a = np.zeros((nIter)),
        term3 = np.zeros((nIter)),
        theta = np.zeros((nIter)),
        phi_star = np.zeros((nIter)),
        phi = np.zeros((nIter)),
        alpha = np.zeros((nIter))
    )
    for i in range(nIter):
        xs[:,i] = x
        A = x[0]
        a = x[1]
        alpha = 1 / math.sqrt(A**2 + (K-1) * a**2)
        theta = math.acos(A * alpha)
        phi_star = math.acos(a * alpha)
        phi = math.acos( (2*A*a + (K-2)*a*a) * (alpha ** 2))

        # Three terms.
        term1 = (math.pi - phi) * ((A - 1) + (K - 1) * a)
        term2A = theta
        term2a = phi_star - phi
        term3 = (K-1) * (alpha * math.sin(phi_star) - math.sin(phi)) + alpha * math.sin(theta)

        res["term1"][i] = term1
        res["term2A"][i] = term2A
        res["term2a"][i] = term2a
        res["term3"][i] = term3
        res["theta"][i] = theta
        res["phi_star"][i] = phi_star
        res["phi"][i] = phi
        res["alpha"][i] = alpha

        dA = 0
        da = 0

        dA += - term1
        da += - term1

        dA += - phi * (A - 1)
        da += - phi * a

        dA += term3 * A
        da += term3 * a

        dA += - term2A
        da += - term2a

        x[0] += eta * dA
        x[1] += eta * da

    errs = np.power(xs[0,:] - 1, 2) + (K - 1) * np.power(xs[1,:], 2)

    return xs, errs, res

nIter = 1000
iter_nums = range(nIter)

all_xs = {}
all_errs = {}
all_res = {}

for K in (2, 5, 10):
    xs, errs, res = run_dynamics2(0.001, K, eta=0.01, nIter=nIter)
    all_xs[K] = xs
    all_errs[K] = errs
    all_res[K] = res

plt.figure()
colors = 'rgb'
for i, K in enumerate((2, 5, 10)):
    plt.plot(iter_nums, all_errs[K], colors[i], label='K = %d' % K)
    plt.plot(iter_nums[::nIter/10], all_errs[K][::nIter/10], 'o' + colors[i])

plt.legend()
plt.savefig('2layer-2d-convergence.pdf')

plt.figure()
x = np.arange(0, 1, 0.01)
y1 = x
plt.plot(x, y1, '--')

for i, K in enumerate((2, 5, 10)):
    xs = all_xs[K]
    plt.plot(xs[0,:], xs[1,:], colors[i], label='K = %d' % K)
    plt.plot(xs[0,::nIter/10], xs[1,::nIter/10], 'o' + colors[i])
    # two reference lines.

    x_s = (math.sqrt(K - 1) - math.acos(1/math.sqrt(K)) + math.pi) / (math.pi * K)
    y2 = x_s * (x - 1) / (x_s - 1)
    plt.plot(x, y2, 'y--')

#plt.axis('equal')
plt.axis([0, 1, 0, 0.6])
plt.legend()
plt.savefig('2layer-2d-traj.pdf')

#for k, v in res.iteritems():
#    plt.plot(v)
#    plt.title(k)
#    plt.show()

# plt.plot(res["phi_star"] - res["phi"])
# plt.title("phi_star - phi")
# plt.show()

# plt.plot(res["theta"])
# plt.title("theta")
# plt.show()

# plt.plot(res["theta"], res["phi_star"] - res["phi"])
# plt.title("theta versus phi_star - phi")
# plt.show()
