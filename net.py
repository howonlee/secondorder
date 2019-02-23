import random
import numpy as np
import numpy.random as npr
import numpy.linalg as npl

def act(net):
    return np.where(net > 0, net, net * 0.5)

def dact(net):
    return np.where(net > 0, 1., 0.5)

def inv_act(h):
    return np.where(h > 0, h, h * 2)

def errfn(h2, ys):
    return 0.5 * np.sum(np.power(h2 - ys, 2))

def rand_onehots(shape):
    onehots = np.zeros(shape)
    cols = onehots.shape[1]
    for row in range(onehots.shape[0]):
        randcol = random.randint(0, cols-1)
        onehots[row, randcol] = 1.
    return onehots


if __name__ == "__main__":
    print("lol")
    xs = npr.randn(3,2)
    ys = rand_onehots((3,3))
    w1 = npr.randn(2,4)
    w2 = npr.randn(4,3)
    v1 = np.zeros_like(w1)
    v2 = np.zeros_like(w2)
    hess1 = np.zeros((v1.size, v1.size))
    hess2 = np.zeros((v2.size, v2.size))
    hidsize = w1.shape[1]
    outsize = w2.shape[1]
    alpha = 1e-1
    r = 1e-9
    net1 = np.dot(xs, w1)
    h1 = act(net1)
    net2 = np.dot(h1, w2)
    h2 = act(net2)
    err = errfn(h2, ys)
    print("err: {}".format(str(err)))
    derr_dh2 = h2 - ys
    derr_dnet2 = derr_dh2 * dact(net2)
    derr_dh1 = np.dot(derr_dnet2, w2.T)
    derr_dnet1 = derr_dh1 * dact(net1)
    derr_dw2 = np.dot(h1.T, derr_dnet2)
    derr_dw1 = np.dot(xs.T, derr_dnet1)
    derr_dx = np.dot(derr_dnet1, w1.T)
    for idx, member in np.ndenumerate(w1):
        hessrow = (idx[0] * hidsize) + idx[1]
        v1[:] = 0.
        v2[:] = 0.
        v1[idx] = 1.
        analytic_rnet1 = np.dot(xs, v1)
        analytic_rh1 = analytic_rnet1 * dact(net1)
        analytic_rnet2 = np.dot(analytic_rh1, w2) + np.dot(h1, v2)
        analytic_rh2 = analytic_rnet2 * dact(net2)
        analytic_derr_dh2 = analytic_rh2
        analytic_derr_dnet2 = (analytic_derr_dh2 * dact(net2))
        analytic_derr_dh1 = np.dot(analytic_derr_dnet2, w2.T) + np.dot(derr_dnet2, v2.T)
        analytic_derr_dnet1 = (analytic_derr_dh1 * dact(net1))
        analytic_derr_dx = np.dot(analytic_derr_dnet1, w1.T) + np.dot(derr_dnet1, v1.T)
        analytic_derr_dw1 = np.dot(xs.T, analytic_derr_dnet1)
        hess1[hessrow, :] = analytic_derr_dw1.ravel()
    for idx, member in np.ndenumerate(w2):
        hessrow = (idx[0] * outsize) + idx[1]
        v1[:] = 0.
        v2[:] = 0.
        v2[idx] = 1.
        analytic_rnet1 = np.dot(xs, v1)
        analytic_rh1 = analytic_rnet1 * dact(net1)
        analytic_rnet2 = np.dot(analytic_rh1, w2) + np.dot(h1, v2)
        analytic_rh2 = analytic_rnet2 * dact(net2)
        analytic_derr_dh2 = analytic_rh2
        analytic_derr_dnet2 = (analytic_derr_dh2 * dact(net2))
        analytic_derr_dh1 = np.dot(analytic_derr_dnet2, w2.T) + np.dot(derr_dnet2, v2.T)
        analytic_derr_dnet1 = (analytic_derr_dh1 * dact(net1))
        analytic_derr_dx = np.dot(analytic_derr_dnet1, w1.T) + np.dot(derr_dnet1, v1.T)
        analytic_derr_dw2 = np.dot(h1.T, analytic_derr_dnet2) + np.dot(analytic_rh1.T, derr_dnet2)
        hess2[hessrow, :] = analytic_derr_dw2.ravel()
    import pdb
    pdb.set_trace()
