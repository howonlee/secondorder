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
    # condition _everything_
    # this doesn't make much sense
    xs = npr.randn(3,3) + np.eye(3) * 0.1
    ys = rand_onehots((3,3)) + np.eye(3) * 0.1
    w1 = npr.randn(3,3) + np.eye(3) * 0.1
    w2 = npr.randn(3,3) + np.eye(3) * 0.1
    v1 = np.zeros_like(w1)
    v2 = np.zeros_like(w2)
    hess1 = np.zeros((v1.size, v1.size))
    hess2 = np.zeros((v2.size, v2.size))
    hidsize = w1.shape[1]
    outsize = w2.shape[1]
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
    # execute the inversion with the still-in dealie. assume bad conditioning got us into a mess
    fd1_derr_dw1 = derr_dw1 * (1. + (r / 2.))
    fd2_derr_dw1 = derr_dw1 * (1. - (r / 2.))
    fd1_derr_dw2 = derr_dw2 * (1. + (r / 2.))
    fd2_derr_dw2 = derr_dw2 * (1. - (r / 2.))
    fd1_derr_dnet1 = np.dot(npl.pinv(xs.T), fd1_derr_dw1)
    fd2_derr_dnet1 = np.dot(npl.pinv(xs.T), fd2_derr_dw1)
    fd1_derr_dnet2 = np.dot(npl.pinv(h1.T), fd1_derr_dw2)
    fd2_derr_dnet2 = np.dot(npl.pinv(h1.T), fd2_derr_dw2)
    fd1_derr_dh1 = fd1_derr_dnet1 / dact(net1)
    fd2_derr_dh1 = fd2_derr_dnet1 / dact(net1)
    fd1_derr_dh2 = fd1_derr_dnet2 / dact(net2)
    fd2_derr_dh2 = fd2_derr_dnet2 / dact(net2)
    fd1_h2 = fd1_derr_dh2 + ys
    fd2_h2 = fd2_derr_dh2 + ys
    fd1_net2 = inv_act(fd1_h2)
    fd2_net2 = inv_act(fd2_h2)
    fd1_h1 = np.dot(fd1_net2, npl.pinv(w2))
    fd2_h1 = np.dot(fd2_net2, npl.pinv(w2))
    fd1_net1 = inv_act(fd1_h1)
    fd2_net1 = inv_act(fd2_h1)
    fd1_w1 = np.dot(npl.pinv(xs), fd1_net1)
    fd2_w1 = np.dot(npl.pinv(xs), fd2_net1)
    fd1_w2 = np.dot(npl.pinv(h1), fd1_net2)
    fd2_w2 = np.dot(npl.pinv(h1), fd2_net2)
    fd_w1 = (fd1_w1 - fd2_w1) / r
    fd_w2 = (fd1_w2 - fd2_w2) / r
    print("to check if this is correct:")
    print("np.dot(npl.inv(hess1), derr_dw1.ravel()).reshape(3,3)")
    print("like so: ")
    print(np.allclose(np.dot(npl.inv(hess1), derr_dw1.ravel()).reshape(3,3), fd_w1, rtol=1e-3, atol=1e-4))
    print("and for w2: ")
    print(np.allclose(np.dot(npl.inv(hess2), derr_dw2.ravel()).reshape(3,3), fd_w2, rtol=1e-3, atol=1e-4))
    print("note awful tolerances. if you actually wanted to run in production you could fix with higher order FD or sitting down and writing analytic solution")
    import pdb
    pdb.set_trace()
