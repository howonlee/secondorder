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

def calc_hess():
    """ dead code for calculating analytic hessian. all of it """
    """
    hess = np.zeros((insize * outsize, insize* outsize))
    for idx, member in np.ndenumerate(w):
        hessrow = (idx[0] * outsize) + idx[1]
        v[:] = 0.
        v[idx] = 1.
        rnet = np.dot(xs, v)
        rh = rnet * dact(net)
        rderr_dh = rh
        # ddact doesn't exist
        rderr_dnet = rderr_dh * dact(net)
        rderr_dw = np.dot(xs.T, rderr_dnet)
        hess[hessrow, :] = rderr_dw.ravel()
    w -= alpha * np.dot(npl.inv(hess), derr_dw.ravel()).reshape(derr_dw.shape)
    """
    pass


if __name__ == "__aain__":
    xs = npr.randn(3,2)
    ys = some shit
    num_epochs = 10
    alpha = 1e-1
    r = 1e-9
    xs, ys = pair
    net = np.dot(xs, w)
    h = act(net)
    err = errfn(h, ys)
    if it % 1 == 0:
        print("it: {}, err: {}".format(str(it), str(err)))
    derr_dh = h - ys
    derr_dnet = derr_dh * dact(net)
    derr_dw = np.dot(xs.T, derr_dnet)
