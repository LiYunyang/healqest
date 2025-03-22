import numpy as np

tr_cg = lambda i: i - 1
tr_cd = lambda i: 0


class CacheMemory(dict):
    """A dictionary-like object that can store things and remove keys
    conveniently.
    """
    def __init__(self):
        pass

    def store(self, key, data):
        [dTAd_inv, searchdirs, searchfwds] = data
        self[key] = [dTAd_inv, searchdirs, searchfwds]

    def restore(self, key):
        return self[key]

    def remove(self, key):
        del self[key]

    def trim(self, keys):
        """Remove a list of keys from the dictionary
        """
        assert set(keys).issubset(list(self.keys()))
        for key in set(self.keys()) - set(keys):
            del self[key]


def cd_solve(
    x,
    b,
    fwd_op,
    pre_ops,
    dot_op,
    criterion,
    tr,
    cache=CacheMemory(),
    roundoff=25,
):
    """Conjugate direction solver
    The customizable conjugate directions loop for x=[fwd_op]^{-1}b.
    The initial value of x is taken as guess.
    fwd_op, pre_op(s) and dot_op mustn't modify the inputs!

    Selected Parameters:
    --------
    x: flexible format (array, customized classes...)
        solve result
    b: flexible format (array, customized classes...)
        b = [fwd_op] x
    pre_ops: list of operators
        operator for generating new search direction
    dot_op: operator
        operator for generating delta from the residual
    criterion: function
        the function that determines whether things have converged
    tr: truncation / restart functions
        Suggest:  Truncated Partial Restart (TPR)
        TR(i) = i - max(P, min( T, mod(i,R) ))
    cache: CacheMemory object
        cacher for search objects.
    """

    n_pre_ops = len(pre_ops)

    residual = b - fwd_op(x)
    searchdirs = [op(residual) for op in pre_ops]

    iter = 0
    while criterion(iter, x, residual) == False:
        searchfwds = [fwd_op(searchdir) for searchdir in searchdirs]
        deltas = [dot_op(searchdir, residual) for searchdir in searchdirs]

        # calculate (D^T A D)^{-1}
        dTAd = np.zeros((n_pre_ops, n_pre_ops))
        for ip1 in range(0, n_pre_ops):
            for ip2 in range(0, ip1 + 1):
                dTAd[ip1, ip2] = dTAd[ip2, ip1] = dot_op(
                    searchdirs[ip1], searchfwds[ip2]
                )
        dTAd_inv = np.linalg.inv(dTAd)

        # search.
        alphas = np.dot(dTAd_inv, deltas)
        for (searchdir, alpha) in zip(searchdirs, alphas):
            x += searchdir * alpha

        # append to cache.
        cache.store(iter, [dTAd_inv, searchdirs, searchfwds])

        # update residual
        iter += 1
        if np.mod(iter, roundoff) == 0:
            residual = b - fwd_op(x)
        else:
            for (searchfwd, alpha) in zip(searchfwds, alphas):
                residual -= searchfwd * alpha

        # initial choices for new search directions.
        searchdirs = [pre_op(residual) for pre_op in pre_ops]

        # orthogonalize w.r.t. previous searches.
        prev_iters = list(range(tr(iter), iter))

        for titer in prev_iters:
            [prev_dTAd_inv, prev_searchdirs, prev_searchfwds] = cache.restore(
                titer
            )

            for searchdir in searchdirs:
                proj = [
                    dot_op(searchdir, prev_searchfwd)
                    for prev_searchfwd in prev_searchfwds
                ]
                betas = np.dot(prev_dTAd_inv, proj)

                for (beta, prev_searchdir) in zip(betas, prev_searchdirs):
                    searchdir -= prev_searchdir * beta

        # clear old keys from cache
        cache.trim(list(range(tr(iter + 1), iter)))

    return iter
