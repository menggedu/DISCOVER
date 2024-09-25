import numpy  as  np

def FiniteDiff(u, dx):
    
    n = u.size
    ux = np.zeros(n)

    # for i in range(1, n - 1):
    ux[1:n-1] = (u[2:n] - u[0:n-2]) / (2 * dx)

    ux[0] = (-3.0 / 2 * u[0] + 2 * u[1] - u[2] / 2) / dx
    ux[n - 1] = (3.0 / 2 * u[n - 1] - 2 * u[n - 2] + u[n - 3] / 2) / dx
    return ux


def FiniteDiff2(u, dx):
    # import pdb;pdb.set_trace()
    n = u.size
    ux = np.zeros(n)

    # for i in range(1, n - 1):
    ux[1:n-1] = (u[2:n] - 2 * u[1:n-1] + u[0:n-2]) / dx ** 2

    ux[0] = (2 * u[0] - 5 * u[1] + 4 * u[2] - u[3]) / dx ** 2
    ux[n - 1] = (2 * u[n - 1] - 5 * u[n - 2] + 4 * u[n - 3] - u[n - 4]) / dx ** 2
    return ux


def Diff(u, dxt,dim, name='x'):
    """
    Here dx is a scalar, name is a str indicating what it is
    """

    n, m = u.shape
    uxt = np.zeros((n, m))


    dxt = dxt[2]-dxt[1]
    for i in range(m):
        uxt[:, i] = FiniteDiff(u[:, i], dxt)

    return uxt


def Diff2(u, dxt,dim, name='x'):
    """
    Here dx is a scalar, name is a str indicating what it is
    """

    n, m = u.shape
    uxt = np.zeros((n, m))


    dxt = dxt[2]-dxt[1]
    for i in range(m):
        uxt[:, i] = FiniteDiff2(u[:, i], dxt)


    return uxt

def Diff3(u, dxt, dim, name='x'):
    """
    Here dx is a scalar, name is a str indicating what it is
    """

    n, m = u.shape
    uxt = np.zeros((n, m))


    dxt = dxt[2]-dxt[1]
    for i in range(m):
        uxt[:, i] = FiniteDiff2(u[:, i], dxt)
        uxt[:,i] = FiniteDiff(uxt[:,i],dxt )


    return uxt

def Diff4(u, dxt,dim, name='x'):
    """
    Here dx is a scalar, name is a str indicating what it is
    """

    n, m = u.shape
    uxt = np.zeros((n, m))

    if name == 'x':
        dxt = dxt[2]-dxt[1]
        for i in range(m):
            uxt[:, i] = FiniteDiff2(u[:, i], dxt)
            uxt[:,i] = FiniteDiff2(uxt[:,i],dxt )
    elif name == 't':
        for i in range(n):
            uxt[i, :] = FiniteDiff2(u[i, :], dxt)
            uxt[i,:] = FiniteDiff2(uxt[:,i],dxt )

    else:
        NotImplementedError()