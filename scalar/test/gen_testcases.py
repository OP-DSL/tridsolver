import numpy as np
np.random.seed(1)
import scipy.sparse as sp
import scipy.sparse.linalg as slg

PRECISION = 15
PRECISION_FLOAT=6

def calc_result(a, b, c, d, axis, precision = 0):
    """
    :precision: if 0 then use the full precision, else round the inputs to
                precision decimals before solving
    """
    non_system_axes = np.arange(len(a.shape))
    non_system_axes = np.delete(non_system_axes, axis)
    N = a.shape[axis]
    axis_permutation = np.r_[non_system_axes, [axis]]
    # Transform the mesh so that it has 2 axes, the second defines the systems,
    # the first groups everything else
    aa = a.transpose(axis_permutation).reshape((-1, N))
    bb = b.transpose(axis_permutation).reshape((-1, N))
    cc = c.transpose(axis_permutation).reshape((-1, N))
    dd = d.transpose(axis_permutation)
    transposed_shape = np.array(dd.shape)
    dd = dd.reshape((-1, N))
    uu = np.zeros(dd.shape)
    if precision > 0:
        aa = aa.round(precision)
        bb = bb.round(precision)
        cc = cc.round(precision)
        dd = dd.round(precision)
    for i in range(aa.shape[0]):
        a_diag = aa[i, 1:]    # a is indexed 1 ... N-1
        b_diag = bb[i, :]     # b is indexed 0 ... N-1
        c_diag = cc[i, :-1]   # c is indexed 0 ... N-2
        coeff_matrix = sp.diags([a_diag, b_diag, c_diag], [-1, 0, 1])
        print('Condition number:', np.linalg.cond(coeff_matrix.toarray(),
                                                  np.inf))
        uu[i, :] = slg.spsolve(coeff_matrix, dd[i, :])
    # transform the mesh (the result part) back
    return uu.reshape(transposed_shape).transpose(np.argsort(axis_permutation))

def write_testcase(fname, a, b, c, d, u, u_float, solvedim):
    with open(fname, mode = 'w') as f:
        # Number of dimensions and solving dimension
        f.write('{} {}\n'.format(len(a.shape), solvedim))
        # Sizes in different dimensions
        f.write(' '.join([str(size) for size in a.shape]))
        f.write('\n')
        # a matrix, in row major format
        f.write(' '.join([str(round(val, PRECISION))
                          for val in a.flatten(order = 'F')]))
        f.write('\n')
        # b matrix, in row major format
        f.write(' '.join([str(round(val, PRECISION))
                          for val in b.flatten(order = 'F')]))
        f.write('\n')
        # c matrix, in row major format
        f.write(' '.join([str(round(val, PRECISION))
                          for val in c.flatten(order = 'F')]))
        f.write('\n')
        # d matrix, in row major format
        f.write(' '.join([str(round(val, PRECISION))
                          for val in d.flatten(order = 'F')]))
        f.write('\n')
        # u matrix, in row major format
        f.write(' '.join([str(round(val, PRECISION))
                          for val in u.flatten(order = 'F')]))
        f.write('\n')
        # u matrix in float precision, in row major format
        f.write(' '.join([str(round(val, PRECISION_FLOAT))
                          for val in u_float.flatten(order = 'F')]))
        f.write('\n')

def gen_testcase(fname, shape, solvedim):
    a = -1 + np.random.rand(*shape) * 0.1
    b = 2 + np.random.rand(*shape)
    c = -1 + np.random.rand(*shape) * 0.1
    d = np.random.rand(*shape)
    u = calc_result(a, b, c, d, solvedim)
    u_float = calc_result(a, b, c, d, solvedim, precision = PRECISION_FLOAT)
    write_testcase(fname, a, b, c, d, u, u_float, solvedim)

def main():
    gen_testcase('one_dim_small', [5], 0)
    gen_testcase('one_dim_large', [200], 0)
    gen_testcase('two_dim_small_solve0', [5, 5], 0)
    gen_testcase('two_dim_small_solve1', [5, 5], 1)
    gen_testcase('two_dim_large_solve0', [20, 20], 0)
    gen_testcase('two_dim_large_solve1', [20, 20], 1)

if __name__ == "__main__":
    main()
