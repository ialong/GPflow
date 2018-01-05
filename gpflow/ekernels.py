from functools import reduce
import warnings
import tensorflow as tf
from . import kernels
from ._settings import settings

from .quadrature import mvhermgauss
from numpy import pi as nppi

int_type = settings.dtypes.int_type
float_type = settings.dtypes.float_type


class RBF(kernels.RBF):
    def eKdiag(self, Xmu, Xcov=None, CI=None):
        """
        <diag(K_{X, X})>_{q(X)}
        :param Xmu: kernel input (NxD)
        :param Xcov: X covariance matrices (NxDxD)
        :param CI: (optional) control inputs (NxE)
        :return: N
        """
        return self.Kdiag(Xmu)

    def eKxz(self, Z, Xmu, Xcov, CI=None):
        """
        <K_{X, Z}>_{q(X)}
        :param Z: inducing inputs (MxD) or (Mx(D+E))
        :param Xmu: X means (NxD)
        :param Xcov: X covariance matrices (NxDxD)
        :param CI: (optional) control inputs (NxE)
        :return: NxM
        """
        D = tf.shape(Xmu)[1]
        E = 0 if CI is None else tf.shape(CI)[1]
        lengthscales = self.lengthscales if self.ARD else \
            tf.zeros((D + E,), dtype=float_type) + self.lengthscales

        chol_L_plus_Xcov = tf.cholesky(tf.matrix_diag(lengthscales[:D] ** 2) + Xcov)  # NxDxD
        all_diffs = tf.transpose(Z[:, :D]) - tf.expand_dims(Xmu, 2)  # NxDxM

        sqrt_det_L = tf.reduce_prod(lengthscales[:D])
        sqrt_det_L_plus_Xcov = tf.exp(tf.reduce_sum(tf.log(tf.matrix_diag_part(chol_L_plus_Xcov)), axis=1))
        determinants = sqrt_det_L / sqrt_det_L_plus_Xcov  # N

        exponent_mahalanobis = tf.matrix_triangular_solve(chol_L_plus_Xcov, all_diffs, lower=True)  # NxDxM
        exponent_mahalanobis = tf.reduce_sum(tf.square(exponent_mahalanobis), 1)  # NxM
        if CI is not None:
            ci_L_inv = CI / lengthscales[D:]  # NxE
            z_L_inv = Z[:, D:] / lengthscales[D:]  # MxE
            exponent_mahalanobis += tf.reduce_sum(tf.square(z_L_inv), 1) \
                                    + tf.expand_dims(tf.reduce_sum(tf.square(ci_L_inv), 1), 1) \
                                    - 2 * tf.matmul(ci_L_inv, z_L_inv, transpose_b=True)

        exponent_mahalanobis = tf.exp(-0.5 * exponent_mahalanobis)  # NxM

        return self.variance * (determinants[:, None] * exponent_mahalanobis)

    def exKxz(self, Z, Xmu, Xcov, CI=None):
        """
        exKxz[n] = <[x_n K_{x_{n-1}, Z}]^T>_q_{x_{n-1,n}}
        :param Z: inducing inputs (MxD) or (Mx(D+E))
        :param Xmu: X means ((N+1)xD)
        :param Xcov: X covariance matrices (2xNxDxD)
        :param CI: (optional) control inputs (NxE)
        :return: NxMxD
        """
        D = tf.shape(Xmu)[1]
        E = 0 if CI is None else tf.shape(CI)[1]
        lengthscales = self.lengthscales if self.ARD else \
            tf.zeros((D+E,), dtype=float_type) + self.lengthscales

        chol_L_plus_Xcov = tf.cholesky(tf.matrix_diag(lengthscales[:D] ** 2) + Xcov[0])  # NxDxD
        all_diffs = tf.transpose(Z[:, :D]) - tf.expand_dims(Xmu[:-1], 2)  # NxDxM

        sqrt_det_L = tf.reduce_prod(lengthscales[:D])
        sqrt_det_L_plus_Xcov = tf.exp(tf.reduce_sum(tf.log(tf.matrix_diag_part(chol_L_plus_Xcov)), axis=1))
        determinants = sqrt_det_L / sqrt_det_L_plus_Xcov  # N

        exponent_mahalanobis = tf.cholesky_solve(chol_L_plus_Xcov, all_diffs)  # NxDxM
        expectation_term = tf.matmul(Xcov[1], exponent_mahalanobis, transpose_a=True)
        expectation_term = tf.transpose(tf.expand_dims(Xmu[1:], 2) + expectation_term, [0, 2, 1])  # NxMxD

        exponent_mahalanobis = tf.reduce_sum(all_diffs * exponent_mahalanobis, 1)  # NxM
        if CI is not None:
            ci_L_inv =  CI / lengthscales[D:]  # NxE
            z_L_inv = Z[:, D:] / lengthscales[D:]  # MxE
            exponent_mahalanobis += tf.reduce_sum(tf.square(z_L_inv), 1) \
                                    + tf.expand_dims(tf.reduce_sum(tf.square(ci_L_inv), 1), 1) \
                                    - 2*tf.matmul(ci_L_inv, z_L_inv, transpose_b=True)
        exponent_mahalanobis = tf.exp(-0.5 * exponent_mahalanobis)  # NxM

        return self.variance * (determinants[:, None] * exponent_mahalanobis)[:, :, None] * expectation_term

    def eKzxKxz(self, Z, Xmu, Xcov, CI=None):
        """
        eKzxKxz[n] = <K_{Z, x_n}K_{x_n, Z}>_{q(x_n)}
        :param Z: inducing inputs (MxD) or (Mx(D+E))
        :param Xmu: X means (NxD)
        :param Xcov: X covariance matrices (NxDxD)
        :param CI: (optional) control inputs (NxE)
        :return: NxMxM
        """
        N = tf.shape(Xmu)[0]
        D = tf.shape(Xmu)[1]
        E = 0 if CI is None else tf.shape(CI)[1]

        squared_lengthscales = self.lengthscales ** 2. if self.ARD else \
            tf.zeros((D+E,), dtype=float_type) + self.lengthscales ** 2.

        sqrt_det_L = tf.reduce_prod(0.5 * squared_lengthscales[:D]) ** 0.5
        chol_L_plus_Xcov = tf.cholesky(0.5 * tf.matrix_diag(squared_lengthscales[:D]) + Xcov)  # NxDxD
        dets = sqrt_det_L / tf.exp(tf.reduce_sum(tf.log(tf.matrix_diag_part(chol_L_plus_Xcov)), axis=1))  # N

        C_inv_mu = tf.matrix_triangular_solve(chol_L_plus_Xcov, tf.expand_dims(Xmu, 2), lower=True)  # NxDx1
        C_inv_z = tf.matrix_triangular_solve(chol_L_plus_Xcov,
                                             tf.tile(tf.expand_dims(tf.transpose(Z[:, :D]) / 2., 0), [N, 1, 1]), lower=True)  # NxDxM
        mu_CC_inv_mu = tf.expand_dims(tf.reduce_sum(tf.square(C_inv_mu), 1), 2)  # Nx1x1
        z_CC_inv_z = tf.reduce_sum(tf.square(C_inv_z), 1)  # NxM
        zm_CC_inv_zn = tf.matmul(C_inv_z, C_inv_z, transpose_a=True)  # NxMxM
        two_z_CC_inv_mu = 2 * tf.matmul(C_inv_z, C_inv_mu, transpose_a=True)  # NxMx1

        if CI is not None:
            ci_twoL_inv = CI / (0.5 * squared_lengthscales[D:]) ** 0.5  # NxE
            z_twoL_inv = Z[:, D:] / (2. * squared_lengthscales[D:]) ** 0.5 # MxE
            mu_CC_inv_mu += tf.reshape(tf.reduce_sum(tf.square(ci_twoL_inv), 1), [N, 1, 1])  # Nx1x1
            z_CC_inv_z += tf.reduce_sum(tf.square(z_twoL_inv), 1)  # M -> NxM
            zm_CC_inv_zn += tf.matmul(z_twoL_inv, z_twoL_inv, transpose_b=True)  # MxM -> NxMxM
            two_z_CC_inv_mu += 2 * tf.expand_dims(tf.matmul(ci_twoL_inv, z_twoL_inv, transpose_b=True), 2)  # NxMx1

        exponent_mahalanobis = mu_CC_inv_mu + tf.expand_dims(z_CC_inv_z, 1) + tf.expand_dims(z_CC_inv_z, 2) + \
                               2 * zm_CC_inv_zn - two_z_CC_inv_mu - tf.transpose(two_z_CC_inv_mu, [0, 2, 1])  # NxMxM
        exponent_mahalanobis = tf.exp(-0.5 * exponent_mahalanobis)  # NxMxM

        return self.variance**1.5 * tf.sqrt(self.K(Z, presliced=True)) * tf.reshape(dets, [N, 1, 1]) * exponent_mahalanobis


class Linear(kernels.Linear):
    def eKdiag(self, Xmu, Xcov, CI=None):
        """
        <diag(K_{X, X})>_{q(X)}
        :param Xmu: kernel input (NxD)
        :param Xcov: X covariance matrices (NxDxD)
        :param CI: (optional) control inputs (NxE)
        :return: N
        """
        D = tf.shape(Xmu)[1]
        E = 0 if CI is None else tf.shape(CI)[1]
        variance = self.variance if self.ARD else tf.zeros((D + E,), dtype=float_type) + self.variance
        eKdiag = tf.reduce_sum(variance[:D] * (tf.matrix_diag_part(Xcov) + tf.square(Xmu)), 1)
        if CI is not None:
            eKdiag += tf.reduce_sum(variance[D:] * tf.square(CI), 1)
        return eKdiag

    def sum_eKdiag(self, Xmu, Xcov, CI=None):
        """
        \sum <diag(K_{X, X})>_{q(X)}
        :param Xmu: kernel input (NxD)
        :param Xcov: X covariance matrices (NxDxD)
        :param CI: (optional) control inputs (NxE)
        :return: ()
        """
        D = tf.shape(Xmu)[1]
        E = 0 if CI is None else tf.shape(CI)[1]
        variance = self.variance if self.ARD else tf.zeros((D + E,), dtype=float_type) + self.variance
        eKdiag = tf.reduce_sum(variance[:D] * tf.reduce_sum(tf.matrix_diag_part(Xcov) + tf.square(Xmu), 0))
        if CI is not None:
            eKdiag += tf.reduce_sum(variance[D:] * tf.reduce_sum(tf.square(CI), 0))
        return eKdiag

    def eKxz(self, Z, Xmu, Xcov, CI=None):
        """
        <K_{X, Z}>_{q(X)}
        :param Z: inducing inputs (MxD) or (Mx(D+E))
        :param Xmu: X means (NxD)
        :param Xcov: X covariance matrices (NxDxD)
        :param CI: (optional) control inputs (NxE)
        :return: NxM
        """
        _X = tf.identity(Xmu) if CI is None else tf.concat([Xmu, CI], 1)
        return tf.matmul(_X, Z * self.variance, transpose_b=True)

    def sum_eKxz(self, Z, Xmu, Xcov, CI=None):
        """
        \sum_n <K_{x_n, Z}>_{q(x_n)}
        :param Z: inducing inputs (MxD) or (Mx(D+E))
        :param Xmu: X means (NxD)
        :param Xcov: X covariance matrices (NxDxD)
        :param CI: (optional) control inputs (NxE)
        :return: M
        """
        _X = tf.reduce_sum(Xmu, 0) if CI is None else tf.reduce_sum(tf.concat([Xmu, CI], 1), 0)
        return tf.matmul(Z, tf.expand_dims(_X * self.variance, 1))[:, 0]

    def exKxz(self, Z, Xmu, Xcov, CI=None):
        """
        exKxz[n] = <[x_n K_{x_{n-1}, Z}]^T>_q_{x_{n-1,n}}
        :param Z: inducing inputs (MxD) or (Mx(D+E))
        :param Xmu: X means ((N+1)xD)
        :param Xcov: X covariance matrices (2xNxDxD)
        :param CI: (optional) control inputs (NxE)
        :return: NxMxD
        """
        N = tf.shape(Xmu)[0] - 1
        D = tf.shape(Xmu)[1]
        var_Z = self.variance * Z
        tiled_Z = tf.tile(tf.expand_dims(var_Z[:, :D], 0), (N, 1, 1))  # NxMxD
        exKxz = tf.matmul(tiled_Z, Xcov[1] + Xmu[:-1][..., None] * Xmu[1:][:, None, :])
        if CI is not None:
            exKxz += tf.matmul(CI, var_Z[:, D:], transpose_b=True)[..., None] * Xmu[1:][:, None, :]
        return exKxz

    def sum_exKxz(self, Z, Xmu, Xcov, CI=None):
        """
        \sum_n <[x_n K_{x_{n-1}, Z}]^T>_q_{x_{n-1,n}}
        :param Z: inducing inputs (MxD) or (Mx(D+E))
        :param Xmu: X means ((N+1)xD)
        :param Xcov: X covariance matrices (2xNxDxD)
        :param CI: (optional) control inputs (NxE)
        :return: MxD
        """
        D = tf.shape(Xmu)[1]
        var_Z = self.variance * Z
        summed_XX = tf.reduce_sum(Xcov[1] + Xmu[:-1][..., None] * Xmu[1:][:, None, :], 0)
        exKxz = tf.matmul(var_Z[:, :D], summed_XX)
        if CI is not None:
            summed_CX = tf.reduce_sum(CI[..., None] * Xmu[1:][:, None, :], 0)
            exKxz += tf.matmul(var_Z[:, D:], summed_CX)
        return exKxz

    def eKzxKxz(self, Z, Xmu, Xcov, CI=None):
        """
        eKzxKxz[n] = <K_{Z, x_n}K_{x_n, Z}>_{q(x_n)}
        :param Z: inducing inputs (MxD) or (Mx(D+E))
        :param Xmu: X means (NxD)
        :param Xcov: X covariance matrices (NxDxD)
        :param CI: (optional) control inputs (NxE)
        :return: NxMxM
        """
        N = tf.shape(Xmu)[0]
        E = 0 if CI is None else tf.shape(CI)[1]
        var_Z = self.variance * Z
        tiled_Z = tf.tile(tf.expand_dims(var_Z, 0), (N, 1, 1))  # NxMx(D+E)
        _X = tf.identity(Xmu) if CI is None else tf.concat([Xmu, CI], 1)
        _Xcov = tf.identity(Xcov) if CI is None else tf.pad(Xcov, [[0,0],[0,E],[0,E]])
        XX = _Xcov + tf.expand_dims(_X, 1) * tf.expand_dims(_X, 2)  # NxDxD
        return tf.matmul(tf.matmul(tiled_Z, XX), tiled_Z, transpose_b=True)

    def sum_eKzxKxz(self, Z, Xmu, Xcov, CI=None):
        """
        \sum_n <K_{Z, x_n}K_{x_n, Z}>_{q(x_n)}
        :param Z: inducing inputs (MxD) or (Mx(D+E))
        :param Xmu: X means (NxD)
        :param Xcov: X covariance matrices (NxDxD)
        :param CI: (optional) control inputs (NxE)
        :return: MxM
        """
        E = 0 if CI is None else tf.shape(CI)[1]
        var_Z = self.variance * Z
        _X = tf.identity(Xmu) if CI is None else tf.concat([Xmu, CI], 1)
        _Xcov = tf.reduce_sum(Xcov, 0) if CI is None else \
            tf.pad(tf.reduce_sum(Xcov, 0), [[0, E], [0, E]])
        XX = _Xcov + tf.reduce_sum(tf.expand_dims(_X, 1) * tf.expand_dims(_X, 2), 0)  # DxD
        return tf.matmul(tf.matmul(var_Z, XX), var_Z, transpose_b=True)


class Add(kernels.Add):
    """
    Add - eKzxKxz is only implemented for Linear + RBF
    """
    def __init__(self, kern_list):
        super(Add, self).__init__(kern_list)

    def eKdiag(self, X, Xcov, CI=None):
        return reduce(tf.add, [k.eKdiag(X, Xcov, CI) for k in self.kern_list])

    def eKxz(self, Z, Xmu, Xcov, CI=None):
        return reduce(tf.add, [k.eKxz(Z, Xmu, Xcov, CI) for k in self.kern_list])

    def exKxz(self, Z, Xmu, Xcov, CI=None):
        return reduce(tf.add, [k.exKxz(Z, Xmu, Xcov, CI) for k in self.kern_list])

    def eKzxKxz(self, Z, Xmu, Xcov, CI=None):
        if len(self.kern_list) != 2:
            raise NotImplementedError
        k_rbf, k_lin = (self.kern_list[0], self.kern_list[1]) if type(self.kern_list[0]) is RBF \
            else (self.kern_list[1], self.kern_list[0])
        assert type(k_lin) is Linear and type(k_rbf) is RBF

        all_sum = reduce(tf.add, [k.eKzxKxz(Z, Xmu, Xcov, CI) for k in self.kern_list])

        N = tf.shape(Xmu)[0]
        D = tf.shape(Xmu)[1]
        E = 0 if CI is None else tf.shape(CI)[1]

        k_lin_variance = k_lin.variance if k_lin.ARD else tf.zeros((D + E,), dtype=float_type) + k_lin.variance

        lengthscales = k_rbf.lengthscales if k_rbf.ARD else \
            tf.zeros((D + E,), dtype=float_type) + k_rbf.lengthscales  ## Begin RBF eKxz

        chol_L_plus_Xcov = tf.cholesky(tf.matrix_diag(lengthscales[:D] ** 2) + Xcov)  # NxDxD
        all_diffs = tf.transpose(Z[:, :D]) - tf.expand_dims(Xmu, 2)  # NxDxM

        sqrt_det_L = tf.reduce_prod(lengthscales[:D])
        sqrt_det_L_plus_Xcov = tf.exp(tf.reduce_sum(tf.log(tf.matrix_diag_part(chol_L_plus_Xcov)), axis=1))
        determinants = sqrt_det_L / sqrt_det_L_plus_Xcov  # N

        exponent_mahalanobis = tf.matrix_triangular_solve(chol_L_plus_Xcov, all_diffs, lower=True)  # NxDxM
        exponent_mahalanobis = tf.reduce_sum(tf.square(exponent_mahalanobis), 1)  # NxM
        if CI is not None:
            ci_L_inv = CI / lengthscales[D:]  # NxE
            z_L_inv = Z[:, D:] / lengthscales[D:]  # MxE
            exponent_mahalanobis += tf.reduce_sum(tf.square(z_L_inv), 1) \
                                    + tf.expand_dims(tf.reduce_sum(tf.square(ci_L_inv), 1), 1) \
                                    - 2 * tf.matmul(ci_L_inv, z_L_inv, transpose_b=True)

        exponent_mahalanobis = tf.exp(-0.5 * exponent_mahalanobis)  # NxM
        eKxz_rbf = k_rbf.variance * (determinants[:, None] * exponent_mahalanobis)  ## End RBF eKxz NxM

        tiled_Z = tf.tile(tf.expand_dims(Z[:, :D], 0), (N, 1, 1))  # NxMxD
        cross_eKzxKxz = tf.cholesky_solve(chol_L_plus_Xcov,
                                          tf.transpose((k_lin_variance[:D] * lengthscales[:D] ** 2.) * tiled_Z, [0, 2, 1]))
        z_L_inv_Xcov = tf.matmul(tiled_Z, Xcov / lengthscales[:D, None] ** 2.)  # NxMxD
        cross_eKzxKxz = tf.matmul((z_L_inv_Xcov + Xmu[:, None, :]) * eKxz_rbf[..., None],
                                  cross_eKzxKxz)  # NxMxM
        if CI is not None:
            cross_eKzxKxz += tf.matmul(CI, Z[:, D:] * k_lin_variance[D:], transpose_b=True)[:, None, :]  # k_lin contribution
        return all_sum + cross_eKzxKxz + tf.transpose(cross_eKzxKxz, [0, 2, 1])


class OldLinear(kernels.Linear):
    def eKdiag(self, X, Xcov):
        if self.ARD:
            raise NotImplementedError
        # use only active dimensions
        X, _ = self._slice(X, None)
        Xcov = self._slice_cov(Xcov)
        return self.variance * (tf.reduce_sum(tf.square(X), 1) + tf.reduce_sum(tf.matrix_diag_part(Xcov), 1))

    def eKxz(self, Z, Xmu, Xcov):
        if self.ARD:
            raise NotImplementedError
        # use only active dimensions
        Z, Xmu = self._slice(Z, Xmu)
        return self.variance * tf.matmul(Xmu, Z, transpose_b=True)

    def exKxz(self, Z, Xmu, Xcov, CI=None):
        if self.ARD:
            raise NotImplementedError
        with tf.control_dependencies([
            tf.assert_equal(tf.shape(Xmu)[1], tf.constant(self.input_dim, int_type),
                            message="Currently cannot handle slicing in exKxz."),
            tf.assert_equal(tf.shape(Xmu)[0]-1, tf.shape(Xcov)[1], name="assert_Xmu_Xcov_shape")
        ]):
            Xmu = tf.identity(Xmu)

        N = tf.shape(Xmu)[0] - 1
        Xmum = Xmu[:-1, :]
        Xmup = Xmu[1:, :]
        op = tf.expand_dims(Xmum, 2) * tf.expand_dims(Xmup, 1) + Xcov[1]  # NxDxD
        return self.variance * tf.matmul(tf.tile(tf.expand_dims(Z, 0), (N, 1, 1)), op)

    def eKzxKxz(self, Z, Xmu, Xcov, CI=None):
        if self.ARD:
            raise NotImplementedError
        # use only active dimensions
        Xcov = self._slice_cov(Xcov)
        Z, Xmu = self._slice(Z, Xmu)
        N = tf.shape(Xmu)[0]
        mom2 = tf.expand_dims(Xmu, 1) * tf.expand_dims(Xmu, 2) + Xcov  # NxDxD
        eZ = tf.tile(tf.expand_dims(Z, 0), (N, 1, 1))  # NxMxD
        return self.variance ** 2.0 * tf.matmul(tf.matmul(eZ, mom2), eZ, transpose_b=True)


class OldAdd(kernels.Add):
    """
    Add
    This version of Add will call the corresponding kernel expectations for each of the summed kernels. This will be
    much better for kernels with analytically calculated kernel expectations. If quadrature is to be used, it's probably
    better to do quadrature on the summed kernel function using `gpflow.kernels.Add` instead.
    """

    def __init__(self, kern_list):
        self.crossexp_funcs = {frozenset([Linear, RBF]): self.Linear_RBF_eKzxKxz}
        # self.crossexp_funcs = {}
        super(OldAdd, self).__init__(kern_list)

    def eKdiag(self, X, Xcov, CI=None):
        return reduce(tf.add, [k.eKdiag(X, Xcov, CI) for k in self.kern_list])

    def eKxz(self, Z, Xmu, Xcov, CI=None):
        return reduce(tf.add, [k.eKxz(Z, Xmu, Xcov, CI) for k in self.kern_list])

    def exKxz(self, Z, Xmu, Xcov, CI=None):
        return reduce(tf.add, [k.exKxz(Z, Xmu, Xcov, CI) for k in self.kern_list])

    def eKzxKxz(self, Z, Xmu, Xcov, CI=None):
        all_sum = reduce(tf.add, [k.eKzxKxz(Z, Xmu, Xcov, CI) for k in self.kern_list])

        if self.on_separate_dimensions and Xcov.get_shape().ndims == 2:
            # If we're on separate dimensions and the covariances are diagonal, we don't need Cov[Kzx1Kxz2].
            crossmeans = []
            eKxzs = [k.eKxz(Z, Xmu, Xcov, CI) for k in self.kern_list]
            for i, Ka in enumerate(eKxzs):
                for Kb in eKxzs[i + 1:]:
                    op = Ka[:, None, :] * Kb[:, :, None]
                    ct = tf.transpose(op, [0, 2, 1]) + op
                    crossmeans.append(ct)
            crossmean = reduce(tf.add, crossmeans)
            return all_sum + crossmean
        else:
            crossexps = []
            for i, ka in enumerate(self.kern_list):
                for kb in self.kern_list[i + 1:]:
                    try:
                        crossexp_func = self.crossexp_funcs[frozenset([type(ka), type(kb)])]
                        crossexp = crossexp_func(ka, kb, Z, Xmu, Xcov)
                    except (KeyError, NotImplementedError) as e:
                        print(str(e))
                        crossexp = self.quad_eKzx1Kxz2(ka, kb, Z, Xmu, Xcov)
                    crossexps.append(crossexp)
            return all_sum + reduce(tf.add, crossexps)

    def Linear_RBF_eKzxKxz(self, Ka, Kb, Z, Xmu, Xcov):
        Xcov = self._slice_cov(Xcov)
        Z, Xmu = self._slice(Z, Xmu)
        k_rbf, k_lin = (Ka, Kb) if type(Ka) is RBF else (Kb, Ka)
        assert type(k_lin) is Linear, "%s is not %s" % (str(type(k_lin)), str(Linear))
        assert type(k_rbf) is RBF, "%s is not %s" % (str(type(k_rbf)), str(RBF))

        N = tf.shape(Xmu)[0]
        D = tf.shape(Xmu)[1]

        k_lin_variance = k_lin.variance if k_lin.ARD else tf.zeros((D,), dtype=float_type) + k_lin.variance

        lengthscales = k_rbf.lengthscales if k_rbf.ARD else \
            tf.zeros((D,), dtype=float_type) + k_rbf.lengthscales  ## Begin RBF eKxz

        chol_L_plus_Xcov = tf.cholesky(tf.matrix_diag(lengthscales ** 2) + Xcov)  # NxDxD
        all_diffs = tf.transpose(Z) - tf.expand_dims(Xmu, 2)  # NxDxM

        sqrt_det_L = tf.reduce_prod(lengthscales)
        sqrt_det_L_plus_Xcov = tf.exp(tf.reduce_sum(tf.log(tf.matrix_diag_part(chol_L_plus_Xcov)), axis=1))
        determinants = sqrt_det_L / sqrt_det_L_plus_Xcov  # N

        exponent_mahalanobis = tf.matrix_triangular_solve(chol_L_plus_Xcov, all_diffs, lower=True)  # NxDxM
        exponent_mahalanobis = tf.reduce_sum(tf.square(exponent_mahalanobis), 1)  # NxM
        exponent_mahalanobis = tf.exp(-0.5 * exponent_mahalanobis)  # NxM
        eKxz_rbf = k_rbf.variance * (determinants[:, None] * exponent_mahalanobis)  ## End RBF eKxz NxM

        tiled_Z = tf.tile(tf.expand_dims(Z, 0), (N, 1, 1))  # NxMxD
        cross_eKzxKxz = tf.cholesky_solve(chol_L_plus_Xcov,
                                          tf.transpose((k_lin_variance * lengthscales ** 2.) * tiled_Z, [0, 2, 1]))
        z_L_inv_Xcov = tf.matmul(tiled_Z, Xcov / lengthscales[:, None] ** 2.)  # NxMxD
        cross_eKzxKxz = tf.matmul((z_L_inv_Xcov + Xmu[:, None, :]) * eKxz_rbf[..., None], cross_eKzxKxz)  # NxMxM
        return cross_eKzxKxz + tf.transpose(cross_eKzxKxz, [0, 2, 1])

    def Linear_RBF_eKzxKxz_old(self, Ka, Kb, Z, Xmu, Xcov):
        Xcov = self._slice_cov(Xcov)
        Z, Xmu = self._slice(Z, Xmu)
        lin, rbf = (Ka, Kb) if type(Ka) is Linear else (Kb, Ka)
        assert type(lin) is Linear, "%s is not %s" % (str(type(lin)), str(Linear))
        assert type(rbf) is RBF, "%s is not %s" % (str(type(rbf)), str(RBF))
        if lin.ARD or type(lin.active_dims) is not slice or type(rbf.active_dims) is not slice:
            raise NotImplementedError("Active dims and/or Linear ARD not implemented. Switching to quadrature.")
        D = tf.shape(Xmu)[1]
        M = tf.shape(Z)[0]
        N = tf.shape(Xmu)[0]
        lengthscales = rbf.lengthscales if rbf.ARD else tf.zeros((D,), dtype=float_type) + rbf.lengthscales
        lengthscales2 = lengthscales ** 2.0

        const = rbf.variance * lin.variance * tf.reduce_prod(lengthscales)

        gaussmat = Xcov + tf.matrix_diag(lengthscales2)[None, :, :]  # NxDxD

        det = tf.matrix_determinant(gaussmat) ** -0.5  # N

        cgm = tf.cholesky(gaussmat)  # NxDxD
        tcgm = tf.tile(cgm[:, None, :, :], [1, M, 1, 1])
        vecmin = Z[None, :, :] - Xmu[:, None, :]  # NxMxD
        d = tf.matrix_triangular_solve(tcgm, vecmin[:, :, :, None])  # NxMxDx1
        exp = tf.exp(-0.5 * tf.reduce_sum(d ** 2.0, [2, 3]))  # NxM
        # exp = tf.Print(exp, [tf.shape(exp)])

        vecplus = (Z[None, :, :, None] / lengthscales2[None, None, :, None] +
                   tf.matrix_solve(Xcov, Xmu[:, :, None])[:, None, :, :])  # NxMxDx1
        mean = tf.cholesky_solve(tcgm,
                                 tf.matmul(tf.tile(Xcov[:, None, :, :], [1, M, 1, 1]), vecplus)
                                 )[:, :, :, 0] * lengthscales2[None, None, :]  # NxMxD
        a = tf.matmul(tf.tile(Z[None, :, :], [N, 1, 1]),
                            mean * exp[:, :, None] * det[:, None, None] * const, transpose_b=True)
        return a + tf.transpose(a, [0, 2, 1])

    def quad_eKzx1Kxz2(self, Ka, Kb, Z, Xmu, Xcov):
        # Quadrature for Cov[(Kzx1 - eKzx1)(kxz2 - eKxz2)]
        self._check_quadrature()
        warnings.warn("gpflow.ekernels.Add: Using numerical quadrature for kernel expectation cross terms.")
        Xmu, Z = self._slice(Xmu, Z)
        Xcov = self._slice_cov(Xcov)
        N, M, HpowD = tf.shape(Xmu)[0], tf.shape(Z)[0], self.num_gauss_hermite_points ** self.input_dim
        xn, wn = mvhermgauss(self.num_gauss_hermite_points, self.input_dim)

        # transform points based on Gaussian parameters
        cholXcov = tf.cholesky(Xcov)  # NxDxD
        Xt = tf.matmul(cholXcov, tf.tile(xn[None, :, :], (N, 1, 1)), transpose_b=True)  # NxDxH**D

        X = 2.0 ** 0.5 * Xt + tf.expand_dims(Xmu, 2)  # NxDxH**D
        Xr = tf.reshape(tf.transpose(X, [2, 0, 1]), (-1, self.input_dim))  # (H**D*N)xD

        cKa, cKb = [tf.reshape(
            k.K(tf.reshape(Xr, (-1, self.input_dim)), Z, presliced=False),
            (HpowD, N, M)
        ) - k.eKxz(Z, Xmu, Xcov)[None, :, :] for k in (Ka, Kb)]  # Centred Kxz
        eKa, eKb = Ka.eKxz(Z, Xmu, Xcov), Kb.eKxz(Z, Xmu, Xcov)

        wr = wn * nppi ** (-self.input_dim * 0.5)
        cc = tf.reduce_sum(cKa[:, :, None, :] * cKb[:, :, :, None] * wr[:, None, None, None], 0)
        cm = eKa[:, None, :] * eKb[:, :, None]
        return cc + tf.transpose(cc, [0, 2, 1]) + cm + tf.transpose(cm, [0, 2, 1])


class Prod(kernels.Prod):
    def eKdiag(self, Xmu, Xcov):
        if not self.on_separate_dimensions:
            raise NotImplementedError("Prod currently needs to be defined on separate dimensions.")  # pragma: no cover
        with tf.control_dependencies([
            tf.assert_equal(tf.rank(Xcov), 2,
                            message="Prod currently only supports diagonal Xcov.", name="assert_Xcov_diag"),
        ]):
            return reduce(tf.multiply, [k.eKdiag(Xmu, Xcov) for k in self.kern_list])

    def eKxz(self, Z, Xmu, Xcov):
        if not self.on_separate_dimensions:
            raise NotImplementedError("Prod currently needs to be defined on separate dimensions.")  # pragma: no cover
        with tf.control_dependencies([
            tf.assert_equal(tf.rank(Xcov), 2,
                            message="Prod currently only supports diagonal Xcov.", name="assert_Xcov_diag"),
        ]):
            return reduce(tf.multiply, [k.eKxz(Z, Xmu, Xcov) for k in self.kern_list])

    def eKzxKxz(self, Z, Xmu, Xcov):
        if not self.on_separate_dimensions:
            raise NotImplementedError("Prod currently needs to be defined on separate dimensions.")  # pragma: no cover
        with tf.control_dependencies([
            tf.assert_equal(tf.rank(Xcov), 2,
                            message="Prod currently only supports diagonal Xcov.", name="assert_Xcov_diag"),
        ]):
            return reduce(tf.multiply, [k.eKzxKxz(Z, Xmu, Xcov) for k in self.kern_list])
