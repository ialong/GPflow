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
    def eKdiag(self, X, Xcov=None):
        """
        Also known as phi_0.
        :param X:
        :return: N
        """
        return self.Kdiag(X)

    def eKxz(self, Z, Xmu, Xcov):
        """
        Also known as phi_1: <K_{x, Z}>_{q(x)}.
        :param Z: MxD inducing inputs
        :param Xmu: X mean (NxD)
        :param Xcov: NxDxD
        :return: NxM
        """
        # use only active dimensions
        Xcov = self._slice_cov(Xcov)
        Z, Xmu = self._slice(Z, Xmu)
        D = tf.shape(Xmu)[1]
        lengthscales = self.lengthscales if self.ARD else tf.zeros((D,), dtype=float_type) + self.lengthscales

        vec = tf.expand_dims(Xmu, 2) - tf.expand_dims(tf.transpose(Z), 0)  # NxDxM
        chols = tf.cholesky(tf.expand_dims(tf.diag(lengthscales ** 2), 0) + Xcov)
        Lvec = tf.matrix_triangular_solve(chols, vec)
        q = tf.reduce_sum(Lvec ** 2, [1])

        chol_diags = tf.matrix_diag_part(chols)  # N x D
        half_log_dets = tf.reduce_sum(tf.log(chol_diags), 1) - tf.reduce_sum(tf.log(lengthscales))  # N,

        return self.variance * tf.exp(-0.5 * q - tf.expand_dims(half_log_dets, 1))

    def exKxz(self, Z, Xmu, Xcov, CI=None):
        """
        <x_t K_{x_{t-1}, Z}>_q_{x_{t-1:t}}
        :param Z: inducing inputs (MxD) or (Mx(D+E)) 
        :param Xmu: X means ((N+1)xD)
        :param Xcov: X covariance matrices (2xNxDxD)
        :param CI: (optional) control inputs (NxE)
        :return: NxMxD
        """
        with tf.control_dependencies([
            tf.assert_equal(tf.shape(Xmu)[0]-1, tf.shape(Xcov)[1], name="assert_Xmu_Xcov_shape")
        ]):
            Xmu = tf.identity(Xmu)

        D = tf.shape(Xmu)[1]
        E = 0 if CI is None else tf.shape(CI)[1]
        squared_lengthscales = self.lengthscales ** 2. if self.ARD else \
            tf.zeros((D+E,), dtype=float_type) + self.lengthscales ** 2.

        chol_L_plus_Xcov = tf.cholesky(tf.diag(squared_lengthscales[:D]) + Xcov[0])  # NxDxD
        all_diffs = tf.transpose(Z[:, :D]) - tf.expand_dims(Xmu[:-1], 2)  # NxDxM

        sqrt_det_L = tf.reduce_prod(squared_lengthscales[:D]) ** 0.5
        sqrt_det_L_plus_Xcov = tf.exp(tf.reduce_sum(tf.log(tf.matrix_diag_part(chol_L_plus_Xcov)), axis=1))
        determinants = sqrt_det_L / sqrt_det_L_plus_Xcov  # N

        exponent_mahalanobis = tf.matrix_triangular_solve(chol_L_plus_Xcov, all_diffs, lower=True)  # NxDxM
        exponent_mahalanobis = tf.reduce_sum(tf.square(exponent_mahalanobis), 1)  # NxM
        if CI is not None:
            ci_L_inv =  CI / squared_lengthscales[D:] ** 0.5  # NxE
            z_L_inv = Z[:, D:] / squared_lengthscales[D:] ** 0.5  # MxE
            exponent_mahalanobis += tf.reduce_sum(tf.square(ci_L_inv), 1) \
                                    + tf.expand_dims(tf.reduce_sum(tf.square(ci_L_inv), 1), 1) \
                                    - 2*tf.matmul(ci_L_inv, z_L_inv, transpose_b=True)

        exponent_mahalanobis = tf.exp(-0.5 * exponent_mahalanobis)  # NxM

        expectation_term = tf.cholesky_solve(chol_L_plus_Xcov, all_diffs)
        expectation_term = tf.matmul(Xcov[1], expectation_term, transpose_a=True)
        expectation_term = tf.transpose(tf.expand_dims(Xmu[1:], 2) + expectation_term, [0, 2, 1])  # NxMxD

        return self.variance * (determinants[:, None] * exponent_mahalanobis)[:, :, None] * expectation_term

    def eKzxKxz(self, Z, Xmu, Xcov, CI=None):
        """
        Also known as Phi_2.
        :param Z: inducing inputs (MxD) or (Mx(D+E))
        :param Xmu: X means (NxD)
        :param Xcov: X covariance matrices (NxDxD)
        :param CI: (optional) control inputs (NxE)
        :return: NxMxM
        """
        # use only active dimensions
        Xcov = self._slice_cov(Xcov)
        Z, Xmu = self._slice(Z, Xmu)

        N = tf.shape(Xmu)[0]
        D = tf.shape(Xmu)[1]
        E = 0 if CI is None else tf.shape(CI)[1]

        squared_lengthscales = self.lengthscales ** 2. if self.ARD else \
            tf.zeros((D+E,), dtype=float_type) + self.lengthscales ** 2.

        sqrt_det_L = tf.reduce_prod(0.5 * squared_lengthscales[:D]) ** 0.5
        chol_L_plus_Xcov = tf.cholesky(0.5 * tf.diag(squared_lengthscales[:D]) + Xcov)  # NxDxD
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

    def exKxz(self, Z, Xmu, Xcov):
        with tf.control_dependencies([
            tf.assert_equal(tf.shape(Xmu)[1], tf.constant(self.input_dim, int_type),
                            message="Currently cannot handle slicing in exKxz."),
            tf.assert_equal(tf.shape(Xmu), tf.shape(Xcov)[1:3], name="assert_Xmu_Xcov_shape")
        ]):
            Xmu = tf.identity(Xmu)

        N = tf.shape(Xmu)[0] - 1
        Xmum = Xmu[:-1, :]
        Xmup = Xmu[1:, :]
        op = tf.expand_dims(Xmum, 2) * tf.expand_dims(Xmup, 1) + Xcov[1, :-1, :, :]  # NxDxD
        return self.variance * tf.matmul(tf.tile(tf.expand_dims(Z, 0), (N, 1, 1)), op)

    def eKzxKxz(self, Z, Xmu, Xcov):
        """
        exKxz
        :param Z: MxD
        :param Xmu: NxD
        :param Xcov: NxDxD
        :return:
        """
        # use only active dimensions
        Xcov = self._slice_cov(Xcov)
        Z, Xmu = self._slice(Z, Xmu)
        N = tf.shape(Xmu)[0]
        mom2 = tf.expand_dims(Xmu, 1) * tf.expand_dims(Xmu, 2) + Xcov  # NxDxD
        eZ = tf.tile(tf.expand_dims(Z, 0), (N, 1, 1))  # NxMxD
        return self.variance ** 2.0 * tf.matmul(tf.matmul(eZ, mom2), eZ, transpose_b=True)


class Add(kernels.Add):
    """
    Add
    This version of Add will call the corresponding kernel expectations for each of the summed kernels. This will be
    much better for kernels with analytically calculated kernel expectations. If quadrature is to be used, it's probably
    better to do quadrature on the summed kernel function using `GPflow.kernels.Add` instead.
    """

    def __init__(self, kern_list):
        self.crossexp_funcs = {frozenset([Linear, RBF]): self.Linear_RBF_eKxzKzx}
        # self.crossexp_funcs = {}
        kernels.Add.__init__(self, kern_list)

    def eKdiag(self, X, Xcov):
        return reduce(tf.add, [k.eKdiag(X, Xcov) for k in self.kern_list])

    def eKxz(self, Z, Xmu, Xcov):
        return reduce(tf.add, [k.eKxz(Z, Xmu, Xcov) for k in self.kern_list])

    def exKxz(self, Z, Xmu, Xcov):
        return reduce(tf.add, [k.exKxz(Z, Xmu, Xcov) for k in self.kern_list])

    def eKzxKxz(self, Z, Xmu, Xcov):
        all_sum = reduce(tf.add, [k.eKzxKxz(Z, Xmu, Xcov) for k in self.kern_list])

        if self.on_separate_dimensions and Xcov.get_shape().ndims == 2:
            # If we're on separate dimensions and the covariances are diagonal, we don't need Cov[Kzx1Kxz2].
            crossmeans = []
            eKxzs = [k.eKxz(Z, Xmu, Xcov) for k in self.kern_list]
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

    def Linear_RBF_eKxzKzx(self, Ka, Kb, Z, Xmu, Xcov):
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

        gaussmat = Xcov + tf.diag(lengthscales2)[None, :, :]  # NxDxD

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
        warnings.warn("GPflow.ekernels.Add: Using numerical quadrature for kernel expectation cross terms.")
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
