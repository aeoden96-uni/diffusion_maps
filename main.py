import math

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import fractional_matrix_power


def fmp(A):
    return fractional_matrix_power(A, -0.5)


# iid sampling of [0,1]
def get_random_uniform_iid(n):
    return np.random.uniform(0, 1, n)


def get_random_uniform(n):
    return np.random.rand(n)


# sphere sampling
def get_random_coordinates(n, dim):
    vec = np.random.randn(dim, n)
    vec /= np.linalg.norm(vec, axis=0)
    return vec


# rank of matrix
def rank(A):
    U, S, V = np.linalg.svd(A)
    return np.sum(S > 1e-10)


def ex(x, y, epsilon):
    return math.exp(- np.linalg.norm(x - y) ** 2 / epsilon)


def make_A1(A_tilda, S1):
    # 2x13 -> 2x2
    # truncate A_tilda to 2x2
    return A_tilda[:, :len(S1)]


def make_A2(A_tilda, S1):
    return A_tilda[:, len(S1):]


def form(s, x):
    print(f'{s: <{10}}', x)


def make_C(A1, A2):
    A1_f = fractional_matrix_power(A1, -0.5)
    return A1 + A1_f.dot(A2).dot(A2.transpose()).dot(A1_f)


def K_tilda(K, S1, N):
    # S1 = {
    #     # first column of F
    #     3: F[:, 3],
    #     8: F[:, 8],
    #
    # }
    K_tilda = np.zeros((len(S1), N))
    S1_list_keys = [key for key in sorted(S1.keys())]

    for i in range(len(S1)):
        K_tilda[i, :] = K[S1_list_keys[i], :]

    return K_tilda


def A_tilda(Q_t, K_t, Q):
    return fmp(Q_t).dot(K_t).dot(fmp(Q))


def Q_tilda(S1, Q, epsilon):
    S1_list = [S1[key] for key in sorted(S1.keys())]
    S1_list_keys = [key for key in sorted(S1.keys())]

    # S1 = {
    #     # first column of F
    #     3: F[:, 3],
    #     8: F[:, 8],
    #
    # }
    Q_tilda = np.diag(np.ones(len(S1)))
    for i in range(len(S1_list)):
        qi = 0
        for j in range(len(S1_list)):
            x = Q[S1_list_keys[i], S1_list_keys[i]] * Q[S1_list_keys[j], S1_list_keys[j]]

            qi += ex(S1_list[i], S1_list[j], epsilon) / x
        Q_tilda[i, i] = qi
    return Q_tilda


def main():
    N = 100

    x = np.array(get_random_coordinates(N, 3))
    # print(x)
    # print(x[0][0] ** 2 + x[1][0] ** 2 + x[2][0] ** 2)

    form("x shape", x.shape)

    M = np.array([get_random_uniform_iid(3) for i in range(17)])

    form("M shape", M.shape)
    form("M rank", rank(M))

    F = M.dot(x)
    # print("F",F)
    form("F shape", F.shape)

    epsilon = 1
    form("Epsilon", epsilon)
    mi = 7.8e-6
    form("Mi", mi)

    # RIJECNICI
    S1 = {
        # first column of F
        0: F[:, 0],

    }

    S2 = {

    }
    for i in range(N - 1):
        S2[i + 1] = F[:, i + 1]

    # diagonal matrix
    Q = np.diag(np.ones(N))

    for i in range(N):
        qi = 0
        for j in range(N):
            qi += ex(F[:, i], F[:, j], epsilon)
        Q[i, i] = qi

    # print matrix Q
    # form("Q", Q)

    form("Q", Q.shape)

    Q_t = Q_tilda(S1, Q, epsilon)

    print("Q_t", Q_t)

    K = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            K[i, j] = ex(F[:, i], F[:, j], epsilon)

    K_t = K_tilda(K, S1, N)

    print("K_t", K_t)

    A_t = A_tilda(Q_t, K_t, Q)

    form("A_tilda", A_t.shape)

    A1 = make_A1(A_t, S1)
    A2 = make_A2(A_t, S1)

    form("A1", A1.shape)
    form("A2", A2.shape)

    C = make_C(A1, A2)

    form("C", C.shape)

    # matrix svd
    fi, lam, _ = np.linalg.svd(C)

    # array to diagonal matrix
    lam = np.diag(lam)
    form("lam", lam)

    lam_f = fmp(lam)

    A1_f = fmp(A1)

    fi_k = np.r_[A1, A2.transpose()].dot(A1_f).dot(fi).dot(lam_f)
    form("fi_k", fi_k.shape)

    ONM = fmp(Q).dot(fi_k).dot(lam)

    form("ONM", ONM.shape)

    for k in range(1, N):
        S1_ = S1.copy()
        S2_ = S2.copy()

        S1_[k] = F[:, k]
        form("S1_", len(S1_))
        form("S2_", len(S2_))
        del S2_[k]
        form("S2_", len(S2_))

        form("S1_", len(S1_))

        Q_tt = Q_tilda(S1_, Q, epsilon)
        K_tt = K_tilda(K, S1_, N)

        A_ = A_tilda(Q_tt, K_tt, Q)

        A1_ = make_A1(A_, S1_)
        A2_ = make_A2(A_, S1_)

        C_ = make_C(A1_, A2_)

        fi_, lam_, _ = np.linalg.svd(C_)
        lam_ = np.diag(lam_)
        lam_f_ = fractional_matrix_power(lam_, -0.5)

        form("A1_", A1_.shape)
        form("A2_", A2_.shape)
        # form("A1_f_", A1_f_.shape)
        # form("C_", C_.shape)
        # form("fi_", fi_.shape)
        # form("lam_", lam_.shape)
        # form("lam_f_", lam_f_.shape)
        A1_f_ = fractional_matrix_power(A1_, -0.5)
        fi_k_ = np.r_[A1_, A2_.transpose()].dot(A1_f_).dot(fi_).dot(lam_f_)

        ONM_ = np.array(fmp(Q).dot(fi_k_).dot(lam_))

        form("ONM_", ONM_.shape)

        S1_list_keys = [key for key in sorted(S1_.keys())]

        B = np.zeros((len(S1_), len(S1_) - 1))
        for i in range(len(S1_list_keys)):
            B[i, :] = ONM[S1_list_keys[i], :]

        form("B", B.shape)

        D = np.zeros((len(S1_), len(S1_)))
        for i in range(len(S1_list_keys)):
            D[i, :]= ONM_[S1_list_keys[i], :]

        form("D", D.shape)

        T = B.transpose().dot(D)

        form("T", T.shape)

        beta = np.linalg.norm(ONM[k].dot(T) - ONM_[k])

        form("beta", beta)


        if beta > mi / 2:
            S1 = S1_
            S2 = S2_
            A1 = A1_.copy()
            A2 = A2_.copy()
            ONM = ONM_





if __name__ == "__main__":
    main()
