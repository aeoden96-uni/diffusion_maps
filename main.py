import math

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import fractional_matrix_power


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
    return math.exp(- np.linalg.norm(x - y) / epsilon)


def make_A1(Q, dictionary, epsilon):
    A1_new = np.zeros((len(dictionary), len(dictionary)))
    # print(dictionary)
    # print(len(dictionary))
    # print("Q", Q.shape)
    i, j = 0, 0
    for key, value in dictionary.items():
        j = 0
        for key2, value2 in dictionary.items():
            # print(i,j,key,key2)
            A1_new[i][j] = ex(value, value2, epsilon) / math.sqrt(Q[key][key] * Q[key2][key2])

            j += 1
        i += 1
    return np.array(A1_new)


def make_A2(Q, S1, S2, epsilon, N):
    A2_new = np.zeros((len(S1), N - len(S1)))
    form ( "A2_new",A2_new.shape)
    form ( "S1" , len(S1))
    form ("S2", len(S2))
    i, j = 0, 0
    for key, value in S1.items():
        j = 0
        for key2, value2 in S2.items():
            Q_ = Q[key][key] * Q[key2][key2]
            new_val = ex(value, value2, epsilon) / math.sqrt(Q_)
            A2_new[i][j] = new_val

            j += 1
        i += 1
    return np.array(A2_new)


def form(s,x):
    print(f'{s: <{10}}',x)


def make_C(A1, A2):
    A1_f = fractional_matrix_power(A1, -0.5)
    return A1 + A1_f.dot(A2).dot(A2.transpose()).dot(A1_f)


def main():
    N = 13

    x = np.array(get_random_coordinates(N, 3))
    # print(x)
    # print(x[0][0] ** 2 + x[1][0] ** 2 + x[2][0] ** 2)

    form("x shape", x.shape)

    M = np.array([get_random_uniform_iid(3) for i in range(17)])


    form("M shape",M.shape)
    form("M rank", rank(M))

    F = M.dot(x)
    # print("F",F)
    form("F shape", F.shape)

    epsilon = 1
    form("Epsilon", epsilon)
    mi = 7.8e-6
    form("Mi", mi)

    S1 = {
        0: F[:, 0]
    }
    S2 = {

    }

    for i in range(N - 1):
        S2[i + 1] = F[:, i + 1]

    # diagonal matrix
    Q = np.diag(np.ones(N))

    for i in range(len(S1)):
        qi = 0
        for j in range(len(S1)):
            qi += ex(F[:, i], F[:, j], epsilon)
        Q[i, i] = qi

    form("Q", Q.shape)

    # empty matrix
    A1 = np.zeros((len(S1), len(S1)))
    A2 = np.zeros((len(S1), N - len(S1)))

    form("A1", A1.shape)
    form("A2", A2.shape)

    # A1[0][0] = ex(S1[0], S1[0], epsilon) / math.sqrt(Q[0][0] * Q[0][0])
    A1 = make_A1(Q, S1, epsilon)
    A2 = make_A2(Q, S1, S2, epsilon, N)

    A1_f = fractional_matrix_power(A1, -0.5)

    form("A1_f", A1_f.shape)

    C = make_C(A1, A2)

    form("C", C.shape)

    # matrix svd
    fi, lam, _ = np.linalg.svd(C)

    # array to diagonal matrix
    lam = np.diag(lam)
    form("lam", lam)

    lam_f = fractional_matrix_power(lam, -0.5)

    fi_k = np.r_[A1, A2.transpose()].dot(A1_f).dot(fi).dot(lam_f)
    form("fi_k", fi_k.shape)

    ONM = fractional_matrix_power(Q, -0.5).dot(fi_k).dot(lam)

    form("ONM", ONM.shape)

    for k in range(1, N):
        S1_ = S1.copy()
        S2_ = S2.copy()

        S1_[k] = F[:, k]
        form("S1_", len(S1_))
        form("S2_", len(S2_))
        del S2_[k]
        form("S2_", len(S2_))
        A1_ = make_A1(Q, S1_, epsilon)
        A2_ = make_A2(Q, S1_, S2_, epsilon, N)
        A1_f_ = fractional_matrix_power(A1_, -0.5)
        C_ = make_C(A1_, A2_)

        fi_, lam_, _ = np.linalg.svd(C_)
        lam_ = np.diag(lam_)

        lam_f_ = fractional_matrix_power(lam_, -0.5)



        form("A1_",A1_.shape)
        form("A2_", A2_.shape)
        # form("A1_f_", A1_f_.shape)
        # form("C_", C_.shape)
        # form("fi_", fi_.shape)
        # form("lam_", lam_.shape)
        # form("lam_f_", lam_f_.shape)

        fi_k_ = np.r_[A1_, A2_.transpose()].dot(A1_f_).dot(fi_).dot(lam_f_)

        ONM_ = np.array(fractional_matrix_power(Q, -0.5).dot(fi_k_).dot(lam_))

        form("ONM_", ONM_.shape)

        T = ONM.transpose().dot(ONM_)

        form("T",T.shape)



        beta = np.linalg.norm(ONM[k].dot(T) - ONM_[k])

        form("beta", beta)

        if beta > mi/2:
            S1 = S1_
            S2 = S2_
            A1 = A1_
            A2 = A2_
            ONM = ONM_

    form("ONM",ONM.shape)




if __name__ == "__main__":
    main()
