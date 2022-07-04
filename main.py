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

    i, j = 0, 0
    for key, value in dictionary.items():
        for key2, value2 in dictionary.items():
            A1_new[i][j] = ex(value, value2, epsilon) / math.sqrt(Q[key][key] * Q[key2][key2])

            j += 1
        i += 1
    return np.array(A1_new)


def make_A2(Q, S1, S2, epsilon, N):
    A2_new = np.zeros((len(S1), N - len(S1)))

    i, j = 0, 0
    for key, value in S1.items():
        for key2, value2 in S2.items():
            A2_new[i][j] = ex(value, value2, epsilon) / math.sqrt(Q[key][key] * Q[key2][key2])

            j += 1
        i += 1
    return np.array(A2_new)


def main():
    N = 5

    x = np.array(get_random_coordinates(N, 3))
    print(x)
    print(x[0][0] ** 2 + x[1][0] ** 2 + x[2][0] ** 2)

    print("x shape", x.shape)

    M = np.array([get_random_uniform_iid(3) for i in range(17)])

    print(M)
    print("M shape", M.shape)
    print("Rank of M:", rank(M))

    F = M.dot(x)
    # print("F",F)
    print("F shape", F.shape)

    epsilon = 1
    print("Epsilon:", epsilon)
    mi = 7.8e-6
    print("Mi:", mi)

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

    print("Q", Q)

    # empty matrix
    A1 = np.zeros((len(S1), len(S1)))
    A2 = np.zeros((len(S1), N - len(S1)))

    print("A1", A1.shape)
    print("A2", A2.shape)

    # A1[0][0] = ex(S1[0], S1[0], epsilon) / math.sqrt(Q[0][0] * Q[0][0])
    A1 = make_A1(Q, S1, epsilon)
    A2 = make_A2(Q, S1, S2, epsilon, N)

    A1_f = fractional_matrix_power(A1, -0.5)

    print("A1_f", A1_f)

    C = A1 + A1_f.dot(A2).dot(A2.transpose()).dot(A1_f)

    print("C", C.shape)

    # matrix svd
    fi, lam, _ = np.linalg.svd(C)

    # array to diagonal matrix
    lam = np.diag(lam)
    print("lam", lam)

    lam_f = fractional_matrix_power(lam,-0.5)

    fi_k = np.r_[A1 , A2.transpose()].dot(A1_f).dot(fi).dot(lam_f)
    print("fi_k", fi_k.shape)

    ONM = fractional_matrix_power(Q, -0.5).dot(fi_k).dot(lam)

    print("ONM", ONM.shape)
    S1_ = S1.copy()
    for k in range(N-1):
        S1_[k] = F[:, k]









if __name__ == "__main__":
    main()
