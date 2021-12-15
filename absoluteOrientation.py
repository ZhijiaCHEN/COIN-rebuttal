##########################################################################################################################################

# Python code for Absolute Orientation of dataset B to dataset A, both with n points each, where for each i in {1,2,...,n}, points A_i and B_i correspond

##########################################################################################################################################

import numpy as np

def absolute_orientation(sourceSpaceMatrix: np.ndarray, targetSpaceMatrix: np.ndarray, matToBeAligned: np.ndarray):
    # centering
    B = sourceSpaceMatrix
    A = targetSpaceMatrix
    assert A.shape == B.shape
    (M, N) = A.shape


    meanA = np.mean(A, 0)
    meanB = np.mean(B, 0)
    meanMat = np.mean(matToBeAligned, 0)
    Aprime = A - meanA
    Bprime = B - meanB
    matPrime = matToBeAligned - meanMat

    # rotation
    Z = np.zeros((N, N))
    for i in range(M):
        Z = Z + np.outer(Bprime[i],Aprime[i])

    U,S,V = np.linalg.svd(Z)
    R = np.matmul(U,V)
    rotB = np.matmul(Bprime, R) 
    rotMat = np.matmul(matPrime, R) 

    # scaling
    s1=0.0
    s2=0.0

    for i in range(0,M):
        s1=s1+np.dot(Aprime[i],rotB[i])
        s2=s2+np.dot(Bprime[i], Bprime[i])

    s = s1/s2

    # output matrix : B oriented onto A
    # newB = s*rotB + np.mean(A,0)
    newMat = s*rotMat + meanA
    return newMat

# A = np.asfarray([[-2, -1, 1, 3], [1, 2, -1, 5], [4, 5, 5, 6]])
# B = np.asfarray([[12, 24, 37, 90], [48, 52, 64, 98], [73, 85, 92, 95]])
# mat = np.asfarray([[12, 24, 37, 90], [48, 52, 64, 98], [73, 85, 92, 95], [73, 85, 92, 95], [48, 52, 64, 98], [12, 24, 37, 90]])
# # A += np.random()
# newMat = absolute_orientation(B, A, mat)
# print(newMat)