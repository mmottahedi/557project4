"""
QDA and LDA for Stat 557

"""

import numpy as np
from scipy.linalg import eig, inv
from numpy import diag, real, transpose, log, dot, sqrt


class DA(object):
    """
    Linear Discriminant Analysis
    """

    def __init__(self, x, y, mode="LDA"):
        """
        inputs:
            x: matrix of predictor
            y: matrix of response
        """
        self.x = x
        self.y = y
        self.mode = mode
        self.xDim = np.shape(x)
        self.yDim = np.shape(y)
        self.classSize = len(set(y))
        self.classMember = set(y)
        self.classMean = {}
        self.priorProbability = {}
        self.classCovMatrix = {}
        self.commonCovMatrix = None
        self.eigen_dict = {}

    def sampleEstimate(self):
        """
        compute the mean and covariance matrix and class prior probability
        """
        for K in self.classMember:
            classK_data = self.x[self.y == K, ]
            self.classMean[K] = np.transpose(np.mean(classK_data, 0))
            self.priorProbability[K] = sum(self.y == K) / len(self.y)
            self.classCovMatrix[K] = np.cov(np.transpose(classK_data))
        self.commonCovMatrix = np.cov(np.transpose(self.x))

    def fit(self):
        self.sampleEstimate()
        self.eigen_dict = self.eigDecomp()

    def LDA(self, x):
        """
        inputs:
            x: numpy array
        output:
            delta_k: dictionary detla for each class
        """
        #self.sampleEstimate()
        delta_k = {}
        D, U = eig(self.commonCovMatrix)

        for K in self.classMember:
            x_sphered = dot(dot(diag(1/sqrt(real(D))), U.T), x)
            trans_centroid = dot(dot(diag(1/sqrt(real(D))), U.T),
                                 self.classMean[K])
            delta_k[K] = 0.5 * sqrt(sum((x_sphered - trans_centroid) ** 2)) - \
                log(self.priorProbability[K])

        return min(delta_k, key=delta_k.get)

    def eigDecomp(self):
        """
        eigen value decomposition for each class
        returns dictionary of eig vectors and eigen values for each class
`        """
        eigen_dict = {}
        for key in self.classCovMatrix:
            eigen_dict[key] = eig(self.classCovMatrix[key])
        return eigen_dict

    def QDA(self, x):
        """
        inputs:
            x: numpy array
        output:
            maximum delta between classes
        """
        delta_k = {}
        # self.sampleEstimate()
        eigen_dict = self.eigen_dict
        for K in self.classMember:
            D_k = diag(real(eigen_dict[K][0]))
            U_k = eigen_dict[K][1]
            log_cov = sum(diag(D_k))
            part2 = dot(dot(transpose(dot(U_k.T,
                                      (x - self.classMean[K]))), inv(D_k)),
                        (dot(U_k.T, (x - self.classMean[K]))))
            part3 = log(self.priorProbability[K])

            delta_k[K] = -0.5 * log_cov - 0.5 * part2 + part3
        return max(delta_k, key=delta_k.get)

    def summary(self):
        self.accuracy = sum(self.y_hat == self.y) / len(self.y)
        print("accuracy: ", self.accuracy)

    def predict(self, X, Y):
        """
        return predicted class and error rate
        parameter:
            X: input numpy array
            Y: class label numpy array
        returns:
            predicted label and error rate
        """
        y_hat = []
        if self.mode == "LDA":
            for x in X:
                y_hat.append(self.LDA(x))
        if self.mode == "QDA":
            for x in X:
                y_hat.append(self.QDA(x))

        error = 1 - sum(Y == y_hat) / len(Y)
        return y_hat, error


if __name__ == "__main__":
    np.random.seed(1234)
    cov = [[5.79, -0.14], [-.14, 1.66]]
    mean1 = [0, 0]
    mean2 = [2, 2]
    Y = np.random.randint(0, 2, 1000)
    X1 = np.random.multivariate_normal(mean1, cov, 475)
    X2 = np.random.multivariate_normal(mean2, cov, 525)
    X = np.zeros((1000, 2))
    X[Y == 0] = X1
    X[Y == 1] = X2
    fit = DA(X, Y, mode="LDA")
    print(fit.xDim, fit.yDim)
    fit.sampleEstimate()
    print(fit.classMean)
    print("priorProbability: \n ", fit.priorProbability)
    print("common covariance matrix: \n ", fit.commonCovMatrix)
    print(type(fit.commonCovMatrix))
    y_hat, error = fit.predict(X, Y)
    print("error: ", error)
