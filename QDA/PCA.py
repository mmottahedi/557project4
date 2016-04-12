"""
Principle Component Analysis
"""

from numpy import shape, mean, var, dot, abs, real, reshape, array
from scipy.linalg import eig
from collections import OrderedDict


class PCA(object):

    """
    Principle Componenet Analysis
    """

    def __init__(self, x):
        """
        inputs:
            numpy array with nxp dimension
        """

        self.x = x
        self.dim = shape(x)
        self.centerd_x = x - mean(x, 0)
        self.means = mean(x, 0)
        self.vars = var(x, 0)

    def fit(self):
        """
        returns:
            sorted eigen values/vectors tuple
         """
        eigval, eigvec = eig(dot(self.centerd_x.T, self.centerd_x))
        # eig_pairs = [(abs(eigval[i]), eigvec[:, i]) for i in range(len(eigval))]

        eig_pairs = {abs(eigval[i]): eigvec[:, i] for i in range(len(eigval))}
        #print(shape(eig_pairs))
        # eig_pairs.sort(reverse=True)

        return OrderedDict(sorted(eig_pairs.items(), reverse=True))

    def summary(self, n=None):

        pairs = self.fit()
        d = [i for i,j in pairs.items()]
        #self.d = reshape(d, (1, self.dim[1]))

        if n is None:
            n = self.dim[1]

        var_exp = (d / sum(d))
        return var_exp[:n]

    def transform(self, n=None):
        pairs = self.fit()
        W = [j for i,j in pairs.items()]
        if n is None:
            n = self.dim[1]
       #W = reshape(W, (self.dim[1], self.dim[1]), order='C')
        W = array(W)
        self.W = W
        X_new = dot(self.centerd_x, W.T[:, :n])
        return X_new

    def decomp(self):
        return self.d, self.W


if __name__ == "__main__":

    import pandas as pd
    import numpy as np

    train = pd.read_csv("Forest/train.csv")
    test = pd.read_csv("Forest/test.csv")

    xtrain = train.iloc[:, 1:-1]
    ytrain = train.iloc[:, -1]

    xtest = test.iloc[:, 1:-1]
    ytest = test.iloc[:, -1]

    pca = PCA(xtrain.values)
    print(pca.summary())