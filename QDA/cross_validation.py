"""
module for cross validation
"""

from numpy.random import shuffle


def cv(model, mode, k_fold, X, Y):
    """
    cross validation

    parameters:
    model: model object used for fitting
    k_fold: number fold to devide the data
    X: input data numpy arraya
    Y: class label numpy array
    """
    if len(X) % k_fold:
        fold_size = int(len(X) / k_fold)
    else:
        fold_size = int(len(X) // k_fold)
    indx = list(range(len(X)))
    shuffle(indx)
    test_indx = [indx[i: i + fold_size] for i in range(0, len(indx), fold_size)]

    test_error = []
    train_error = []

    for fold in test_indx:
        train_indx = list(set(indx) - set(fold))
        X_train = X[train_indx]
        Y_train = Y[train_indx]
        X_test = X[fold]
        Y_test = Y[fold]

        fit = model(X_train, Y_train, mode)
        train_error.append(fit.predict(X_train, Y_train)[1])
        test_error.append(fit.predict(X_test, Y_test)[1])
    return train_error, test_error

if __name__ == "__main__":

    from LDA import DA
    import numpy as np

    np.random.seed(1234)
    cov1 = [[2, 5], [0, 1]]
    cov2 = [[1, 1], [0, 1]]
    mean1 = [0, 0]
    mean2 = [-3, -2]
    Y = np.random.randint(0, 2, 1000)
    X1 = np.random.multivariate_normal(mean1, cov1, 475)
    X2 = np.random.multivariate_normal(mean2, cov2, 525)
    X = np.zeros((1000, 2))
    X[Y == 0] = X1
    X[Y == 1] = X2

    train_e, test_e = cv(DA, "LDA", 3, X, Y)
    print(train_e, test_e)
