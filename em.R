library(mvtnorm)


# DATA CLEANING -----------------------------------------------------------


# read data in and format it
data <- read.csv('http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data', header = F, na.strings = '?')
names(data) <- c('id','thickness','size.unif','shape.unif','adhesion','size.cell','nuclei','chromatin','nucleoli','mitoses','class')
data$class <- as.factor(data$class)
levels(data$class) <- c('benign','malignant')
data <- data[-which(is.na(data$nuclei)),-1] # leave ID column out and remove NAs


# FUNCTIONS ---------------------------------------------------------------


### split data into test and training sets
# df:         data set
# train_amt:  proportion of data to use for training
# resp_idx:   which column the class variable is (for breast cancer data this is 10)
split_data <- function(df, train_amt, resp_idx) {
  N <- floor(nrow(df) * train_amt)
  ind <- sample.int(nrow(df), size = N)
  x_train <- as.matrix(df[ind,-resp_idx])
  x_test <- as.matrix(df[-ind,-resp_idx])
  class_train <- df[ind,resp_idx]
  class_test <- df[-ind,resp_idx]
  return(list(x_train = x_train,
              x_test = x_test,
              class_train = class_train,
              class_test = class_test))
}

### EM algorithm
# x:      covariates (as a matrix, one row = one observation)
# y:      response (as a factor)
# R:      number of subclasses for each class
# tol:    condition for convergence
# maxit:  maximum number of iterations for EM to run
EM <- function(x, y, R, tol = 1e-6, maxit = 10) {
  labs <- levels(y)
  K <- length(labs)
  M <- ncol(x)
  N <- nrow(x)
  pi <- array(0, dim = c(K, R))
  mu <- array(0, dim = c(K, M, R))
  sigma <- diag(M)
  y <- as.numeric(y)
  ### come up with starting points using kmeans
  for(k in 1:K) {
    ind <- which(y == k)
    mu[k, , ] <- t(kmeans(x[ind, ], R)$centers)
    pi[k, ] <- rep(1/R, length = R)
  }
  converged <- F
  ### main EM loop
  # while(!converged) {
  for(it in 1:maxit) {
    # E step
    p <- array(0, dim = c(N, R))
    for(i in 1:N) {
      k <- y[i]
      for(r in 1:R) {
        p[i,r] <- pi[k,r] * dmvnorm(x[i, ], mu[k, ,r], sigma)
      }
      p[i, ] <- p[i, ]/sum(p[i, ])
    }
    # M step
    for(k in 1:K) {
      ind <- which(y == k)
      for(r in 1:R) {
        pi[k,r] <- sum(p[ind,r])/length(ind)
        mu[k, ,r] <- colSums(x[ind, ]*p[ind,r])/sum(p[ind,r])
      }
    }
    tally <- 0
    for(i in 1:N) {
      z <- y[i]
      for(r in 1:R) {
        tmp <- x[i, ] - mu[z, ,r]
        tally <- tally + p[i,r] * (tmp %o% tmp)
      }
    }
    sigma <- tally/N
    # check convergence

  }
  return(list(pi = pi,
              mu = mu,
              sigma = sigma,
              class_labels = labs))
}

### predict classes
# em: object from the result of EM function above
# x: new data (as a matrix, one row = one observation)
em_predict <- function(em, x) {
  pi <- em$pi
  mu <- em$mu
  sigma <- em$sigma
  labs <- em$class_labels
  K <- dim(mu)[1]
  R <- dim(mu)[3]
  classes <- rep(NA, nrow(x))

  for(i in 1:nrow(x)) {
    probs <- rep(0, K)
    for(k in 1:K) {
      for(r in 1:R) {
        probs[k] <- probs[k] + pi[k,r] * dmvnorm(x[i, ], mean = mu[k, ,r], sigma = sigma)
      }
    }
    classes[i] <- which.max(probs)
  }
  return(factor(classes, labels = labs))
}


# SCRIPT ------------------------------------------------------------------
require(DAAG)

R <- 3 # number of subclasses
training_amount <- 0.3

X <- split_data(data, training_amount, 10)
x_train <- X$x_train
x_test <- X$x_test
class_train <- X$class_train
class_test <- X$class_test

# run MDA
fit <- EM(x_train, class_train, R)
# classify test data
fit_pred <- em_predict(fit, x_test)
# resulting confusion matrix and error rate
confusion(fit_pred, class_test)

# try PCA
x <- rbind(x_train, x_test)
x_scaled <- apply(x, 2, function(x) (x - mean(x))/sd(x))
eig <- eigen(cov(x_scaled))
eig$values/sum(eig$values)
V <- eig$vectors[ ,1:2]
x_new <- as.matrix(x_scaled %*% V)
x_train_new <- x_new[1:nrow(x_train), ]
x_test_new <- x_new[-(1:nrow(x_train)), ]

fit2 <- EM(x_train_new, class_train, R)
fit_pred2 <- em_predict(fit2, x_test_new)
confusion(fit_pred2, class_test)

plot(x_train_new, pch = 10, alpha = 0.5, col = as.numeric(class_train)+1)
points(t(fit2$mu[1, , ]), pch = 8, cex = 3, lwd = 3, col = 2)
points(t(fit2$mu[2, , ]), pch = 8, cex = 3, lwd = 3, col = 3)
plot(x_test_new, pch = 10, col = as.numeric(class_test)+1)
points(t(fit2$mu[1, , ]), pch = 8, cex = 3, lwd = 3, col = 2)
points(t(fit2$mu[2, , ]), pch = 8, cex = 3, lwd = 3, col = 3)
