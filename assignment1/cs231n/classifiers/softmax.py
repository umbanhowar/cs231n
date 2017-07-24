import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    losses = []
    for i in xrange(X.shape[0]):
        ex = X[i, :]
        scores = ex.dot(W)
        # to prevent numerical instability
        scores_adj = scores - np.max(scores)
        loss_i = -np.log(np.exp(scores_adj[y[i]]) / np.sum(np.exp(scores_adj)))
        losses.append(loss_i)
        for j in xrange(W.shape[1]):
            if j == y[i]:
                dW[:, j] -= ex
            dW[:, j] += (1/np.sum(np.exp(scores))) * ex * np.exp(scores[j])

    loss += np.mean(losses)
    # regularization
    loss += 0.5 * reg * np.sum(W * W)

    dW /= X.shape[0]
    dW += reg * W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W) # N x C
  scores_adj = scores - np.max(scores, axis=1)[:, None] # N x C
  losses = -np.log(np.exp(scores_adj[np.arange(scores_adj.shape[0]), y]) / np.sum(np.exp(scores_adj), axis=1)) # N x 1
  loss = np.mean(losses) + 0.5 * reg * np.sum(W * W) # 1 x 1

  a = np.ones_like(scores)
  b = a / np.sum(np.exp(scores), axis=1)[:,None]
  c = b * np.exp(scores)
  c[np.arange(b.shape[0]), y] -= 1
  dW = X.T.dot(c)

  dW /= X.shape[0]
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
