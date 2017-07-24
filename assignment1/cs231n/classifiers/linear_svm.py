import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero
  sample_gradient = np.zeros(W.shape)
  reg_gradient = reg * W
  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  # iterate through the training examples
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    # Iterate over the different classes
    for j in xrange(num_classes):
      if j == y[i]:
        continue

      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        # We know here that j != y_i
        sample_gradient[:, j] += X[i]
        sample_gradient[:, y[i]] -= X[i]
        loss += margin
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  sample_gradient = sample_gradient / float(X.shape[0])
  dW = sample_gradient + reg_gradient
  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  scores = X.dot(W)
  correct_scores = scores[np.arange(scores.shape[0]),y]
  diffs = scores - correct_scores[:, None]
  ones_mask = np.ones_like(diffs)
  ones_mask[np.arange(ones_mask.shape[0]), y] = 0
  hinged = np.maximum(diffs + ones_mask, 0)
  loss_by_ex = np.sum(hinged, axis=1)
  loss = np.sum(loss_by_ex, axis=0) / float(X.shape[0])

  # add regularization
  loss += 0.5 * reg * np.sum(W * W)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
  # Hinged holds a 1 if that margin is positive, 0 else
  hinged[hinged > 0] = 1
  # set the correct score column in hinged to be minus the number of margins above zero
  # This is because when we take the dot product, this location will be multiplied
  # by the corresponding example which in the loss sum is subtracted however many
  # times the margin is above zero.
  hinged[np.arange(hinged.shape[0]), y] = -1 * np.sum(hinged, axis=1)
  dW = X.T.dot(hinged)
  # divide by the number of training examples
  dW = dW / num_train
  # Add regularization
  dW += reg * W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
