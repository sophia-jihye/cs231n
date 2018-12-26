import numpy as np
from random import shuffle

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
  num_train = X.shape[0]
  num_classes = W.shape[1]
    
  for i in range(num_train):
    scores = X[i].dot(W)
    scores -= np.max(scores) # f = np.array([123, 456, 789]) -> f becomes [-666, -333, 0]
    p = np.exp(scores) / np.sum(np.exp(scores)) # safe to do, gives the correct answer
    loss += -np.log(p[y[i]])   # Even if it predicted correctly with high probability but smaller than 1, it is loss.

    for j in range(num_classes):
      dW[:, j] += (p[j] - (j==y[i])) * X[i]

  loss = loss / num_train + reg * np.sum(W * W)
  dW = dW / num_train + reg * W 
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
  num_train = X.shape[0]
  num_classes = W.shape[1]
    
  scores = X.dot(W)  
  scores -= np.max(scores, axis=1).reshape(-1,1) # f = np.array([123, 456, 789]) -> f becomes [-666, -333, 0]
  p = np.exp(scores) / np.sum(np.exp(scores), axis=1).reshape(-1,1) # safe to do, gives the correct answer
  loss = -np.sum(np.log(p[range(num_train), y]))  
  
  p_mask = p.copy()
  p_mask[range(num_train), y] -= 1
  dW = np.dot(np.transpose(X), p_mask)
    
  loss = loss / num_train + reg * np.sum(W * W)
  dW = dW / num_train + reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

