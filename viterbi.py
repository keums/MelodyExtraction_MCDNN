# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 21:47:18 2016

@author: keums
"""
import numpy as np

def viterbi(posterior, transition_matrix=None, prior=None, penalty=0,
            scaled=True):
    """Find the optimal Viterbi path through a posteriorgram.
    Ported closely from Tae Min Cho's MATLAB implementation.
    Parameters
    ----------
    posterior: np.ndarray, shape=(num_obs, num_states)
        Matrix of observations (events, time steps, etc) by the number of
        states (classes, categories, etc), e.g.
          posterior[t, i] = Pr(y(t) | Q(t) = i)
    transition_matrix: np.ndarray, shape=(num_states, num_states)
        Transition matrix for the viterbi algorithm. For clarity, each row
        corresponds to the probability of transitioning to the next state, e.g.
          transition_matrix[i, j] = Pr(Q(t + 1) = j | Q(t) = i)
    prior: np.ndarray, default=None (uniform)
        Probability distribution over the states, e.g.
          prior[i] = Pr(Q(0) = i)
    penalty: scalar, default=0
        Scalar penalty to down-weight off-diagonal states.
    scaled : bool, default=True
        Scale transition probabilities between steps in the algorithm.
        Note: Hard-coded to True in TMC's implementation; it's probably a bad
        idea to change this.
    Returns
    -------
    path: np.ndarray, shape=(num_obs,)
        Optimal state indices through the posterior.
    """

    # Infer dimensions.
    num_obs, num_states = posterior.shape

    # Define the scaling function
    scaler = normalize if scaled else lambda x: x
    # Normalize the posterior.
    posterior = normalize(posterior, axis=1)

    if transition_matrix is None:
        transition_matrix = np.ones([num_states]*2)

    transition_matrix = normalize(transition_matrix, axis=1)

    # Apply the off-axis penalty.
    offset = np.ones([num_states]*2, dtype=float)
    offset -= np.eye(num_states, dtype=np.float)
    penalty = offset * np.exp(penalty) + np.eye(num_states, dtype=np.float)
    transition_matrix = penalty * transition_matrix

    # Create a uniform prior if one isn't provided.
    prior = np.ones(num_states) / float(num_states) if prior is None else prior

    # Algorithm initialization
    delta = np.zeros_like(posterior)
    psi = np.zeros_like(posterior)
    path = np.zeros(num_obs, dtype=int)

    idx = 0
    delta[idx, :] = scaler(prior * posterior[idx, :])

    for idx in range(1, num_obs):
        res = delta[idx - 1, :].reshape(1, num_states) * transition_matrix
        delta[idx, :] = scaler(np.max(res, axis=1) * posterior[idx, :])
        psi[idx, :] = np.argmax(res, axis=1)

    path[-1] = np.argmax(delta[-1, :])
    for idx in range(num_obs - 2, -1, -1):
        path[idx] = psi[idx + 1, path[idx + 1]]
    return path


def normalize(x, axis=None):
    """Normalize the values of an ndarray to sum to 1 along the given axis.
    Parameters
    ----------
    x : np.ndarray
        Input multidimensional array to normalize.
    axis : int, default=None
        Axis to normalize along, otherwise performed over the full array.
    Returns
    -------
    z : np.ndarray, shape=x.shape
        Normalized array.
    """
    if not axis is None:
        shape = list(x.shape)
        shape[axis] = 1
        scalar = x.astype(float).sum(axis=axis).reshape(shape)
        scalar[scalar == 0] = 1.0
    else:
        scalar = x.sum()
        scalar = 1 if scalar == 0 else scalar
    return x / scalar


