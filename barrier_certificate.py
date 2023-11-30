import numpy as np
from cvxopt.solvers import qp, options
from cvxopt import matrix, sparse
from scipy.special import comb

def barrier_certificate(dxi, x, barrier_gain=100, safety_radius=0.05, magnitude_limit=0.25):
    """
    barrier_gain: double (controls how quickly agents can approach each other.  lower = slower)
    safety_radius: double (how far apart the agents will stay)
    magnitude_limit: how fast the robot can move linearly.

    -> function (the barrier certificate function)
    source https://github.com/robotarium/robotarium_python_simulator/blob/master/rps/utilities/barrier_certificates.py
    """

    #Check user input types
    assert isinstance(dxi, np.ndarray), "In the function created by the create_single_integrator_barrier_certificate function, the single-integrator robot velocity command (dxi) must be a numpy array. Recieved type %r." % type(dxi).__name__
    assert isinstance(x, np.ndarray), "In the function created by the create_single_integrator_barrier_certificate function, the robot states (x) must be a numpy array. Recieved type %r." % type(x).__name__

    #Check user input ranges/sizes
    assert x.shape[0] == 2, "In the function created by the create_single_integrator_barrier_certificate function, the dimension of the single integrator robot states (x) must be 2 ([x;y]). Recieved dimension %r." % x.shape[0]
    assert dxi.shape[0] == 2, "In the function created by the create_single_integrator_barrier_certificate function, the dimension of the robot single integrator velocity command (dxi) must be 2 ([x_dot;y_dot]). Recieved dimension %r." % dxi.shape[0]
    assert x.shape[1] == dxi.shape[1], "In the function created by the create_single_integrator_barrier_certificate function, the number of robot states (x) must be equal to the number of robot single integrator velocity commands (dxi). Recieved a current robot pose input array (x) of size %r x %r and single integrator velocity array (dxi) of size %r x %r." % (x.shape[0], x.shape[1], dxi.shape[0], dxi.shape[1])

    
    # Initialize some variables for computational savings
    N = dxi.shape[1]
    num_constraints = int(comb(N, 2))
    A = np.zeros((num_constraints, 2*N))
    b = np.zeros(num_constraints)
    H = sparse(matrix(2*np.identity(2*N)))

    count = 0
    for i in range(N-1):
        for j in range(i+1, N):
            error = x[:, i] - x[:, j]
            h = (error[0]*error[0] + error[1]*error[1]) - np.power(safety_radius, 2)

            A[count, (2*i, (2*i+1))] = -2*error
            A[count, (2*j, (2*j+1))] = 2*error
            b[count] = barrier_gain*np.power(h, 3)

            count += 1

    # Threshold control inputs before QP
    norms = np.linalg.norm(dxi, 2, 0)
    idxs_to_normalize = (norms > magnitude_limit)
    dxi[:, idxs_to_normalize] *= magnitude_limit/norms[idxs_to_normalize]

    f = -2*np.reshape(dxi, 2*N, order='F')
    result = qp(H, matrix(f), matrix(A), matrix(b), options={"show_progress": False})['x']

    return np.reshape(result, (2, -1), order='F')
