import numpy as np
import matplotlib.pyplot as plt

num_states = 8

def compute_true_states(initial_state, repetitions, n_shots, cdf_matrix):
    """Simulate quantum state transitions over multiple shots without readout.
    
    Simulates the true (unobserved) quantum state trajectory for each repetition,
    evolving according to the transition probability matrix defined by cdf_matrix.
    No readout is performed; this is pure Markovian evolution.
    
    Parameters
    ----------
    initial_state : int
        Initial quantum state (0 to num_states-1) for all repetitions.
    repetitions : int
        Number of independent experimental repetitions to simulate.
    n_shots : int
        Number of evolution steps per repetition.
    cdf_matrix : ndarray, shape (num_states, num_states)
        Cumulative distribution functions for state transitions.
        cdf_matrix[:, i] contains the CDF for transitions FROM state i.
    
    Returns
    -------
    true_states : ndarray, shape (repetitions, n_shots)
        True quantum state at each time step. Element [r, i] is the state
        of repetition r after shot i.
    """
    # Preallocate output
    true_states = np.zeros((repetitions, n_shots+1), dtype=int)
    true_states[:, 0] = initial_state

    # Pre-generate random values for transitions
    rand_vals = np.random.rand(repetitions, n_shots)

    for i in range(1, n_shots+1):
        prev_states = true_states[:, i - 1]  # shape (repetitions,)

        # Gather relevant CDFs for each repetition
        cdfs = cdf_matrix[:, prev_states].T  # shape (repetitions, num_states)

        # Use searchsorted manually across rows (vectorized via list comprehension)
        true_states[:, i] = np.array([
            np.searchsorted(cdfs_row, rand_val)
            for cdfs_row, rand_val in zip(cdfs, rand_vals[:, i - 1])
        ])

    return true_states[:, 1:]

def compute_true_states_with_readout_triggered_transitions(
    initial_state,
    repetitions,
    n_shots,
    true_pos,
    false_pos,
    cdf_matrix
):
    """Simulate state evolution with readout-triggered transitions.
    
    Each readout event (positive signal) triggers one additional state transition.
    False positives can occur on wrong states, and true positives occur on the
    correct state with probability true_pos.
    
    Parameters
    ----------
    initial_state : int
        Initial quantum state (0 to num_states-1) for all repetitions.
    repetitions : int
        Number of independent experimental repetitions to simulate.
    n_shots : int
        Number of measurement/evolution steps per repetition.
    true_pos : float
        Probability of readout success on the true/desired state (0 to 1).
    false_pos : float
        Probability of false readout on an incorrect state (0 to 1).
    cdf_matrix : ndarray, shape (num_states, num_states)
        Cumulative distribution functions for state transitions.
        cdf_matrix[:, i] contains the CDF for transitions FROM state i.
    
    Returns
    -------
    true_states : ndarray, shape (repetitions, n_shots)
        True quantum state at each time step after readout-triggered evolution.
    """
    num_states = cdf_matrix.shape[0]
    true_states = np.zeros((repetitions, n_shots + 1), dtype=int)
    binary_traces = np.zeros((repetitions, n_shots + 1, num_states), dtype=int)
    true_states[:, 0] = initial_state

    rand_trans = np.random.rand(repetitions, n_shots)
    rand_readout_event = np.random.rand(repetitions, n_shots, num_states)
    rand_readout_choice = np.random.rand(repetitions, n_shots)

    for i in range(1, n_shots + 1):
        current_states = true_states[:, i - 1]

        # Build readout probability vector for each repetition
        readout_probs = np.full((repetitions, num_states), false_pos)
        readout_probs[np.arange(repetitions), current_states] = true_pos

        # Determine if a readout occurred
        binary_traces[:, i, :] = (rand_readout_event[:, i - 1, :] < readout_probs).astype(int)

        # Default: no transition → same state
        num_readouts = binary_traces[:, i, :].sum(axis=1)  

        next_states = current_states.copy()

        for _ in range(num_readouts.max()):
            indices = np.where(num_readouts > 0)[0]
            if len(indices) == 0:
                break
            prev = next_states[indices]
            cdfs = cdf_matrix[:, prev].T
            new_rand = np.random.rand(len(indices))
            next_states[indices] = np.array([
                np.searchsorted(cdf_row, rand)
                for cdf_row, rand in zip(cdfs, new_rand)
            ])
            num_readouts[indices] -= 1  # decrement transitions left

        true_states[:, i] = next_states

    return true_states[:, 1:] #, binary_traces[:, 1:, :]

# Vectorized detection simulation
# Generate a random array for all traces
def include_readout_errors(true_states, true_pos, false_pos):
    """Add readout measurement errors to ideal quantum states.
    
    Simulates imperfect readout by introducing false positives (wrong state
    detected) and false negatives (correct state not detected). The readout
    outcome for each state is probabilistic based on whether that state is
    the true state or not.
    
    Parameters
    ----------
    true_states : ndarray, shape (repetitions, n_shots)
        True quantum states before readout (integers 0 to num_states-1).
    true_pos : float
        Probability of correctly detecting the true state (0 to 1).
    false_pos : float
        Probability of false positive on any non-true state (0 to 1).
    
    Returns
    -------
    raw_traces : ndarray, shape (repetitions, n_shots, num_states)
        Binary readout traces. raw_traces[r, i, s] = 1 if state s is detected
        at shot i of repetition r, 0 otherwise.
    """
    raw_traces = np.zeros([true_states.shape[0], true_states.shape[1], num_states])
    rand_matrix = np.random.rand(true_states.shape[0], true_states.shape[1], num_states)

    # Broadcast ground truth state mask
    for s in range(num_states):
        is_true = (true_states == s)
        raw_traces[:, :, s] = np.where(
            is_true,
            rand_matrix[:, :, s] < true_pos,  # true state: high chance of 1
            rand_matrix[:, :, s] < false_pos  # false state: low chance of 1
        )
    return raw_traces

def simulate_active_feedback_readout(
    initial_state,
    repetitions,
    n_errorspace,
    threshold,
    true_pos,
    false_pos,
    cdf_matrix
):
    """Simulate adaptive readout protocol with active feedback control.
    
    Implements an error-space adaptive readout: an initial readout selects a
    state hypothesis. The system then evolves and is re-measured in an
    'error space' (all states except the hypothesis). If too many errors are
    detected, the cycle repeats; otherwise the hypothesis is confirmed.
    
    Parameters
    ----------
    initial_state : int
        Initial quantum state (0 to num_states-1) for all repetitions.
    repetitions : int
        Number of independent experimental repetitions to simulate.
    n_errorspace : int
        Number of error-space measurements to perform per protocol iteration.
    threshold : int
        Maximum number of errors allowed to accept the state hypothesis.
    true_pos : float
        Probability of correctly detecting a state (0 to 1).
    false_pos : float
        Probability of false positive on a wrong state (0 to 1).
    cdf_matrix : ndarray, shape (num_states, num_states)
        Cumulative distribution functions for state transitions.
        cdf_matrix[:, i] contains the CDF for transitions FROM state i.
    
    Returns
    -------
    confirmed_states : ndarray, shape (repetitions,)
        The state hypothesis confirmed at the end of the protocol.
    states_after_protocol : ndarray, shape (repetitions,)
        The true quantum state after all protocol evolution steps complete.
    num_transitions : ndarray, shape (repetitions,)
        Number of complete protocol loops before confirmation for each repetition.
    raw_traces : ndarray, shape (repetitions, 1, num_states)
        Initial readout traces (first positive detection per repetition).
    """
    transition_cdf = cdf_matrix
    num_states = transition_cdf.shape[0]
    confirmed_states = np.zeros(repetitions, dtype=int)
    states_after_protocol = np.zeros(repetitions, dtype=int)
    num_transitions = np.zeros(repetitions, dtype=int)
    raw_traces = np.zeros((repetitions, 1, num_states), dtype=int)
    error_traces = np.zeros((repetitions, n_errorspace))

    for r in range(repetitions):
        current_state = initial_state
        transitions = 1

        while True:
            # Step 1: Initial readout attempt
            readout_probs = np.full(num_states, false_pos)
            readout_probs[current_state] = true_pos
            readout_event = np.random.rand(num_states) < readout_probs
            raw_traces[r, 0, :] = readout_event

            if not np.any(readout_event):
                continue  # No positive → retry

            # Find first positive readout event
            first_positive_index = np.flatnonzero(readout_event)[0]

            # All other states except the first positive
            states_to_check = np.setdiff1d(np.arange(num_states), [first_positive_index])

            # go through one iteration of transition matrix
            probs = transition_cdf[:, current_state]
            rand = np.random.rand()
            current_state = np.searchsorted(probs, rand)

            # Step 2: Error check on these states
            error_check = np.zeros(n_errorspace, dtype=int)
            for i in range(n_errorspace):
                if current_state in states_to_check:
                    error_check[i] = np.random.rand() < true_pos
                else:
                    error_check[i] = np.random.rand() < false_pos

            error_traces[r, :] = error_check

            # Count positives in error-space
            positives_in_error_space = error_check.sum()

            for i in range(positives_in_error_space):
                if i < threshold:
                    probs = transition_cdf[:, current_state]
                    rand = np.random.rand()
                    current_state = np.searchsorted(probs, rand)

            if positives_in_error_space < threshold: 
                # Accept the state read out in step 1
                confirmed_states[r] = first_positive_index
                states_after_protocol[r] = current_state
                num_transitions[r] = transitions
                break
            else:
                # Evolve the system
                transitions += 1

    return confirmed_states, states_after_protocol, num_transitions, raw_traces