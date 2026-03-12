import parameters as p

import pickle
import numpy as np


def extract_distance_matrix():
    """
    Load the precomputed correlation tensor and convert it into a
    z-scored distance matrix, keeping memory usage under control.

    Expected pkl shape : (T, 463, 463)  — 463 stocks, 6 will be removed.
    Output shape       : (T, 457, 457)  — matches model input size.
    """
    with open(
        "../Quasi_Differentiation_High_Temporal_Resolution_Cross_Correlations/Codes/"
        "Extract distance matrix (2017-2022) from pkl file/IQDw{}.pkl".format(p.w),
        "rb",
    ) as pkl_file:
        # Load as float32 immediately to halve memory vs float64
        data1 = pickle.load(pkl_file).astype(np.float32)

    # Clamp correlations to [−1, 1] before distance conversion.
    # Upper clamp prevents imaginary sqrt; lower clamp prevents distances > 2.
    data1 = np.clip(data1, -1.0, 1.0)

    # Convert correlation → distance:  d = sqrt(2 * (1 − corr))
    # Result is in [0, 2]; 0 = identical, 2 = perfectly anti-correlated.
    distance_matrix = (2 * (1 - data1)) ** np.float32(0.5)

    # We no longer need the original correlation tensor
    del data1

    # Assert expected raw stock count before deletion (463 − 6 = 457)
    assert distance_matrix.shape[1] == 457 + 6, (
        f"Unexpected stock count {distance_matrix.shape[1]} in pkl; "
        f"expected 463 (457 stocks + 6 to be removed)."
    )

    # Remove nan-prone rows/cols — same indices on both stock axes.
    # Using a single shared list ensures rows and cols are always in sync.
    bad_indices = [6, 111, 128, 169, 170, 225]
    distance_matrix = np.delete(distance_matrix, bad_indices, axis=1)
    distance_matrix = np.delete(distance_matrix, bad_indices, axis=2)

    # Assert final shape matches model expectation
    assert distance_matrix.shape[1] == 457 and distance_matrix.shape[2] == 457, (
        f"Distance matrix stock dimensions {distance_matrix.shape[1:]} != (457, 457) "
        f"after removing bad indices."
    )

    # Global z-score standardisation (all T days and all N×N entries together).
    # NOTE: This uses the global mean/std across all days — confirm with supervisor
    # that global normalisation is intended rather than per-day z-scoring.
    mean = np.mean(distance_matrix, dtype=np.float32)
    std  = np.std(distance_matrix,  dtype=np.float32)
    distance_matrix = (distance_matrix - mean) / std

    return distance_matrix
