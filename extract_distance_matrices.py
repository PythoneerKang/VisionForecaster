import parameters as p

import pickle
import numpy as np
# import datetime as dt


def extract_distance_matrix():
    """
    Load the precomputed correlation tensor and convert it into a
    z-scored distance matrix, keeping memory usage under control.
    """
    with open(
        "../Quasi_Differentiation_High_Temporal_Resolution_Cross_Correlations/Codes/"
        "Extract distance matrix (2017-2022) from pkl file/IQDw{}.pkl".format(p.w),
        "rb",
    ) as pkl_file:
        # Load as float32 immediately to halve memory vs float64
        data1 = pickle.load(pkl_file).astype(np.float32)

    # Clamp correlations to [−∞, 1] then convert to distances
    data1[data1 > 1] = 1
    distance_matrix = (2 * (1 - data1)) ** np.float32(0.5)

    # We no longer need the original correlation tensor
    del data1

    # Clean up distance matrix due to nan-prone rows/cols
    rows_to_delete = [6, 111, 128, 169, 170, 225]
    arr_rows_deleted = np.delete(distance_matrix, rows_to_delete, axis=1)

    cols_to_delete = [6, 111, 128, 169, 170, 225]
    distance_matrix = np.delete(arr_rows_deleted, cols_to_delete, axis=2)

    # Z-score standardisation (still float32)
    mean = np.mean(distance_matrix, dtype=np.float32)
    std = np.std(distance_matrix, dtype=np.float32)
    distance_matrix = (distance_matrix - mean) / std

    #output = np.argwhere(np.isnan(distance_matrix))
    #np.savetxt('loc_of_nan.txt', output, delimiter=',', fmt='%d')
    # np.savetxt(datearray[t] + '_distanceMatrix.txt', distance_matrix, delimiter=',')
    return distance_matrix