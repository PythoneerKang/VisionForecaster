import parameters as p

import pickle
import numpy as np
# import datetime as dt

def extract_distance_matrix():
    pkl_file = open('../Quasi_Differentiation_High_Temporal_Resolution_Cross_Correlations/Codes/Extract distance matrix (2017-2022) from pkl file/IQDw{}.pkl'.format(p.w), 'rb')
    data1 = pickle.load(pkl_file)
    # date_array = np.loadtxt('stock_time.txt', delimiter='\n')

    # with open('stock_time.txt', 'r') as file:
    #     lines = file.readlines()
    #     # Each element in 'lines' will include the newline character (\n)
    #     # To remove newline characters:
    #     datearray = [line.strip().replace("/", " ") for line in lines]
    #     print(datearray)
    #     datearray = [dt.datetime.strptime(i, "%Y %m %d") for i in datearray]
    #     datearray = [dt.datetime.strftime(i, "%Y%m%d") for i in datearray]
    #
    # file.close()

    # totaltime = np.arange(0, len(datearray))
    # distance_matrix = np.empty(shape=(len(datearray), len(data1[0, :, :]), len(data1[0, :, :])))
    #
    # for t in totaltime:
    #     for i in range(len(data1[t, :, :])):
    #         for j in range(len(data1[t, :, :])):
    #             if data1[t, i, j] > 1: data1[t, i, j] = 1
    #             distance_matrix[t][i][j] = (2 * (1 - data1[t, i, j])) ** 0.5
    data1[data1>1]=1
    distance_matrix = (2 * (1- data1)) ** 0.5

    #Clean up distance matrix due to nan values
    rows_to_delete = [6,111,128,169,170,225]
    arr_rows_deleted = np.delete(distance_matrix, rows_to_delete, axis=1)

    cols_to_delete = [6,111,128,169,170,225]
    distance_matrix = np.delete(arr_rows_deleted, cols_to_delete, axis=2)

    #Z-score standardization
    mean = np.mean(distance_matrix)
    std = np.std(distance_matrix)

    # Apply z-score formula
    distance_matrix = (distance_matrix - mean) / std

    #output = np.argwhere(np.isnan(distance_matrix))
    #np.savetxt('loc_of_nan.txt', output, delimiter=',', fmt='%d')
    # np.savetxt(datearray[t] + '_distanceMatrix.txt', distance_matrix, delimiter=',')
    return distance_matrix