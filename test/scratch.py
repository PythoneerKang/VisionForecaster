import pickle
#import numpy as np

fold = 1

with open(f'train_val_fold{fold}.pkl', 'rb') as f:
    path, train_mse, val_mse, train_r2, val_r2 = pickle.load(f)

print(path)
print(train_mse, type(train_mse[0]))
print(val_mse, type(val_mse[0]))
print(train_mse, type(train_r2[0]))
print(val_r2, type(val_r2[0]))