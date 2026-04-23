import numpy as np
import pickle

with open("../Quasi_Differentiation_High_Temporal_Resolution_Cross_Correlations/Codes/Extract distance matrix (2017-2022) from pkl file/IQDw180.pkl", "rb") as f:
    raw = pickle.load(f).astype(np.float32)

data = np.clip(raw, -1.0, 1.0)
dist = np.sqrt(2.0 * (1.0 - data)).astype(np.float32)
adj = (dist <= 0.6).astype(np.float32)
adj[:, np.arange(463), np.arange(463)] = 0.0
bad_indices = [6, 111, 128, 169, 170, 225]
adj = np.delete(adj, bad_indices, axis=1)
adj = np.delete(adj, bad_indices, axis=2)

triu_i, triu_j = np.triu_indices(457, k=1)
density = adj[:, triu_i, triu_j].mean(axis=1)

# The TRUE zero-overlap lag
k = 73
target = density[k:]
feature = density[:-k]
corr_73 = np.corrcoef(target, feature)[0, 1]
print(f"Correlation at 72 weeks (1-day overlap): 0.5051")
print(f"Correlation at 73 weeks (zero overlap): {corr_73:.4f}")