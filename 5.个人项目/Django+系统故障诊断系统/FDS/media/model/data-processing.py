import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import nan_euclidean_distances
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

df = pd.read_csv('./datashet/train_10000.csv')
X = df.iloc[:, :-1]
X_list = X.values.tolist()

imputer = KNNImputer(n_neighbors=1)
result = imputer.fit_transform(X_list)
y = pd.DataFrame(result)
df.iloc[:,:-1]=y
df.to_csv('./datashet/output.csv')

