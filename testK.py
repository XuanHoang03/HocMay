import numpy as np
import pandas as pd
import matplotlib.pyplot as plt #datavisualization
from sklearn.datasets import make_blobs #synthetic dataset
from sklearn.neighbors import KNeighborsClassifier #kNN classifier
from sklearn.model_selection import train_test_split #train and test sets 
from sklearn.model_selection import GridSearchCV
from scipy.io import arff

data = arff.loadarff('diabetes.arff')
#file csv convert sang arff
df = pd.DataFrame(data[0])
X = df.drop('Outcome', axis=1)
y = df['Outcome']
knn_grid = GridSearchCV(estimator = KNeighborsClassifier(), param_grid={'n_neighbors': np.arange(1,10)}, cv=10)
knn_grid.fit(X,y)

print(knn_grid.best_params_)