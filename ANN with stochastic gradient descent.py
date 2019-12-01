import numpy as np
import pandas as pd

dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y= dataset.iloc[:, 13].values


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_X_1 = LabelEncoder()
X[:, 1] = labelEncoder_X_1.fit_transform(X[:, 1])
labelEncoder_X_2 = LabelEncoder()
X[:, 2] = labelEncoder_X_2.fit_transform(X[:, 2])
df_X=pd.DataFrame(X)
oneHotEncoder = OneHotEncoder(categorical_features = [1])
X=oneHotEncoder.fit_transform(X).toarray()
X = X[:, 1:]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from tensorflow.python.framework import ops
ops.reset_default_graph()

classifier = SequentiIal()

classifier.add(Dense(6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))

classifier.add(Dense(6, kernel_initializer = 'uniform', activation = 'relu'))

classifier.add(Dense(1, kernel_initializer = 'uniform', activation = 'sigmoid'))

classifier.compile()

df_X1=pd.DataFrame(X)