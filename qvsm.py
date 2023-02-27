# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 10:40:22 2023

@author: Manjula
"""
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
#There are 150 samples, with 4 attributes (same units, all numeric)
#The class distribution is balanced, with 50 samples per class.
#There's no missing data.
data = load_iris()
df = pd.DataFrame(data=data.data, columns=data.feature_names)
#Let's load the data into a DataFrame for future use and reference

X = data.data
y = data.target
#The training data features are loaded onto X, and the corresponding class labels onto y.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
print('There are {} samples in the training set and {} samples in the test set'.format(
X_train.shape[0], X_test.shape[0]))
#We perform dataset splitting to obtain training and test datasets.

#Performing feature scaling and Principal Component Analysis, and then MinMax Scaling of data that would reduce the dimensionality of features.
#This would also keep a check on overfitting/underfitting.
std_scaler = StandardScaler().fit(X_train)
X_train = std_scaler.transform(X_train)
X_test = std_scaler.transform(X_test)
    
pca = PCA(n_components=2).fit(X_train)
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)

samples = np.append(X_train, X_test, axis=0)
minmax_scaler = MinMaxScaler((-1, 1)).fit(samples)
X_train = minmax_scaler.transform(X_train)
X_test = minmax_scaler.transform(X_test)
#Creating a dictionary for catregorical labels with the dimensionally reduced data present in it.
labels = ['setosa', 'versicolor', 'virginica']
training_input = {key: (X_train[y_train == k, :]) for k, key in enumerate(labels)}
test_input = {key: (X_test[y_test == k, :]) for k, key in enumerate(labels)}

''' I WAS UNABLE TO USE AllPairs() method to train seperate classifiers on each label because of 
some compatibility issues. Invocation of the AllPairs() method requires importing QuantumInstance
from aqua\quantum_instance.py. But, it then again requires Backend and BaseBackend from qiskit.
providers, which has been removed. '''
from qiskit.circuit.library import ZZFeatureMap, PauliFeatureMap
from qiskit.primitives import Sampler
from qiskit.algorithms.state_fidelities import ComputeUncompute
from qiskit_machine_learning.kernels import FidelityQuantumKernel 
from time import time

'''There are several issues with qiskit.aqua as of now. It requires migration corrections and version controls extensively. 
QSVM and backend simulations are not compatible due to the dependency on BasicAer and BasicBackend. 
'''
#from qiskit.aqua.components.multiclass_extensions.multiclass_extension.MulticlassExtension import AllPairs
feature_map = ZZFeatureMap(feature_dimension=2, reps=2, entanglement="linear")
feature_map2 = PauliFeatureMap(feature_dimension=2, reps=2, paulis = ['Z','Y','ZZ'])
sampler = Sampler()
fidelity = ComputeUncompute(sampler=sampler)
kernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=feature_map)

# svc=SVC(kernel=kernel.evaluate)
# t0=time()
# svc.fit(X_train, y_train)
# svc_score = svc.score(X_test, y_test)
#Comes out to be 0.822222, same as QSVC score
# t1=time()


from qiskit_machine_learning.algorithms import QSVC
qsvc = QSVC(quantum_kernel=kernel)
qsvc.fit(X_train, y_train)
y_pred = qsvc.predict(X_test)
qsvc_score = qsvc.score(X_test, y_test)
print(f"QSVC classification test score using kernel1: {qsvc_score}")
print('Accuracy Score:', accuracy_score(y_test, y_pred))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
print('Classification Report:\n', classification_report(y_test, y_pred))
print("----------------------------------------------------------------------")
''' The ZZ Feature Map works quite accurately on the given dataset.'''



'''The following code and the code commented above using PauliFeatureMap may perform better or worse, depending on the number of qubits, the reps, etc. '''
# qsvm = QSVM(feature_map=feature_map2, training_dataset=training_input, test_dataset=test_input)
# qsvm.fit(X_train, y_train)
# y_pred2 = qsvm.predict(X_test)
# qsvm_score = qsvm.score(X_test, y_test)
# print(f"QSVM classification test score using kernel2: {qsvm_score}")
# print('Accuracy Score:', accuracy_score(y_test, y_pred2))
# print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred2))
# print('Classification Report:\n', classification_report(y_test, y_pred2))
# from qiskit import BasicAer
# from qiskit.circuit.library import ZZFeatureMap
# from qiskit.aqua import QuantumInstance, aqua_globals
# from qiskit.aqua.algorithms import QSVM

# from qiskit.aqua.utils.dataset_helper import get_feature_dimension

# seed = 10599
# aqua_globals.random_seed = seed
# qsvm = QSVM(feature_map, training_input, multiclass_extension=AllPairs())

# backend = BasicAer.get_backend('qasm_simulator')
# quantum_instance = QuantumInstance(backend, shots=1024, seed_simulator=seed, seed_transpiler=seed)

# result = qsvm.run(quantum_instance)
# for k,v in result.items():
#     print(f'{k} : {v}')