import streamlit as st 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

#app ki heading
st.write("""
# Explore different ML models and datasets
which one is goooood?
""")

# dataset k nam ak box may daal k sidebar lga do
classifier_name = st.sidebar.selectbox(
    'Select Dataset',
    ('Iris', 'Breast Cancer', 'Wine')
)
# or isi k nichy classifier k nam ak box may dal do
classifier_name = st.sidebar.selectbox(
    'select classifier',
    ('KNN', 'SVM', 'Random Forest')
)

def get_dataset(dataset_name):
    data = None
    if dataset_name == "Iris":
        data = datasets.load_iris()
    elif dataset_name == "Wine":
        data = datasets.load_wine()
    else:
        data = datasets.load_breast_cancer()
    x = data.data
    y = data.target
    return x, y

X, y = get_dataset('Iris')
X, y = get_dataset('Wine')
X, y = get_dataset('breast_cancer')

#make shape of dataset
st.write('Shape of datset:', X.shape)
st.write('number of classes:', len(np.unique(y)))

#using various classifier parameters
def add_parameter_ui(classifier_name):
    params = dict()
    if classifier_name == 'SVM':
        C = st.sidebar.slider('C', 0.01, 10.0)
        params['C'] = C
    elif classifier_name == 'KNN':
        K = st.sidebar.slider('K', 1, 15)
        params['K'] = K
    else:
        max_depth = st.sidebar.slider('max_depth', 2, 15)
        params['max_depth'] = max_depth
        n_estimators = st.sidebar.slider('n_estimators', 1, 100)
        params['n_estimators'] = n_estimators
    return params
#ab is function ko bulana h
params = add_parameter_ui(classifier_name)

#ab hum classifier ka base bnaien gy
def get_classifier(classifier_name, params):
    clf = None
    if classifier_name == 'SVM':
        clf = SVC(C=params['C'])
    elif classifier_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=params['K'])
    else:
        clf = clf = RandomForestClassifier(n_estimators=params['n_estimators'],
            max_depth=params['max_depth'], random_state=1234)
    return clf

#call the function
clf = get_classifier(classifier_name, params)

#split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

#train the data
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

#accuracy score
acc = accuracy_score(y_test, y_pred)
st.write(f'Classifier = {classifier_name}')
st.write(f'Accuracy = {acc}')


#add plot
pca = PCA(2)
X_projected = pca.fit_transform(X)

x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

fig = plt.figure()
plt.scatter(x1, x2, c=y, alpha=0.8, cmap='viridis')

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar()

#plt show
st.pyplot(fig)