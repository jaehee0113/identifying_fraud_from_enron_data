#!/usr/bin/python

import sys
import pickle
import pandas as pd
import numpy as np

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from feature_selection import parse_nan, get_poi_interaction_ratio, get_restricted_stock_ratio, get_total_received_cash, remove_outliers
from sklearn import preprocessing


# k-fold cross-validation, which are often used in practice
def evaluate_kfold_cross_validation(clf, features, labels):
	print clf
	from sklearn.cross_validation import KFold
	from sklearn.metrics import accuracy_score, precision_score, recall_score
	kf=KFold(len(labels), n_folds = 3)
	for train_indices, test_indices in kf:
		features_train = [features[ii] for ii in train_indices]
		features_test = [features[ii] for ii in test_indices]
		labels_train = [labels[ii] for ii in train_indices]
		labels_test = [labels[ii] for ii in test_indices]
	clf.fit(features_train, labels_train)
	predictions = clf.predict(features_test)
	accuracy = accuracy_score(labels_test, predictions)
	precision = precision_score(labels_test, predictions)
	recall = recall_score(labels_test, predictions)
	print("accuracy: %f" % (accuracy))
	print("precision: %f" % (precision))
	print("recall: %f" % (recall))

# Tasks

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi"self.
features_list = ['poi', 'cash_received', 'poi_interaction_ratio']

# This is the final classification algorithm for my dataset.
chosen_clf = 'dtc'

### Load the dictionary containing the dataset
with open("../data/final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
outliers = ['TOTAL', 'THE TRAVEL AGENCY IN THE PARK', 'LOCKHART EUGENE E']
remove_outliers(data_dict, outliers)

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
get_restricted_stock_ratio(data_dict)
get_total_received_cash(data_dict)
get_poi_interaction_ratio(data_dict)

my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# To apply feature scaling we use min max scaler for consistency!
scaler = preprocessing.MinMaxScaler()
features = scaler.fit_transform(features)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

if(chosen_clf == 'kmeans'):
	from sklearn.cluster import KMeans
	clf = KMeans(n_clusters = 2, max_iter = 800)

if(chosen_clf == 'gnb'):
	from sklearn.naive_bayes import GaussianNB
	clf = GaussianNB()

if(chosen_clf == 'knc'):
	from sklearn.neighbors import KNeighborsClassifier
	clf = KNeighborsClassifier(algorithm = 'auto',
		leaf_size = 30,
		metric = 'minkowski',
		n_neighbors = 5,
		p = 2,
		weights = 'uniform')

if(chosen_clf == 'dtc'):
	from sklearn.tree import DecisionTreeClassifier
	clf = DecisionTreeClassifier(criterion = 'gini',
		max_depth = 6,
		max_features = None,
		min_samples_leaf = 2, 
		min_samples_split = 7, 
		splitter = 'best',
		random_state = 48)


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Other than stratified shuffle split cross validation, we will use kfold cross validation as well (to try out various validation techniques)
evaluate_kfold_cross_validation(clf, features, labels)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)