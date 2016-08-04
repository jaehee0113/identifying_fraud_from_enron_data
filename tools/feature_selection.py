# We will now add some useful features and delete useless features that were found after outlier detection and via inspecting csv data.
import pandas as pd
import numpy as np
import pprint

import sys
import pickle
from sklearn.feature_selection import SelectKBest
from feature_format import featureFormat, targetFeatureSplit
from sklearn import preprocessing
sys.path.append("../tools/")

def parse_nan(data):
	if( data == 'NaN'):
		data = 0

	return data

#Total number of message sent / received with POI DIVIDED BY Total number of messages sent / received 
def get_poi_interaction_ratio(enron_dict):

	for key, value in enron_dict.iteritems():
		total_messages = parse_nan(value['from_messages']) + parse_nan(value['to_messages'])
		poi_messages = parse_nan(value['from_poi_to_this_person']) + parse_nan(value['from_this_person_to_poi'])

		if(total_messages != 0):
			poi_interaction_ratio = float(poi_messages) / total_messages
		else:
			poi_interaction_ratio = 'NaN'

		value['poi_interaction_ratio'] = poi_interaction_ratio

	return enron_dict

def get_restricted_stock_ratio(enron_dict):
    
    for key, value in enron_dict.iteritems():
    	#To prevent undefined error

    	value['total_stock_value'] = parse_nan(value['total_stock_value'])
    	value['restricted_stock'] = parse_nan(value['restricted_stock'])

    	if(value['total_stock_value'] == 0 or value['total_stock_value'] == ''):
    		restricted_stock_ratio = 'NaN'
    	else:
    		restricted_stock_ratio = float(value['restricted_stock']) / float(value['total_stock_value'])

    	#Adding new fields
        value['restricted_stock_ratio'] = restricted_stock_ratio

    return enron_dict

def get_total_received_cash(enron_dict):
	for key, value in enron_dict.iteritems():
		cash_received = float(parse_nan(value['salary'])) + float(parse_nan(value['bonus'])) + float(parse_nan(value['loan_advances'])) + float(parse_nan(value['exercised_stock_options']))
		value['cash_received'] = cash_received

def remove_outliers(enron_dict, outliers):
    for outlier in outliers:
        enron_dict.pop(outlier, 0)

if __name__ == '__main__':
	#loading the dataset
	enron_dict = pickle.load(open("../data/final_project_dataset.pkl", "r"))

	#removing outliers spotted
	outliers = ['TOTAL', 'THE TRAVEL AGENCY IN THE PARK', 'LOCKHART EUGENE E']

	remove_outliers(enron_dict, outliers)

	#existing features to be considered are restricted_stock and total_stock_value
	features = ['restricted_stock', 'total_stock_value']

	#adding new potentially useful features to our dictionary
	get_restricted_stock_ratio(enron_dict)
	get_total_received_cash(enron_dict)
	get_poi_interaction_ratio(enron_dict)

	#CODE USED FOR CONFIRMING THE INCLUSION OF NEW FEATURES
	#for key, value in enron_dict.iteritems():
	#	print str(value['restricted_stock_ratio']) + " and " + str(value['cash_received']) + " and " + str(value['poi_interaction_ratio'])

	#To determine which features are good, I have decided to use SelectKBest algorithm in Scikit-Learn

	features_list = ['poi','bonus','cash_received','deferral_payments','deferred_income','director_fees','exercised_stock_options','expenses','loan_advances','long_term_incentive','other','restricted_stock_ratio','restricted_stock','restricted_stock_deferred','salary','total_payments','total_stock_value','from_messages','to_messages','poi_interaction_ratio','from_poi_to_this_person','from_this_person_to_poi','shared_receipt_with_poi']
	data = featureFormat(enron_dict, features_list)
	labels, features = targetFeatureSplit(data)

	# To apply feature scaling we use min max scaler for consistency!
	scaler = preprocessing.MinMaxScaler()
	features = scaler.fit_transform(features)

	# we use SelectKBest to do this
	k_best = SelectKBest(k='all')
	k_best.fit(features, labels)
	scores = k_best.scores_
	scores_by_features = zip(features_list[1:], scores)
	scores_by_features.sort(key = lambda features_list: features_list[1], reverse = True)
	pprint.pprint(scores_by_features)