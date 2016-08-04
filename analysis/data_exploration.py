# Explores data from final_project_dataset.pkl

import sys
import pickle
import pandas as pd
sys.path.append("../tools/")

def set_panda_dataframe(enron_dict):
	df = pd.DataFrame()
	for key, value in enron_dict.iteritems():
		#the value would become a row for the dataframe
		row = value.copy()
		#We would have the name of each person as 'name' column
		row['name'] = key
		#append to dataframe
		df = df.append(row, ignore_index = True)

	return df


def count_features_nanval(enron_dict):
	#as features are contained as a value of each key in enron_dict,
	#I will first create a dictionary with a key that is the name of each feature
	feature_dict = {}

	#Just getting first value from enron_dict
	for key in enron_dict.itervalues().next():
		feature_dict[key] = 0

	#We now iterate to count the missing value (i.e. NaN for each feature)
	for key, value in enron_dict.iteritems():
		for feature in feature_dict:
			if value[feature] == 'NaN':
				feature_dict[feature] += 1;

	return feature_dict

#def get_feature_list():

#Invoked when running this file (this should generate data exploration report!)
if __name__ == '__main__':
	#loading the dataset
	enron_dict = pickle.load(open("../data/final_project_dataset.pkl", "r"))

	#Getting the total number of people (as keys are person's name)
	
	#I want to double check that we are excluding duplicates.
	unique_enron_people = set();
	for key in enron_dict:
		unique_enron_people.add(key)

	unique_pois = set();
	for key, value in enron_dict.iteritems():
		if(value['poi'] == True):
			unique_pois.add(key)


	print "We have " + str(len(unique_enron_people)) + " people in this dataset."
	print "We have " + str(len(unique_pois)) + " pois in this dataset."


	features_with_missing_count = count_features_nanval(enron_dict)

	print "We have " + str(len(features_with_missing_count)) + " features for each person in this dataset."

	unique_features = set();
	for key, value in features_with_missing_count.iteritems():
		unique_features.add(key)
		print "Feature " + str(key) + " has " + str(value) + " missing values."

	print "In fact, we have the following features: "
	counter = 1
	for value in unique_features:
		print str(counter) + '. ' + value
		counter += 1

	#For effective outlier analysis, I would like to convert pkl file into csv file so that I can use
	#techniques I used for titantic data analysis to analyze the data!

	df = set_panda_dataframe(enron_dict)

	file_out = "../data/enron_data.csv"
	df.to_csv(file_out)

