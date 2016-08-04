#Outlier detection using csv generated during the data exploration process!
import pandas as pd
import numpy as np
import pprint
import matplotlib.pyplot as plt
import matplotlib.colors as clrs
from matplotlib import cm

filename = 'enron_data.csv'

enron_data = pd.read_csv('../data/' + filename)

#list of features [THE feature THE TRAVEL AGENCY IN THE PARK and LOCKHART EUGENE E were deemed to be useless]
features = ['bonus', 'deferral_payments', 'deferred_income', 
			'director_fees', 'exercised_stock_options', 
			'expenses', 'loan_advances', 'long_term_incentive', 
			'restricted_stock', 'restricted_stock_deferred', 
			'salary', 'total_payments', 'total_stock_value']

def parse_nan(data):
	if(np.isnan(data)):
		return 0
	else:
		return data

for feature in features:
	enron_data[feature] = enron_data[feature].apply(parse_nan)

# By drawing a graph, we would like to know whether outliers exist first.
plt.figure(1)

for i in range(1, len(features) + 1):	
	plt.subplot(5,3,i)
	plt.hist(enron_data[features[i - 1]], normed=1)
	plt.title(features[i - 1])

plt.show()


#From this graph, we can clearly see that outliers exist
#We can analyze bonus feature to extract the top five data and can print out the name
enron_data_by_bonus = enron_data.sort('bonus', ascending = False);
for i in range(5):
	print enron_data_by_bonus.iloc[i]['name']

#From here we can find out some unusual data! the name is TOTAL! It seems that this is a non-useful outlier so we will drop this row from our data
necessary_enron_data = enron_data[enron_data.name != 'TOTAL']
enron_data_by_bonus = necessary_enron_data.sort('bonus', ascending = False);
for i in range(5):
	print enron_data_by_bonus.iloc[i]['name']
#After checking we can see that TOTAL row is gone!


#After deleting the TOTAL row we can clearly see that the graph is more legible and makes sense.

'''
plt.figure(1)

for i in range(1, len(features) + 1):	
	plt.subplot(5,3,i)
	plt.hist(necessary_enron_data[features[i - 1]], normed=1)
	plt.title(features[i - 1])

plt.show()
'''

#Now I will create another csv, which is the dataset that discarded TOTAL row
file_out = "../data/processed_enron_data.csv"
necessary_enron_data.to_csv(file_out)