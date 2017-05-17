import pandas as pd
import numpy as np
np.random.seed(1)

dc_listings = pd.read_csv('dc_airbnb.csv')
# here is to randomize the data 
dc_listings = dc_listings.loc[np.random.permutation(len(dc_listings))]
stripped_commas = dc_listings['price'].str.replace(',', '')
stripped_dollars = stripped_commas.str.replace('$', '')
dc_listings['price'] = stripped_dollars.astype('float')

# Here is to return a summary of the dataframe and non-null object to the table
print(dc_listings.info())

# here is to drop columns which we are not interested
drop_columns = ['room_type', 'city', 'state', 'latitude', 'longitude', 'zipcode', 'host_response_rate', 'host_acceptance_rate', 'host_listings_count']
dc_listings = dc_listings.drop(drop_columns, axis=1)

# now displaying the number of columns which are not null
print(dc_listings.isnull().sum())


#dropping cleaning fee and security deposit 
dc_listings = dc_listings.drop(['cleaning_fee', 'security_deposit'], axis=1)

#dropping row which has null value
dc_listings = dc_listings.dropna(axis=0)
print(dc_listings.isnull().sum())

#now we are to normalize the data (express things in terms of mean and standard deviation)
normalized_listings = (dc_listings - dc_listings.mean())/(dc_listings.std())
normalized_listings['price'] = dc_listings['price']
print(normalized_listings.head(3))


#Calculate the Euclidean distance using only the accommodates and bathrooms features between the first row and fifth row in normalized_listings using the distance.euclidean() function.
from scipy.spatial import distance
first_listing = normalized_listings.iloc[0][['accommodates', 'bathrooms']]
fifth_listing = normalized_listings.iloc[4][['accommodates', 'bathrooms']]
first_fifth_distance = distance.euclidean(first_listing, fifth_listing)
# this is just to calculate the distanc between first and fifth
print(first_fifth_distance)


#from now on using sci kit learn 
print('*********')
print('From here using the sci kit learn library')
print('*********')

from sklearn.neighbors import KNeighborsRegressor

train_df = normalized_listings.iloc[0:2792]
test_df = normalized_listings.iloc[2792:]
train_columns = ['accommodates', 'bathrooms']

# Instantiate ML model.
# knn = KNeighborsRegressor(n_neighbors=5, algorithm='brute')

# Fit model to data.
# knn.fit(train_df[train_columns], train_df['price'])

# Use model to make predictions.
# predictions = knn.predict(test_df[train_columns])



from sklearn.metrics import mean_squared_error

# Enter the number of columns wishing to be trained here
train_columns = ['accommodates', 'bathrooms', 'bedrooms']
# train_columns = ['bathrooms']
# train_columns = ['accommodates']
knn = KNeighborsRegressor(n_neighbors=5, algorithm='brute', metric='euclidean')
knn.fit(train_df[train_columns], train_df['price'])
predictions = knn.predict(test_df[train_columns])
from sklearn.metrics import mean_squared_error

two_features_mse = mean_squared_error(test_df['price'], predictions)
two_features_rmse = two_features_mse ** (0.5)
print('two_features_mse')
print(two_features_mse)
print('two_features_rmse')
print(two_features_rmse)

print('testing')
