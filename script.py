#1 Intro
print("This is K-nearest neighbors")

# Find a few similar listings.
# Calculate the average nightly rental price of these listings.
# Set the average price as the price for our listing.

#2 reading the file 
import pandas as pd
dc_listings = pd.read_csv('dc_airbnb.csv')
print(dc_listings.iloc[0])

#3 
# Calculate the Euclidean distance between our living space, which can accommodate 3 people, and the first living space in the dc_listings Dataframe.
# Assign the result to first_distance and display the value using the print function.
import numpy as np
our_acc_value = 3
first_living_space_value = dc_listings.iloc[0]['accommodates']
print(first_living_space_value)
first_distance = np.abs(first_living_space_value - our_acc_value)
print(first_distance)

#4 
# Calculate the distance between each value in the accommodates column from dc_listings and the value 3, which is the number of people our listing accommodates:
# Use the apply method to calculate the absolute value between each value in accommodates and 3 and return a new Series containing the distance values.
# Assign the distance values to the distance column.
# Use the Series method value_counts and the print function to display the unique value counts for the distance column.

new_listing = 3
dc_listings['distance'] = dc_listings['accommodates'].apply(lambda x: np.abs(x - new_listing))
print(dc_listings['distance'].value_counts())

#5
# Randomize the order of the rows in dc_listings:
# Use the np.random.permutation() function to return a NumPy array of shuffled index values.
# Use the Dataframe method loc[] to return a new Dataframe containing the shuffled order.
# Assign the new Dataframe back to dc_listings.
# After randomization, sort dc_listings by the distance column.
# Display the first 10 values in the price column using the print function.

np.random.seed(1)
dc_listings = dc_listings.loc[np.random.permutation(len(dc_listings))]
dc_listings = dc_listings.sort_values('distance')
print(dc_listings.iloc[0:10]['price'])






