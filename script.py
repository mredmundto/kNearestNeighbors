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

#6
# Remove the commas (,) and dollar sign characters ($) from the price column:
# Use the str accessor so we can apply string methods to each value in the column followed by the string method replace to replace all comma characters with the empty character: stripped_commas = dc_listings['price'].str.replace(',', '')
# Repeat to remove the dollar sign characters as well.
# Convert the new Series object containing the cleaned values to the float datatype and assign back to the price column in dc_listings.
# Calculate the mean of the first 5 values in the price column and assign to mean_price.
# Use the print function or the variable inspector below to display mean_price.

stripped_commas = dc_listings['price'].str.replace(',', '')
stripped_dollars = stripped_commas.str.replace('$', '')
dc_listings['price'] = stripped_dollars.astype('float')
mean_price = dc_listings.iloc[0:5]['price'].mean()
print(mean_price)



#7
# Write a function named predict_price that can use the k-nearest neighbors machine learning technique to calculate the suggested price for any value for accommodates. This function should:
# Take in a single parameter, new_listing, that describes the number of bedrooms.
# Assign dc_listings to a new Dataframe named temp_df so we aren't constantly modifying the original dataset each time we call the function.
# Calculate the distance between each value in the accommodates column and the new_listing value that was passed in. Assign the resulting Series object to the distance column in temp_df.
# Sort temp_df by the distance column and select the first 5 values in the price column. Don't randomize the ordering of temp_df.
# Calculate the mean of these 5 values and use that as the return value for the entire predict_price function.
# Use the predict_price function to suggest a price for a living space that:
# accommodates 1 person, assign the suggested price to acc_one.
# accommodates 2 people, assign the suggested price to acc_two.
# accommodates 4 people, assign the suggested price to acc_four.
# Brought along the changes we made to the `dc_listings` Dataframe.
dc_listings = pd.read_csv('dc_airbnb.csv')
stripped_commas = dc_listings['price'].str.replace(',', '')
stripped_dollars = stripped_commas.str.replace('$', '')
dc_listings['price'] = stripped_dollars.astype('float')
dc_listings = dc_listings.loc[np.random.permutation(len(dc_listings))]

def predict_price(new_listing):
    temp_df = dc_listings
    temp_df['distance'] = temp_df['accommodates'].apply(lambda x: np.abs(x - new_listing))
    temp_df = temp_df.sort_values('distance')
    nearest_neighbors = temp_df.iloc[0:5]['price']
    predicted_price = nearest_neighbors.mean()
    return(predicted_price)

acc_one = predict_price(1)
acc_two = predict_price(2)
acc_four = predict_price(4)
print('This is the predicted price for one, two and four bedrooms')
print(acc_one)
print(acc_two)
print(acc_four)


