import numpy as np
import pandas as pd

dc_listings = pd.read_csv("dc_airbnb.csv")
stripped_commas = dc_listings['price'].str.replace(',', '')
stripped_dollars = stripped_commas.str.replace('$', '')
dc_listings['price'] = stripped_dollars.astype('float')
shuffled_index = np.random.permutation(dc_listings.index)
dc_listings = dc_listings.reindex(shuffled_index)

# now this is we are splitting half and half here
split_one = dc_listings.iloc[0:1862]
split_two = dc_listings.iloc[1862:]

from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

train_one = split_one
test_one = split_two
train_two = split_two
test_two = split_one
# First half
model = KNeighborsRegressor()
model.fit(train_one[["accommodates"]], train_one["price"])
test_one["predicted_price"] = model.predict(test_one[["accommodates"]])
iteration_one_rmse = mean_squared_error(test_one["price"], test_one["predicted_price"])**(0.5)

# Second half
model.fit(train_two[["accommodates"]], train_two["price"])
test_two["predicted_price"] = model.predict(test_two[["accommodates"]])
iteration_two_rmse = mean_squared_error(test_two["price"], test_two["predicted_price"])**(0.5)

avg_rmse = np.mean([iteration_two_rmse, iteration_one_rmse])

print('Comparing the error between using first half as training and vice versa')
print(iteration_one_rmse, iteration_two_rmse, avg_rmse)


dc_listings.set_value(dc_listings.index[0:744], "fold", 1)
dc_listings.set_value(dc_listings.index[744:1488], "fold", 2)
dc_listings.set_value(dc_listings.index[1488:2232], "fold", 3)
dc_listings.set_value(dc_listings.index[2232:2976], "fold", 4)
dc_listings.set_value(dc_listings.index[2976:3722], "fold", 5)



from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error


# # Training
# model = KNeighborsRegressor()
# train_iteration_one = dc_listings[dc_listings["fold"] != 1]
# test_iteration_one = dc_listings[dc_listings["fold"] == 1]
# model.fit(train_iteration_one[["accommodates"]], train_iteration_one["price"])

# # Predicting
# labels = model.predict(test_iteration_one[["accommodates"]])
# test_iteration_one["predicted_price"] = labels
# iteration_one_mse = mean_squared_error(test_iteration_one["price"], test_iteration_one["predicted_price"])
# iteration_one_rmse = iteration_one_mse ** (0.5)



# Use np.mean to calculate the mean.
import numpy as np
fold_ids = [1,2,3,4,5]
def train_and_validate(df, folds):
    fold_rmses = []
    for fold in folds:
        # Train
        model = KNeighborsRegressor()
        train = dc_listings[dc_listings["fold"] != fold]
        test = dc_listings[dc_listings["fold"] == fold]
        model.fit(train[["accommodates"]], train["price"])
        # Predict
        labels = model.predict(test[["accommodates"]])
        test["predicted_price"] = labels
        mse = mean_squared_error(test["price"], test["predicted_price"])
        rmse = mse**(0.5)
        fold_rmses.append(rmse)
    return(fold_rmses)

print('this is to use K-folds for K = 5, f-fold means testing different data set')
rmses = train_and_validate(dc_listings, fold_ids)
print(rmses)
avg_rmse = np.mean(rmses)
print(avg_rmse)


print('**** from now on we use K-Folds from sci kit learn****')


from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
kf = KFold(len(dc_listings), 5, shuffle=True, random_state=1)
model = KNeighborsRegressor()
mses = cross_val_score(model, dc_listings[["accommodates"]], dc_listings["price"], scoring="mean_squared_error", cv=kf)
rmses = [np.sqrt(np.absolute(mse)) for mse in mses]
avg_rmse = np.mean(rmses)

print(rmses)
print(avg_rmse)

from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
num_folds = [3, 5, 7, 9, 10, 11, 13, 15, 17, 19, 21, 23]

for fold in num_folds:
    kf = KFold(len(dc_listings), fold, shuffle=True, random_state=1)
    model = KNeighborsRegressor()
    mses = cross_val_score(model, dc_listings[["accommodates"]], dc_listings["price"], scoring="mean_squared_error", cv=kf)
    rmses = [np.sqrt(np.absolute(mse)) for mse in mses]
    avg_rmse = np.mean(rmses)
    std_rmse = np.std(rmses)
    print(str(fold), "folds: ", "avg RMSE: ", str(avg_rmse), "std RMSE: ", str(std_rmse))





