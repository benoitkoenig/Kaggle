import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor

def is_not_zero(number):
    if number == 0:
        return 0
    return 1

def get_dummies_and_handle_missing(data):
    data = pd.get_dummies(data)
    myImputer = SimpleImputer()

    extImpData = data.copy()
    colsWithMissing = (col for col in extImpData.columns if extImpData[col].isnull().any())
    for col in colsWithMissing:
        extImpData[col + '_is_missing'] = extImpData[col].isnull()
    cols_persist = extImpData.columns
    extImpData = pd.DataFrame(myImputer.fit_transform(extImpData))
    extImpData.columns = cols_persist
    extImpData["DateSold"] = extImpData["YrSold"] + extImpData["MoSold"] / 12.
    return extImpData

data = pd.read_csv("./train.csv")
data = get_dummies_and_handle_missing(data)

real_data = pd.read_csv("./test.csv")
real_data = get_dummies_and_handle_missing(real_data)

data, real_data = data.align(real_data, join='left', axis=1)
real_data = real_data.drop(["SalePrice"], axis=1) # SalePrice was wrongfully added by data.align

y = data["SalePrice"]
X = data.drop(["SalePrice"], axis=1)

# Use a pipeline so I kinda see how to use it, and also in case it's later needed
my_pipeline = make_pipeline(XGBRegressor(random_state=1, early_stopping_rounds=5, learning_rate=0.1))

# Use cross_val_score to rate my pipeline
scores = cross_val_score(my_pipeline, X, y, scoring='neg_mean_absolute_error')
print(scores)

# Use the pipeline to get real predictions
my_pipeline.fit(X, y)
predictions = my_pipeline.predict(real_data)
output = pd.DataFrame({'Id': real_data.Id, 'SalePrice': predictions})
output.to_csv('submission.csv', index=False)
