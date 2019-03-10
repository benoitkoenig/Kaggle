import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

data = pd.read_csv("./melb_data.csv")

#data = pd.get_dummies(data)
data = data.select_dtypes(exclude=["object"])

dropnaData = data.dropna(axis=1)

myImputer = SimpleImputer()
imputedData = pd.DataFrame(myImputer.fit_transform(data))
imputedData.columns = data.columns

extImpData = data.copy()
colsWithMissing = (col for col in extImpData.columns if extImpData[col].isnull().any())
for col in colsWithMissing:
    extImpData[col + '_is_missing'] = extImpData[col].isnull()
cols_persist = extImpData.columns
extImpData = pd.DataFrame(myImputer.fit_transform(extImpData))
extImpData.columns = cols_persist

datas = [dropnaData, imputedData, extImpData]
for data in datas:
    y = data["Price"]
    X = data.drop(["Price"], axis=1)
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
    model = RandomForestRegressor(random_state=1)
    model.fit(train_X, train_y)
    predictions = model.predict(val_X)
    print(mean_absolute_error(val_y, predictions))
