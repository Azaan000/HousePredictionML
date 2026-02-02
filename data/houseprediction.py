from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

df = pd.read_csv('data/housing.csv')
df.dropna(inplace=True)

X = df.drop('median_house_value', axis=1)
y = df['median_house_value']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

traindata = X_train.join(y_train)
numeric = traindata.select_dtypes(include=[np.number])

traindata['total_rooms'] = np.log(traindata['total_rooms'] + 1)
traindata['total_bedrooms'] = np.log(traindata['total_bedrooms'] + 1)
traindata['population'] = np.log(traindata['population'] + 1)
traindata['households'] = np.log(traindata['households'] + 1)

plt.figure(figsize=(10,6))
sns.heatmap(numeric.corr(), annot = True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

traindata.hist(bins=30, figsize=(15,10))
plt.tight_layout()
plt.show()

ocean = pd.get_dummies(traindata['ocean_proximity'], dtype = int)
                       
traindata = traindata.drop('ocean_proximity', axis = 1)
traindata = traindata.join(ocean)

sns.scatterplot(x = 'latitude', y = 'longitude', hue='median_house_value', data = traindata, palette = 'coolwarm')
plt.title('Geographical Distribution of House Values')
plt.show()

traindata['bedroom_ratio'] = traindata['total_bedrooms'] / traindata['total_rooms']
traindata['household_rooms'] = traindata['total_rooms'] / traindata['households']

plt.figure(figsize=(15,6))
sns.heatmap(traindata.corr(), annot = True, cmap='coolwarm')
plt.title('Enhanced Correlation Heatmap')
plt.show()

model = LinearRegression()

X_train = traindata.drop('median_house_value', axis = 1)
y_train = traindata['median_house_value']

model.fit(X_train, y_train)

testdata = X_test.join(y_test)
numeric = testdata.select_dtypes(include=[np.number])

testdata['total_rooms'] = np.log(testdata['total_rooms'] + 1)
testdata['total_bedrooms'] = np.log(testdata['total_bedrooms'] + 1)
testdata['population'] = np.log(testdata['population'] + 1)
testdata['households'] = np.log(testdata['households'] + 1)

ocean = pd.get_dummies(testdata['ocean_proximity'], dtype = int)
                       
testdata = testdata.drop('ocean_proximity', axis = 1)
testdata = testdata.join(ocean)

testdata['bedroom_ratio'] = testdata['total_bedrooms'] / testdata['total_rooms']
testdata['household_rooms'] = testdata['total_rooms'] / testdata['households']

X_test = testdata.drop('median_house_value', axis = 1)
y_test = testdata['median_house_value']

score = model.score(X_test, y_test)
print(f'Model R^2 Score: {score}')
#Score on test data: 0.68 using linear regression without scaling the data

forest = RandomForestRegressor(n_estimators=100, random_state=42)
forest.fit(X_train, y_train)
pred = forest.predict(X_test)
accuracy_score = forest.score(X_test, y_test)
print(f'Model R^2 Score: {accuracy_score}')

#Score on test data: 0.81 using Random Forest Regressor without scaling the data


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'data/house_prediction_model.pkl')

joblib.dump(forest, model_path)

print("\nRandom Forest model saved as house_prediction_model.pkl")


