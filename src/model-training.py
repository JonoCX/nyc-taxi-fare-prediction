import pandas as pd 
import numpy as np 
import tensorflow as tf 

from tensorflow import keras 
from tensorflow.keras import layers
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold


data_location = '/media/jonathan/HDD/data/new-york-city-taxi-fare/feature-engineered-csvs/'
train_csv = data_location + 'train/train_features.csv'
test_csv = data_location + 'test/test_features.csv'

training_data = pd.read_csv(train_csv)
test_data = pd.read_csv(test_csv)

# drop na's
training_data = training_data.dropna(how='any', axis='rows')

# one hot encode catagorical columns
def ohe(data, columns):
    for col in columns:
        data = pd.concat([data, pd.get_dummies(data[col], prefix=col)], axis=1)
    return data

ohe_cols = ['passenger_count', 'day_of_week', 'time_of_day']
training_data = ohe(training_data, ohe_cols)
test_data = ohe(test_data, ohe_cols)

training_data = training_data.drop(ohe_cols, axis=1)
test_data = test_data.drop(ohe_cols, axis=1)

# normalize distance
def normalize(data, column):
    data[column] = (data[column] - data[column].mean()) / data[column].std()
    return data

training_data = normalize(training_data, 'distance_km')
test_data = normalize(test_data, 'distance_km')

y = training_data['fare_amount']
X = training_data.drop(['key', 'fare_amount'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape)

from sklearn.linear_model import LinearRegression

reg = LinearRegression(fit_intercept=False).fit(X_train, y_train)

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_pred - y_true)**2))

y_pred = reg.predict(X_test)
print(rmse(y_test, y_pred))

def baseline_model():
    model = keras.Sequential()
    model.add(layers.Dense(64, input_dim=36, activation=tf.nn.relu))
    model.add(layers.Dense(64, activation=tf.nn.relu))
    model.add(layers.Dense(1))
    model.compile(loss=rmse, optimizer=tf.keras.optimizers.RMSprop())
    return model

estimator = KerasRegressor(build_fn=baseline_model, epochs=10, verbose=1)

kfold = KFold(n_splits=10, random_state=42)
results = cross_val_score(estimator, X_train, y_train, cv=kfold)
print('Result: %.2f' % (np.sqrt(results.mean())))

# model = keras.Sequential([
#     layers.Dense(64, activation=tf.nn.relu, input_shape=X_train.shape),
#     layers.Dense(64, activation=tf.nn.relu),
#     layers.Dense(1)
# ])

# optimizer = tf.keras.optimizers.RMSprop(0.001)

# model.compile(loss=rmse, optimizer=optimizer, metrics=['mean_absolute_error', 'mean_squared_error'])

# # The patience parameter is the amount of epochs to check for improvement
# early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

# history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), callbacks=[early_stop])
