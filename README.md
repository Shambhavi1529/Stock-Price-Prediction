# Recurrent Neural Network

#### Prediction of the 10-day closing stock price for Walmart using RNN, LTSM, Gru, and Conv1D techniques

import numpy as np
import pandas as pd
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
import tensorflow as tf
from tensorflow import keras

# Set random seed
np.random.seed(42)

# Configure plotting settings
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
%matplotlib inline

# Additional imports
from sklearn.metrics import mean_squared_error

### Loading the dataset

# Load the data into a pandas dataframe
df = pd.read_csv("/Users/shambhavimishra/Downloads/WMT.csv')

df.head()

# Set the date column as the index
df.set_index('Date', inplace=True)

df = df.sort_values(by='Date')

#Drop the columns we don't need
df = df.drop(['Open', 'High', 'Low', 'Adj Close', 'Volume'], axis=1)

df

#Plot stock price
plt.plot(df['Close'])

plt.show()

### Data Cleaning

# Remove all null values
df = df.dropna()

df

close_price = df["Close"][-100:]

df = pd.DataFrame({'Date': close_price.index, 'Close': close_price.values})
df

### Reshape the dataset

#reshape the data
df.shape[0]/10

df.groupby(['Date']).count()

df_count = pd.DataFrame(df.groupby(['Date']).count()['Close'])

df_count

df_temp = np.array(df['Close']).reshape(10,10)

df_temp

df_convert = pd.DataFrame(df_temp, columns=np.arange(0,10,1))

df_convert

row_count = df.shape[0]
row_count

close_prices = df['Close'].values

print(close_prices)

### Reshape for standardizing data

# standardization the data
df_feature = np.array(df_convert).ravel().reshape(-1,1)

df_feature.shape

df_feature

# Scale the data between 0 and 1

scaler=MinMaxScaler(feature_range=(0,1))
close_prices=scaler.fit_transform(np.array(close_prices).reshape(-1,1))

print(close_prices)

# Reshaping the data
df_reshaped = close_prices.reshape(10,10)
df_reshaped.shape

pd.DataFrame(df_reshaped, columns=np.arange(0,10,1))

### Splitting the data

## Splitting dataset into train and test split
training_size = int(len(close_prices)*0.70)
test_size = len(close_prices)-training_size
train_data,test_data = close_prices[0:training_size,:],close_prices[training_size:len(close_prices),:1]

training_size,test_size

train_data

## Create Input and Target values

import numpy as np

# convert an array of values into a dataset matrix

def create_dataset(dataset, look_back=9):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0] 
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

## Adding one more dimension to make it ready for RNNs

# reshape data
look_back = 9
X_train, y_train = create_dataset(train_data, look_back)
X_test, y_test = create_dataset(test_data, look_back)

print(X_train.shape), print(y_train.shape)

print(X_test.shape), print(y_test.shape)

# reshape input to be which is required for LSTM
X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

X_train, X_test, y_train, y_test

print (X_train.shape)
print (y_train.shape)
print (X_test.shape)
print (y_test.shape)

## A normal (cross-sectional) NN

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[9, 1]),
    keras.layers.Dense(27, activation='relu'),
    keras.layers.Dense(1, activation=None)
])

model.compile(loss="mse", optimizer='Adam')

history = model.fit(X_train, y_train, epochs=100)

#Predict:
y_pred = model.predict(X_test)


# Remember, these are standardized values. 

comparison = pd.DataFrame()

comparison['actual'] = scaler.inverse_transform([y_test]).flatten()
comparison['predicted'] = scaler.inverse_transform(y_pred).flatten()


comparison

mean_squared_error(comparison['actual'], comparison['predicted'])


plt.plot(comparison['actual'], label = 'actual')
plt.plot(comparison['predicted'], label = 'predicted')

plt.legend()

plt.show()

## Simplest RNN

model = keras.models.Sequential([
    keras.layers.SimpleRNN(27, activation='relu', input_shape=[9, 1]),
    keras.layers.Dense(1, activation=None)
])

np.random.seed(42)
tf.random.set_seed(42)

model.compile(loss="mse", optimizer='Adam')

history = model.fit(X_train, y_train, epochs=100)

### Predictions

#Predict:
y_pred = model.predict(X_test)

#Remember, these are standardized values. 

comparison = pd.DataFrame()

comparison['actual'] = scaler.inverse_transform([y_test]).flatten()
comparison['predicted'] = scaler.inverse_transform(y_pred).flatten()

comparison.head(10)

mean_squared_error(comparison['actual'], comparison['predicted'])

plt.plot(comparison['actual'], label = 'actual')
plt.plot(comparison['predicted'], label = 'predicted')

plt.legend()

plt.show()

## Simple RNN with more layers

model = keras.models.Sequential([
    keras.layers.SimpleRNN(27, activation='relu', return_sequences=True, input_shape=[9, 1]),
    keras.layers.SimpleRNN(27, activation='relu', return_sequences=False),
    keras.layers.Dense(1, activation=None)
])

np.random.seed(42)
tf.random.set_seed(42)

model.compile(loss="mse", optimizer='Adam')

history = model.fit(X_train, y_train, epochs=100)

### Predictions

#Predict:
y_pred = model.predict(X_test)

#Remember, these are standardized values. 

comparison = pd.DataFrame()

comparison['actual'] = scaler.inverse_transform([y_test]).flatten()
comparison['predicted'] = scaler.inverse_transform(y_pred).flatten()

comparison.head(10)

mean_squared_error(comparison['actual'], comparison['predicted'])

plt.plot(comparison['actual'], label = 'actual')
plt.plot(comparison['predicted'], label = 'predicted')

plt.legend()

plt.show()

## LSTM neural network with one layer 

model = keras.models.Sequential([
    keras.layers.LSTM(54, activation='relu', input_shape=[9, 1]),
    keras.layers.Dense(1, activation=None)
])

np.random.seed(42)
tf.random.set_seed(42)

model.compile(loss="mse", optimizer='Adam')

history = model.fit(X_train, y_train, epochs=100)

### Predictions

#Predict:
y_pred = model.predict(X_test)

#Remember, these are standardized values. 

comparison = pd.DataFrame()

comparison['actual'] = scaler.inverse_transform([y_test]).flatten()
comparison['predicted'] = scaler.inverse_transform(y_pred).flatten()

comparison.head(10)

mean_squared_error(comparison['actual'], comparison['predicted'])

plt.plot(comparison['actual'], label = 'actual')
plt.plot(comparison['predicted'], label = 'predicted')

plt.legend()

plt.show()

## LSTM neural network with more layer 

model = keras.models.Sequential([
    keras.layers.LSTM(36, activation='tanh', return_sequences=True, input_shape=[9, 1]),
    keras.layers.LSTM(36, activation='tanh', return_sequences=False),
    keras.layers.Dense(1, activation=None)
])

np.random.seed(42)
tf.random.set_seed(42)

model.compile(loss="mse", optimizer='Adam')

history = model.fit(X_train, y_train, epochs=100)

### Predictions

#Predict:
y_pred = model.predict(X_test)

#Remember, these are standardized values. 

comparison = pd.DataFrame()

comparison['actual'] = scaler.inverse_transform([y_test]).flatten()
comparison['predicted'] = scaler.inverse_transform(y_pred).flatten()

comparison.head(10)

mean_squared_error(comparison['actual'], comparison['predicted'])

plt.plot(comparison['actual'], label = 'actual')
plt.plot(comparison['predicted'], label = 'predicted')

plt.legend()

plt.show()

## GRU Neural network with multiple layers 

model = keras.models.Sequential([
    keras.layers.GRU(27, activation='relu', return_sequences=True, input_shape=[9, 1]),
    keras.layers.GRU(27, activation='relu', return_sequences=False),
    keras.layers.Dense(1, activation=None)
])

np.random.seed(42)
tf.random.set_seed(42)

model.compile(loss="mse", optimizer='RMSprop')

history = model.fit(X_train, y_train, epochs=100)

### Predictions

#Predict:
y_pred = model.predict(X_test)

#Remember, these are standardized values. 

comparison = pd.DataFrame()

comparison['actual'] = scaler.inverse_transform([y_test]).flatten()
comparison['predicted'] = scaler.inverse_transform(y_pred).flatten()

comparison.head(10)

mean_squared_error(comparison['actual'], comparison['predicted'])

plt.plot(comparison['actual'], label = 'actual')
plt.plot(comparison['predicted'], label = 'predicted')

plt.legend()

plt.show()

## 1D Conv neural network using Keras 

model = keras.models.Sequential([
    keras.layers.Conv1D(filters=20, kernel_size=3, strides=1, padding="valid", input_shape=[9, 1]),
    keras.layers.GRU(27, activation='relu', return_sequences=True),
    keras.layers.GRU(27, activation='relu', return_sequences=False),
    keras.layers.Dense(1, activation=None)
])

np.random.seed(42)
tf.random.set_seed(42)

model.compile(loss="mse", optimizer='Adam')

history = model.fit(X_train, y_train, epochs=100)

### Predictions

#Predict:
y_pred = model.predict(X_test)

#Remember, these are standardized values. 

comparison = pd.DataFrame()

comparison['actual'] = scaler.inverse_transform([y_test]).flatten()
comparison['predicted'] = scaler.inverse_transform(y_pred).flatten()

comparison.head(10)

mean_squared_error(comparison['actual'], comparison['predicted'])

plt.plot(comparison['actual'], label = 'actual')
plt.plot(comparison['predicted'], label = 'predicted')

plt.legend()

plt.show()

# Conclusion: 

#### After analyzing the various models trained for stock price prediction, it seems that the "Simple RNN" with more layers performed the best. This model achieved a mean squared error of 3.92, which is the lowest of all the models tested.

#### One important difference between the Simple RNN with more layers and the other models is the number of layers used in the model. The Simple RNN with more layers used multiple layers which could capture more complex patterns in the data, resulting in a lower mean squared error. In contrast, some of the other models used only one layer, which might not be enough to fully capture the nuances of the data.

##### A normal(corss-sectional) NN had mean squared error of 4.89
##### Simplest RNN had mean squared error of 5.23
##### Simplest RNN with more layers had mean squared error of 3.92
##### LSTM Neural Networ with one layer had mean squared error of 10.25
##### LSTM Neural Networ with more layers had mean squared error of 15.53
##### GRU Neural Networ with multiple layers had mean squared error of 7.55
##### 1D Conv Neural Networ using Keras had mean squared error of 8.80


#### All of the models performed relatively well, with mean squared errors. However, the Simple RNN with more layers was the most effective, achieving the lowest mean squared error.

#### Thus, the Simple RNN with more layers performed the best in your stock price prediction task, achieving the lowest mean squared error. This model was able to capture complex patterns in the data due to the multiple layers used in the model. However, all of the models tested showed promising results, suggesting that deep learning models can be effective tools for stock price prediction.

