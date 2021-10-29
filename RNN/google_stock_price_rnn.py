import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, log_loss

data_df = pd.read_csv("data/Google_Stock_Price_Train.csv")

X = np.array(data_df["Open"].values)
y = np.array(data_df["Close"].values)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

train = X_train.reshape(X_train.shape[0], 1)  # define a dimenssão nx1 = 1x1
y_train = y_train.reshape(y_train.shape[0], 1)
test = X_test.reshape(X_test.shape[0], 1)
y_test = y_test.reshape(y_test.shape[0], 1)

scaler = MinMaxScaler(feature_range=(0, 1))  # escala de 0 até 1

train_scaled = scaler.fit_transform(train)
y_train_scaled = scaler.fit_transform(y_train)
test_scaled = scaler.fit_transform(test)

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import SimpleRNN
from keras.layers import Dropout

RNN = Sequential()  # Iniciando o RNN

RNN.add(SimpleRNN(
    units=50,
    activation="tanh",
    return_sequences=True,
    input_shape=(train.shape[1], 1)
))  # primeiro RNN
RNN.add(Dropout(0.2))  # abandono de 20% dos dados

RNN.add(SimpleRNN(
    units=50,
    activation="tanh",
    return_sequences=True
))  # segundo RNN
RNN.add(Dropout(0.2))

RNN.add(SimpleRNN(
    units=50,
    activation="tanh",
    return_sequences=True
))  # terceiro RNN
RNN.add(Dropout(0.2))

RNN.add(SimpleRNN(
    units=50
))  # quarto RNN
RNN.add(Dropout(0.2))

RNN.add(Dense(units=1))  # Camada de saída

RNN.compile(optimizer="adam", loss="mean_squared_error")

X = train_scaled.reshape(train_scaled.shape[0], train_scaled.shape[1], 1)  # array 1x3
X_test = test_scaled.reshape(test_scaled.shape[0], test_scaled.shape[1], 1)  # array 1x3

RNN.fit(X, y_train_scaled, epochs=100, batch_size=32)

predicted = scaler.inverse_transform(RNN.predict(X_test))

y_predicted = predicted.reshape(predicted.shape[0], 1)

import matplotlib.pyplot as plt

plt.figure(figsize=(20, 10), dpi=80, facecolor='w', edgecolor='k')
plt.plot(test, color="orange", label="Real value")
plt.plot(predicted, color="c", label="RNN predicted result")
plt.legend()
plt.xlabel("Days")
plt.ylabel("Values")
plt.grid(True)
plt.show()
