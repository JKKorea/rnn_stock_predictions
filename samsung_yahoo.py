import tensorflow as tf
import numpy as np
import matplotlib
import time
import os

tf.set_random_seed(777)  # reproducibility

if "DISPLAY" not in os.environ:
    # remove Travis CI Error
    matplotlib.use('Agg')

import matplotlib.pyplot as plt

def MinMaxScaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # noise term prevents the zero division
    return numerator / (denominator + 1e-7)

# train Parameters
timesteps = seq_length = 8
data_dim = 6
hidden_dim = 10
output_dim = 1
learing_rate = 0.01
iterations = 500

# Choose stock
stock = "SSNLF"

# start time setting
startTime = time.time()

# data scrolling parts
import pandas_datareader.data as web
import datetime

start = datetime.datetime(2010, 1, 2)
end = datetime.datetime(2017, 7, 14)
df = web.DataReader(
	stock, # name
	'yahoo', # data source
	start, # start
	end # end
)

# Convert pandas dataframe to numpy array
xy = df.as_matrix()

# Open, High, Low, Volume, Close
test_min = np.min(xy,0)
test_max = np.max(xy,0)
denom = test_max - test_min

xy = MinMaxScaler(xy)
x = xy
y = xy[:, [-3]]  # Close as label

# data for Prediction
start = datetime.datetime(2017, 7, 18)
end = datetime.datetime(2017, 7, 26)
df = web.DataReader(
	stock, # name
	'yahoo', # data source
	start, # start
	end # end
)


test_last_X = df.as_matrix().reshape(1,8,6);

test_last_min = np.min(test_last_X, 0)
test_last_max = np.max(test_last_X, 0)
test_last_denom = test_last_max - test_last_min



# real Prediction data
start = datetime.datetime(2017, 7, 27)
end = datetime.datetime(2017, 7, 27)
df = web.DataReader(
	stock, # name
	'yahoo', # data source
	start, # start
	end # end
)

real_stock = df.as_matrix()

# build a dataset
dataX = []
dataY = []
for i in range(0, len(y) - seq_length):
    _x = x[i:i + seq_length]
    _y = y[i + seq_length]  # Next close price
   # print(_x, "->", _y)
    dataX.append(_x)
    dataY.append(_y)

# train/test split 70 / 30
train_size = int(len(dataY) * 0.7)
test_size = len(dataY) - train_size
trainX, testX = np.array(dataX[0:train_size]), np.array(
    dataX[train_size:len(dataX)])
trainY, testY = np.array(dataY[0:train_size]), np.array(
    dataY[train_size:len(dataY)])

# input place holders
X = tf.placeholder(tf.float32, [None, seq_length, data_dim], name='input_X')
Y = tf.placeholder(tf.float32, [None, 1], name='intput_Y')

# build a LSTM network
cell = tf.contrib.rnn.BasicLSTMCell(
    num_units=hidden_dim, state_is_tuple=True, activation=tf.tanh)
outputs, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

Y_pred = tf.contrib.layers.fully_connected(
    outputs[:, -1], output_dim, activation_fn=None)  # We use the last cell's output

# cost/loss
loss = tf.reduce_sum(tf.square(Y_pred - Y), name='losses_sum')  # sum of the squares

# optimizer
optimizer = tf.train.AdamOptimizer(learing_rate)
train = optimizer.minimize(loss, name='train')

# RMSE
targets = tf.placeholder(tf.float32, [None, 1], name='targets')
predictions = tf.placeholder(tf.float32, [None, 1], name='predictions')
rmse = tf.sqrt(tf.reduce_mean(tf.square(targets - predictions)), name='rmse')

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    # Tensorboard
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("./tensorflowlog", sess.graph)
    
    losslist = [];    
    # Training step
    for i in range(iterations):
        _, step_loss = sess.run([train, loss], feed_dict={
                                X: trainX, Y: trainY})
        print("[step: {}] loss: {}".format(i, step_loss))
        losslist = np.append(losslist, step_loss)

    # Test step
    test_predict = sess.run(Y_pred, feed_dict={X: testX})
    rmse = sess.run(rmse, feed_dict={
                    targets: testY, predictions: test_predict})
    print("RMSE: {}".format(rmse))

    # Print train_size, test_size
    print("train_size : {}".format(train_size))
    print("test_size : {}".format(test_size))

    # Predictions test
    prediction_test = sess.run(Y_pred, feed_dict={X: test_last_X})
    print("real stock price : ", end='')
    real_value = real_stock[0][-2] 
    print(real_value)
    
    print("prediction stock price : ", end='')
    prediction_value = (prediction_test*test_last_denom + test_last_min)[-1][-2]
    print(prediction_value)

    print("Error rate : ", end='')
    print(abs(prediction_value - real_value)/prediction_value * 100)

     # end time setting, print time
    elapsedTime = time.time() - startTime
    print("it took " + "%.3f"%(elapsedTime) + " s.")
    
    # Plot losss
    plt.figure(1)
    plt.plot(losslist, color ="green", label ="Error");
    plt.xlabel("Iteration Number")
    plt.ylabel("Sum of the Squarred Error")
    plt.legend(loc='upper right', frameon=False)

    # Plot predictions
    plt.figure(2) 
    plt.plot(testY, color ="red", label ="Real")
    plt.plot(test_predict, color ="blue", label ="Prediction")
    plt.xlabel("Time Period")
    plt.ylabel("Stock Price")
    plt.legend(loc='upper left', frameon=False)
    plt.xticks([])
    plt.yticks([])
    plt.show()