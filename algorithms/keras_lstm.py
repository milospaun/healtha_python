import time

import numpy as np
import matplotlib.pyplot as plt
from keras.engine.saving import load_model

import util

from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation, Dropout, SimpleRNN, Conv1D, Reshape

from keras.utils import plot_model


def init_params(x, y, batch_size, lstm_size):
    return None


def create_model(x, prob_coef):
    model = Sequential()
    model.add(Dense(128, activation='relu', input_dim=x.shape[1]))
    model.add(Dropout(prob_coef))
    model.add(Dense(32, activation='relu'))
    # model.add(Dropout(prob_coef))
    model.add(Dense(10, activation='softmax'))
    return model


def create_model_lstm(x, prob_coef, time_steps, lstm_size):
    model = Sequential()

    model.add(Dense(128, activation='relu', input_dim=x.shape[1]))
    model.add(Dropout(prob_coef))
    model.add(Dense(32, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(lstm_size, activation='relu'))

    # The LSTM and Conv1D input layer must be 3D. (batchSize, length, channels)
    #   batchSize = number of sentences
    #   length = number of words in each sentence
    #   channels = dimension of the embedding's output.
    model.add(Reshape((lstm_size, 1)))
    model.add(Conv1D(filters=1, kernel_size=time_steps, padding='valid', activation='relu', strides=1))
    model.add(LSTM(lstm_size, input_shape=(time_steps, lstm_size), return_sequences=False))
    model.add(Activation('softmax'))
    return model


def train_model(x_train, y_train, x_test, y_test, learning_rate=0.001,
                num_epochs=250, batch_size=60, prob_coef=1, lstm_size=10, time_steps=5,
                print_cost=True, filename="model.mod"):
    """

    :param time_steps:
    :param batch_size:
    :param lstm_size:
    :param prob_coef: dropout parameter
    :param x_train:
    :param y_train: one shot array
    :param x_test:
    :param y_test: one shot array
    :param learning_rate:
    :param num_epochs:
    :param print_cost:
    :param filename:
    :return:
    """
    time.clock()
    t0 = time.time()

    model = create_model(x_train, prob_coef)
    # model = create_model_lstm(x_train, prob_coef, time_steps=time_steps, lstm_size=lstm_size)

    model.compile(loss='categorical_crossentropy',
                  optimizer='Adam',
                  metrics=['categorical_accuracy'])

    # x_train and y_train are Numpy arrays --just like in the Scikit-Learn API.
    hist = model.fit(x_train, y_train, epochs=num_epochs, batch_size=batch_size, verbose=2)

    model.summary()
    print(hist.history)
    costs = hist.history['loss']

    loss_and_metrics = model.evaluate(x_train, y_train, batch_size=batch_size)
    acc_train = loss_and_metrics[1]

    loss_and_metrics = model.evaluate(x_test, y_test, batch_size=batch_size)
    acc_test = loss_and_metrics[1]

    pred_train = model.predict(x_train)
    pred_train = np.argmax(pred_train, axis=-1)
    pred_test = model.predict(x_test)
    pred_test = np.argmax(pred_test, axis=-1)
    y_train = np.argmax(y_train, axis=-1)
    y_test = np.argmax(y_test, axis=-1)

    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    util.log("\nTrain metrics:##########################################")
    util.print_metrics_per_class(y_train, pred_train)
    util.log("\nTest metrics:##########################################")
    util.print_metrics_per_class(y_test, pred_test)

    # save execution time
    util.log("\n Total program time: " + str(time.clock()))
    train_time = str(int(time.time() - t0))
    util.log("Total process training time: " + train_time)
    util.log_table_value("{} {} {} {} {} {}".format(num_epochs, acc_train, acc_test, train_time, batch_size, prob_coef,
                                                    str(x_train.shape)))

    # visualize and save model to a file
    # To DO install graphviz
    # plot_model(model, to_file='..//Tmp//model.png')
    return model


def testData(model_path, xtest, ytest):
    """
    Get predictions with statistics.
    :param filename: path to saved model that should be used for testing
    :param xtest:
    :param ytest:   ones hot array
    :param xtrain:
    :param ytrain: one shot array
    :return:
    """

    model = load_model(model_path)
    ycapa = model.predict(xtest, batch_size=480)

    ycapa = np.argmax(ycapa, axis=-1)
    # ytest = np.argmax(ytest, axis=-1)

    util.log(f"\nTest metrics: {ytest.shape} {ycapa.shape}")
    util.print_metrics_per_class(ytest, ycapa)

    return 1


def runData(model_path, x):
    """
    Get predictions without showing any statistics.
    :param model_path:
    :param x:
    :return:
    """
    model = load_model(model_path)
    ycapa = model.predict(x)

    return ycapa
