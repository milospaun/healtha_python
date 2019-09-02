import datetime
import time

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import util


def init_params(x, y):
    # init parameters
    """

    :type y: np 2D array
    :param y:
    :type x: np 2D array
    """
    n_features = x.shape[1]
    y_cat = y.shape[1]
    W1 = tf.get_variable("W1", [n_features, 128], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b1 = tf.get_variable("b1", [1, 128], initializer=tf.zeros_initializer())
    W2 = tf.get_variable("W2", [128, 32], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b2 = tf.get_variable("b2", [1, 32], initializer=tf.zeros_initializer())
    W3 = tf.get_variable("W3", [32, y_cat], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b3 = tf.get_variable("b3", [1, y_cat], initializer=tf.zeros_initializer())

    param = {"W1": W1,
             "b1": b1,
             "W2": W2,
             "b2": b2,
             "W3": W3,
             "b3": b3}

    return param


def forward_propagation(x, parameters, keep_prob):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']

    print("X: " + str(x.shape))
    print("W1: " + str(W1.shape))
    print("b1: " + str(b1.shape))
    print("W2: " + str(W2.shape))
    print("b2: " + str(b2.shape))
    print("W3: " + str(W3.shape))
    print("b3: " + str(b3.shape))

    Z1 = tf.matmul(x, W1) + b1  # Z1 = np.dot(W1, X) + b1
    A1 = tf.nn.relu(Z1)  # A1 = relu(Z1)
    drop_out1 = tf.nn.dropout(A1, keep_prob)
    Z2 = tf.matmul(drop_out1, W2) + b2  # Z2 = np.dot(W2, A1) + b2
    A2 = tf.nn.relu(Z2)  # A2 = relu(Z2)
    Z3 = tf.matmul(A2, W3) + b3  # Z3 = np.dot(W3,A2) + b3
    return Z3


def compute_cost(Z3, Y):
    # logits = tf.transpose(Z3)
    # labels = tf.transpose(Y)
    # changed softmax from v1 to v2
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=Z3, labels=Y))
    return cost


def train_model(x_train, y_train, x_test, y_test, learning_rate=0.001,
                num_epochs=250, minibatch_size=32, prob_coef=1, print_cost=True, filename="model.mod"):
    """

    :param prob_coef: dropout parameter
    :param x_train:
    :param y_train: one shot array
    :param x_test:
    :param y_test: one shot array
    :param learning_rate:
    :param num_epochs:
    :param minibatch_size:
    :param print_cost:
    :param filename:
    :return:
    """
    time.clock()
    t0 = time.time()
    msg = "num_epoch: {}, minibatch_size: {}, prob_coef: {}, learning_rate: {}" \
        .format(num_epochs, minibatch_size, prob_coef, learning_rate)
    util.log(msg)
    tf.reset_default_graph()  # to be able to rerun the model without overwriting tf variables
    (n_x, m) = x_train.shape  # (n_x: input size, m : number of examples in the train set)
    n_y = y_train.shape[0]  # n_y : output size
    costs = []  # To keep track of the cost

    # Create Placeholders of shape (n_x, n_y)
    X = tf.placeholder(tf.float32)
    Y = tf.placeholder(tf.float32)
    keep_prob = tf.placeholder(tf.float32)

    # Initialize parameters
    parameters = init_params(x_train, y_train)

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    # Forward propagation: Build the forward propagation in the tensorflow graph
    Z3 = forward_propagation(X, parameters, keep_prob)

    # Cost function: Add cost function to tensorflow graph
    cost = compute_cost(Z3, Y)

    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Initialize all the variables
    init = tf.global_variables_initializer()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        sess.run(init)  # Run the initialization
        for epoch in range(num_epochs):
            epoch_cost = 0.  # Defines a cost related to an epoch
            num_minibatches = int(m / minibatch_size)  # number of minibatches of size minibatch_size in the train set
            minibatches = util.random_mini_batches(x_train, y_train, minibatch_size)

            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch  # Select a minibatch
                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the "optimizer" and "cost", the feedict should contain a minib. for (X,Y).
                _, minibatch_cost = sess.run([optimizer, cost],
                                             feed_dict={X: minibatch_X, Y: minibatch_Y, keep_prob: prob_coef})
                epoch_cost += minibatch_cost / num_minibatches

            # Print the cost every epoch
            if print_cost is True and epoch % 10 == 0:
                print("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost is True and epoch % 5 == 0:
                costs.append(epoch_cost)

        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        save_path = saver.save(sess, filename)
        print("Model saved in file: %s" % save_path)

        # Calculate accuracy on the test set
        Z3_max = tf.argmax(Z3, axis=1)
        Y_max = tf.argmax(Y, axis=1)
        correct_prediction = tf.equal(Z3_max, Y_max)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        # Confusion matrix and Accuracy
        acc_train = sess.run(accuracy, feed_dict={X: x_train, Y: y_train, keep_prob: 1})
        acc_test = accuracy.eval({X: x_test, Y: y_test, keep_prob: 1})
        print("Train TF Accuracy:", acc_train)
        print("Test TF Accuracy:", acc_test)
        # np.savetxt("Tmp\ytest.out", np.array(Z3capa))

        Z3capa_train = Z3_max.eval(feed_dict={X: x_train, Y: y_train, keep_prob: 1})
        Ycapa_train = Y_max.eval(feed_dict={X: x_train, Y: y_train, keep_prob: 1})
        Z3capa_test = Z3_max.eval(feed_dict={X: x_test, Y: y_test, keep_prob: 1})
        Ycapa_test = Y_max.eval(feed_dict={X: x_test, Y: y_test, keep_prob: 1})

    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    util.log("\nTrain metrics:##########################################")
    util.print_metrics_per_class(Z3capa_train, Ycapa_train)
    util.log("\nTest metrics:##########################################")
    util.print_metrics_per_class(Z3capa_test, Ycapa_test)

    # save execution time
    util.log("\n Total program time: " + str(time.clock()))
    util.log("Total process training time: " + str(int(time.time() - t0)))
    util.log_table_value("{} {} {} {}".format(num_epochs, acc_train, acc_test, str(int(time.time() - t0))))

    return parameters


def testData(filename, xtest, ytest, xtrain, ytrain):
    """
    Get predictions with statistics.
    :param filename: path to saved model that should be used for testing
    :param xtest:
    :param ytest:   ones hot array
    :param xtrain:
    :param ytrain: one shot array
    :return:
    """
    # Initialize parameters and create Variables
    tf.reset_default_graph()
    param = init_params(xtest, ytest)

    # Create some placeholders for input output.
    X = tf.placeholder(tf.float32)
    Y = tf.placeholder(tf.float32)
    keep_prob = tf.placeholder(tf.float32)

    # Calculate the correct predictions
    Z3 = forward_propagation(X, param, keep_prob)
    Z3_max = tf.argmax(Z3, axis=1)
    Y_max = tf.argmax(Y, axis=1)
    correct_prediction = tf.equal(Z3_max, Y_max)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    # Later, launch the model, use the saver to restore variables from disk, and do some work with the model.
    with tf.Session() as sess:
        # Restore variables from disk.
        saver.restore(sess, filename)
        print("Model restored.")

        # Calculate accuracy on the test set
        print("Train Accuracy:", sess.run(accuracy, feed_dict={X: xtrain, Y: ytrain, keep_prob: 0.5}))

        acc = accuracy.eval({X: xtest, Y: ytest})
        print("Test Accuracy:", acc)
        Z3capa_train = Z3_max.eval(feed_dict={X: xtrain, Y: ytrain, keep_prob: 0.5})
        Ycapa_train = Y_max.eval(feed_dict={X: xtrain, Y: ytrain, keep_prob: 0.5})
        Z3capa_test = Z3_max.eval(feed_dict={X: xtest, Y: ytest, keep_prob: 1})
        Ycapa_test = Y_max.eval(feed_dict={X: xtest, Y: ytest, keep_prob: 1})

    filepath_report = filename + "report.txt"
    print("\nTrain metrics:##################################")
    util.print_metrics_per_class(Z3capa_train, Ycapa_train)
    print("\nTest metrics:##################################")
    util.print_metrics_per_class(Z3capa_test, Ycapa_test)

    np.savetxt("..\\Tmp\\ytest.out", np.array(Ycapa_test), fmt="%d")
    np.savetxt("..\\Tmp\\xtest.out", np.array(xtest), fmt="%.9f")
    return acc


def runData(model_path, x, y):
    """
    Get predictions without showing any statistics.
    :param model_path:
    :param x:
    :param y:
    :return:
    """
    # Initialize parameters and create Variables
    tf.reset_default_graph()
    param = init_params(x, y)

    # Create some placeholders for input output.
    X = tf.placeholder(tf.float32)
    keep_prob = tf.placeholder(tf.float32)

    # Calculate the correct predictions
    Z3 = forward_propagation(X, param, keep_prob)
    Z3_max = tf.argmax(Z3, axis=1)

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    # Later, launch the model, use the saver to restore variables from disk, and do some work with the model.
    sess = tf.Session()
    with tf.Session() as sess:
        # Restore variables from disk.
        saver.restore(sess, model_path)
        print("Model restored.")

        # Calculate y
        Y = Z3_max.eval(feed_dict={X: x, keep_prob: 1})

    Y = Y + 1
    print("YCapa: " + str(Y[0:200]))
    np.savetxt("..\\Tmp\\ytest.out", np.array(Y), fmt='%d')
    np.savetxt("..\\Tmp\\xtest.out", np.array(x), fmt="%.9f")
    return Y


