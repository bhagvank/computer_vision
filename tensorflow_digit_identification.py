import numpy as nump
import tensorflow as tenf


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

X_training, Y_training = mnist.train.next_batch(5000)
X_test, Y_test = mnist.test.next_batch(200)

xtr = tenf.placeholder("float", [None, 784])
xte = tenf.placeholder("float", [784])

distance = tenf.reduce_sum(tenf.abs(tenf.add(xtr, tenf.negative(xte))), reduction_indices=1)


pred = tenf.argmin(distance, 0)

accuracy = 0

init = tenf.global_variables_initializer()

with tenf.Session() as sess:
    sess.run(init)

    for i in range(len(X_test)):
        nn_index = sess.run(pred, feed_dict={xtr: X_training, xte: X_test[i, :]})
        print("Test value", i, "Prediction value", nump.argmax(Y_training[nn_index]), \
            "True Class is ", nump.argmax(Y_test[i]))
        if nump.argmax(Y_training[nn_index]) == nump.argmax(Y_test[i]):
            accuracy += 1./len(X_test)
    print("Completed")
    print("Accuracy is", accuracy)
