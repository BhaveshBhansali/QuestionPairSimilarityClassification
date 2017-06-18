import numpy as np
import pickle
import tensorflow as tf
from random import sample
import sys
import os
from Multilayer_Perceptron import next_data,dev_model

def main():

    error_out_file=open(sys.argv[5]+os.sep+'LEARNING_RATE_'+str(sys.argv[1]+'_'+'TRAINING_EPOCHS_'+sys.argv[2]+'_'+'BATCH_SIZE_'+sys.argv[3])+'logistic_regression.txt',mode='a')

    with open(sys.argv[4]+os.sep+'embedding_array.p', 'rb') as fp:
        embedding_array=pickle.load(fp)

    with open(sys.argv[4]+os.sep+'train_data_modelling.p', 'rb') as fp:
        train_data=pickle.load(fp)

    with open(sys.argv[4]+os.sep+'dev_data.p', 'rb') as fp:
        dev_data = pickle.load(fp)

    ######### Building logistic regression model

    # Parameters
    learning_rate = float(sys.argv[1])
    training_epochs = int(sys.argv[2])
    #batch_size = 100
    display_step = 1


    # Network Parameters
    n_input = 600  # question words data have input (img shape: 300+300)
    n_classes = 1  # probability value

    batch_size = int(sys.argv[3])
    # tf Graph Input
    x = tf.placeholder(tf.float32, [None, n_input])  # normalized word vector dimension 300+300=600
    #x = tf.Variable(tf.float32, [batchSize, n_input])  # normalized word vector dimension 300+300=600
    y = tf.placeholder(tf.float32, [None, n_classes])  # probability value

    w1 = tf.Variable(tf.random_normal([n_input, n_classes]))
    b1 = tf.Variable(tf.random_normal([n_classes]))



    logit = tf.matmul(x, w1) + b1

    out_prob = tf.nn.sigmoid(logit)
    output_label = tf.cast(out_prob > 0.5, tf.float32)

    # Output layer with linear activation
    out = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logit)

    # Define loss and optimizer
    cost = tf.reduce_mean(out)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


    # Initializing the variables
    init = tf.global_variables_initializer()

    # Create a saver object
    saver = tf.train.Saver()

    #Launch the graph

    accuracy = tf.reduce_mean(tf.cast(tf.equal(output_label,y),tf.float32))

    with tf.Session() as sess:
        sess.run(init)

        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.

            ## shuffle training data
            train_data=sample(train_data, len(train_data))

            start_index=0
            last_index=batch_size
            no_of_batch = int(len(train_data)/batch_size)

            # Loop over all batches
            for i in range(no_of_batch):

                batch_x, batch_y = next_data(start_index,last_index,embedding_array,train_data)

                # Run optimization op (backprop) and cost op (to get loss value)
                _, c, prediction = sess.run([optimizer, cost, accuracy], feed_dict={x: batch_x,y: batch_y})
                # Compute average loss
                #avg_cost += c / total_batch
                #avg_cost += c / (1.0*batchSize)
                #print("Iteration:", '%04d' % (i + 1), "cost=","{:.9f}".format(c), "Prediction=", "{:.9f}".format(prediction) )
                #error_out_file.write("Iteration:"+str(i+1)+' cost='+str(c) + " prediction" + str(prediction))
                #error_out_file.write('\n')
                start_index=last_index
                last_index+=batch_size

            # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c),"Prediction=", "{:.9f}".format(prediction))
                #error_out_file.write("Epoch:" + str(epoch+1) + ' cost=' + str(c)+ " prediction" + str(prediction))
                #error_out_file.write('\n')

            dev_model(sess,dev_data,100,embedding_array,out_prob,cost,accuracy,error_out_file,x,y)

        #Now, save the graph
        saver.save(sess, sys.argv[5]+os.sep+str(sys.argv[1]+sys.argv[2]+sys.argv[3]+'_logistic_regression'))
        print("Optimization Finished!")

        # calling function to test accuracy on dev data on current state of trained model
        dev_model(sess,dev_data,100,embedding_array,out_prob,cost,accuracy,error_out_file,x,y)




if __name__ == '__main__':
    main()
