import numpy as np
import pickle
import tensorflow as tf
from random import sample
import sys
import os


def next_data(start_index,last_index,embedding_array,training_data):

    """
    extract data for batch processing

     Parameters:
    -----------
    :param start_index: start index of batch data
    :param last_index: last index of bacth data
    :param embedding_array: array of word vectors
    :param training data: training data


    Returns:
    --------
    :return: feature and label array
    """
    y_data=[]
    x_data=[]

    ## for each question pair, create vector of length (1,600) dimension
    for i in range(start_index,last_index):

        wordId1List = training_data[i][0]
        wordId2List = training_data[i][1]

        # create vectors of all the words of question1
        x1 = embedding_array[wordId1List,:]

        # create vectors of all the words of question1
        x2 = embedding_array[wordId2List,:]

        # Take mean of all the vectors of the words of quesiton2 and reduces it to (1,300) dimension (normalization of vectors
        # of all words of question1)
        x1_in = np.mean(x1,axis=0)

        # Take mean of all the vectors of the words of quesiton2 and reduces it to (1,300) dimension (normalization of vectors
        # of all words of question2)
        x2_in = np.mean(x2,axis=0)

        ## concatenate normalized vectors of question1 and question2, gives (1,600) dimension vectors
        # this 600 vector represent similarity between question pairs
        x_d = np.concatenate([x1_in,x2_in])

        # append vector of (1,600) dimension to x_data
        x_data.append(x_d)

        # append example's label to y_data
        y_data.append([training_data[i][2]])


    x_data = np.array(x_data)
    y_data = np.array(y_data)

    return x_data,y_data

def dev_model(sess,data,batch_size,embedding_array,out_prob,cost,accuracy,file_location,x,y):
    """
    compute accuracy of dev data on current state of trained model

    """

    start_index = 0
    last_index = batch_size
    no_of_Batch = int(len(data) / batch_size)

    total_prediction = 0.0
    total_cost = 0.0
    b_count = 0
    output = []
    a = 0.0

    #data=sample(data, len(data))

    for i in range(no_of_Batch):
        batch_x, batch_y = next_data(start_index, last_index, embedding_array, data)

        # Run optimization op (backprop) and cost op (to get loss value)
        out_sim, c , prediction = sess.run([out_prob, cost, accuracy], feed_dict={x: batch_x, y: batch_y})
        b_count += 1
        total_prediction += prediction
        total_cost += c

        output.append(out_sim)

        start_index=last_index
        last_index+=batch_size


    prediction = total_prediction/(1.0*(b_count))
    c = total_cost/(1.0*(b_count))
    #a = a/(1.0*(bCount))


    print("********************************************")
    print("Test Accuracy = "+str(prediction))
    print("Test Cost = " + str(c))
    #print("Output Accuracy = "+str(a))
    print("********************************************")

    file_location.write("********************************************")
    file_location.write("Test Accuracy = "+str(prediction))
    file_location.write('\n')
    file_location.write("Test Cost = " + str(c))
    file_location.write("********************************************")


def main():


    error_out_file=open(sys.argv[6]+os.sep+'LEARNING_RATE_'+str(sys.argv[1]+'_'+'TRAINING_EPOCHS_'+sys.argv[2]+'_'+'N_HIDDEN_'+sys.argv[3]+'_'+'BATCH_SIZE_'+sys.argv[4])+'HIDDEN_LOGISTIC.txt',mode='a')

    with open(sys.argv[5]+os.sep+'embedding_array.p', 'rb') as fp:
        embedding_array=pickle.load(fp)

    with open(sys.argv[5]+os.sep+'train_data_modelling.p', 'rb') as fp:
        train_data=pickle.load(fp)

    with open(sys.argv[5]+os.sep+'dev_data.p', 'rb') as fp:
        dev_data = pickle.load(fp)

    ######### Building multilayer perceptron neural network

    # Parameters

    learning_rate = float(sys.argv[1])
    training_epochs = int(sys.argv[2])
    #batch_size = 100
    display_step = 1


    # Network Parameters
    n_hidden_1 = int(sys.argv[3])  # 1st layer number of features
    n_input = 600  # question words data have input (img shape: 300+300)
    n_classes = 1  # probability value

    batchSize = int(sys.argv[4])

    # tf Graph Input
    x = tf.placeholder(tf.float32, [None, n_input])  # normalized word vector dimension 300+300=600
    #x = tf.Variable(tf.float32, [batchSize, n_input])  # normalized word vector dimension 300+300=600
    y = tf.placeholder(tf.float32, [None, n_classes])  # probability value

    w1 = tf.Variable(tf.random_normal([n_input, n_hidden_1]))
    b1 = tf.Variable(tf.random_normal([n_hidden_1]))
    w2 = tf.Variable(tf.random_normal([n_hidden_1,n_classes]))
    b2 = tf.Variable(tf.random_normal([n_classes]))


    pre_activation = tf.matmul(x, w1) + b1

    # recitified linear unit to squash input between the range of (0,max_value)
    activation = tf.nn.relu(pre_activation)
    logit = tf.matmul(activation,w2) + b2

    # sigmoid function to calultae probability
    out_prob = tf.nn.sigmoid(logit)

    # store ouput label: if class is of label1 or label 0
    outputLabel = tf.cast(out_prob > 0.5, tf.float32)

    # Output layer with linear activation
    out = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logit)

    # Define loss and optimizer
    cost = tf.reduce_mean(out)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    #Launch the graph
    accuracy = tf.reduce_mean(tf.cast(tf.equal(outputLabel,y),tf.float32))



    # Initializing the variables
    init = tf.global_variables_initializer()

    # Create a saver object
    saver = tf.train.Saver()


    # Construct model
    with tf.Session() as sess:
        sess.run(init)

        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.

            ## shuffle training data
            train_data=sample(train_data, len(train_data))

            start_index=0
            last_index=batchSize
            no_of_batch = int(len(train_data)/batchSize)

            # Loop over all batches
            for i in range(no_of_batch):

                batch_x, batch_y = next_data(start_index,last_index,embedding_array,train_data)

                # Run optimization op (backprop) and cost op (to get loss value)
                _, c, prediction = sess.run([optimizer, cost, accuracy], feed_dict={x: batch_x,y: batch_y})
                # Compute average loss
                #avg_cost += c / total_batch
                #avg_cost += c / (1.0*batchSize)
                #print("Iteration:", '%04d' % (i + 1), "cost=","{:.9f}".format(c), "Prediction=", "{:.9f}".format(prediction) )
                #error_out_file.write("Iteration:"+str(i+1)+' cost='+str(c) + " prediction = " + str(prediction))
                #error_out_file.write('\n')
                start_index=last_index
                last_index+=batchSize

            ## Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c), "Prediction=", "{:.9f}".format(prediction))
                #error_out_file.write("Epoch:" + str(epoch+1) + ' cost=' + str(c)+ " prediction" + str(prediction))
                #error_out_file.write('\n')

            dev_model(sess,dev_data,100,embedding_array,out_prob,cost,accuracy,error_out_file,x,y)

        #Now, save the graph
        saver.save(sess, sys.argv[6]+os.sep+str(sys.argv[1]+sys.argv[2]+sys.argv[3]+sys.argv[4]+'_hidden_layer_logistic'))
        print("Optimization Finished!")


        # calling function to test accuracy on dev data on current state of trained model
        dev_model(sess,dev_data,100,embedding_array,out_prob,cost,accuracy,error_out_file,x,y)






if __name__ == '__main__':
    main()
