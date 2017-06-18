import tensorflow as tf
import numpy as np
import os
import sys
import pickle
from Multilayer_Perceptron import next_data


def testModel(sess,data,batch_size,embedding_array,outProb,cost,accuracy,file_location,x,y,outDir):

    """
    compute similar probabilities for test data and writing it into the file

    """

    start_index = 0
    last_index = batch_size
    no_of_batch = int(len(data) / batch_size)

    total_cost = 0.0
    b_count = 0
    output = []
    id_list = []

    for i in range(no_of_batch):
        print(i)
        batch_x, batch_y = next_data(start_index, last_index, embedding_array, data)
        #batch_y is questionID list

        # Test Model
        out_sim, c = sess.run([outProb, cost], feed_dict={x: batch_x,y:batch_y})
        b_count += 1
        total_cost += c
        output.append(out_sim)
        id_list.extend(batch_y)

        start_index=last_index
        last_index+=batch_size

    c = total_cost/(1.0*(b_count))
    output = np.concatenate(output,axis=0)
    idList = np.array(id_list)

    result = np.hstack([idList,output])

    ## saving question id and probabilities into file
    fileName = outDir + os.sep + "test_probabilities.txt"
    np.savetxt(fileName,result,fmt=['%d','%0.4f'],delimiter=',',header='test_id,is_duplicate')

    print("********************************************")
    print("Test Cost = " + str(c))
    print("********************************************")

    file_location.write("********************************************")
    file_location.write('\n')
    file_location.write("Test Cost = " + str(c))
    file_location.write("********************************************")



def main():

    out_file=open(sys.argv[3]+os.sep+'_Model_Restore_20_epoch.txt',mode='a')

    with open(sys.argv[1]+os.sep+'embedding_array.p', 'rb') as fp:
        embedding_array=pickle.load(fp)

    with open(sys.argv[1]+os.sep+'test_data.p', 'rb') as fp:
        test_data = pickle.load(fp)


    # Network Parameters
    n_hidden_1=1000
    #n_hidden_1 = sys.argv[3]  # 1st layer number of features
    n_input = 600  # question words data have input (img shape: 300+300)
    n_classes = 1  # probability value

    #batchSize = sys.argv[4]
    batchSize=10
    # tf Graph Input
    x = tf.placeholder(tf.float32, [None, n_input])  # normalized word vector dimension 300+300=600
    #x = tf.Variable(tf.float32, [batchSize, n_input])  # normalized word vector dimension 300+300=600
    y = tf.placeholder(tf.float32, [None, n_classes])  # probability value

    w1 = tf.Variable(tf.random_normal([n_input, n_hidden_1]))
    b1 = tf.Variable(tf.random_normal([n_hidden_1]))
    w2 = tf.Variable(tf.random_normal([n_hidden_1,n_classes]))
    b2 = tf.Variable(tf.random_normal([n_classes]))


    pre_activation = tf.matmul(x, w1) + b1
    activation = tf.nn.relu(pre_activation)
    logit = tf.matmul(activation,w2) + b2

    out_prob = tf.nn.sigmoid(logit)
    output_label = tf.cast(out_prob > 0.5, tf.float32)

    # Output layer with linear activation
    out = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logit)

    # Define loss and optimizer
    cost = tf.reduce_mean(out)

    #Launch the graph
    accuracy = tf.reduce_mean(tf.cast(tf.equal(output_label,y),tf.float32))

    init = tf.global_variables_initializer()

    saver = tf.train.Saver()


    ## restoring trained model
    print("Starting session...")
    with tf.Session() as sess:
        # Initialize variables
        sess.run(init)

        # Restore model weights from previously saved model
        saver.restore(sess, sys.argv[2]+os.sep+'./0.01201000200_hidden_layer_logistic')
        #saver.restore(sess,'./data/epoch_20/0.01201000200_hidden_layer_logistic')

        print("Model restored from file")
        #print(sess.run(w1))

        testModel(sess,test_data,batchSize,embedding_array,out_prob,cost,accuracy,out_file,x,y,sys.argv[3])

if __name__ == '__main__':
    main()