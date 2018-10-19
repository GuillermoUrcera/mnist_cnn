# -*- coding: utf-8 -*-
"""
GUILLERMO URCERA MARTÍN
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Constants
LOGS_PATH="/tmp/mnist_cnn_logs"
LEARNING_RATE=1e-3
MAX_EPOCH=600
MINIBATCH_SIZE=256
L2_LAMBDA=1e-3
EPOCHS_PER_LOG=10

# Load data
mnist=tf.keras.datasets.mnist
(images_train, labels_train_1d),(images_test, labels_test_1d)=mnist.load_data()
image_size=len(images_train[0])
train_elements=len(labels_train_1d)
test_elements=len(labels_test_1d)

# Take a look at the data
fig, axes = plt.subplots(nrows=1, ncols=2)

axes[0].hist(labels_train_1d,rwidth=0.8)
axes[0].set_xlabel("Number")
axes[0].set_ylabel("Count")
axes[0].set_title("Train data")

axes[1].hist(labels_test_1d,rwidth=0.8)
axes[1].set_xlabel("Number")
axes[1].set_ylabel("Count")
axes[1].set_title("Test data")

# Let's render a sample image
fig=plt.figure()
index=np.random.randint(train_elements)
image=images_train[index]
plt.imshow(image)
print "Shape:",image.shape

# Normalize data
images_train=images_train/255.0
images_test=images_test/255.0

labels_train=np.zeros((train_elements,10))
labels_train[np.arange(train_elements),labels_train_1d]=1

labels_test=np.zeros((test_elements,10))
labels_test[np.arange(test_elements),labels_test_1d]=1

# Tensorflow init
tf.reset_default_graph()
sess=tf.Session()

# Create CNN
input_tensor=tf.placeholder(tf.float32, shape=(None,image_size,image_size,1),name="input_tensor")
label_tensor=tf.placeholder(tf.float32,shape=(None,10),name="label_tensor")

conv1=tf.layers.conv2d(inputs=input_tensor,filters=64,kernel_size=3,padding='valid',activation=tf.nn.relu,name="conv_layer_1")
pool1=tf.layers.max_pooling2d(conv1,2,2,name="max_pool_layer_1")
conv2=tf.layers.conv2d(inputs=pool1,filters=128,kernel_size=3,padding='valid',activation=tf.nn.relu,name="conv_layer_2")
pool2=tf.layers.max_pooling2d(conv2,2,2,name="max_pool_layer_2")
flat=tf.layers.flatten(pool2)
dense1=tf.layers.dense(flat,256,activation=tf.nn.relu,name="dense_layer_1")
logits=tf.layers.dense(dense1,10,activation=None,name="logits")
output=tf.nn.softmax(logits)

weights=tf.trainable_variables() 
lossL2=tf.add_n([tf.nn.l2_loss(v) for v in weights if 'bias' not in v.name ])*L2_LAMBDA
cost=tf.losses.softmax_cross_entropy(onehot_labels=label_tensor,logits=logits)+lossL2

train=tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)

# Tensorboard
writer=tf.summary.FileWriter(LOGS_PATH,sess.graph)
loss_ph=tf.placeholder('float',name='Training_loss')
loss_sum=tf.summary.scalar("Loss", loss_ph)
accuracy_ph=tf.placeholder('float',name='Accuracy')
accuracy_sum=tf.summary.scalar("Test_accuracy", accuracy_ph)
test_loss_ph=tf.placeholder('float',name='test_loss')
test_loss_sum=tf.summary.scalar("Test_loss", test_loss_ph)
summaryMerged=tf.summary.merge_all()
saver = tf.train.Saver()

# Finalise tensorflow init
init_op=tf.global_variables_initializer()
tf.get_default_graph().finalize()
sess.run(init_op)

images_train=np.reshape(images_train,(train_elements,image_size,image_size,1))
images_test=np.reshape(images_test,(test_elements,image_size,image_size,1))

# Generate minibatches
def generate_minibatches(images,labels):
    index=np.random.permutation(len(labels_train))
    x=images_train[index]
    y=labels_train[index]
    minibatch_list=[]
    for i in range(len(y)/MINIBATCH_SIZE):
        minibatch_list.append([x[i*MINIBATCH_SIZE:i*MINIBATCH_SIZE+MINIBATCH_SIZE,:,:],y[i*MINIBATCH_SIZE:i*MINIBATCH_SIZE+MINIBATCH_SIZE]])
    minibatch_list.append([x[i*MINIBATCH_SIZE:len(x),:,:],y[i*MINIBATCH_SIZE:len(y)]])
    return minibatch_list

acc_loss=0
for epoch in range(MAX_EPOCH):
    minibatch_list=generate_minibatches(images_train,labels_train)
    for minibatch in minibatch_list:
        minibatch_X,minibatch_Y=minibatch
        loss,_=sess.run([cost,train],feed_dict={input_tensor:minibatch_X,label_tensor:minibatch_Y})
        acc_loss+=loss
    if epoch%EPOCHS_PER_LOG==0:
        print "Epoch",epoch
        # Record accumulated loss
        mean_loss=float(acc_loss)/(EPOCHS_PER_LOG*train_elements)
        summary_loss=sess.run(loss_sum,feed_dict={loss_ph:mean_loss})
        writer.add_summary(summary_loss,epoch)
        acc_loss=0
        # Record test loss
        test_loss=sess.run(cost,feed_dict={input_tensor:images_test,label_tensor:labels_test})/float(test_elements)
        summary_test_loss=sess.run(test_loss_sum,feed_dict={test_loss_ph:test_loss})
        writer.add_summary(summary_test_loss,epoch)
        # Record test accuracy
        predicted_value=np.argmax(sess.run(output,feed_dict={input_tensor:images_test}),-1)
        test_accuracy=1-float(np.count_nonzero(predicted_value-labels_test_1d))/test_elements             
        summary_accuracy=sess.run(accuracy_sum,feed_dict={accuracy_ph:test_accuracy})
        writer.add_summary(summary_accuracy,epoch)

        
    
    
    
