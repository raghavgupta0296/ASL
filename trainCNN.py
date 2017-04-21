import numpy as np
import tensorflow as tf
import cv2

im_ht = 100
im_wid = 100
no_labels = 2

x = []
y = []

filepath = "./TrainingImages/"
filepaths = []
letters = [chr(i) for i in range(65,67)] #91
for l in letters:
    for num in range(1,21,1):
        filepaths.append(filepath+l+"/"+"%s.png"%str(num))
        y.append(l)

for f in filepaths:
    im = cv2.imread(f,-1)
    im = cv2.resize(im,(100,100))
    x.append(im)
x = np.array(x)
y= np.array(y)

def one_hot(y):
    al2num = dict(zip(letters,range(len(letters))))
    new_y = np.zeros(shape=(len(y),len(letters)))
    for i in range(len(y)):
        new_y[i][al2num[y[i]]] = 1
    return new_y
y = one_hot(y)

x = np.reshape(x,(x.shape[0],x.shape[1],x.shape[2],1))
print(x.shape,y.shape)

X = tf.placeholder(tf.float32,shape=[None,im_ht,im_wid,1])
Y = tf.placeholder(tf.float32,shape=[None,no_labels])
dropout_keepProb = tf.placeholder(tf.float32)

def ini_wt(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)
def ini_bias(shape):
    initial = tf.constant(0.1,shape = shape)
    return tf.Variable(initial)

def conv_2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
def max_pool(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

# conv-pool layer1
W1 = ini_wt([5,5,1,32])
b1 = ini_bias([32])
conv = tf.nn.relu(conv_2d(X,W1)+b1)
maxPool = max_pool(conv)

# conv-pool layer2
W2 = ini_wt([5,5,32,64])
b2 = ini_bias([64])
conv = tf.nn.relu(conv_2d(maxPool,W2)+b2)
maxPool = max_pool(conv)

# Dense Layer1
W3 = ini_wt([int(im_ht/4*im_wid/4*64),1024])
b3 = ini_bias([1024])
r = tf.reshape(maxPool,[-1,int(im_ht/4*im_wid/4*64)])
r = tf.nn.relu(tf.matmul(r,W3)+b3)

# dropout
r = tf.nn.dropout(r,keep_prob=dropout_keepProb)

# Dense Layer2 - label
W4 = ini_wt([1024,no_labels])
b4 = ini_bias([no_labels])
r = tf.matmul(r,W4)+b4

# Loss
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y,logits=r))
# Optimizer
optimizer = tf.train.AdamOptimizer().minimize(cross_entropy)

accuracy = tf.equal(tf.argmax(r,1),tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(accuracy,tf.float32))

saver = tf.train.Saver()
best_acc = 0

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(100):
        print("Epoch : %d / 100 "%i)
        optimizer.run(feed_dict={X:x,Y:y,dropout_keepProb:0.5})
        if i%20==0:
            print ("\n Training Acc : ")
            acc = sess.run(accuracy,feed_dict={X:x,Y:y,dropout_keepProb:1})
            print (acc*100,"%")
            if(acc>best_acc):
                best_acc = acc
                saver.save(sess,"./tfWeights") #./output/tfWeights
