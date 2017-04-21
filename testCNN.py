import numpy as np
import tensorflow as tf
import cv2

class testing:

    def ini_wt(self,shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)
    def ini_bias(self,shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)
    def conv_2d(self,x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    def max_pool(self,x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def __init__(self):
        self.im_ht = 100
        self.im_wid = 100
        self.no_labels = 2

        self.W1 = self.ini_wt([5, 5, 1, 32])
        self.b1 = self.ini_bias([32])
        self.W2 = self.ini_wt([5, 5, 32, 64])
        self.b2 = self.ini_bias([64])
        self.W3 = self.ini_wt([int(self.im_ht / 4 * self.im_wid / 4 * 64), 1024])
        self.b3 = self.ini_bias([1024])
        self.W4 = self.ini_wt([1024, self.no_labels])
        self.b4 = self.ini_bias([self.no_labels])

        init_op = tf.global_variables_initializer()
        saver = tf.train.Saver()

        self.sess = tf.Session()
        self.sess.run(init_op)
        saver.restore(self.sess, "./tfWeights")
        
    def test_im(self,im):
        im = cv2.resize(im,(self.im_ht,self.im_wid))
    
        X = tf.placeholder(tf.float32, shape=[None, self.im_ht, self.im_wid, 1])
    
        im = np.reshape(im,(1,im.shape[0],im.shape[1],1))

        # conv-pool layer1
        conv = tf.nn.relu(self.conv_2d(X, self.W1) + self.b1)
        maxPool = self.max_pool(conv)

        # conv-pool layer2
        conv = tf.nn.relu(self.conv_2d(maxPool, self.W2) + self.b2)
        maxPool = self.max_pool(conv)

        # Dense Layer1
        r = tf.reshape(maxPool, [-1, int(self.im_ht / 4 * self.im_wid / 4 * 64)])
        r = tf.nn.relu(tf.matmul(r, self.W3) + self.b3)

        # Dense Layer2 - label
        r = tf.matmul(r, self.W4) + self.b4

        r = tf.argmax(r,1)[0]

        r = self.sess.run(r,feed_dict={X:im})

        letters = [chr(i) for i in range(65, 67)]  # 91
        num2al = dict(zip(range(len(letters)),letters))
        r = num2al[r]
        print (r)

if __name__ == '__main__':
    im = cv2.imread("20.png",-1)
    t = testing()
    t.test_im(im)