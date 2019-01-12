import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import random






def plot(samples):
    fig = plt.figure()
    plt.gca().set_color_cycle(['blue', 'red','green', 'black'])
    plt.plot(samples[0],linewidth=2.0)
    plt.show()


    return fig

## Noise for the GAN

def sample_Z(m, n):
    return np.random.uniform(-100., 100., size=[m, n])





## Load the Data

npzfile = np.load("xSet.npz")        
Train=  npzfile["train"]
Add = npzfile["add"]

## Batch Size
mb_size = 20


## Noise Dimension
Z_dim = 10000
X_dim = Train.shape[1]

## Number of epochs
num_epochs = 100000

y_dim = Add.shape[1]

## Hidden dimensions
h_dim = 1000
h2_dim = 500
h3_dim = 250
random.seed()

## Learning Rate
lr = 0.1


## For putting outputs in a specific directory
if not os.path.exists('out/'):
    os.makedirs('out/')

i = 0


## Create random batches

def rand_batch(size):
    global Train
    global Add
    s_size = Train.shape[0]
    mybatch = []
    count = 0
    X_mb = []
    y_mb = []
    while count < size:
        rn = random.randint(0,s_size-1)
        if rn not in mybatch:
            mybatch.append(rn)
            count +=1
    for i in mybatch:
        X_mb.append(Train[i])
        y_mb.append(Add[i])

    return (X_mb,y_mb)



def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


""" Discriminator Net model """
X = tf.placeholder(tf.float32, shape=[None, X_dim])
y = tf.placeholder(tf.float32, shape=[None, y_dim])

D_W1 = tf.Variable(xavier_init([X_dim + y_dim, h_dim]))
D_b1 = tf.Variable(tf.ones(shape=[h_dim]))



D_W2 = tf.Variable(xavier_init([h_dim, h2_dim]))
D_b2 = tf.Variable(tf.ones(shape=[h2_dim]))

D_W3 = tf.Variable(xavier_init([h2_dim, h3_dim]))
D_b3 = tf.Variable(tf.ones(shape=[h3_dim]))


D_W4 = tf.Variable(xavier_init([h3_dim, 1]))
D_b4 = tf.Variable(tf.ones(shape=[1]))

theta_D = [D_W1, D_W2, D_W3, D_W4, D_b1, D_b2, D_b3, D_b4]



def discriminator(x, y):
    inputs = tf.concat(axis=1, values=[x, y])
    D_h1 = tf.nn.tanh(tf.matmul(inputs, D_W1) + D_b1)
    D_h2 = tf.nn.tanh(tf.matmul(D_h1, D_W2) + D_b2)
    D_h3 = tf.nn.tanh(tf.matmul(D_h2, D_W3) + D_b3)
    D_logit = tf.matmul(D_h3, D_W4) + D_b4
    D_prob = tf.nn.sigmoid(D_logit)
    return  D_prob,D_logit






""" Generator Net model """
Z = tf.placeholder(tf.float32, shape=[None, Z_dim])

G_W1 = tf.Variable(xavier_init([Z_dim + y_dim, h_dim]))
G_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

G_W2 = tf.Variable(xavier_init([h_dim, h2_dim]))
G_b2 = tf.Variable(tf.zeros(shape=[h2_dim]))

G_W3 = tf.Variable(xavier_init([h2_dim, h3_dim]))
G_b3 = tf.Variable(tf.zeros(shape=[h3_dim]))

G_W4 = tf.Variable(xavier_init([h3_dim, X_dim]))
G_b4 = tf.Variable(tf.zeros(shape=[X_dim]))

theta_G = [G_W1, G_W2, G_W3, G_W4, G_b1, G_b2, G_b3, G_b4]


def generator(z, y):
    inputs = tf.concat(axis=1, values=[z, y])
    G_h1 = tf.nn.tanh(tf.matmul(inputs, G_W1) + G_b1)
    G_h2 = tf.nn.tanh(tf.matmul(G_h1, G_W2) + G_b2)
    G_h3 = tf.nn.tanh(tf.matmul(G_h2, G_W3) + G_b3)
    G_log_prob = tf.matmul(G_h3,G_W4)+G_b4
    G_prob = tf.nn.sigmoid(G_log_prob)

    return G_log_prob





G_sample = generator(Z, y)

D_real, D_logit_real = discriminator(X, y)
D_fake, D_logit_fake = discriminator(G_sample, y)

D_loss_real = tf.reduce_mean(- (tf.log((1 - D_fake)+1e-10)+tf.log(D_real+1e-10)  ))

D_loss_fake = tf.reduce_mean(- tf.log(D_fake+1e-10))
D_loss = D_loss_real
G_loss = D_loss_fake

D_solver = tf.train.AdagradOptimizer(learning_rate=lr).minimize(D_loss, var_list=theta_D)
G_solver = tf.train.AdagradOptimizer(learning_rate=lr).minimize(G_loss, var_list=theta_G)






with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for epoch_idx in range(num_epochs):
        if epoch_idx % 10000 == 0:
            n_sample = 1

            Z_sample = sample_Z(n_sample, Z_dim)
            y_sample = np.ones(shape=[n_sample, y_dim])
            y_sample[0][0] = 0.0
            y_sample[0][1] = 50.0
            samples = sess.run(G_sample, feed_dict={Z: Z_sample, y:y_sample})
            print samples
            fig = plot(samples)
        X_mb, y_mb = rand_batch(mb_size)

        Z_sample = sample_Z(mb_size, Z_dim)

        A,B = sess.run([D_solver, D_loss], feed_dict={X: X_mb, Z: Z_sample, y:y_mb})
        C,D = sess.run([G_solver, G_loss], feed_dict={Z: Z_sample, y:y_mb})
        
        if epoch_idx % 100 == 0:
            print('Iter: {}'.format(epoch_idx))
            print('D loss: {}'.format(B))
            print('G loss: {}'.format(D))
            print()
            print(D_W1.eval())
            print(G_W1.eval())




        

    
    

        


