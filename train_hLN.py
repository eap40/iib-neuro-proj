import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import urllib
from sim_hLN import *
from utils import *

class Model(object):
  def __init__(self):
    # Initialize the weights to `5.0` and the bias to `0.0`
    # In practice, these should be initialized to random values (for example, with `tf.random.normal`)
    self.W = tf.Variable(5.0)
    self.b = tf.Variable(0.0)
    self.params = [self.W, self.b]

  def __call__(self, x):
    return self.W * x + self.b

model = Model()

assert model(3.0).numpy() == 15.0

###hLN model training, following same structure as above###

class hLN_model(object):
  #   will need fleshing out/editing according to form of sim_hLN/general Python class for hLN model
  def __init__(self):
    # Initialize the parameters in some way
    # In practice, these should be initialized to random values (for example, with `tf.random.normal`)
    self.Jw = 0
    self.Wwe = tf.Variable(0.0)
    self.Wwi = tf.Variable(0.0)
    self.Tau_e = 0
    self.Tau_i = 0
    self.Th = 0
    self.Delay = 0
    self.params = [self.Jw, self.Wwe, self.WWi, self.Tau_e, self.Tau_i, self.Th, self.Delay]


  def __call__(self, x):
    return sim_hLN(X=x, dt=1, params=self.params)


# define loss and gradient function

def loss(predicted_v, target_v):
    """loss function is mean squared error between hLN model output and target membrane potential"""
    return tf.reduce_mean(tf.square(predicted_v - target_v))

# use gradient tape to calculate gradients used to optimise model
def grad(model, inputs, targets):
    """find value of loss function and its gradient with respect to the trainable parameters of the model"""
    with tf.GradientTape() as tape:
        loss_value = loss(model(inputs), targets)
    return loss_value, tape.gradient(loss_value, model.params)

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.1)

# first try new syntax with same regression problem as before - i.e. using gradient tape, optimiser etc.

TRUE_W = 3.0
TRUE_b = 2.0
NUM_EXAMPLES = 1000

inputs = tf.random.normal(shape=[NUM_EXAMPLES])
noise = tf.random.normal(shape=[NUM_EXAMPLES])
outputs = inputs * TRUE_W + TRUE_b + noise

Ws, bs = [], []

epochs = range(20)
for epoch in epochs:
    Ws.append(model.W.numpy())
    bs.append(model.b.numpy())

    loss_value, grads = grad(model, inputs, outputs)
    optimizer.apply_gradients(zip(grads, model.params))










