import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import urllib
from sim_hLN import *
from utils import *

class RegressionModel(object):
  def __init__(self):
    # Initialize the weights to `5.0` and the bias to `0.0`
    # In practice, these should be initialized to random values (for example, with `tf.random.normal`)
    self.W = tf.Variable(5.0)
    self.b = tf.Variable(0.0)
    self.params = (self.W, self.b)

  def __call__(self, x):
    return self.W * x + self.b

model = RegressionModel()

assert model(3.0).numpy() == 15.0

###hLN model training, following same structure as above###

class hLN_Model(object):
    #   will need fleshing out/editing according to form of sim_hLN/general Python class for hLN model
    # to define hLN model we just need its structure (Jc) and how the input neurons connect to its subunits
    def __init__(self, Jc, Wci, Wce):
        # Initialize the parameters in some way
        # In practice, these should be initialized to random values (for example, with `tf.random.normal`)
        M = len(Jc)
        self.Jc, self.Wci, self.Wce = Jc, Wci, Wce
        self.Jw = np.full([M,1], 1) #coupling weights 1 for all branches intially
        self.Wwe = [np.ones(X_e.shape[0])]
        self.Wwi = [np.full(X_i.shape[0], -1)]
        self.Tau_e = [np.full(X_e.shape[0], 1)]
        self.Tau_i = [np.full(X_i.shape[0], 1)]
        self.Th = [1]
        self.Delay = 0
        self.v0 = 0
        self.params = [self.v0, self.Jw, self.Wwe, self.Wwi, self.Tau_e, self.Tau_i, self.Th]

    def __call__(self, x):
        return sim_hLN(X=x, dt=1, Jc=self.Jc, Wce=self.Wce, Wci=self.Wci, params=self.params)


# define loss and gradient function

def loss(predicted_v, target_v):
    """loss function is mean squared error between hLN model output and target membrane potential"""
    return tf.reduce_mean(tf.square(predicted_v - target_v))

# use gradient tape to calculate gradients used to optimise model
def grad(model, inputs, targets):
    """find value of loss function and its gradient with respect to the trainable parameters of the model"""
    with tf.GradientTape() as tape:
        tape.watch(model.params)
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

epochs = range(10)
for epoch in epochs:
    Ws.append(model.W.numpy())
    bs.append(model.b.numpy())

    loss_value, grads = grad(model, inputs, outputs)
    optimizer.apply_gradients(zip(grads, model.params))

# plot regression output
# plt.plot(epochs, Ws, 'r',
#          epochs, bs, 'b')
# plt.plot([TRUE_W] * len(epochs), 'r--',
#          [TRUE_b] * len(epochs), 'b--')
# plt.legend(['W', 'b', 'True W', 'True b'])
# plt.show()


#### now apply same format to hLN model instead of regression model ####

# first we need our inputs and target output
E_spikes, I_spikes = gen_realistic_inputs(Tmax=3000)
X_e = spikes_to_input(E_spikes, Tmax=48000)
X_i = spikes_to_input(I_spikes, Tmax=48000)
X_tot = np.vstack((X_e, X_i)) #this is our final input


# true parameters to produce output:
N_soma = 420
Jc_sing = np.array([0])
M=len(Jc_sing)
Jw_sing = np.full([M,1], 1) #coupling weights 1 for all branches intially
M = len(Jc_sing) #number of subunits
Wce_sing = [np.arange(0, X_e.shape[0], 1)] #all input excitatory neurons connected to root subunit
Wwe_sing = [np.ones(X_e.shape[0])] #weighting matrix - all excitatory neurons connected with weight 1
Wci_sing = [np.arange(N_soma, N_soma + X_i.shape[0] -1, 1)] #all input inhibitory neurons connected to root subunit
Wwi_sing = [np.full(X_i.shape[0], -1)] #weighting matrix - all inhibitory neurons connected with weight -1
Tau_e = [np.full(X_e.shape[0], 1)] #all excitatory time constants 1
Tau_i = [np.full(X_i.shape[0], 1)] #all inhibitory time constants 1
Th = [1] #no offset in all sigmoids
v0 = 0 #no offset in membrane potential

params_sing = [v0, Jw_sing, Wwe_sing, Wwi_sing, Tau_e, Tau_i, Th]

target = sim_hLN(X=X_tot, dt=1, Jc=Jc_sing, Wce=Wce_sing, Wci=Wci_sing, params=params_sing)

# initialise model
hLN_model = hLN_Model(Jc=Jc_sing, Wci=Wci_sing, Wce=Wce_sing)


epochs = range(10)
for epoch in epochs:
    # loss_value, grads = grad(model=hLN_model, inputs=X_tot, targets=target)
    # optimizer.apply_gradients(zip(grads, model.params))
    model_output = hLN_model(X_tot)
    loss_value = loss(model_output, target)
    mse = (np.square(model_output - target)).mean(axis=None)
    print(f"Loss value = {loss_value}, MSE = {mse}")


# plt.plot(model_output[4900:5000], label="Model Output")
# plt.plot(target[4900:5000], label="Target")
# plt.title("Root subunit response to in vivo like input patterns")
# plt.xlabel("Time (ms)")
# plt.ylabel("Root subunit response")
# plt.ylim(bottom=-0.1)
# plt.legend()
# plt.show()



