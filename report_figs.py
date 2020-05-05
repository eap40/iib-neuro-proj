### File for creating output to use for Figures in final report. Small runs usually so should be able to run alongside
### big training runs on fields, still much quicker than doing on laptop

from train_hLN import *
from gen_inputs import *

def run():
    """Procedure to get figure data here"""

    # try the following GPU code to fix problem with cudnn errors
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    # Lets generate some inputs this time
    E_spikes, I_spikes = gen_realistic_inputs(Tmax=3000)
    #
    X_e = spikes_to_input(E_spikes, Tmax=48000)
    X_i = spikes_to_input(I_spikes, Tmax=48000)
    X_tot = np.vstack((X_e, X_i))

    # X_tot = tf.convert_to_tensor(np.load('Data/real_inputs.npy'), dtype=tf.float32)  # real inputs made earlier
    inputs = tf.convert_to_tensor(X_tot, dtype=tf.float32)

    # define target model for validate_fit function
    # list of lists for to define hierarchical clustering
    clusts = [[[[[0, 1], [2]], [[3, 4], [5, 6]]], [[[7, 8], [9]], [[10, 11], [12]]]]]
    Jc_1l = np.array([0])
    Jc_2n = np.array([0, 1, 1])
    Jc_3n = np.array([0, 1, 1, 2, 2, 3, 3])
    Jc_4n = np.array([0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7])
    Wce_1l, Wci_1l = create_weights(Jc_1l, n_levels=1, clusts=clusts)
    Wce_2n, Wci_2n = create_weights(Jc_2n, n_levels=2, clusts=clusts)
    Wce_3n, Wci_3n = create_weights(Jc_3n, n_levels=3, clusts=clusts)
    Wce_4n, Wci_4n = create_weights(Jc_4n, n_levels=4, clusts=clusts)
    hln_1l = hLN_Model(Jc=Jc_1l, Wce=Wce_1l, Wci=Wci_1l, sig_on=tf.constant([False]))
    hln_1n = hLN_Model(Jc=Jc_1l, Wce=Wce_1l, Wci=Wci_1l, sig_on=tf.constant([True]))
    hln_2n = hLN_Model(Jc=Jc_2n, Wce=Wce_2n, Wci=Wci_2n, sig_on=tf.constant([True, True, True]))
    hln_3n = hLN_Model(Jc=Jc_3n, Wce=Wce_3n, Wci=Wci_3n, sig_on=tf.constant([True, True, True,
                                                                             True, True, True, True]))

    Wce_sing, Wci_sing = np.array([np.array([0])], dtype=np.int32), np.array([np.array([])], dtype=np.int32)

    # training_build showed we could get good results from simple delta input - now try with SSSS model and real inputs
    # could run into problems as now we're doing minibatch training due to long time series
    hln_adam_1s = hLN_Model(Jc=Jc_1l, Wce=Wce_sing, Wci=Wci_sing, sig_on=tf.constant([False]))
    train_accs_1s, test_accs_1s, trained_plist_1s, target_plist_1s = test_recovery(model=hln_adam_1s,
                                                                                   inputs=inputs, num_sims=20,
                                                                                   n_attempts=1, num_epochs=20000,
                                                                                   learning_rate=0.001)

    # save data
    np.savez_compressed('/scratch/eap40/tr_adam_1s', a=train_accs_1s, b=test_accs_1s, c=trained_plist_1s,
                        d=target_plist_1s, e=inputs)

    return


def test_recovery(model, inputs, num_sims, n_attempts, num_epochs, learning_rate, enforce_params=False):
    """Function to test quality of parameter recovery for a a specified hLN model. For each simulation, function
    will generate a new target based on random parameters and attempt to fit. The final quality of fit will then
    be evaluated, as well as statistics concerning parameter recovery."""

    # split input data into training and test sets, 80/20 initially
    split = 0.8
    L = inputs.shape[1]
    n_train = int(L * split)
    train_inputs = inputs[:, :n_train]
    test_inputs = inputs[:, n_train:]

    # create empty lists to store stats we want function to return
    train_accuracies = []
    test_accuracies = []
    target_params_list = []
    trained_params_list = []

    # generate a number of different targets
    for sim in range(num_sims):
        # randomise the parameters before generating target
        model.randomise_parameters()

        # if we want to enforce some target parameters then do so here
        if enforce_params:
            model.logTaue.assign(np.log(np.full(2, np.random.uniform(low=10, high=20))))

        # generate target with new parameters
        train_target = model(train_inputs)

        # store parameters of the target model, and split the target into training and test sets
        test_target = model(test_inputs)
        target_params = [param.numpy() for param in model.params]
        target_params_list.append(target_params)

        # for each simulation (i.e. each target generated), we have multiple training attempts with different initial
        # conditions each time. We then take the model with the best training accuracy out of these attempts for
        # investigation of parameter recovery
        # n_attempts = 5

        # for each attempt, we want to store the final training accuracy, test accuracy and final model parameters
        attempt_train_accuracies = [0]*n_attempts
        attempt_test_accuracies = [0]*n_attempts
        attempt_parameters = [0]*n_attempts

        for attempt in range(n_attempts):

            # Now try and recover parameters from this model: first randomise them again:
            model.randomise_parameters()

            # Debugging inability to achieve 100% accuracy: randomise one parameter at a time and check recoverability
            # first v0
            # model.v0.assign(tf.random.uniform(shape=(), minval=-0.1, maxval=0.1, dtype=tf.float32))
            # model.Wwe.assign(tf.random.uniform(shape=[model.n_e], minval=0.05, maxval=0.15, dtype=tf.float32))
            # model.Taue = tf.random.uniform(shape=[model.n_e], minval=10, maxval=20, dtype=tf.float32)
            # model.logTaue.assign(tf.math.log(model.Taue))
            # model.Delay = tf.random.uniform(shape=[1], minval=0.1, maxval=5, dtype=tf.float32)
            # model.logDelay.assign(tf.math.log(model.Delay))

            # # define optimizer - vanilla
            # optimizer_1l = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=learning_rate)

            # adam optimizer
            optimizer_adam = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999,
                                                      epsilon=1e-07, amsgrad=False)

            # train model with SGD
            loss_values, accuracies = train_sgd(model=model, num_epochs=num_epochs, optimizer=optimizer_adam,
                                                inputs=train_inputs, target=train_target)

            # # train without SGD on a whole dataset
            # loss_values, accuracies = train(model=model, num_epochs=num_epochs, optimizer=optimizer_adam,
            #                                     inputs=train_inputs, target=train_target)

            # compute final test and training losses, and store for later
            test_loss = loss(predicted_v=model(test_inputs), target_v=test_target)
            test_accuracy = 100 * (1 - (test_loss / np.var(test_target)))
            train_loss = loss(predicted_v=model(train_inputs), target_v=train_target)
            train_accuracy = 100 * (1 - (train_loss / np.var(train_target)))
            attempt_test_accuracies[attempt] = test_accuracy
            attempt_train_accuracies[attempt] = train_accuracy

            # now store parameters
            trained_params = [param.numpy() for param in model.params]
            attempt_parameters[attempt] = trained_params

        # now find attempt that produced maximum training accuracy, and use this for evaluation
        max_index = attempt_train_accuracies.index(max(attempt_train_accuracies))
        trained_params = attempt_parameters[max_index]
        test_accuracies.append(attempt_test_accuracies[max_index])
        train_accuracies.append(attempt_train_accuracies[max_index])

        # save the final trained parameters
        trained_params_list.append(trained_params)



    return train_accuracies, test_accuracies, trained_params_list, target_params_list


if __name__ == '__main__':
    print('Beginning figure procedure')
    run()