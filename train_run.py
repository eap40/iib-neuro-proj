### File for running more complex training procedures, neater than a Jupyter notebook
from train_hLN import *
from gen_inputs import *

def run():
    """Training procedure here"""


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
    E_spikes, I_spikes = gen_realistic_inputs(Tmax=15000)
    #
    X_e = spikes_to_input(E_spikes, Tmax=240000)
    X_i = spikes_to_input(I_spikes, Tmax=240000)
    X_tot = np.vstack((X_e, X_i))

    # X_tot = tf.convert_to_tensor(np.load('Data/real_inputs.npy'), dtype=tf.float32)  # real inputs made earlier
    inputs=tf.convert_to_tensor(X_tot, dtype=tf.float32)


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
    hln_1l_tied = hLN_TiedModel(Jc=Jc_1l, Wce=Wce_1l, Wci=Wci_1l, sig_on=tf.constant([False]))

    # validate_fit function
    target_params_list, trained_params_list = validate_fit(target_model=hln_1n, num_sims=5, inputs=inputs)

    # validate_fit_data function
    # target_params_list, trained_params_list = validate_fit_data(target_model=hln_1l, num_sims=5, inputs=inputs)

    # training debug
    # target_params, trained_params, train_losses, val_losses = debug_training(target_model=hln_1l, inputs=inputs, nSD=1)

    # compare tied model routine
    # tied_train_accuracies, tied_test_accuracies, untied_train_accuracies, untied_test_accuracies = compare_tied(
    #     target_model=hln_2n, untied_model=hln_1l, tied_model=hln_1l_tied, inputs=inputs, num_sims=5, n_attempts=5,
    #                                                                               num_epochs=5000, learning_rate=0.001)

    # save data
    np.savez_compressed('/scratch/eap40/val_tu_1n5', a=target_params_list, b=trained_params_list, c=inputs)

    print("Procedure finished")


def validate_fit(target_model, num_sims, inputs):
    """Function to validate the model fitting procedure, producing output similar to that in Figure S2 of the
    Ujfalussy paper. Finds the performance of different models (1L-4N) in approximating a target signal generated
    by hLN model target_model. Repeats the procedure for num_sims settings of the target model parameters."""

    ### Define the different hLN architectures we will be using:
    # 1L
    Jc_1l = np.array([0])
    # 1N
    Jc_1n = np.array([0])
    # 2N
    Jc_2n = np.array([0, 1, 1])
    # 3N
    Jc_3n = np.array([0, 1, 1, 2, 2, 3, 3])
    # 4N
    Jc_4n = np.array([0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7])

    # list of lists for to define hierarchical clustering
    clusts = [[[[[0, 1], [2]], [[3, 4], [5, 6]]], [[[7, 8], [9]], [[10, 11], [12]]]]]

    Wce_1l, Wci_1l = create_weights(Jc_1l, n_levels=1, clusts=clusts)
    Wce_2n, Wci_2n = create_weights(Jc_2n, n_levels=2, clusts=clusts)
    Wce_3n, Wci_3n = create_weights(Jc_3n, n_levels=3, clusts=clusts)
    Wce_4n, Wci_4n = create_weights(Jc_4n, n_levels=4, clusts=clusts)

    # split input data into training, validation and test sets
    L = inputs.shape[1]
    train_split = 0.7
    n_train = int(L * train_split)
    train_inputs = inputs[:, :n_train]
    val_split = 0.1
    n_val = int(L * val_split)
    val_inputs = inputs[:, n_train:n_train + n_val]
    n_test = L - n_train - n_val
    test_inputs = inputs[:, -n_test:]

    target_params_list = []
    trained_params_list = []

    # define nSDs: when we step up a model, we'll initialise multiple models with different nSDs and then pick the
    # one with lowest training error after training
    nSDs = [4, 8]

    # repeat procedure multiple times
    for sim in range(num_sims):

        # randomise parameters, and generate the target trace
        target_model.randomise_parameters()
        train_target = target_model(train_inputs)
        val_target = target_model(val_inputs)
        test_target = target_model(test_inputs)

        # start off with 1L model, and train until some performance on validation set
        print("Beginning 1L training")
        hln_1l = hLN_Model(Jc=Jc_1l, Wce=Wce_1l, Wci=Wci_1l, sig_on=tf.constant([False]))
        train_until(model=hln_1l, train_inputs=train_inputs, train_target=train_target,
                                                    val_inputs=val_inputs, val_target=val_target)


        # continue procedure with more complex models: 1N:
        print("1L training finished, beginning 1N training")
        hln_1n = hLN_Model(Jc=Jc_1l, Wce=Wce_1l, Wci=Wci_1l, sig_on=tf.constant([True]))
        best_loss_1n = 1000  # initialise best loss big - only save models if they beat the current best loss
        best_params_1n = [param.numpy() for param in hln_1n.params]
        for nSD in nSDs:
            init_nonlin(X=inputs, model=hln_1n, lin_model=hln_1l, nSD=nSD)
            train_until(model=hln_1n, train_inputs=train_inputs, train_target=train_target,
                                                        val_inputs=val_inputs, val_target=val_target)
            final_loss = loss(hln_1n(train_inputs), train_target).numpy()
            if final_loss < best_loss_1n:
                best_loss_1n = final_loss
                best_params_1n = [param.numpy() for param in hln_1n.params]

        for i in range(len(hln_1n.params)):
            hln_1n.params[i].assign(best_params_1n[i])


        # continue procedure with more complex models: 2N:
        print("1N training finished, beginning 2N training")
        hln_2l = hLN_Model(Jc=Jc_2n, Wce=Wce_2n, Wci=Wci_2n, sig_on=tf.constant([True, False, False]))
        update_arch(prev_model=hln_1n, next_model=hln_2l)
        hln_2n = hLN_Model(Jc=Jc_2n, Wce=Wce_2n, Wci=Wci_2n, sig_on=tf.constant([True, True, True]))
        best_loss_2n = 1000  # initialise best loss big - only save models if they beat the current best loss
        best_params_2n = [param.numpy() for param in hln_2n.params]
        for nSD in nSDs:
            init_nonlin(X=inputs, model=hln_2n, lin_model=hln_2l, nSD=nSD)
            train_until(model=hln_2n, train_inputs=train_inputs, train_target=train_target,
                        val_inputs=val_inputs, val_target=val_target)
            final_loss = loss(hln_2n(train_inputs), train_target).numpy()
            if final_loss < best_loss_2n:
                best_loss_2n = final_loss
                best_params_2n = [param.numpy() for param in hln_2n.params]

        for i in range(len(hln_2n.params)):
            hln_2n.params[i].assign(best_params_2n[i])


        # continue procedure with more complex models: 3N:
        print("2N training finished, beginning 3N training")
        hln_3l = hLN_Model(Jc=Jc_3n, Wce=Wce_3n, Wci=Wci_3n, sig_on=tf.constant([True, True, True,
                                                                                 False, False, False, False]))
        update_arch(prev_model=hln_2n, next_model=hln_3l)
        hln_3n = hLN_Model(Jc=Jc_3n, Wce=Wce_3n, Wci=Wci_3n, sig_on=tf.constant([True, True, True,
                                                                                 True, True, True, True]))
        best_loss_3n = 1000  # initialise best loss big - only save models if they beat the current best loss
        best_params_3n = [param.numpy() for param in hln_3n.params]
        for nSD in nSDs:
            init_nonlin(X=inputs, model=hln_3n, lin_model=hln_3l, nSD=nSD)
            train_until(model=hln_3n, train_inputs=train_inputs, train_target=train_target,
                        val_inputs=val_inputs, val_target=val_target)
            final_loss = loss(hln_3n(train_inputs), train_target).numpy()
            if final_loss < best_loss_3n:
                best_loss_3n = final_loss
                best_params_3n = [param.numpy() for param in hln_3n.params]

        for i in range(len(hln_3n.params)):
            hln_3n.params[i].assign(best_params_3n[i])


        # continue procedure with more complex models: 4N:
        print("3N training finished, beginning 4N training")
        hln_4l = hLN_Model(Jc=Jc_4n, Wce=Wce_4n, Wci=Wci_4n, sig_on=tf.constant([True, True, True, True, True, True, True,
                                                                                 False, False, False, False, False, False,
                                                                                 False, False]))
        update_arch(prev_model=hln_3n, next_model=hln_4l)
        hln_4n = hLN_Model(Jc=Jc_4n, Wce=Wce_4n, Wci=Wci_4n, sig_on=tf.constant([True, True, True, True, True, True, True,
                                                                                 True, True, True, True, True, True, True,
                                                                                 True]))
        best_loss_4n = 1000  # initialise best loss big - only save models if they beat the current best loss
        best_params_4n = [param.numpy() for param in hln_4n.params]
        for nSD in nSDs:
            init_nonlin(X=inputs, model=hln_4n, lin_model=hln_4l, nSD=nSD)
            train_until(model=hln_4n, train_inputs=train_inputs, train_target=train_target,
                        val_inputs=val_inputs, val_target=val_target)
            final_loss = loss(hln_4n(train_inputs), train_target).numpy()
            if final_loss < best_loss_4n:
                best_loss_4n = final_loss
                best_params_4n = [param.numpy() for param in hln_4n.params]


        print("4N training finished, procedure ending")

        # # return parameters of all trained models (tied and untied), and parameters of target model
        params_1l = [param.numpy() for param in hln_1l.params]
        target_params = [param.numpy() for param in target_model.params]

        # tied_trained_params = [params_1l, best_params_1n, best_params_2n, best_params_3n, best_params_4n]
        # untied_trained_params = [untied_params_1l, untied_params_1n, untied_params_2n,untied_params_3n,untied_params_4n]

        trained_params = [params_1l, best_params_1n, best_params_2n, best_params_3n, best_params_4n]


        # add recovered and target parameters to list we will later save
        target_params_list.append(target_params)
        trained_params_list.append(trained_params)

    return target_params_list, trained_params_list


def validate_fit_data(target_model, num_sims, inputs):
    """Function to validate the model fitting procedure, producing output similar to that in Figure S2 of the
    Ujfalussy paper. Finds the performance of different models (1L-4N) in approximating a target signal generated
    by hLN model target_model. Repeats the procedure for num_sims settings of the target model parameters. Instead
    of complex procedure of intialising and tying etc, instead uses more input/output data to achieve performance"""

    ### Define the different hLN architectures we will be using:
    # 1L
    Jc_1l = np.array([0])
    # 1N
    Jc_1n = np.array([0])
    # 2N
    Jc_2n = np.array([0, 1, 1])
    # 3N
    Jc_3n = np.array([0, 1, 1, 2, 2, 3, 3])
    # 4N
    Jc_4n = np.array([0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7])

    # list of lists for to define hierarchical clustering
    clusts = [[[[[0, 1], [2]], [[3, 4], [5, 6]]], [[[7, 8], [9]], [[10, 11], [12]]]]]

    Wce_1l, Wci_1l = create_weights(Jc_1l, n_levels=1, clusts=clusts)
    Wce_2n, Wci_2n = create_weights(Jc_2n, n_levels=2, clusts=clusts)
    Wce_3n, Wci_3n = create_weights(Jc_3n, n_levels=3, clusts=clusts)
    Wce_4n, Wci_4n = create_weights(Jc_4n, n_levels=4, clusts=clusts)

    # split input data into training, validation and test sets
    L = inputs.shape[1]
    train_split = 0.7
    n_train = int(L * train_split)
    train_inputs = inputs[:, :n_train]
    val_split = 0.1
    n_val = int(L * val_split)
    val_inputs = inputs[:, n_train:n_train + n_val]
    n_test = L - n_train - n_val
    test_inputs = inputs[:, -n_test:]

    target_params_list = []
    trained_params_list = []

    # repeat procedure multiple times
    for sim in range(num_sims):

        # randomise parameters, and generate the target trace
        target_model.randomise_parameters()
        train_target = target_model(train_inputs)
        val_target = target_model(val_inputs)
        test_target = target_model(test_inputs)

        # start off with 1L model, and train until some performance on validation set
        print("Beginning 1L training")
        # after tied model trained, untie parameters and train until val loss goes down
        hln_1l = hLN_Model(Jc=Jc_1l, Wce=Wce_1l, Wci=Wci_1l, sig_on=tf.constant([False]))
        train_until(model=hln_1l, train_inputs=train_inputs, train_target=train_target,
                    val_inputs=val_inputs, val_target=val_target)


        # continue procedure with more complex models: 1N:
        print("1L training finished, beginning 1N training")
        hln_1n = hLN_Model(Jc=Jc_1l, Wce=Wce_1l, Wci=Wci_1l, sig_on=tf.constant([True]))
        train_until(model=hln_1n, train_inputs=train_inputs, train_target=train_target,
                                                        val_inputs=val_inputs, val_target=val_target)


        # continue procedure with more complex models: 2N:
        print("1N training finished, beginning 2N training")
        hln_2n = hLN_Model(Jc=Jc_2n, Wce=Wce_2n, Wci=Wci_2n, sig_on=tf.constant([True, True, True]))
        train_until(model=hln_2n, train_inputs=train_inputs, train_target=train_target,
                        val_inputs=val_inputs, val_target=val_target)


        # continue procedure with more complex models: 3N:
        print("2N training finished, beginning 3N training")
        hln_3n = hLN_Model(Jc=Jc_3n, Wce=Wce_3n, Wci=Wci_3n, sig_on=tf.constant([True, True, True,
                                                                                 True, True, True, True]))
        train_until(model=hln_3n, train_inputs=train_inputs, train_target=train_target,
                        val_inputs=val_inputs, val_target=val_target)


        # continue procedure with more complex models: 4N:
        hln_4n = hLN_Model(Jc=Jc_4n, Wce=Wce_4n, Wci=Wci_4n, sig_on=tf.constant([True, True, True, True, True, True, True,
                                                                                 True, True, True, True, True, True, True,
                                                                                 True]))
        train_until(model=hln_4n, train_inputs=train_inputs, train_target=train_target,
                        val_inputs=val_inputs, val_target=val_target)


        print("4N training finished, procedure ending")

        # # return parameters of all trained models (tied and untied), and parameters of target model
        params_1l = [param.numpy() for param in hln_1l.params]
        params_1n = [param.numpy() for param in hln_1n.params]
        params_2n = [param.numpy() for param in hln_2n.params]
        params_3n = [param.numpy() for param in hln_3n.params]
        params_4n = [param.numpy() for param in hln_4n.params]

        target_params = [param.numpy() for param in target_model.params]

        trained_params = [params_1l, params_1n, params_2n, params_3n, params_4n]

        # add recovered and target parameters to list we will later save
        target_params_list.append(target_params)
        trained_params_list.append(trained_params)

    return target_params_list, trained_params_list


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

            # define optimizer - vanilla
            optimizer_1l = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=learning_rate)

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


def compare_tied(target_model, untied_model, tied_model, inputs, num_sims, n_attempts, num_epochs, learning_rate):
    """Function to compare training procedures with and without tied models (where synapses share weights and time
    constants"""


    # First compare 1L tied and untied to a 1L target - does the tied model converge faster / get to a better final
    # accuracy
    # split input data into training and test sets, 80/20 initially
    split = 0.8
    L = inputs.shape[1]
    n_train = int(L * split)
    train_inputs = inputs[:, :n_train]
    test_inputs = inputs[:, n_train:]

    # create empty lists to store stats we want function to return
    tied_train_accuracies = []
    untied_train_accuracies = []
    tied_test_accuracies = []
    untied_test_accuracies = []
    target_params_list = []
    trained_params_list = []

    # generate a number of different targets
    for sim in range(num_sims):
        # randomise the parameters before generating target
        target_model.randomise_parameters()

        # generate target with new parameters
        train_target = target_model(train_inputs)

        # store parameters of the target model, and split the target into training and test sets
        test_target = target_model(test_inputs)
        target_params = [param.numpy() for param in target_model.params]
        target_params_list.append(target_params)

        # for each simulation (i.e. each target generated), we have multiple training attempts with different initial
        # conditions each time. We then take the model with the best training accuracy out of these attempts for
        # investigation of parameter recovery

        # for each attempt, we want to store the final training accuracy, test accuracy and final model parameters
        tied_attempt_train_losses = [0]*n_attempts
        tied_attempt_test_losses = [0]*n_attempts
        untied_attempt_train_losses = [0] * n_attempts
        untied_attempt_test_losses = [0] * n_attempts
        # attempt_parameters = [0]*n_attempts

        for attempt in range(n_attempts):

            # Now try and train tied and untied models, first randomise them again:
            untied_model.randomise_parameters()
            tied_model.randomise_parameters()


            # adam optimizer
            untied_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999,
                                                      epsilon=1e-07, amsgrad=False)
            tied_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999,
                                                        epsilon=1e-07, amsgrad=False)

            # train tied model with SGD
            loss_values, accuracies = train_sgd(model=tied_model, num_epochs=num_epochs, optimizer=tied_optimizer,
                                                inputs=train_inputs, target=train_target)

            # train untied model with SGD
            loss_values, accuracies = train_sgd(model=untied_model, num_epochs=num_epochs, optimizer=untied_optimizer,
                                                inputs=train_inputs, target=train_target)


            # compute final test and training losses, and store for later
            tied_test_loss = loss(predicted_v=tied_model(test_inputs), target_v=test_target)
            untied_test_loss = loss(predicted_v=untied_model(test_inputs), target_v=test_target)
            # test_accuracy = 100 * (1 - (test_loss / np.var(test_target)))
            tied_train_loss = loss(predicted_v=tied_model(train_inputs), target_v=train_target)
            untied_train_loss = loss(predicted_v=untied_model(train_inputs), target_v=train_target)
            # train_accuracy = 100 * (1 - (train_loss / np.var(train_target)))
            tied_attempt_test_losses[attempt] = tied_test_loss
            untied_attempt_test_losses[attempt] = untied_test_loss
            tied_attempt_train_losses[attempt] = tied_train_loss
            untied_attempt_train_losses[attempt] = untied_train_loss

            # # now store parameters
            # trained_params = [param.numpy() for param in model.params]
            # attempt_parameters[attempt] = trained_params

        # now find attempt that produced minimum training loss, and use this for evaluation
        tied_min_index = tied_attempt_train_losses.index(min(tied_attempt_train_losses))
        untied_min_index = untied_attempt_train_losses.index(min(untied_attempt_train_losses))

        tied_test_accuracies.append(get_accs(tied_attempt_test_losses[tied_min_index], test_target))
        tied_train_accuracies.append(get_accs(tied_attempt_train_losses[tied_min_index], train_target))
        untied_test_accuracies.append(get_accs(untied_attempt_test_losses[untied_min_index], test_target))
        untied_train_accuracies.append(get_accs(untied_attempt_train_losses[untied_min_index], train_target))

        # # save the final trained parameters
        # trained_params_list.append(trained_params)



    return tied_train_accuracies, tied_test_accuracies, untied_train_accuracies, untied_test_accuracies


def debug_training(target_model, inputs, nSD):
    """Function to perform similar process to validate fit, but returns loss values during training on the
    train and validation set in order to investigate optimizer and hyperparameter investigation"""
    ### Define the different hLN architectures we will be using:
    # 1L
    Jc_1l = np.array([0])
    # 1N
    Jc_1n = np.array([0])
    # 2N
    Jc_2n = np.array([0, 1, 1])
    # 3N
    Jc_3n = np.array([0, 1, 1, 2, 2, 3, 3])
    # 4N
    Jc_4n = np.array([0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7])

    # list of lists for to define hierarchical clustering
    clusts = [[[[[0, 1], [2]], [[3, 4], [5, 6]]], [[[7, 8], [9]], [[10, 11], [12]]]]]

    Wce_1l, Wci_1l = create_weights(Jc_1l, n_levels=1, clusts=clusts)
    Wce_2n, Wci_2n = create_weights(Jc_2n, n_levels=2, clusts=clusts)
    Wce_3n, Wci_3n = create_weights(Jc_3n, n_levels=3, clusts=clusts)
    Wce_4n, Wci_4n = create_weights(Jc_4n, n_levels=4, clusts=clusts)

    # split input data into training, validation and test sets
    L = inputs.shape[1]
    train_split = 0.7
    n_train = int(L * train_split)
    train_inputs = inputs[:, :n_train]
    val_split = 0.1
    n_val = int(L * val_split)
    val_inputs = inputs[:, n_train:n_train + n_val]
    n_test = L - n_train - n_val
    test_inputs = inputs[:, -n_test:]

    target_params_list = []
    trained_params_list = []

    # randomise parameters, and generate the target trace
    target_model.randomise_parameters()
    train_target = target_model(train_inputs)
    val_target = target_model(val_inputs)
    test_target = target_model(test_inputs)

    # start off with 1L model, and train until some performance on validation set
    print("Beginning 1L training")
    hln_1l = hLN_TiedModel(Jc=Jc_1l, Wce=Wce_1l, Wci=Wci_1l, sig_on=tf.constant([False]))
    train_losses_1l, val_losses_1l = train_until(model=hln_1l, train_inputs=train_inputs, train_target=train_target,
                                                val_inputs=val_inputs, val_target=val_target)



    # continue procedure with more complex models: 1N:
    print("1L training finished, beginning 1N training")
    hln_1n = hLN_TiedModel(Jc=Jc_1l, Wce=Wce_1l, Wci=Wci_1l, sig_on=tf.constant([True]))
    init_nonlin_tied(X=inputs, model=hln_1n, lin_model=hln_1l, nSD=nSD)
    train_losses_1n, val_losses_1n=train_until(model=hln_1n, train_inputs=train_inputs, train_target=train_target,
                                                val_inputs=val_inputs, val_target=val_target)

    # continue procedure with more complex models: 2N:
    print("1N training finished, beginning 2N training")
    hln_2l = hLN_TiedModel(Jc=Jc_2n, Wce=Wce_2n, Wci=Wci_2n, sig_on=tf.constant([True, False, False]))
    update_arch_tied(prev_model=hln_1n, next_model=hln_2l)
    hln_2n = hLN_TiedModel(Jc=Jc_2n, Wce=Wce_2n, Wci=Wci_2n, sig_on=tf.constant([True, True, True]))
    init_nonlin_tied(X=inputs, model=hln_2n, lin_model=hln_2l, nSD=nSD)
    train_losses_2n, val_losses_2n = train_until(model=hln_2n, train_inputs=train_inputs, train_target=train_target,
                                                val_inputs=val_inputs, val_target=val_target)

    # continue procedure with more complex models: 3N:
    print("2N training finished, beginning 3N training")
    hln_3l = hLN_TiedModel(Jc=Jc_3n, Wce=Wce_3n, Wci=Wci_3n, sig_on=tf.constant([True, True, True,
                                                                             False, False, False, False]))
    update_arch_tied(prev_model=hln_2n, next_model=hln_3l)
    hln_3n = hLN_TiedModel(Jc=Jc_3n, Wce=Wce_3n, Wci=Wci_3n, sig_on=tf.constant([True, True, True,
                                                                             True, True, True, True]))
    init_nonlin_tied(X=inputs, model=hln_3n, lin_model=hln_3l, nSD=nSD)
    train_losses_3n, val_losses_3n=train_until(model=hln_3n, train_inputs=train_inputs, train_target=train_target,
                                                val_inputs=val_inputs, val_target=val_target)


    # continue procedure with more complex models: 4N:
    print("3N training finished, beginning 4N training")
    hln_4l = hLN_TiedModel(Jc=Jc_4n, Wce=Wce_4n, Wci=Wci_4n,
                       sig_on=tf.constant([True, True, True, True, True, True, True,
                                           False, False, False, False, False, False,
                                           False, False]))
    update_arch_tied(prev_model=hln_3n, next_model=hln_4l)
    hln_4n = hLN_TiedModel(Jc=Jc_4n, Wce=Wce_4n, Wci=Wci_4n,
                       sig_on=tf.constant([True, True, True, True, True, True, True,
                                           True, True, True, True, True, True, True,
                                           True]))
    init_nonlin_tied(X=inputs, model=hln_4n, lin_model=hln_4l, nSD=nSD)
    train_losses_4n, val_losses_4n = train_until(model=hln_4n, train_inputs=train_inputs, train_target=train_target,
                                                val_inputs=val_inputs, val_target=val_target)


    print("4N training finished, procedure ending")

    # return parameters of all trained models, and parameters of target model
    params_1l = [param.numpy() for param in hln_1l.params]
    params_1n = [param.numpy() for param in hln_1n.params]
    params_2n = [param.numpy() for param in hln_2n.params]
    params_3n = [param.numpy() for param in hln_3n.params]
    params_4n = [param.numpy() for param in hln_4n.params]

    target_params = [param.numpy() for param in target_model.params]

    trained_params = [params_1l, params_1n, params_2n, params_3n, params_4n]

    train_losses = [train_losses_1l, train_losses_1n, train_losses_2n,train_losses_3n, train_losses_4n]
    val_losses = [val_losses_1l, val_losses_1n, val_losses_2n, val_losses_3n, val_losses_4n]

    return target_params, trained_params, train_losses, val_losses




if __name__ == '__main__':
    print('Beginning training procedure')
    run()
