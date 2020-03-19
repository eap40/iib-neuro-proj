### File for running more complex training procedures, neater than a Jupyter notebook
from train_hLN import *

def run():
    """Training procedure here"""

    # define hierarchical clustering of input ensembles
    clusts = [[[[[0, 1], [2]], [[3, 4], [5, 6]]], [[[7, 8], [9]], [[10, 11], [12]]]]]

    ### Define the different hLN architectures we will be using:
    # 1N
    Jc_1n = np.array([0])
    # 2N
    Jc_2n = np.array([0, 1, 1])
    # 3N
    Jc_3n = np.array([0, 1, 1, 2, 2, 3, 3])
    # 4N
    Jc_4n = np.array([0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7])

    # Get some realistic inputs
    X_tot = tf.convert_to_tensor(np.load('Data/real_inputs.npy'), dtype=tf.float32)  # real inputs made earlier
    X_e = X_tot[:629]  # 629 excitatory inputs, in 13 ensembles
    X_i = X_tot[629:]  # 120 inhibitory inputs, also in 13 ensembles
    # remember 1st inhibitory inputs is the somatic input - must always go to root subunit
    inputs = X_tot


    # Split the data into training and test sets, 80/20 initially
    split = 0.8
    L = inputs.shape[1]
    n_train = int(L * split)
    train_inputs = inputs[:, :n_train]
    test_inputs = inputs[:, n_train:]

    # create the Wcs based on the model and clusts
    Wce_1l, Wci_1l = create_weights(Jc_1l, n_levels=1, clusts=clusts)
    Wce_2n, Wci_2n = create_weights(Jc_2n, n_levels=2, clusts=clusts)
    Wce_3n, Wci_3n = create_weights(Jc_3n, n_levels=3, clusts=clusts)
    Wce_4n, Wci_4n = create_weights(Jc_4n, n_levels=4, clusts=clusts)

    # initialise a known version each of the models to generate data with: save the parameter somewhere
    hln_1l = hLN_Model(Jc=Jc_1n, Wce=Wce_1l, Wci=Wci_1l, sig_on=tf.constant([False]))
    params_1l = [param.numpy() for param in hln_1l.params]
    np.save("Data/params_1l.npy", np.array(params_1l))

    # select the parameters of the model that generated the data
    target_params = params_1l

    # generate output data from realistic inputs
    target_1l = hln_1l(inputs)
    # target_1l = tf.convert_to_tensor(np.load('../Data/target.npy'), dtype=tf.float32)  # real output made earlier
    np.save("Data/target_1l.npy", target_1l.numpy())

    # split target into training and test data
    train_target = target_1l[:n_train]
    test_target = target_1l[n_train:]


    # randomise the parameters of hln_1l, start training
    hln_1l.randomise_parameters()
    # initialise 1L model, and optimise for data
    # define optimizer
    optimizer_1l = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.005)
    # train model with SGD
    loss_values_1l, accuracies_1l = train_lin_sgd(model=hln_1l, num_epochs=1000, optimizer=optimizer_1l,
                                                  inputs=train_inputs, target=train_target)

    # when to stop training? when performance on test data doesnt increase significantly each say 1000 epochs

    # visualise training graph and save without viewing, same for parameter graphs, also save stats
    # visualise results from 1L training
    plot_loss_acc(loss_values=loss_values_1l, accuracies=accuracies_1l, save=True, name="Figures/loss_acc_1l.png")

    plot_params(model=hln_1l, target_params=target_params)



    # initialise 1n model to approximate 1l model, train

    # initialise 2n model to approximate 1n model, train

    # initialise 3n model to approximate 2n model, train

    # for continued training


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
        target = model(train_inputs)

        # store parameters of the target model, and split the target into training and test sets
        train_target = target[:n_train]
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

            # define optimizer
            optimizer_1l = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=learning_rate)
            # train model with SGD
            # loss_values, accuracies = train_sgd(model=model, num_epochs=num_epochs, optimizer=optimizer_1l,
            #                                     inputs=train_inputs, target=train_target)

            # # train without SGD on a whole dataset
            loss_values, accuracies = train(model=model, num_epochs=num_epochs, optimizer=optimizer_1l,
                                                inputs=train_inputs, target=train_target)

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


        # # convert log parameters to normal for plotting
        # log_params = [1, 4, 5, 7]
        # for ind in log_params:
        #     trained_params[ind] = np.exp(trained_params[ind])
        #     target_params[ind] = np.exp(target_params[ind])
        #
        # # for single subunit linear model, only want these 4 parameters
        # lin_indices = [0, 2, 4, 7]
        # param_names = ["v0", "Wwe", "Taue", "Delay"]
        #
        # sim_stats=[]
        # for i in range(len(param_names)):
        #     p_trained, p_target = trained_params[lin_indices[i]].flatten(), target_params[lin_indices[i]].flatten()
        #     if len(p_trained) > 1:
        #         var_explained = 1 - ((p_trained - p_target) ** 2).mean() / np.var(p_target)
        #         sim_stats.append(var_explained)
        #     elif len(p_trained) == 1:
        #         error = np.abs((p_trained[0] - p_target[0]) / p_target[0]) * 100
        #         sim_stats.append(error)
        #
        # param_stats.append(sim_stats)

    return train_accuracies, test_accuracies, trained_params_list, target_params_list

# if __name__ == '__main__':
#     print('Beginning training procedure')
#     run()
