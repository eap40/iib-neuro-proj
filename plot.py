### File containing useful plotting functions we will use multiple times in training

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from train_hLN import *
matplotlib.rcParams["savefig.dpi"] = 200


def plot_loss_acc(model, loss_values, accuracies, inputs, target, name="name", save=False):
    """Plotting function for losses and accuracies"""

    fig, ax = plt.subplots(ncols=2)
    fig.set_size_inches(12, 6)

    ax[0].plot(loss_values)
    ax[0].set_title('Training loss value')
    ax[0].set_xlabel('Epoch number')

    ax[1].plot(accuracies)
    ax[1].set_title('Prediction accuracy (%)')
    ax[1].set_xlabel('Epoch number')

    #     plt.title('Loss and accuracy during linear training')

    if save:
        plt.savefig(name)

    plt.show()

    # display final training loss and accuracy:
    print(f"Final training loss: {loss_values[-1]:.2E}, Final training accuracy:{accuracies[-1]:.2f}%")

    # also display test loss and accuracy:
    test_loss = loss(predicted_v=model(inputs), target_v=target)
    test_accuracy = 100 * (1 - (test_loss / np.var(target)))
    print(f"Test loss: {test_loss:.2E}, Test accuracy:{test_accuracy:.2f}%")

    return


def plot_params(model, target_params, name="name", save=False):
    """create plots to compare trained parameters with target parameters"""

    trained_params = [param.numpy() for param in model.params]
    np.save("../Data/trained_params.npy", np.array(trained_params))

    fig, ax = plt.subplots(nrows=2, ncols=4)
    fig.set_size_inches(20, 10)

    # convert log parameters to normal for plotting
    log_params = [1, 4, 5, 7]
    for ind in log_params:
        trained_params[ind] = np.exp(trained_params[ind])
        target_params[ind] = np.exp(target_params[ind])

    i = 0
    param_names = ["v0", "Jw", "Wwe", "Wwi", "Taue", "Taui", "Th", "Delay"]
    for col in range(4):
        for row in range(2):
            p_trained, p_target = trained_params[i].flatten(), target_params[i].flatten()
            if len(p_trained) > 0:
                ax[row, col].scatter(p_target, p_trained)
                x = np.linspace(min(p_target), max(p_target), 100)
                ax[row, col].plot(x, x, color='red', label='Perfect recovery')
                ax[row, col].set_xlabel("Truth")
                ax[row, col].set_ylabel("Recovered")
                if len(p_trained) > 1:
                    var_explained = 1 - ((p_trained - p_target) ** 2).mean() / np.var(p_target)
                    ax[row, col].set_title(param_names[i] + f", ve = {var_explained:.2f}%")
                elif len(p_trained) == 1:
                    error = np.abs((p_trained[0] - p_target[0]) / p_target[0]) * 100
                    ax[row, col].set_title(param_names[i] + f", error ={error:.2f}%")
                ax[row, col].legend()
            i += 1

    if save:
        plt.savefig(name)

    plt.show()

    return


def plot_params_1l(model, target_params, name="name", save=False):
    """create plots to compare trained parameters with target parameters for single subunit linear model"""

    trained_params = [param.numpy() for param in model.params]
    np.save("../Data/trained_params.npy", np.array(trained_params))

    fig, ax = plt.subplots(nrows=2, ncols=2)
    fig.set_size_inches(15, 15)

    # convert log parameters to normal for plotting
    log_params = [1, 4, 5, 7]
    for ind in log_params:
        trained_params[ind] = np.exp(trained_params[ind])
        target_params[ind] = np.exp(target_params[ind])

    i = 0
    lin_indices = [0, 2, 4, 7]
    param_names = ["v0", "Wwe", "Taue", "Delay"]
    for row in range(2):
        for col in range(2):
            print(param_names[i])
            p_trained, p_target = trained_params[lin_indices[i]].flatten(), target_params[lin_indices[i]].flatten()
            if len(p_trained) > 0:
                ax[row, col].scatter(p_target, p_trained)
                x = np.linspace(min(p_trained), max(p_trained), 100)
                ax[row, col].plot(x, x, color='red', label='Perfect recovery')
                ax[row, col].set_xlabel("Truth")
                ax[row, col].set_ylabel("Recovered")
                ax[row, col].yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.3f'))
                ax[row, col].xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.3f'))
                if len(p_trained) > 1:
                    var_explained = 1 - ((p_trained - p_target) ** 2).mean() / np.var(p_target)
                    ax[row, col].set_title(param_names[i] + f", ve = {var_explained:.2f}%")
                elif len(p_trained) == 1:
                    error = np.abs((p_trained[0] - p_target[0]) / p_target[0]) * 100
                    ax[row, col].set_title(param_names[i] + f", error ={error:.2f}%")
                ax[row, col].legend()

            i += 1

    if save:
        plt.savefig(name)

    plt.show()

    return

