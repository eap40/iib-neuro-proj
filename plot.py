### File containing useful plotting functions we will use multiple times in training

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams["savefig.dpi"] = 200


def plot_loss_acc(loss_values, accuracies, name, save=False):
    plt.figure(1)

    plt.subplot(1, 2, 1)
    plt.plot(loss_values)
    plt.title('Loss value')
    plt.xlabel('Epoch number')

    plt.subplot(1, 2, 2)
    plt.plot(accuracies)
    plt.title('Prediction accuracy (%)')
    plt.xlabel('Epoch number')

    plt.title('Loss and accuracy during linear training')
    plt.tight_layout()

    if save:
        plt.savefig(name)

    return


def plot_params(model, target_params, save=False):
    # create plots to compare trained parameters with target parameters:
    trained_params = [param.numpy() for param in model.params]
    np.save("../Data/trained_params.npy", np.array(trained_params))

    fig, ax = plt.subplots(nrows=2, ncols=4)
    fig.set_size_inches(20, 10)
    i = 0
    param_names = ["v0,", "logJw", "Wwe", "Wwi", "logTaue", "logTaui", "Th", "logDelay"]
    for col in range(4):
        for row in range(2):
            p_trained, p_target = trained_params[i].flatten(), target_params[i].flatten()
            ax[row, col].scatter(p_trained, p_target)
            x = np.linspace(min(p_trained), max(p_trained), 100)
            ax[row, col].plot(x, x, color='red')
            ax[row, col].set_title(param_names[i] + ", ve = var_explained")
            var_explained = 1 - ((p_trained - p_target) ** 2).mean() / np.var(p_target)
            ax[row, col].set_title(param_names[i] + f", ve = {var_explained}")

            # print(param_names[i], "variance explained:", min(0, var_explained))
            i += 1

    if save:
        plt.savefig("some_name.png")

    return

