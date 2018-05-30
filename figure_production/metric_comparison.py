"""
The idea is to have 32 48 and 64 architectures and compare the metrics on them.

"""


import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


tag_names = ('distances/FD', 'distances/FID', "distances/IS_mean",
             "distances/KID", "distances/L2", "distances/chi_square",
             "distances/cos", "distances/scalar_mean_fake",
             "distances/scalar_mean_real", "distances/scalar_sdev_fake",
             "distances/scalar_sdev_real", "distances/IS_std",
             "performance/D_loss", "performance/gradient_penalty")
tags = [x.replace("/", "_") for x in tag_names]
tags
# print(tags)
len(tag_names)


def read_data():
    csv_dir = "./outputs/csv_data"
    files_to_read = os.listdir(csv_dir)
    data_to_load = ["32", "48", "64", "80", "128"]
    data_dicts = []
    for index in data_to_load:
        gan_data = [x for x in files_to_read if str(index + "_" + index) in x]
        gan = {}
        for i, data in enumerate(gan_data):
            r = 25
            if int(index) > 99:
                r = 27
            gan[data[r:]] = pd.read_csv(os.path.join(
                csv_dir, data), usecols=(1, 2), header=0)
        data_dicts.append(gan)
    frames = {}
    for i, index in enumerate(data_to_load):
        frames[index] = data_dicts[i]
    return frames


def plot_data(data):
    data = read_data()
    keys = sorted(list(data.keys()))
    key_ints = [int(x) for x in keys]
    keys= [x for _,x in sorted(zip(key_ints,keys))]
    markers = ["o", "v", "*", "D",  ">", "<" ]
    key_to_marker = {}
    for i, key in enumerate(keys):
        key_to_marker[key] = markers[i]

    fig, ax = plt.subplots(nrows=5, ncols=2, sharex=True, figsize  = (8,11))
    # figsize=(4, 10)
    plt.subplot(5, 2, 1)
    figure_name = "distances_FD"
    hands = []
    for key in keys:
        repr = "-" + key_to_marker[key]
        steps = data[key][figure_name].iloc[:, 0]
        values = data[key][figure_name].iloc[:, 1]
        hands.append(plt.plot(steps, values, repr, lw=2,
                              label="Network width = " + key)[0])
        # plt.fill_between(indexes, mu + std, mu - std, alpha=0.5)
    # plt.legend(handles=hands)
    plt.title("Frechet distance")
    plt.subplot(5, 2, 2)
    figure_name = "distances_L2"
    hands = []
    for key in keys:
        repr = "-" + key_to_marker[key]
        steps = data[key][figure_name].iloc[:, 0]
        values = data[key][figure_name].iloc[:, 1]
        hands.append(plt.plot(steps, values, repr, lw=2,
                              label="Network width = " + key)[0])
        # plt.fill_between(indexes, mu + std, mu - std, alpha=0.5)
    # plt.legend(handles=hands)
    plt.title("L2 distance")
    plt.subplot(5, 2, 3)
    figure_name = "distances_cos"
    hands = []
    for key in keys:
        repr = "-" + key_to_marker[key]
        steps = data[key][figure_name].iloc[:, 0]
        values = data[key][figure_name].iloc[:, 1]
        hands.append(plt.plot(steps, values, repr, lw=2,
                              label="Network width = " + key)[0])
        # plt.fill_between(indexes, mu + std, mu - std, alpha=0.5)
    # plt.legend(handles=hands)
    plt.title("Cosine distance")
    plt.subplot(5, 2, 4)
    figure_name = "distances_chi_square"
    hands = []
    for key in keys:
        repr = "-" + key_to_marker[key]
        steps = data[key][figure_name].iloc[:, 0]
        values = data[key][figure_name].iloc[:, 1]
        hands.append(plt.plot(steps, values, repr, lw=2,
                              label="Network width = " + key)[0])
        # plt.fill_between(indexes, mu + std, mu - std, alpha=0.5)
    plt.title("Chi squared distance")
    plt.legend(handles=hands, prop={'size': 10})
    plt.subplot(5, 2, 5)
    figure_name = "distances_FID"
    hands = []
    for key in keys:
        repr = "-" + key_to_marker[key]
        steps = data[key][figure_name].iloc[:, 0]
        values = data[key][figure_name].iloc[:, 1]
        hands.append(plt.plot(steps, values, repr, lw=2,
                              label="Network width = " + key)[0])
        # plt.fill_between(indexes, mu + std, mu - std, alpha=0.5)
    # plt.legend(handles=hands)
    plt.title("Frechet Inception Distance")
    plt.subplot(5, 2, 6)
    figure_name = "distances_KID"
    hands = []
    for key in keys:
        repr = "-" + key_to_marker[key]
        steps = data[key][figure_name].iloc[:, 0]
        values = data[key][figure_name].iloc[:, 1]
        hands.append(plt.plot(steps, values, repr, lw=2,
                              label="Network width = " + key)[0])
        # plt.fill_between(indexes, mu + std, mu - std, alpha=0.5)
    # plt.legend(handles=hands)
    plt.title("Kernel Inception Distance")
    plt.subplot(5, 2, 7)
    figure_name = "distances_IS_mean"
    hands = []
    for key in keys:
        repr = "-" + key_to_marker[key]
        steps = data[key][figure_name].iloc[:, 0]
        values = data[key][figure_name].iloc[:, 1]
        std = data[key]["distances_IS_std"].iloc[:, 1]
        hands.append(plt.plot(steps, values, repr, lw=2,
                              label="Network width = " + key)[0])
        plt.fill_between(steps, values + std, values - std, alpha=0.5)
    # plt.legend(handles=hands)
    plt.title("Inception Score")
    plt.subplot(5, 2, 8)
    figure_name = "Wasserstein Distance"
    hands = []
    for key in keys:
        repr = "-" + key_to_marker[key]
        steps = data[key]["performance_D_loss"].iloc[:, 0]
        values = -1 * (data[key]["performance_D_loss"].iloc[:, 1] + data[key]["performance_gradient_penalty"].iloc[:, 1])
        # data["32"]["performance_D_loss"].iloc[:, 1]
        # data["32"]["performance_gradient_penalty"].iloc[:, 1]
        hands.append(plt.plot(steps, values, repr, lw=2,
                              label="Network width = " + key)[0])
        # plt.fill_between(indexes, mu + std, mu - std, alpha=0.5)
    # plt.legend(handles=hands)
    plt.ylim(0, 10)
    plt.title(figure_name)
    plt.subplot(5, 2, 9)
    figure_name = "distances_scalar_mean_fake"
    # TODO: Add sdevs
    hands = []
    for key in keys:
        repr = "-" + key_to_marker[key]
        steps = data[key][figure_name].iloc[:, 0]
        values = data[key][figure_name].iloc[:, 1]
        std = data[key]["distances_scalar_sdev_fake"].iloc[:, 1]
        hands.append(plt.plot(steps, values, repr, lw=2,
                              label="Network width = " + key)[0])
        plt.fill_between(steps, values + std, values - std, alpha=0.5)
    # plt.legend(handles=hands)
    plt.ylim(-5, 25)
    plt.title("Scalar output for fake data")
    plt.subplot(5, 2, 10)
    figure_name = "distances_scalar_mean_real"
    hands = []
    for key in keys:
        repr = "-" + key_to_marker[key]
        steps = data[key][figure_name].iloc[:, 0]
        values = data[key][figure_name].iloc[:, 1]
        std = data[key]["distances_scalar_sdev_real"].iloc[:, 1]
        hands.append(plt.plot(steps, values, repr, lw=2,
                              label="Network width = " + key)[0])
        plt.fill_between(steps, values + std, values - std, alpha=0.5)
    # plt.legend(handles=hands)
    plt.ylim(-5, 25)
    plt.title("Scalar output for real data")
    for i in range(1,11):
        plt.subplot(5, 2, i)
        plt.grid(True)
        plt.xlim(0, 8000)
    plt.subplots_adjust(wspace=0.07, hspace=0.05)
    # plt.suptitle('LIME Explanations for G.dim = {}, D.dim = {}'.format(G_dim, D_dim))
    plt.tight_layout()
    save_path = "./outputs/metrics/"
    save_name = "metrics_comparison.png"
    plt.savefig(save_path + save_name, dpi=None, format="png",)
    plt.show()

# ['distances_FD', 'distances_FID', 'distances_IS_mean', 'distances_KID', 'distances_L2', 'distances_chi_square', 'distances_cos', 'distances_scalar_mean_fake',
#     'distances_scalar_mean_real', 'distances_scalar_sdev_fake', 'distances_scalar_sdev_real', 'distances_IS_std', 'performance_D_loss', 'performance_gradient_penalty']


def main():
    # data = read_data()
    plot_data(read_data())


if __name__ == '__main__':
    main()
