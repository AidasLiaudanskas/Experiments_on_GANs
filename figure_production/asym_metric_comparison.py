"""
Plots the metrics in csv format for assymetric architectures.

Need to specify:
* gen_dim
* data_to_load array (i.e. the discriminators for fixed generator)
"""


import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
os.chdir("/home/aidas/Experiments_on_GANs/figure_production")


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



# two functions: one for assymetric cases, other for symmetric.
"""
    ("DCGAN_4_4", 4, 4, 1, 1, "LSUN", True),
    ("DCGAN_8_8", 8, 8, 1, 1, "LSUN", True),
    ("DCGAN_6_6", 6, 6, 1, 1, "LSUN", True),
    ("DCGAN_12_12", 12, 12, 1, 1, "LSUN", True),
    ("DCGAN_16_4", 16, 4, 1, 1, "LSUN", True),
    ("DCGAN_16_8", 16, 8, 1, 1, "LSUN", True),
    ("DCGAN_16_12", 16, 12, 1, 1, "LSUN", True),
    ("DCGAN_16_16", 16, 16, 1, 1, "LSUN", True),
    ("DCGAN_20_20", 20, 20, 1, 1, "LSUN", True),
    ("DCGAN_24_24", 24, 24, 1, 1, "LSUN", True),
    ("DCGAN_32_32", 32, 32, 1, 1, "LSUN", True),
    ("DCGAN_28_28", 28, 28, 1, 1, "LSUN", True),
    ("DCGAN_32_8", 32, 8, 1, 1, "LSUN", True),
    ("DCGAN_32_16", 32, 16, 1, 1, "LSUN", True),
    ("DCGAN_32_24", 32, 24, 1, 1, "LSUN", True),
    ("DCGAN_32_40", 32, 40, 1, 1, "LSUN", True),
    ("DCGAN_32_48", 32, 48, 1, 1, "LSUN", True),
    ("DCGAN_64_16", 64, 16, 1, 1, "LSUN", True),
    ("DCGAN_64_32", 64, 32, 1, 1, "LSUN", True),
    ("DCGAN_64_48", 64, 48, 1, 1, "LSUN", True),
    ("DCGAN_40_40", 40, 40, 1, 1, "LSUN", True),
    ("DCGAN_48_48", 48, 48, 1, 1, "LSUN", True),
    ("DCGAN_56_56", 56, 56, 1, 1, "LSUN", True),
    ("DCGAN_64_64", 64, 64, 1, 1, "LSUN", True),
    ("DCGAN_72_72", 72, 72, 1, 1, "LSUN", True),
    ("DCGAN_80_80", 80, 80, 1, 1, "LSUN", True),

"""

# assymetric pairings:

# gen_dim = 32
# data_to_load = ["8", "16", "24", "32", "40", "48"]

# gen_dim = 16
# data_to_load = ["4", "8", "12", "16"]

gen_dim = 64
data_to_load = ["16", "32", "48", "64"]

def read_data():
    csv_dir = "./outputs/csv_data"
    files_to_read = os.listdir(csv_dir)
    # data_to_load = ["16", "32", "48", "64"]
    data_dicts = []
    for index in data_to_load:
        gan_data = [x for x in files_to_read if str(
            str(gen_dim) + "_" + index) in x]
        gan = {}
        for i, data in enumerate(gan_data):
            r = 32
            if int(index) > 99:
                r = 33
            if int(index) < 10:
                r = 31
            gan[data[r:]] = pd.read_csv(os.path.join(
                csv_dir, data), usecols=(1, 2), header=0)
        data_dicts.append(gan)
    # data_dicts
    frames = {}
    for i, index in enumerate(data_to_load):
        frames[index] = data_dicts[i]
    # frames
    return frames


def plot_data(data):
    markers = ["o", "v", "*", "D",  ">", "<", "p",  "X", "0"]
    # markers = ["o",  "*", "D", "0",  ">", "<"]
    data = read_data()
    keys = sorted(list(data.keys()))
    key_ints = [int(x) for x in keys]
    keys = [x for _, x in sorted(zip(key_ints, keys))]
    # keys
    # data["32"].keys()
    key_to_marker = {}
    for i, key in enumerate(keys):
        key_to_marker[key] = markers[i]
    # keys
    # data["64"]
    fig, ax = plt.subplots(nrows=5, ncols=2, sharex=True, figsize=(8, 11))
    # figsize=(4, 10)
    plt.subplot(4, 2, 1)
    figure_name = "distances_FD"
    hands = []
    for key in keys:
        print("key is ", key)
        steps = data[key][figure_name].iloc[:, 0]
        values = data[key][figure_name].iloc[:, 1]
        repr = "-" + key_to_marker[key]
        hands.append(plt.plot(steps, values, repr, lw=2,
                              label="Critic's width = " + key)[0])
        # plt.fill_between(indexes, mu + std, mu - std, alpha=0.5)
    # plt.legend(handles=hands)
    plt.title("Frechet distance")
    plt.subplot(4, 2, 2)
    figure_name = "distances_L2"
    hands = []
    for key in keys:
        steps = data[key][figure_name].iloc[:, 0]
        values = data[key][figure_name].iloc[:, 1]
        repr = "-" + key_to_marker[key]
        hands.append(plt.plot(steps, values, repr, lw=2,
                              label="Critic's width = " + key)[0])
        # plt.fill_between(indexes, mu + std, mu - std, alpha=0.5)
    plt.legend(handles=hands, loc=1,  prop={'size': 8})
    plt.title("L2 distance")
    plt.subplot(4, 2, 3)
    figure_name = "distances_FID"
    hands = []
    for key in keys:
        repr = "-" + key_to_marker[key]
        steps = data[key][figure_name].iloc[:, 0]
        values = data[key][figure_name].iloc[:, 1]
        hands.append(plt.plot(steps, values, repr, lw=2,
                              label="Critic's width = " + key)[0])
        # plt.fill_between(indexes, mu + std, mu - std, alpha=0.5)
    # plt.legend(handles=hands)
    plt.title("Frechet Inception Distance")
    plt.subplot(4, 2, 4)
    figure_name = "distances_KID"
    hands = []
    for key in keys:
        repr = "-" + key_to_marker[key]
        steps = data[key][figure_name].iloc[:, 0]
        values = data[key][figure_name].iloc[:, 1]
        hands.append(plt.plot(steps, values, repr, lw=2,
                              label="Critic's width = " + key)[0])
        # plt.fill_between(indexes, mu + std, mu - std, alpha=0.5)
    # plt.legend(handles=hands)
    plt.title("Kernel Inception Distance")
    plt.subplot(4, 2, 5)
    figure_name = "distances_IS_mean"
    hands = []
    for key in keys:
        repr = "-" + key_to_marker[key]
        steps = data[key][figure_name].iloc[:, 0]
        values = data[key][figure_name].iloc[:, 1]
        std = data[key]["distances_IS_std"].iloc[:, 1]
        hands.append(plt.plot(steps, values, repr, lw=2,
                              label="Critic's width = " + key)[0])
        plt.fill_between(steps, values + std, values - std, alpha=0.5)
    # plt.legend(handles=hands)
    plt.title("Inception Score")
    plt.subplot(4, 2, 6)
    figure_name = "Wasserstein Distance"
    hands = []
    for key in keys:
        repr = "-" + key_to_marker[key]
        steps = data[key]["performance_D_loss"].iloc[:, 0]
        values = -1 * (data[key]["performance_D_loss"].iloc[:, 1] +
                       data[key]["performance_gradient_penalty"].iloc[:, 1])
        # data["32"]["performance_D_loss"].iloc[:, 1]
        # data["32"]["performance_gradient_penalty"].iloc[:, 1]
        hands.append(plt.plot(steps, values, repr, lw=2,
                              label="Critic's width = " + key)[0])
        # plt.fill_between(indexes, mu + std, mu - std, alpha=0.5)
    # plt.legend(handles=hands)
    plt.ylim(0, 10)
    plt.title(figure_name)
    plt.subplot(4, 2, 7)
    figure_name = "distances_scalar_mean_fake"
    # TODO: Add sdevs
    hands = []
    for key in keys:
        repr = "-" + key_to_marker[key]
        steps = data[key][figure_name].iloc[:, 0]
        values = data[key][figure_name].iloc[:, 1]
        std = data[key]["distances_scalar_sdev_fake"].iloc[:, 1]
        hands.append(plt.plot(steps, values, repr, lw=2,
                              label="Critic's width = " + key)[0])
        plt.fill_between(steps, values + std, values - std, alpha=0.5)
    # plt.legend(handles=hands)
    # plt.ylim(-5, 25)
    plt.title("Scalar output for fake data")
    plt.subplot(4, 2, 8)
    figure_name = "distances_scalar_mean_real"
    hands = []
    for key in keys:
        repr = "-" + key_to_marker[key]
        steps = data[key][figure_name].iloc[:, 0]
        values = data[key][figure_name].iloc[:, 1]
        std = data[key]["distances_scalar_sdev_real"].iloc[:, 1]
        hands.append(plt.plot(steps, values, repr, lw=2,
                              label="Critic's width = " + key)[0])
        plt.fill_between(steps, values + std, values - std, alpha=0.5)
    # plt.legend(handles=hands)
    # plt.ylim(-5, 25)
    plt.title("Scalar output for real data")
    for i in range(1, 9):
        plt.subplot(4, 2, i)
        plt.grid(True)
        # plt.xlim(0, 8000)
    plt.subplots_adjust(wspace=0.01, hspace=0.01)
    # plt.suptitle('LIME Explanations for G.dim = {}, D.dim = {}'.format(G_dim, D_dim))
    plt.tight_layout()
    save_path = "./outputs/metrics/"
    save_name = "asy_metrics_comparison_{}.png".format(gen_dim)
    plt.savefig(save_path + save_name, dpi=None, format="png",)
    plt.show()

# ['distances_FD', 'distances_FID', 'distances_IS_mean', 'distances_KID', 'distances_L2', 'distances_chi_square', 'distances_cos', 'distances_scalar_mean_fake',
#     'distances_scalar_mean_real', 'distances_scalar_sdev_fake', 'distances_scalar_sdev_real', 'distances_IS_std', 'performance_D_loss', 'performance_gradient_penalty']


def main():
    # data = read_data()
    plot_data(read_data())


if __name__ == '__main__':
    main()
