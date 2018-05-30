"""
Get the final value of training metrics for each network and plot it.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats.stats import pearsonr



tag_names = ('distances/FD', 'distances/FID', "distances/IS_mean",
             "distances/KID", "distances/L2", "distances/chi_square",
             "distances/cos", "distances/scalar_mean_fake",
             "distances/scalar_mean_real", "distances/scalar_sdev_fake",
             "distances/scalar_sdev_real", "distances/IS_std",
             "performance/D_loss", "performance/gradient_penalty")
tags = [x.replace("/", "_") for x in tag_names]
tags
# print(tags)
metrics_to_show = ['distances/FID', "distances/IS_mean", "distances/KID" ]
len(tag_names)

# show ratio of KID and FID.

def read_data():
    csv_dir = "./outputs/csv_data"
    files_to_read = os.listdir(csv_dir)
    # files_to_read
    # x = files_to_read[0]
    # x.split("_")
    data_to_load = ["32", "48", "64", "80", "128"]
    sym_archs = set([x.split("_")[1] for x in files_to_read if x.split("_")[1] == x.split("_")[2]])
    int_list = [int(x) for x in sym_archs]
    data_to_load = sorted(int_list)

    # sym_archs = [x for _,x in sorted(zip(int_list, sym_archs))]
    # sym_archs
    data_dicts = []
    for index in data_to_load:
        gan_data = [x for x in files_to_read if (int(x.split("_")[1]) == int(x.split("_")[2]) == index)]
        gan_data
        gan = {}
        for i, data in enumerate(gan_data):
            r = 25
            if index > 99:
                r = 27
            if index < 10:
                r = 23
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
    plt.title("FID")
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
    plt.title("KID")
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
    plt.title("IS")
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
    data = read_data()
    metrics_to_show = ['distances/FID', "distances/IS_mean", "distances/KID" ]
    # plot_data(read_data())
    data_dims = sorted(list(data.keys()))
    data_dims
    data[4].keys()
    data[4]["distances_FID"].iloc[-1, 1]
    fids = [ data[x]['distances_FID'].iloc[-1, 1] for x in data_dims  ]
    kids = [ data[x]['distances_KID'].iloc[-1, 1] for x in data_dims  ]
    iss =  [ data[x]['distances_IS_mean'].iloc[-1, 1] for x in data_dims  ]
    fid_to_kid = [ x/y for (x,y) in zip(fids,kids) ]
    np.corrcoef(fids,kids)
    pearsonr(fids,kids)
    fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True,
                           sharey=True, figsize=(8, 5))
    plt.subplot(2, 2, 1)
    plt.plot(data_dims, fids, "-o", label="G dim = ", linewidth=3)
    plt.title("Frechet Inception Distance (FID)")
    plt.grid(True)
    plt.subplot(2, 2, 2)
    plt.plot(data_dims, kids, "-o", label="G dim = ", linewidth=3)
    plt.title("Kernel Inception Distance (KID)")
    plt.grid(True)
    plt.subplot(2, 2, 3)
    plt.plot(data_dims, iss, "-o", label="G dim = ", linewidth=3)
    plt.title("Inception score")
    plt.grid(True)
    plt.subplot(2, 2, 4)
    plt.plot(data_dims, fid_to_kid, "-o", label="G dim = ", linewidth=3)
    plt.title("FID/KID")
    plt.grid(True)
    fig.text(0.5, 0.01, "Width of both networks",
             ha='center', fontsize=12)

    # plt.legend(handles=hands, prop={'size': legend_size})
    # plt.xlim(0, 130)
    # plt.ylim(0, 1)
    # plt.xlabel("Width of discriminator network")
    # plt.ylabel("Proportion of times discriminator judges the image as real")
    # fig.text(0.5, 0.01, "Width of discriminator network",
    #          ha='center', fontsize=12)
    # fig.text(0.09, 0.5, "Proportion of times discriminator judges the image as real",
    #          va='center', rotation='vertical', fontsize=14)
    # # fig.suptitle("Title centered above all subplots", fontsize=14)
    # plt.suptitle(
    #     'Discriminator Output for different network widths on fake data', y=1, fontsize=12)
    plt.tight_layout()
    save_dir = "/home/aidas/Dropbox/MEng_Thesis/Figures/Figures_30th_May"
    plt.savefig(save_dir + "/FID_KID.png",dpi=None, format="png",)
    plt.show()

    #
    # for metric in metrics_to_show:
    #     for dim in D_dims:
    #         plot stuff

if __name__ == '__main__':
    main()
