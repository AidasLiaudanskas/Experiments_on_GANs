""""
Produces figures based on training data
"""

import os
import numpy as np
import argparse
import matplotlib.pyplot as plt


stats_file = "./mean_offsets.npz"
f = np.load(stats_file)
means_matrix = f["means_matrix"]
marginalized_means = np.mean(means_matrix, axis=0)
save_dir = "/home/aidas/Dropbox/MEng_Thesis/Figures/Figures_30th_May"

markers = ["o", "p", "X", "*", "D", "0",  ">", "<"]


def real_data_test(account_for_offset=False):
    # TODO: Get rid of generator graphs.
    current_dir = os.getcwd()
    print("Current_dir = ", current_dir)
    os.chdir("/home/aidas/GAN_Experiments/progressive_test")
    stats_file = current_dir + "/real_evaluation_stats.npz"
    f = np.load(stats_file)
    results_tensor = f["results_tensor"][:]
    save_files = f["save_files"][:]
    indexes = [int(x.split("_")[1]) for x in save_files]
    save_files = [x for _, x in sorted(zip(indexes, save_files))]
    l = len(save_files)
    overall_means = np.mean(results_tensor, axis=(1, 2))
    overall_stds = np.std(results_tensor, axis=(1, 2))
    batch_means = np.mean(np.mean(results_tensor, axis=2), axis=1)
    batch_stds = np.std(np.std(results_tensor, axis=2), axis=1)
    if account_for_offset:
        overall_means = overall_means - marginalized_means
    width = 3.0
    no_samples = results_tensor.shape[1] * results_tensor.shape[2]
    proportions = [(x > 0).sum() / no_samples for x in results_tensor]
    if account_for_offset:
        proportions = [(x > marginalized_means[i]).sum() /
                       no_samples for i, x in enumerate(results_tensor)]
    np.shape(proportions)
    indexes.sort()
    j = 0
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=False,
                           sharey=False, figsize=(8, 5))
    # Generators Graph
    plt.subplot(2, 1, 1)
    plt.plot(indexes, proportions, '-o',  linewidth=width)
    plt.title('Discriminator Output for different network widths on real data')
    plt.grid(True)
    std = overall_stds
    mu = overall_means
    plt.subplot(2, 1, 2)
    plt.plot(indexes, mu,  '-o', lw=2)
    plt.fill_between(indexes, mu + std, mu - std, alpha=0.5)
    plt.xlabel("Width of discriminator network")
    plt.title('Discriminator scores for different network widths on real data')
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(save_dir + "/real_stats.png", dpi=None, format="png",)
    plt.show()


def fake_data_test(account_for_offset=False):
    legend_size = 8
    current_dir = os.getcwd()
    print("Current_dir = ", current_dir)
    os.chdir("/home/aidas/GAN_Experiments/progressive_test")
    stats_file = current_dir + "/evaluation_stats.npz"
    f = np.load(stats_file)
    results_tensor = f["results_tensor"][:]
    save_files = f["save_files"][:]
    indexes = [int(x.split("_")[1]) for x in save_files]
    save_files = [x for _, x in sorted(zip(indexes, save_files))]
    l = len(save_files)
    overall_means = np.mean(results_tensor, axis=(1, 2)).reshape([l, l])
    overall_stds = np.std(results_tensor, axis=(1, 2)).reshape([l, l])
    batch_means = np.mean(np.mean(results_tensor, axis=2),
                          axis=1).reshape([l, l])
    batch_stds = np.std(np.std(results_tensor, axis=2), axis=1).reshape([l, l])
    if account_for_offset:
        overall_means = overall_means - means_matrix
    width = 3.0
    no_samples = results_tensor.shape[1] * results_tensor.shape[2]
    proportions = np.reshape(
        [(x > 0).sum() / no_samples for x in results_tensor], [l, l])
    if account_for_offset:
        proportions = np.reshape(
            [(x > means_matrix[i // l][i % l]).sum() / no_samples for i, x in enumerate(results_tensor)], [l, l])
    """
    In table vertical axis is Generator depth
    Horizontal axis is Discriminator depth

    """
    indexes.sort()
    # Generators Graph
    # Make 4 subplots as there are too many lines to show on one plot.
    j = 0
    l = 0
    subplt_len = len(proportions) // 4
    fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True,
                           sharey=True, figsize=(8, 5))
    for k in range(4):
        hands = []
        l += 1
        props = proportions[(l - 1) * subplt_len:l * subplt_len]
        plt.subplot(2, 2, l)
        for iter, i in enumerate(props):
            repr = "-" + markers[iter]
            hands.append(
                plt.plot(indexes, i, repr, label="G dim = " + str(indexes[j]), linewidth=width)[0])
            j += 1
        plt.legend(handles=hands, prop={'size': legend_size})
        plt.grid(True)
        plt.xlim(0, 130)
        plt.ylim(0, 1)
    fig.text(0.5, 0.01, "Width of discriminator network",
             ha='center', fontsize=12)
    plt.suptitle(
        'Discriminator Output for different network widths on fake data', y=1, fontsize=12)
    plt.tight_layout()
    plt.savefig(save_dir + "/disc_prop_x_disc.png",
                dpi=None, format="png",)
    plt.show()
    hands = []
    # Discriminators graph
    # Make 4 subplots as there are too many lines to show on one plot.
    j = 0
    l = 0
    propst = proportions.transpose()
    subplt_len = len(propst) // 4
    fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True,
                           sharey=True, figsize=(8, 5))
    for k in range(4):
        hands = []
        l += 1
        props = propst[(l - 1) * subplt_len:l * subplt_len]
        plt.subplot(2, 2, l)
        for iter, i in enumerate(props):
            repr = "-" + markers[iter]
            hands.append(
                plt.plot(indexes, i, repr, label="D dim = " + str(indexes[j]), linewidth=width)[0])
            j += 1
        plt.legend(handles=hands, prop={'size': legend_size})
        plt.grid(True)
        plt.xlim(0, 130)
        plt.ylim(0, 1)

    fig.text(0.5, 0.01, "Width of generator network",
             ha='center', fontsize=12)
    plt.tight_layout()
    plt.suptitle(
        'Discriminator Output for different network widths on fake data', y=1, fontsize=12)
    plt.savefig(save_dir + "/disc_prop_x_gen.png",
                dpi=None, format="png",)
    plt.show()
    hands = []
    j = 0
    l = 0
    subplt_len = len(proportions) // 4
    fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True,
                           sharey=True, figsize=(8, 5))
    for k in range(4):
        hands = []
        l += 1
        means = overall_means[(l - 1) * subplt_len:l * subplt_len]
        stds = overall_stds[(l - 1) * subplt_len:l * subplt_len]
        plt.subplot(2, 2, l)
        for i, mu in enumerate(means):
            repr = "-" + markers[i]
            std = stds[i]
            hands.append(plt.plot(indexes, mu, repr, lw=2,
                                  label="G dim = " + str(indexes[i + subplt_len * (l - 1)]))[0])
            plt.fill_between(indexes, mu + std, mu - std, alpha=0.5)
        plt.legend(handles=hands, prop={'size': legend_size})
        plt.grid(True)
        plt.xlim(0, 130)
    fig.text(0.5, 0.01, "Width of discriminator network",
             ha='center', fontsize=12)
    plt.tight_layout()
    plt.suptitle(
        'Discriminator scores for varying generator widths on fake data', y=1, fontsize=12)
    plt.savefig(save_dir + "/disc_score_x_disc.png",
                dpi=None, format="png",)
    plt.show()
    hands = []
    # scores +- stds generator
    j = 0
    l = 0
    subplt_len = len(proportions) // 4
    fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True,
                           sharey=True, figsize=(8, 5))
    for k in range(4):
        hands = []
        l += 1
        means = overall_means.transpose()[(l - 1) * subplt_len:l * subplt_len]
        stds = overall_stds.transpose()[(l - 1) * subplt_len:l * subplt_len]
        plt.subplot(2, 2, l)
        for i, mu in enumerate(means):
            repr = "-" + markers[i]
            std = stds[i]
            hands.append(plt.plot(indexes, mu, repr, lw=2,
                                  label="D dim = " + str(indexes[i + subplt_len * (l - 1)]))[0])
            plt.fill_between(indexes, mu + std, mu - std, alpha=0.5)
        plt.legend(handles=hands, prop={'size': legend_size})
        plt.grid(True)
        plt.xlim(0, 130)
    fig.text(0.5, 0.01, "Width of generator network",
             ha='center', fontsize=12)
    plt.tight_layout()
    plt.suptitle(
        'Discriminator scores for varying generator widths on fake data', y=1, fontsize=12)
    plt.savefig(save_dir + "/disc_score_x_gen.png",
                dpi=None, format="png",)
    plt.show()


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Add min max generator indexes to process')
    # Allow parallelization of the script
    parser.add_argument("--real_stats", type=str, nargs='?',
                        const=True, default="asd",
                        help="Process real stats?")
    args = parser.parse_args()
    real_stats = str2bool(args.real_stats)
    if real_stats:
        real_data_test()
    else:
        fake_data_test()
