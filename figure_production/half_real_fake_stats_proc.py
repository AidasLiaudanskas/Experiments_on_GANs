"""
TL;DR:
Calculates mean offsets.

Explanation:
WGAN-GP discriminator's scalar outputs are not symmetric about 0.
This script find's the average scalar value for certain discriminator by:
Averaging the outputs of 20 purely real and 20 purely fake batches.
"""

import os
import numpy as np
import matplotlib
import matplotlib.colors as colors

# matplotlib.use('Agg')
import matplotlib.pyplot as plt
np.set_printoptions(precision=2, linewidth=150)


def real_data_test():
    # TODO: Get rid of generator graphs.
    # os.chdir("/home/aidas/Experiments_on_GANs/figure_production")
    current_dir = os.getcwd()
    print("Current_dir = ", current_dir)
    # os.chdir("/home/aidas/GAN_Experiments/progressive_test")
    stats_file = "./real_fake_half.npz"
    f = np.load(stats_file)
    f.keys()
    results_tensor = f["results_tensor"]
    side_length = int(np.sqrt(results_tensor.shape[0]))
    # side_length
    results_tensor = results_tensor.reshape([side_length, side_length, 6])
    save_files = f["save_files"]
    indexes = [int(x.split("_")[1]) for x in save_files]
    # indexes[10]
    # data = results_tensor[14]
    # save_files[14]
    # data
    # The data is stored like fake, real, half means
    # fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=False)
    # Generators Graph
    width = 3.0
    labels = ["fake", "real", "half mixed"]
    # for j in range(3):
    #     # plt.subplot(3, 1, j+1)
    #     # for i, ind in enumerate(indexes):
    #     mu = data[:,j]
    #     std = data[:,j+3]
    #     plt.plot(indexes, mu, '-o',  linewidth=width , label =labels[j])
    #     plt.fill_between(indexes, mu + std, mu - std, alpha=0.5)
    #     # plt.xlabel("Width of discriminator network")
    #     plt.ylabel("Proportion of times discriminator judges the image as real")
    #     plt.title('Discriminator Output for different network widths on real data')
    #     plt.grid(True)

    # data = results_tensor[15]
    # mu = data[:,1]
    # std = data[:,1+3]
    # plt.plot(indexes, mu, '-o',  linewidth=width , label =labels[j])
    # plt.fill_between(indexes, mu + std, mu - std, alpha=0.5)
    #
    # data = results_tensor[10]
    # mu = data[:,1]
    # std = data[:,1+3]
    # plt.plot(indexes, mu, '-o',  linewidth=width , label =labels[j])
    # plt.fill_between(indexes, mu + std, mu - std, alpha=0.5)

    # plt.legend()
    # # plt.xticks([])
    # plt.show()

    def sigmoid(x):
        return (1 / (1 + np.exp(-x)))
    big_id = side_length//2
    print("Gen width = {}:".format(indexes[big_id]))
    for j in range(side_length):
        real_mean = np.mean(results_tensor[big_id][j][1])
        fake_mean = np.mean(results_tensor[big_id][j][0])
        half = np.mean(results_tensor[big_id][j][2])
        avg_mean = (real_mean + fake_mean) / 2
        sg = sigmoid(avg_mean)
        print("Disc_width = {}; Avg = {:.2f}; real = {:.2f}; fake = {:.2f}; half = {:.2f};  sigmoid_ref = {:.5f}".format(
            indexes[j], avg_mean, real_mean, fake_mean, half, sg))
    # marginalise the mean over all generators (first dimension) :
    # for i in range(20):
    avg_means = (results_tensor[:, :, 0] + results_tensor[:, :, 1]) / 2
    avg_means.shape
    print(avg_means)
    print(np.mean(avg_means, axis=0))
    marginalized_means = np.mean(avg_means, axis=0)

    # np.mean(results_tensor[16][10:][2])
    # np.mean(results_tensor[17][10:][2])
    # np.mean(results_tensor[18][10:][2])
    avg_means = (results_tensor[:, :, 0] + results_tensor[:, :, 1]) / 2
    avg_means.shape
    avg_means[14, 14]
    output_path = './mean_offsets.npz'
    to_print = np.around(avg_means, 1)
    np.savez_compressed(output_path, means_matrix=avg_means,
                        save_files=save_files)
    # print(" \\\\ \\hline \n ".join([" & ".join(map(str,line)) for line in to_print]))
    try:
        plt.imshow(avg_means, cmap='plasma', interpolation='nearest', norm=colors.SymLogNorm(linthresh=1,
        vmin=np.min(avg_means), vmax=np.max(avg_means)),)
        plt.colorbar()
        plt.xticks(range(len(indexes)), indexes, fontsize=12)
        plt.yticks(range(len(indexes)), indexes, fontsize=12)
        plt.xlabel("Discriminator's width")
        plt.ylabel("Generator's width")
        plt.show()
    except Exception as e:
        print("Trying to display figures without a screen.")


if __name__ == '__main__':
    real_data_test()
