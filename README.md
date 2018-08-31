# GAN Experiments for CUED IIB Project

This is the code I wrote for my Master's thesis. This is a public branch, meaning it's stripped of all the bigger files (datasets, models, summaries, outputs). Thesis abstract describing the experiments is in the pdf file.


What it does:
Code for training diffrent versions of GANs with few different architecture for Generator and Discriminator
The code is built on a fork of [this repository](https://github.com/igul222/improved_wgan_training).
Inspiration for the code style/structure is taken from [here](https://github.com/tdeboissiere/DeepLearningImplementations/tree/master/WGAN-GP) and [here](https://github.com/YuguangTong/improved_wgan_training).
Code is written in Python3 and has no python2 support at the moment.

<!-- ## To get started:
To initiate a compatible docker this command should work:
```
 nvidia-docker run -it -p <your_port_for_tensorboard>:6006 --name <your_name> -w /root -v /share/Downloads:/share/Downloads -v $HOME:/root/<your_home_dir> -v /share/logs:/share/logs -v /share/models:/share/models gcr.io/tensorflow/tensorflow:1.3.0-gpu-py3 bash
``` -->
Install python dependencies (virtual environment recommended):
```
pip install tqdm colorama sklearn scipy Pillow matplotlib
```
TensorFlow installation assumed. This code was tested on TF 1.4 and 1.5


Most of the training parameters are set in the `flags.py` file.

<!-- Can be used with the command line as well if you prefer it this way. -->
<!--
At the moment, the architecture for the Discriminator and Generator should be manually chosen in the `DandG.py`. The file is self explanatory. -->

The codebase is designed to train many models in a row, the hiperparameter set can be tinkered with in the file mix_and_match.py

The architectures for G and D are defined in file DCGANS.py

To start training all the models with hyperparameters specified (one by one, in sequence), run:
```
python3 mix_and_match.py
```
<!-- With architectures to train specified in mix_and_match.py -->

It trains many architectures in a row by spawning separate processes to avoid TensorFlow's complaints about variables being on the same graph.

Some files are obsolete, as the main goal of the codebase changed with time.

<!-- [Description of the original repository](https://github.com/igul222/improved_wgan_training) -->
======================================


## Prerequisites

- Python, NumPy, TensorFlow, SciPy, Matplotlib
- A recent NVIDIA GPU
- Datasets in ./data folder. They must be in tfrecords format, resolutions specified in flags.py


## How to run the experiments:

1. Specify desired hyperparameters in mix_and_match.py. Archietures can be changed in DCGANS.py
2. Wait for training to finish.
3. Run score_avg_test.py to calculate mean offsets (real_fake_half.npz).
4.  Run half_real_fake_stats_proc.py  
5.  Run real_images_test.py
6. Run mix_and_match_test.py to see which discriminators were fooled
7. Generate figures
8. Generate LIME explanations

Final check before generating all stat figures:
Need 4 .npz files:
* evaluation_stats.npz (by mix_and_match_test.py)
* mean_offsets.npz (by half_real_fake_stats_proc.py)
* real_evaluation_stats.npz (by real_images_test.py)
* real_fake_half.npz (by score_avg_test.py)





### Latest Updates:
fid.py in minst test is the most up to date - fixed inefficient loading.

same with mix_and_match_test.py -- fixed memory hogging.

mnist test has MMD hanging problem fixed.
