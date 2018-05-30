from scipy.misc import imread
from random import shuffle
import time
from tqdm import tqdm

import tensorflow as tf
from glob import glob
from utils import get_image, colorize
import matplotlib.pyplot as plt
# this is based on tensorflow tutorial code
# https://github.com/tensorflow/tensorflow/blob/r0.8/tensorflow/examples/how_tos/reading_data/convert_to_records.py
# TODO: it is probably very wasteful to store these images as raw numpy
# strings, because that is not compressed at all.
# i am only doing that because it is what the tensorflow tutorial does.
# should probably figure out how to store them as JPEG.

IMSIZE = 64

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def main(argv):
    input_folder = "./train_data_128/*.jpg"
    files = glob(input_folder)
    assert len(files) > 0
    # assert len(files) > 1000000, len(files)
    shuffle(files)

    # dirs = glob("/home/ian/imagenet/ILSVRC2012_img_train_t1_t2/n*")
    # assert len(dirs) == 1000, len(dirs)
    # dirs = [d.split('/')[-1] for d in dirs]
    # dirs = sorted(dirs)
    # str_to_int = dict(zip(dirs, range(len(dirs))))

    outfile = './' + "lsun_bedrooms_64x64" + '.tfrecords'
    writer = tf.python_io.TFRecordWriter(outfile)
    print("Number of files to process:", len(files))
    for f in tqdm(files, desc="Processing images", mininterval=0.5):
        # print (i)
        image = get_image(f, IMSIZE, is_crop=True, resize_w=IMSIZE)
        image = colorize(image)
        assert image.shape == (IMSIZE, IMSIZE, 3)
        image += 1.
        image *= (255. / 2.)
        image = image.astype('uint8')
        print( image.min(), image.max())
        wait = input("PRESS ENTER TO CONTINUE.")
        # from pylearn2.utils.image import save
        # save('foo.png', (image + 1.) / 2.)
        # if (i%1000==0):
            # print(i)
            # imgplot = plt.imshow(image)
            # plt.show()
        image_raw = image.tostring()
        # class_str = f.split('/')[-2]
        # label = str_to_int[class_str]
        # if i % 1 == 0:
        #     print( i, '\t',label)
        example = tf.train.Example(features=tf.train.Features(feature={
            # 'height': _int64_feature(IMSIZE),
            # 'width': _int64_feature(IMSIZE),
            # 'depth': _int64_feature(3),
            'image_raw': _bytes_feature(image_raw),
            # 'label': _int64_feature(label) # No need for labels in this case
            }))
        writer.write(example.SerializeToString())

    writer.close()

if __name__ == "__main__":
    tf.app.run()
