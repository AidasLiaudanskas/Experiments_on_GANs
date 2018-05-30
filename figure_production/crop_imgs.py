import os
from PIL import Image


def crop_64s():
    image_dir = "/home/aidas/GAN_data/lime_all/outputs/filtered_lime_run5"
    imgs_in_dir = os.listdir(image_dir)
    imgs_in_dir = [x for x in imgs_in_dir if ("G_64" in x)]
    img_path = os.path.join(image_dir, imgs_in_dir[0])
    img_path
    img = Image.open(img_path)
    w, h = img.size
    top_half = (0, 0, w, h // 2)
    cropped_img = img.crop(top_half)
    save_name = image_dir + "/top_G_64.png"
    cropped_img.save(save_name)
    # cropped_img.show()
    for file in imgs_in_dir:
        img_path = os.path.join(image_dir, file)
        img = Image.open(img_path)
        w, h = img.size
        # The crop rectangle, as a (left, upper, right, lower)-tuple.
        bottom_half = (0, h // 2, w, h)
        cropped_img = img.crop(bottom_half)
        word_parts = file.split(".")[0].split("_")
        d_extension = word_parts[-2] + "_" + word_parts[-1]
        if "norm" in file:
            save_name = image_dir + "/cropped/bot_norm_G_64_" + d_extension + ".png"
        else:
            save_name = image_dir + "/cropped/bot_G_64_" + d_extension + ".png"
        # print(save_name)
        cropped_img.save(save_name)


def crop_reals():
    image_dir = "/home/aidas/GAN_data/lime_all/outputs/filtered_lime_run5"
    imgs_in_dir = os.listdir(image_dir)
    imgs_in_dir = [x for x in imgs_in_dir if ("real" in x)]
    img_path = os.path.join(image_dir, imgs_in_dir[0])
    img = Image.open(img_path)
    w, h = img.size
    top_half = (0, 0, w, h // 2)
    cropped_img = img.crop(top_half)
    save_name = image_dir + "/top_real.png"
    cropped_img.save(save_name)
    # cropped_img.show()
    for file in imgs_in_dir:
        img_path = os.path.join(image_dir, file)
        img = Image.open(img_path)
        w, h = img.size
        # The crop rectangle, as a (left, upper, right, lower)-tuple.
        bottom_half = (0, h // 2, w, h)
        cropped_img = img.crop(bottom_half)
        word_parts = file.split(".")[0].split("_")
        d_extension = word_parts[-2] + "_" + word_parts[-1]
        if "norm" in file:
            save_name = image_dir + "/cropped/bot_norm_real_" + d_extension + ".png"
        else:
            save_name = image_dir + "/cropped/bot_real" + d_extension + ".png"
        cropped_img.save(save_name)


if __name__ == '__main__':
    crop_64s()
    crop_reals()
